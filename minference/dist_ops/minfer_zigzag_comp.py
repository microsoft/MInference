import os
import torch
import triton
import torch.distributed as dist

from typing import List, Tuple, Dict

from MTraining.ops.ring_attn.core.utils import (
    RingComm, TIMING_LOGGER,
    shuffle_zigzag_input, recover_zigzag_output,
    single_gather_tensor, shuffle_block_mask_zigzag
)
from MTraining.ops.minfer import (
    block_attn_fwd, block_attn_bwd, bar_attn_fwd, bar_attn_bwd, block_bar_attn_fwd, convert_blockmask,
    minference_flash_attn_func, build_index
) 


def compute_sr_flops(
    block_mask_offset: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, num_blocks]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    step: int,
    granularity: int,
    q_len: int,
    head_dim: int,
    fwd: bool=True,
):
    num_blocks = triton.cdiv(q_len, granularity)
    bh = block_mask_offset.shape[0] * block_mask_offset.shape[1]

    # -------------------------------------
    # Block Compute
    total_num_blocks = bh * num_blocks * num_blocks / 2
    num_active_blocks = block_mask_offset.sum(dtype=torch.float32).item()
    if step == 0: 
        num_active_blocks -= bh * num_blocks / 2
    block_ratio = num_active_blocks / total_num_blocks
    block_flops = num_active_blocks * (granularity * granularity) * head_dim * 2 * 2

    # -------------------------------------
    # Bar Compute
    bar_cnt_step = (bar_cnt[..., step + 1] - bar_cnt[..., step]).sum(dtype=torch.float32).item()
    bar_ratio = bar_cnt_step / (granularity * total_num_blocks)
    bar_flops = bar_cnt_step * granularity * head_dim * 2 * 2

    # -------------------------------------
    # Sparsity Ratio and FLOPs
    sparsity_ratio = 1 - block_ratio - bar_ratio
    flops = block_flops + bar_flops

    if not fwd: 
        flops, block_flops, bar_flops = 2.5 * flops, 2.5 * block_flops, 2.5 * bar_flops
    return block_ratio, bar_ratio, sparsity_ratio, block_flops, bar_flops, flops

def compute_sr_by_heads(
    block_mask_offset: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, num_blocks]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    step: int,
    granularity: int,
    q_len: int,
):
    batch_size, num_heads = block_mask_offset.shape[0], block_mask_offset.shape[1]
    num_blocks = triton.cdiv(q_len, granularity)
    
    # -------------------------------------
    # Block Compute
    total_num_blocks = batch_size * num_blocks * num_blocks / 2
    total_num_blocks_by_heads = torch.tensor([total_num_blocks for _ in range(num_heads)], dtype=torch.float32).to(block_mask_offset.device)
    num_active_blocks = block_mask_offset.sum(-1).sum(-1).sum(0, dtype=torch.float32) # [num_qo_heads]
    if step == 0:
        num_active_blocks -= batch_size * num_blocks / 2
    block_ratio_by_heads = num_active_blocks / total_num_blocks_by_heads

    # -------------------------------------
    # Bar Compute
    bar_cnt_step = (bar_cnt[..., step + 1] - bar_cnt[..., step]).sum(dim=-1).sum(dim=-1).sum(0, dtype=torch.float32) # [num_qo_heads]
    bar_ratio_by_heads = bar_cnt_step / total_num_blocks_by_heads / granularity 

    # -------------------------------------
    # Sparsity Ratio
    sparsity_ratio_by_heads = 1 - block_ratio_by_heads - bar_ratio_by_heads
    return sparsity_ratio_by_heads.detach().cpu().numpy().tolist()

def minfer_zigzag_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    layer_idx: int,
    softmax_scale: float,
    block_mask: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int = 128,
):
    comm = RingComm(process_group, zigzag=True)
    ring_list = comm.ring_list
    ring_index = ring_list.index(comm.rank)

    out, lse = None, None
    block_idx, block_cnt = convert_blockmask(block_mask, block_size_M=granularity, block_size_N=64)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        block_causal = step == 0
        offset = (ring_index - step) % comm.world_size

        out, lse = block_bar_attn_fwd(
            q, k, v, out, lse, softmax_scale,
            bar_idx, bar_cnt, block_idx[offset], block_cnt[offset],
            granularity=granularity,
            step=offset,
            causal=block_causal,
        )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    # lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def minfer_zigzag_backward(
    process_group: dist.ProcessGroup,
    dout: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    out: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim] 
    softmax_lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    layer_idx: int,
    softmax_scale: float,
    block_mask: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int = 128,
):
    kv_comm = RingComm(process_group, zigzag=True)
    d_kv_comm = RingComm(process_group, zigzag=True)
    ring_list = kv_comm.ring_list
    ring_index = ring_list.index(kv_comm.rank)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
        block_causal = step == 0
        offset = (ring_index - step) % kv_comm.world_size

        # Block Mask
        step_dq, step_dk, step_dv = block_attn_bwd(
            dout, q, k, v, out,
            softmax_lse, softmax_scale,
            block_mask[offset],
            granularity=granularity,
            deterministic=False,
            causal=block_causal,
        )

        # Bar Mask
        step_dq, step_dk, step_dv = bar_attn_bwd(
            dout, q, k, v, out, step_dq, step_dk, step_dv,
            softmax_lse, softmax_scale,
            bar_idx, bar_cnt,
            granularity=granularity,
            deterministic=False,
            step=offset,
        )

        # Update dQ, dK, dV
        if step == 0:
            # TODO: check if float32 is necessary
            dq = step_dq.to(torch.float32)
            dk = step_dk.to(torch.float32)
            dv = step_dv.to(torch.float32)
        else:
            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            dq += step_dq
            dk += step_dk
            dv += step_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v
        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class MInferZigzagAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        v_size,
        s_size,
        layer_idx,
        softmax_scale,
        granularity,
        return_softmax,
        group,
    ):
        if softmax_scale is None: softmax_scale = q.shape[-1] ** (-0.5)
        batch_size, num_tokens_local, num_qo_heads, head_dim = q.shape

        # Indexing
        TIMING_LOGGER.start('indexing')
        block_mask, bar_idx, bar_cnt, bar_pos, v_idx, v_cnt = build_index(
            q, k, v_size, s_size, num_tokens_local, 
            stripe_transform=False,
            zigzag_transform=True,
            granularity=granularity, group=group
        )
        TIMING_LOGGER.end('indexing')

        # Shuffle
        TIMING_LOGGER.start('shfl-fwd-input')
        q = shuffle_zigzag_input(to_send=q, dim=1, process_group=group)
        k = shuffle_zigzag_input(to_send=k, dim=1, process_group=group)
        v = shuffle_zigzag_input(to_send=v, dim=1, process_group=group)
        TIMING_LOGGER.end('shfl-fwd-input')

        # Compute
        TIMING_LOGGER.start('forward')
        out, softmax_lse = minfer_zigzag_forward(
            group, q, k, v, 
            layer_idx, softmax_scale,
            block_mask, bar_idx, bar_cnt,
            granularity=granularity,
        )
        TIMING_LOGGER.end('forward')

        # Saving tensors for backward
        ctx.save_for_backward(q, k, v, out, softmax_lse, block_mask, bar_idx, bar_cnt)
        ctx.softmax_scale = softmax_scale
        ctx.granularity = granularity
        ctx.group = group
        ctx.layer_idx = layer_idx

        # Recover outputs
        TIMING_LOGGER.start('shfl-fwd-output')
        out = recover_zigzag_output(out, dim=1, process_group=group)
        if return_softmax:
            softmax_lse = recover_zigzag_output(softmax_lse, dim=2, process_group=group)
        TIMING_LOGGER.end('shfl-fwd-output')

        # Output and Return
        if return_softmax:
            return (out, softmax_lse, None)
        return out

    @staticmethod
    def backward(ctx, dout, *args):        
        q, k, v, out, softmax_lse, block_mask, bar_idx, bar_cnt = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        granularity = ctx.granularity
        layer_idx = ctx.layer_idx
        group = ctx.group

        # Shuffle
        TIMING_LOGGER.start('shfl-bwd-input')
        dout = shuffle_zigzag_input(to_send=dout, dim=1, process_group=group)
        TIMING_LOGGER.end('shfl-bwd-input')

        # Compute
        TIMING_LOGGER.start('backward')
        dq, dk, dv = minfer_zigzag_backward(
            group, dout, q, k, v, out, softmax_lse,
            layer_idx, softmax_scale,
            block_mask, bar_idx, bar_cnt,
            granularity=granularity,
        )
        TIMING_LOGGER.end('backward')

        # Recover
        TIMING_LOGGER.start('shfl-bwd-output')
        dq = recover_zigzag_output(dq, dim=1, process_group=group)
        dk = recover_zigzag_output(dk, dim=1, process_group=group)
        dv = recover_zigzag_output(dv, dim=1, process_group=group)
        TIMING_LOGGER.end('shfl-bwd-output')

        return dq, dk, dv, None, None, None, None, None, None, None


def minfer_zigzag_qkvpacked_func(
    qkv: torch.Tensor,  # [batch_size, num_tokens, 3, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    layer_idx: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[int, int] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
):
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic
    return MInferZigzagAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        v_size,
        s_size,
        layer_idx,
        softmax_scale,
        granularity,
        return_attn_probs,
        group,
    )


def minfer_zigzag_kvpacked_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    kv: torch.Tensor,  # [batch_size, num_tokens, 2, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    layer_idx: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[int, int] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
):
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic
    return MInferZigzagAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        v_size,
        s_size,
        layer_idx,
        softmax_scale,
        granularity,
        return_attn_probs,
        group,
    )


def minfer_zigzag_func( # the one used for nnscaler training
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    layer_idx: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[int, int] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
) -> torch.Tensor:
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic

    return MInferZigzagAttnFunc.apply(
        q,
        k,
        v,
        v_size,
        s_size,
        layer_idx,
        softmax_scale,
        granularity,
        return_attn_probs,
        group,
    )
