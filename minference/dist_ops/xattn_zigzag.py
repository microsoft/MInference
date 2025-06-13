import os
import math
import torch
import triton
import torch.distributed as dist
from typing import List, Tuple, Dict, Any, Optional

from .utils import (
    RingComm, update_out_and_lse,
    shuffle_zigzag_input, recover_zigzag_output,
    shuffle_block_mask_zigzag,
)

from minference.ops.op_utils.xattn_utils import LN2, find_blocks_chunked
from minference.ops.op_utils.vertical_slash_utils import convert_blockmask
from minference.ops.pit_sparse_flash_attention_v3 import block_attn_fwd, block_attn_bwd
from minference.ops.xattention_fa import flat_group_gemm_fuse_reshape, softmax_fuse_block_sum
from minference.ops.pit_sparse_flash_attention_v3_triton import triton_block_attn_fwd, triton_block_attn_bwd


def xattn_zigzag_estimate(
    query_states: torch.Tensor, # (batch_size, num_q_head, q_len, head_dim)
    key_states: torch.Tensor, # (batch_size, num_kv_head, k_len, head_dim)
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    group: dist.group = None,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len_local, head_dim = key_states.shape
    batch_size, num_q_head, q_len_local, head_dim = query_states.shape

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    k_gather_list = [torch.empty_like(key_states) for _ in range(world_size)]   
    dist.all_gather(k_gather_list, key_states.contiguous(), group=group)
    k_gathered = torch.cat(k_gather_list, dim=2)
    k_len = k_gathered.shape[2]

    if num_q_head > num_kv_head:
        k_gathered = torch.repeat_interleave(k_gathered.contiguous(), num_q_head // num_kv_head, dim=1)

    chunk_size = q_len_local // 2
    q_chunk_num = 2
    q_block_num = q_len_local // block_size
    q_block_num_per_chunk = chunk_size // block_size

    # assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    num_strides_in_k = k_len // stride
    num_strides_per_chunk = chunk_size // stride
    num_strides_per_block = block_size // stride
    num_blocks_per_chunk = num_strides_per_chunk // num_strides_per_block

    attn_weight_slices = [None, None]
    for chunk_idx in range(q_chunk_num):
        global_chunk_idx = rank * 2 + chunk_idx

        # Local start index
        q_chunk_start = chunk_idx * chunk_size
        q_chunk_end =  (chunk_idx + 1) * chunk_size

        # Global start index (stride-level)
        q_chunk_start_stride_global = global_chunk_idx * num_strides_per_chunk
        q_chunk_end_stride_global = (global_chunk_idx + 1) * num_strides_per_chunk

        # attn_weights_slice: (batch_size, num_heads, chunk_size // stride, kv_len // stride)
        # (i.e. the attention sum of each SxS stride block)
        # This step is agnostic to block size and just computes the attention sum in each stride block
        attn_weight_slice = flat_group_gemm_fuse_reshape(
            # query_states, key_states, stride, chunk_start, chunk_end, is_causal=True
            query_states[:, :, q_chunk_start : q_chunk_end, :,],
            k_gathered,
            stride,
            q_chunk_start_stride_global, q_chunk_end_stride_global,
            is_causal=causal,
        )
        attn_weight_slices[chunk_idx] = attn_weight_slice
    del k_gathered, k_gather_list

    for chunk_idx in range(q_chunk_num):
        global_chunk_idx = rank * 2 + chunk_idx
        
        # Local start index
        q_chunk_start = chunk_idx * chunk_size
        q_chunk_end =  (chunk_idx + 1) * chunk_size

        # Global start index (block-level)
        q_block_start = global_chunk_idx * q_block_num_per_chunk
        q_block_end   = (global_chunk_idx + 1) * q_block_num_per_chunk

        # Global start index (stride-level)
        q_chunk_start_stride_global = global_chunk_idx * num_strides_per_chunk
        q_chunk_end_stride_global = (global_chunk_idx + 1) * num_strides_per_chunk

        attn_weight_slice = attn_weight_slices[chunk_idx]

        # (batch_size, num_heads, q_block_num, k_block_num),
        attn_sum = softmax_fuse_block_sum(
            attn_weight_slice, # (batch_size, num_heads, chunk_size // stride, kv_len // stride)
            num_strides_per_block,
            min(4096, num_strides_per_block),
            q_chunk_start_stride_global, q_chunk_end_stride_global,
            num_strides_in_k,
            1 / LN2 / math.sqrt(head_dim) / stride / norm,
            is_causal=causal,
        )
        
        # (batch_size, head_num, num_blocks_per_chunk, block_num)
        simple_mask = find_blocks_chunked(
            attn_sum,
            global_chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        del attn_weight_slice
        if causal:
            simple_mask[:, :, :, q_block_start:q_block_end] = torch.where(
                torch.tril(
                    torch.ones(
                        q_block_num_per_chunk, q_block_num_per_chunk, 
                        dtype=bool, device=key_states.device
                    ),
                    diagonal=0,
                ),
                simple_mask[:, :, :, q_block_start:q_block_end],
                False,
            )
            simple_mask[:, :, :, q_block_end:] = 0
        if keep_sink:
            simple_mask[:, :, 0, :] = True
        if keep_recent:
            eye_matrix = torch.eye(q_block_num_per_chunk, device=simple_mask.device, dtype=bool)
            eye_matrix_expanded = (
                eye_matrix.unsqueeze(0)
                .unsqueeze(0)
                .expand(1, num_kv_head, q_block_num_per_chunk, q_block_num_per_chunk)
            )
            simple_mask[:, :, :, q_block_start:q_block_end] = torch.where(
                eye_matrix_expanded, True, simple_mask[:, :, :, q_block_start:q_block_end]
            )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2) # (batch_size, head_num, q_local_block_num, k_global_block_num)
    return attn_sums, simple_masks

def use_triton():
    return torch.version.hip is not None or os.getenv("FORCE_TRITON", "0") == "1"

def xattn_zigzag_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    block_mask: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    layer_idx: int,
    softmax_scale: float,
    granularity: int = 128,
    block_idx: Optional[torch.Tensor] = None,
    block_cnt: Optional[torch.Tensor] = None,
):
    comm = RingComm(process_group, zigzag=True)
    out, lse = None, None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        # [batch_size, num_qo_heads, num_blocks_local, num_blocks_local]
        block_mask_step = block_mask[step]
        block_causal = step == 0

        if use_triton():
            # TODO: block_mask here needs to be converted to block_idx before passing to triton
            block_out, block_lse = triton_block_attn_fwd(
                q, k, v, 
                block_idx=block_idx[step], block_cnt=block_cnt[step],
                softmax_scale=softmax_scale,
                granularity=granularity,
                causal=block_causal,
                step=step,
            )
        else:
            block_out, block_lse = block_attn_fwd(
                q, k, v, 
                block_mask=block_mask_step,
                softmax_scale=softmax_scale,
                granularity=granularity,
                causal=block_causal,
                step_idx=step,
            )

        out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

def xattn_zigzag_backward(
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
    granularity: int = 128,
    block_idx: Optional[torch.Tensor] = None, # [world_size, batch_size, num_qo_heads, num_blocks_local, num_blocks]
    block_cnt: Optional[torch.Tensor] = None, # [world_size, batch_size, num_qo_heads, num_blocks_local]
):
    kv_comm = RingComm(process_group, zigzag=True)
    d_kv_comm = RingComm(process_group, zigzag=True)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        block_causal = step == 0
        block_mask_step = block_mask[step]

        # --------------------------------
        # Block Mask
        if use_triton():
            step_dq, step_dk, step_dv = triton_block_attn_bwd(
                dout, q, k, v, out,
                softmax_lse, softmax_scale,
                block_idx[step], block_cnt[step],
                granularity=granularity,
                deterministic=False,
                causal=block_causal,
                step=step,
            )
        else:
            step_dq, step_dk, step_dv = block_attn_bwd(
                dout, q, k, v, out,
                softmax_lse, softmax_scale,
                block_mask_step,
                granularity=granularity,
                deterministic=False,
                causal=block_causal,
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

class XAttnZigzagFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx, 
        xattn_params, # Dict[str, Any] 
        granularity,
        causal,
        softmax_scale,
        return_softmax,
        deterministic,
        group,
    ):
        if softmax_scale is None: softmax_scale = q.shape[-1] ** (-0.5)

        # ----------------------------------------------
        # Index Building
        # block_mask [batch_size, num_qo_heads, num_blocks_local, num_blocks]
        _, block_mask = xattn_zigzag_estimate(
            q.transpose(1, 2), k.transpose(1, 2),
            block_size=granularity,
            **xattn_params
        )

        # ------------------------------------------------------------------
        # QKV Shuffling
        q = shuffle_zigzag_input(to_send=q, dim=1, process_group=group)
        k = shuffle_zigzag_input(to_send=k, dim=1, process_group=group)
        v = shuffle_zigzag_input(to_send=v, dim=1, process_group=group)

        # ------------------------------------------------------------------
        # Index Shuffling
        block_mask = shuffle_block_mask_zigzag(
            block_mask, num_blocks_per_chunk=q.shape[1] // 2 // granularity,
            group=group
        ).to(q.device)
        if use_triton():
            block_idx, block_cnt = convert_blockmask(block_mask, block_size_M=granularity, block_size_N=64)
        else:
            block_idx, block_cnt = None, None 
        block_mask = block_mask.contiguous()

        # ----------------------------------------------
        # Compute 
        out, softmax_lse = xattn_zigzag_forward(
            group,
            q, k, v,
            block_mask,
            layer_idx,
            softmax_scale,
            granularity=granularity,
            block_idx=block_idx, block_cnt=block_cnt,
        )

        # ----------------------------------------------
        # Recover outputs
        recovered_out = recover_zigzag_output(out, dim=1,  process_group=group)
        if return_softmax:
            recovered_softmax_lse = recover_zigzag_output(softmax_lse, dim=2, process_group=group)

        # -------------------------------
        # Variale Saving
        if use_triton():
            ctx.save_for_backward(q, k, v, out, softmax_lse, block_mask, block_idx, block_cnt)
        else:
            ctx.save_for_backward(q, k, v, out, softmax_lse, block_mask)
        ctx.softmax_scale = softmax_scale
        ctx.granularity = granularity
        ctx.group = group
        ctx.layer_idx = layer_idx

        # -------------------------------
        # Recover outputs
        if return_softmax:
            return (recovered_out, recovered_softmax_lse, None)
        return recovered_out

    @staticmethod
    def backward(ctx, dout, *args):
        if use_triton():
            q, k, v, out, softmax_lse, block_mask, block_idx, block_cnt = ctx.saved_tensors
        else:
            q, k, v, out, softmax_lse, block_mask = ctx.saved_tensors
            block_idx, block_cnt = None, None
        softmax_scale = ctx.softmax_scale
        granularity = ctx.granularity
        layer_idx = ctx.layer_idx
        group = ctx.group


        dout = shuffle_zigzag_input(to_send=dout, dim=1, process_group=group) 

        # ----------------------------------------------
        # Compute
        dq, dk, dv = xattn_zigzag_backward(
            group,
            dout, q, k, v, 
            out, softmax_lse,
            layer_idx, 
            softmax_scale,
            block_mask,
            granularity,
            block_idx=block_idx, block_cnt=block_cnt,
        )

        dq = recover_zigzag_output(dq, dim=1, process_group=group)
        dk = recover_zigzag_output(dk, dim=1, process_group=group)
        dv = recover_zigzag_output(dv, dim=1, process_group=group)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def xattn_zigzag_qkvpacked_func(
    qkv: torch.Tensor,  # [batch_size, num_tokens, 3, num_heads, head_dim]
    layer_idx: int,
    xattn_params: Dict[str, Any], 
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
):
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic
    return XAttnZigzagFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        layer_idx,
        xattn_params,
        granularity,
        causal,
        softmax_scale,
        return_attn_probs,
        deterministic,
        group,
    )


def xattn_zigzag_kvpacked_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    kv: torch.Tensor,  # [batch_size, num_tokens, 2, num_heads, head_dim]
    layer_idx: int,
    xattn_params: Dict[str, Any], 
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
):
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic

    return XAttnZigzagFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        layer_idx,
        xattn_params,
        granularity,
        causal,
        softmax_scale,
        return_attn_probs,
        deterministic,
        group,
    )


def xattn_zigzag_func( # the one used for nnscaler training
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    layer_idx: int,
    xattn_params: Dict[str, Any], 
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
) -> torch.Tensor:
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic

    return XAttnZigzagFunc.apply(
        q, k, v,
        layer_idx,
        xattn_params,
        granularity,
        causal,
        softmax_scale,
        return_attn_probs,
        deterministic,
        group,
    )
