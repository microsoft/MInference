import os
import torch
import torch.distributed as dist
from typing import List, Tuple, Dict

from .utils import (
    RingComm, 
    shuffle_striped_input, recover_striped_output,
)
from minference.ops.utils import build_index, convert_blockmask
from minference.ops.minference_attn_triton import block_bar_attn_fwd, block_bar_attn_bwd


def sparse_stripe_flash_attn_triton_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    layer_idx: int,
    softmax_scale: float,
    block_idx: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks]
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int = 128,
):
    comm = RingComm(process_group)
    out, lse = None, None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        block_causal = step == 0
        offset = (comm.rank - step) % comm.world_size

        
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

    return out, lse


def sparse_stripe_flash_attn_triton_backward(
    process_group: dist.ProcessGroup,
    dout: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    out: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    softmax_lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    layer_idx: int,
    softmax_scale: float,
    block_idx: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks]
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int = 128,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
        block_causal = step == 0
        offset = (kv_comm.rank - step) % kv_comm.world_size

        dq, step_dk, step_dv = block_bar_attn_bwd(
            dout, q, k, v, out, dq, None, None,
            softmax_lse, softmax_scale,
            bar_idx, bar_cnt, block_idx[offset], block_cnt[offset],
            granularity=granularity,
            deterministic=False,
            step=offset,
            causal=block_causal,
        )

        # Update dQ, dK, dV
        if step == 0:
            dk = step_dk
            dv = step_dv
        else:
            d_kv_comm.wait()

            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv
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


class SparseStripeFlashAttnTritonFunc(torch.autograd.Function):
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
        batch_size, num_tokens_local, num_qo_heads, head_dim = q.shape
        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        # built block_idx: [world_size, batch_size, num_qo_heads, num_blocks_local, num_blocks_local]
        block_mask, bar_idx, bar_cnt, _, _, _ = build_index(q, k, v_size, s_size, num_tokens_local, granularity=granularity, group=group)
        block_idx, block_cnt = convert_blockmask(block_mask, block_size_M=granularity, block_size_N=64)

        q = shuffle_striped_input(to_send=q, dim=1, granularity=granularity, process_group=group)
        k = shuffle_striped_input(to_send=k, dim=1, granularity=granularity, process_group=group)
        v = shuffle_striped_input(to_send=v, dim=1, granularity=granularity, process_group=group)

        # slash attn
        out, softmax_lse = sparse_stripe_flash_attn_triton_forward(
            group, q, k, v, 
            layer_idx, softmax_scale,
            block_idx, block_cnt, bar_idx, bar_cnt,
            granularity=granularity,
        )

        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse, block_idx, block_cnt, bar_idx, bar_cnt)
        ctx.softmax_scale = softmax_scale
        ctx.granularity = granularity
        ctx.group = group
        ctx.layer_idx = layer_idx

        out = recover_striped_output(out, dim=1, granularity=granularity, process_group=group)
        if return_softmax:
            softmax_lse = recover_striped_output(softmax_lse, dim=2, granularity=granularity, process_group=group)
            return (out, softmax_lse, None)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        layer_idx = ctx.layer_idx
        dout = shuffle_striped_input(to_send=dout, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        q, k, v, out, softmax_lse, block_idx, block_cnt, bar_idx, bar_cnt = ctx.saved_tensors

        dq, dk, dv = sparse_stripe_flash_attn_triton_backward(
            ctx.group, dout, q, k, v, out, softmax_lse,
            layer_idx, ctx.softmax_scale,
            block_idx, block_cnt, bar_idx, bar_cnt,
            granularity=ctx.granularity,
        )
    
        dq = recover_striped_output(dq, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        dk = recover_striped_output(dk, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        dv = recover_striped_output(dv, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        
        return dq, dk, dv, None, None, None, None, None, None, None


def sparse_stripe_flash_attn_triton_qkvpacked_func(
    qkv: torch.Tensor,  # [batch_size, num_tokens, 3, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = False,
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
    return SparseStripeFlashAttnTritonFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        v_size,
        s_size,
        softmax_scale,
        granularity,
        return_attn_probs,
        group,
    )


def sparse_stripe_flash_attn_triton_kvpacked_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    kv: torch.Tensor,  # [batch_size, num_tokens, 2, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = False,
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
    return SparseStripeFlashAttnTritonFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        v_size,
        s_size,
        softmax_scale,
        granularity,
        return_attn_probs,
        group,
    )


def sparse_stripe_flash_attn_triton_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    layer_idx: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = False,
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

    return SparseStripeFlashAttnTritonFunc.apply(
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
