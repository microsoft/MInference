import os
import torch
import torch.distributed as dist

from typing import List, Tuple

from .utils import (
    RingComm, 
    shuffle_striped_input, recover_striped_output,
    get_inner_ring, get_outer_ring
)
from minference.ops.utils import build_index, convert_blockmask
from minference.ops.minference_attn_triton import block_bar_attn_fwd, block_bar_attn_bwd

def minfer_dr_stripe_triton_forward_inner(
    process_group: dist.ProcessGroup,
    outer_step: int,
    outer_offset: int,
    inner_ring: List[int],
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    out: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    softmax_scale: float,
    block_idx: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks]
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int = 128,
):
    inner_comm = RingComm(process_group, False, inner_ring)
    inner_rank = inner_ring.index(inner_comm.rank)
    num_inner_steps = len(inner_ring)

    next_k, next_v = None, None

    for inner_step in range(num_inner_steps):
        if inner_step + 1 != num_inner_steps:
            next_k, next_v = inner_comm.send_recv_kv(k, v)

        block_causal = (outer_step == 0) and (inner_step == 0)
        offset = outer_offset * num_inner_steps + (inner_rank - inner_step) % num_inner_steps

        out, lse = block_bar_attn_fwd(
            q, k, v, out, lse, softmax_scale,
            bar_idx, bar_cnt, block_idx[offset], block_cnt[offset],
            granularity=granularity,
            step=offset,
            causal=block_causal,
        )

        if inner_step + 1 != num_inner_steps:
            inner_comm.wait()
            k, v = next_k, next_v

    return out, lse


def minfer_dr_stripe_triton_forward_outer(
    process_group: dist.ProcessGroup,
    outer_ring: List[int],
    inner_ring: List[int],
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    softmax_scale: float,
    block_idx: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks]
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int = 128,
):
    outer_comm = RingComm(process_group, False, outer_ring)
    outer_rank = outer_ring.index(outer_comm.rank)
    num_outer_steps = len(outer_ring)

    out = None
    lse = None

    next_k, next_v = None, None
    for outer_step in range(num_outer_steps):
        if outer_step + 1 != num_outer_steps:
            next_k, next_v = outer_comm.send_recv_kv(k, v)

        outer_offset = (outer_rank - outer_step) % num_outer_steps
        out, lse = minfer_dr_stripe_triton_forward_inner(
            process_group, outer_step, outer_offset, inner_ring,
            q, k, v, out, lse, softmax_scale,
            block_idx, block_cnt, bar_idx, bar_cnt,
            granularity,
        )

        if outer_step + 1 != num_outer_steps:
            outer_comm.wait()
            k, v = next_k, next_v

    # out = out.to(q.dtype)
    # lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def minfer_dr_stripe_triton_backward_inner(
    process_group: dist.ProcessGroup,
    outer_step: int,
    outer_offset: int,
    inner_ring: List[int],
    dout: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    out: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    softmax_lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    softmax_scale: float,
    block_idx: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks]
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int = 128,
):
    inner_kv_comm = RingComm(process_group, False, inner_ring)
    inner_d_kv_comm = RingComm(process_group, False, inner_ring)
    inner_rank = inner_ring.index(inner_kv_comm.rank)
    num_inner_steps = len(inner_ring)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    for inner_step in range(num_inner_steps):
        if inner_step + 1 != num_inner_steps:
            next_k, next_v = inner_kv_comm.send_recv_kv(k, v)

        block_causal = (outer_step == 0) and (inner_step == 0)
        offset = outer_offset * num_inner_steps + (inner_rank - inner_step) % num_inner_steps

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
        if inner_step == 0:
            # TODO: check if float32 is necessary
            dk = step_dk.to(torch.float32)
            dv = step_dv.to(torch.float32)
        else:
            inner_d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv
            dk += step_dk
            dv += step_dv

        if inner_step + 1 != num_inner_steps:
            inner_kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = inner_d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    inner_d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


def minfer_dr_stripe_triton_backward_outer(
    process_group: dist.ProcessGroup,
    outer_ring: List[int],
    inner_ring: List[int],
    dout: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    out: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    softmax_lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    softmax_scale: float,
    block_idx: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks]
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int = 128,
):
    outer_kv_comm = RingComm(process_group, False, outer_ring)
    outer_d_kv_comm = RingComm(process_group, False, outer_ring)
    outer_rank = outer_ring.index(outer_kv_comm.rank)
    num_outer_steps = len(outer_ring)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    for outer_step in range(num_outer_steps):
        if outer_step + 1 != num_outer_steps:
            next_k, next_v = outer_kv_comm.send_recv_kv(k, v)

        outer_offset = (outer_rank - outer_step) % num_outer_steps
        step_dq, step_dk, step_dv = minfer_dr_stripe_triton_backward_inner(
            process_group, outer_step, outer_offset, inner_ring,
            dout, q, k, v, out, softmax_lse, softmax_scale,
            block_idx, block_cnt, bar_idx, bar_cnt, granularity,
        )

        if outer_step == 0:
            # TODO: check if float32 is necessary
            dq = step_dq.to(torch.float32)
            dk = step_dk.to(torch.float32)
            dv = step_dv.to(torch.float32)
        else:
            dq += step_dq
            outer_d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv
            dk += step_dk
            dv += step_dv

        if outer_step + 1 != num_outer_steps:
            outer_kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = outer_d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    outer_d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class MInferDRStripeTritonFunc(torch.autograd.Function):
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

        # build index TODO: move convert_indices() into the first step
        block_mask, bar_idx, bar_cnt = build_index(q, k, v_size, s_size, num_tokens_local, granularity=granularity, group=group)
        block_idx, block_cnt = convert_blockmask(block_mask, block_size_M=granularity, block_size_N=64)

        # TODO: remove shuffle
        q = shuffle_striped_input(to_send=q, dim=1, granularity=granularity, process_group=group)
        k = shuffle_striped_input(to_send=k, dim=1, granularity=granularity, process_group=group)
        v = shuffle_striped_input(to_send=v, dim=1, granularity=granularity, process_group=group)

        inner_ring = get_inner_ring(group)
        outer_ring = get_outer_ring(group)
        out, softmax_lse = minfer_dr_stripe_triton_forward_outer(
            group, outer_ring, inner_ring,
            q, k, v, softmax_scale,
            block_idx, block_cnt, bar_idx, bar_cnt, granularity,
        )

        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse, block_idx, block_cnt, bar_idx, bar_cnt)
        ctx.softmax_scale = softmax_scale
        ctx.granularity = granularity
        ctx.group = group
        ctx.inner_ring = inner_ring
        ctx.outer_ring = outer_ring
        ctx.layer_idx = layer_idx

        out = recover_striped_output(out, dim=1, granularity=granularity, process_group=group)
        if return_softmax:
            softmax_lse = recover_striped_output(softmax_lse, dim=2, granularity=granularity, process_group=group)
            return (out, softmax_lse, None)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        dout = shuffle_striped_input(to_send=dout, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        q, k, v, out, softmax_lse, block_idx, block_cnt, bar_idx, bar_cnt = ctx.saved_tensors

        dq, dk, dv = minfer_dr_stripe_triton_backward_outer(
            ctx.group, ctx.outer_ring, ctx.inner_ring,
            dout, q, k, v, out, softmax_lse, ctx.softmax_scale,
            block_idx, block_cnt, bar_idx, bar_cnt, ctx.granularity,
        )
        dq = recover_striped_output(dq, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        dk = recover_striped_output(dk, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        dv = recover_striped_output(dv, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        return dq, dk, dv, None, None, None, None, None, None, None


def minfer_dr_stripe_triton_qkvpacked_func(
    qkv: torch.Tensor,  # [batch_size, num_tokens, 3, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    layer_idx,
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
    return MInferDRStripeTritonFunc.apply(
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


def minfer_dr_stripe_triton_kvpacked_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    kv: torch.Tensor,  # [batch_size, num_tokens, 2, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    layer_idx,
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
    return MInferDRStripeTritonFunc.apply(
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


def minfer_dr_stripe_triton_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    layer_idx,
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
    return MInferDRStripeTritonFunc.apply(
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
