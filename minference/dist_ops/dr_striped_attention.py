import os
import torch
import torch.distributed as dist

from typing import List
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import (
    RingComm, update_out_and_lse, get_default_args, 
    shuffle_striped_input, recover_striped_output
)

def get_inner_group():
    rank = dist.get_rank()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE"))
    inner_group = [i for i in range(local_world_size)]

    rank_offset = (rank // local_world_size) * local_world_size
    inner_group = [rank_offset + i for i in inner_group]

    return inner_group

def get_outer_group(): 
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE"))
    world_size  = dist.get_world_size()

    outer_group = []
    i = local_rank 
    while i < world_size:
        outer_group.append(i)
        i += local_world_size
    
    return outer_group

def stripe_fwd_inner(
    process_group,
    outer_step: int, 
    outer_rank: int,
    inner_ring_list: List[int],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    granularity=1,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    inner_comm = RingComm(process_group, False, inner_ring_list)
    inner_rank = int(os.environ["LOCAL_RANK"])
    num_inner_steps = len(inner_ring_list)

    out, lse = None, None
    next_k, next_v = None, None

    for inner_step in range(num_inner_steps):
        if inner_step + 1 != num_inner_steps:
            next_k, next_v = inner_comm.send_recv_kv(k, v)

        def forward(q_, k_, v_, dropout_p_, softmax_scale_, causal_, alibi_slopes_):
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {
                    "q": q_,
                    "k": k_,
                    "v": v_,
                    "dropout_p": dropout_p_,
                    "softmax_scale": softmax_scale_,
                    "causal": causal_,
                    "alibi_slopes": alibi_slopes_,
                    "return_softmax": True and dropout_p_ > 0,
                }
            )

            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )

            outputs = _flash_attn_forward(**params)
            if len(outputs) == 8:
                block_out, _, _, _, _, block_lse, _, _ = outputs
            else:
                assert len(outputs) == 4
                block_out, block_lse, _, _ = outputs
            return block_out, block_lse

        if outer_step == 0 and inner_step > inner_rank:
            block_out, block_lse = forward(
                q[:, granularity:], k[:, :-granularity], v[:, :-granularity], 
                dropout_p, softmax_scale, causal, alibi_slopes, 
            )
            out, lse = update_out_and_lse(
                out, lse, block_out, block_lse, slice_=(slice(None), slice(granularity, None))
            )
        else:
            block_out, block_lse = forward(
                q, k, v, 
                dropout_p, softmax_scale, causal, alibi_slopes, 
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if inner_step + 1 != num_inner_steps:
            inner_comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

def stripe_fwd_outer(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    granularity=1,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    inner_ring_list: List[int]=None,
    outer_ring_list: List[int]=None,
):
    assert causal, "stripe flash attn only supports causal attention, if not causal, use ring flash attn instead"
    outer_comm = RingComm(process_group, False, outer_ring_list)
    
    global_rank = dist.get_rank()
    outer_rank = outer_ring_list.index(global_rank)
    num_outer_steps = len(outer_ring_list)

    out = None
    lse = None

    next_k, next_v = None, None
    for outer_step in range(num_outer_steps):
        if outer_step + 1 != num_outer_steps:
            next_k, next_v = outer_comm.send_recv_kv(k, v)

        if outer_step <= outer_rank:
            block_out, block_lse = stripe_fwd_inner(
                process_group, outer_step, outer_rank, inner_ring_list, 
                q, k, v,
                softmax_scale,
                granularity,
                dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            # Before the step index goes beyond the current rank, the received KV indices are not greater than those of the Q in the current rank
            # After the step index goes beyond the current rank, only the KV indices before the last granularity are no greater than those of the Q after the first granularity
            # this conclusion holds after the step index goes beyond the current rank (not just step index == current rank)
            block_out, block_lse = stripe_fwd_inner(
                process_group, outer_step, outer_rank, inner_ring_list, 
                q[:, granularity:], k[:, :-granularity], v[:, :-granularity],
                softmax_scale,
                granularity,
                dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
            )
            out, lse = update_out_and_lse(
                out, lse, block_out, block_lse, slice_=(slice(None), slice(granularity, None))
            )  

        if outer_step + 1 != num_outer_steps:
            outer_comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def stripe_backward_inner(
    process_group,
    outer_step, inner_ring_list,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    granularity=1,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert (
        causal
    ), "stripe flash attn only supports causal attention, if not causal, ring flash attn instead"
    kv_comm = RingComm(process_group, False, inner_ring_list)
    d_kv_comm = RingComm(process_group, False, inner_ring_list)

    inner_rank = int(os.environ["LOCAL_RANK"])
    num_inner_step = len(inner_ring_list)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    for inner_step in range(num_inner_step):
        if inner_step + 1 != num_inner_step:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        shift_causal = outer_step == 0 and inner_step > inner_rank
        softmax_lse_1 = None

        def backward(
            dout_,
            q_, k_, v_, out_, softmax_lse_,
            block_dq_buffer_, block_dk_buffer_, block_dv_buffer_,
        ):
            params = get_default_args(_flash_attn_backward).copy()
            params.update(
                {
                    "dout": dout_,
                    "q": q_,
                    "k": k_,
                    "v": v_,
                    "out": out_,
                    "softmax_lse": softmax_lse_,
                    "dq": block_dq_buffer_,
                    "dk": block_dk_buffer_,
                    "dv": block_dv_buffer_,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                    "alibi_slopes": alibi_slopes,
                    "deterministic": deterministic,
                }
            )
            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )
            _flash_attn_backward(**params)

        if not shift_causal:
            backward(
                dout, q, k, v, out, softmax_lse, 
                block_dq_buffer, block_dk_buffer, block_dv_buffer
            )
        else:
            if softmax_lse_1 is None:
                # lazy init, since the last rank does not need softmax_lse_1
                softmax_lse_1 = softmax_lse[:, :, granularity:].contiguous()
            backward(
                dout[:, granularity:], 
                q[:, granularity:], k[:, :-granularity], v[:, :-granularity], 
                out[:, granularity:], softmax_lse_1,
                block_dq_buffer[:, granularity:], block_dk_buffer[:, :-granularity], block_dv_buffer[:, :-granularity]
            )

        if dq is None:
            dq = block_dq_buffer.to(torch.float32)
            dk = block_dk_buffer.to(torch.float32)
            dv = block_dv_buffer.to(torch.float32)
        else:
            if not shift_causal:
                dq += block_dq_buffer
            else:
                dq[:, granularity:] += block_dq_buffer[:, granularity:]
    
            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if not shift_causal:
                dk = block_dk_buffer + dk
                dv = block_dv_buffer + dv
            else:
                dk[:, :-granularity] += block_dk_buffer[:, :-granularity]
                dv[:, :-granularity] += block_dv_buffer[:, :-granularity]

        if inner_step + 1 != num_inner_step:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

def stripe_backward_outer(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    inner_ring_list: List[int],
    outer_ring_list: List[int],
    granularity=1,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert (
        causal
    ), "stripe flash attn only supports causal attention, if not causal, ring flash attn instead"

    outer_kv_comm = RingComm(process_group, False, outer_ring_list)
    outer_dkv_comm = RingComm(process_group, False, outer_ring_list)
    
    global_rank = dist.get_rank()
    outer_rank = outer_ring_list.index(global_rank)
    num_outer_steps = len(outer_ring_list)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    for outer_step in range(num_outer_steps):
        if outer_step + 1 != num_outer_steps:
            next_k, next_v = outer_kv_comm.send_recv_kv(k, v)

        softmax_lse_1 = None
        outer_shift = outer_step > outer_rank

        if not outer_shift:
            block_dq_buffer, block_dk_buffer, block_dv_buffer = stripe_backward_inner(
                process_group, outer_step, inner_ring_list,
                dout, q, k, v, out, 
                softmax_lse, softmax_scale, granularity, dropout_p, 
                causal, window_size, alibi_slopes, deterministic,
            )
        else:
            if softmax_lse_1 is None:
                # lazy init, since the last rank does not need softmax_lse_1
                softmax_lse_1 = softmax_lse[:, :, granularity:].contiguous()
            block_dq_buffer, block_dk_buffer, block_dv_buffer = stripe_backward_inner(
                process_group, outer_step, inner_ring_list,
                dout[:, granularity:], 
                q[:, granularity:], k[:, :-granularity], v[:, :-granularity], out[:, granularity:], 
                softmax_lse_1, softmax_scale, granularity, dropout_p, 
                causal, window_size, alibi_slopes, deterministic,
            ) 

        if dq is None:
            dq = block_dq_buffer.to(torch.float32)
            dk = block_dk_buffer.to(torch.float32)
            dv = block_dv_buffer.to(torch.float32)
        else:
            if not outer_shift:
                dq += block_dq_buffer
            else:
                dq[:, granularity:] += block_dq_buffer

            outer_dkv_comm.wait()

            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if not outer_shift:
                dk = block_dk_buffer + dk
                dv = block_dv_buffer + dv
            else:
                dk[:, :-granularity] += block_dk_buffer
                dv[:, :-granularity] += block_dv_buffer

        if outer_step + 1 != num_outer_steps:
            outer_kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = outer_dkv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    outer_dkv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

class DRStripeFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        granularity,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        inner_ring_list = get_inner_group() # ranks in the current node, length = num of cards within this node
        outer_ring_list = get_outer_group() # corresponding ranks in other nodes, length = num of nodes

        q = shuffle_striped_input(to_send=q, dim=1, granularity=granularity, process_group=group)
        k = shuffle_striped_input(to_send=k, dim=1, granularity=granularity, process_group=group)
        v = shuffle_striped_input(to_send=v, dim=1, granularity=granularity, process_group=group)

        out, softmax_lse = stripe_fwd_outer(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            granularity=granularity,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            inner_ring_list=inner_ring_list,
            outer_ring_list=outer_ring_list,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.inner_ring_list = inner_ring_list
        ctx.outer_ring_list = outer_ring_list
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.granularity = granularity
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        out = recover_striped_output(out, dim=1, granularity=granularity, process_group=group)
        if return_softmax:
            softmax_lse = recover_striped_output(softmax_lse, dim=2, granularity=granularity, process_group=group)
            return (out, softmax_lse, None)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        dout = shuffle_striped_input(to_send=dout, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        q, k, v, out, softmax_lse = ctx.saved_tensors
        inner_ring_list, outer_ring_list = ctx.inner_ring_list, ctx.outer_ring_list
        dq, dk, dv = stripe_backward_outer(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            ctx.softmax_scale,
            inner_ring_list=inner_ring_list,
            outer_ring_list=outer_ring_list,
            granularity=ctx.granularity,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )

        dq = recover_striped_output(dq, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        dk = recover_striped_output(dk, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        dv = recover_striped_output(dv, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def dr_stripe_flash_attn_qkvpacked_func(
    qkv, # [B, N, 3, H, D]
    dropout_p=0.0,
    softmax_scale=None,
    granularity=1,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return DRStripeFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        granularity,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def dr_stripe_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    granularity=1,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return DRStripeFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        granularity,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def dr_stripe_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    granularity=1,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return DRStripeFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        granularity,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
