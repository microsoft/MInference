import torch
import torch.distributed as dist
from typing import List, Tuple, Dict
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward

from .utils import (
    RingComm,
    update_out_and_lse, get_default_args, 
    shuffle_striped_input, recover_striped_output
)

def stripe_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_idx: int,
    softmax_scale,
    granularity=1,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal, "stripe flash attn only supports causal attention, if not causal, use ring flash attn instead"
    comm = RingComm(process_group)
    bsz, seq_len, num_heads, head_dim = q.shape

    out, lse = None, None
    next_k, next_v = None, None

    def forward(q_, k_, v_, causal_):
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q_,
                "k": k_,
                "v": v_,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal_,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
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


    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        shift = 1 if step > comm.rank else 0
        if shift == 0:
            step_out, step_lse = forward(q, k, v, causal)
            out, lse = update_out_and_lse(out, lse, step_out, step_lse)
        else:
            # Before the step index goes beyond the current rank, the received KV indices are not greater than those of the Q in the current rank
            # After the step index goes beyond the current rank, only the KV indices before the last granularity are no greater than those of the Q after the first granularity
            # this conclusion holds after the step index goes beyond the current rank (not just step index == current rank)
            step_out, step_lse = forward(
                q[:, granularity:], k[:, :-granularity], v[:, :-granularity], causal
            )
            out, lse = update_out_and_lse(
                out, lse, step_out, step_lse, slice_=(slice(None), slice(granularity, None))
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def stripe_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    layer_idx: int,
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
    bsz, seq_len, num_heads, head_dim = q.shape

    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)


    def backward(
        granularity_,
    ):
        if granularity_ == 0:
            k_, v_ = k, v
            dk_, dv_ = block_dk_buffer, block_dv_buffer
        else:
            k_, v_ = k[:, :-granularity_], v[:, :-granularity_]
            dk_, dv_ = block_dk_buffer[:, :-granularity_], block_dv_buffer[:, :-granularity_]
        params = get_default_args(_flash_attn_backward).copy()
        params.update(
            {
                "dout": dout[:, granularity_:],
                "q": q[:, granularity_:],
                "k": k_,
                "v": v_,
                "out": out[:, granularity_:],
                "softmax_lse": softmax_lse[:, :, granularity_:].contiguous(),
                "dq": block_dq_buffer[:, granularity_:],
                "dk": dk_, 
                "dv": dv_,
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
        params.update({"rng_state": torch.zeros((2, ), dtype=torch.int64, device=q.device)})
        _flash_attn_backward(**params)

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        shift_causal = 1 if step > kv_comm.rank else 0
        if shift_causal == 0:
            backward(granularity_=0)
        else:
            backward(granularity_=granularity)

        if dq is None:
            dq = block_dq_buffer.to(torch.float32)
            dk = block_dk_buffer.to(torch.float32)
            dv = block_dv_buffer.to(torch.float32)
        else:
            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if shift_causal == 0:
                dq += block_dq_buffer
                dk = block_dk_buffer + dk
                dv = block_dv_buffer + dv
            else:
                dq[:, granularity:] += block_dq_buffer[:, granularity:]
                dk[:, :-granularity] += block_dk_buffer[:, :-granularity]
                dv[:, :-granularity] += block_dv_buffer[:, :-granularity]

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class StripeFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        layer_idx,
        dropout_p,
        softmax_scale,
        granularity,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        assert alibi_slopes is None
        
        # -----------------------------------------
        # Shuffle
        q = shuffle_striped_input(to_send=q, dim=1, granularity=granularity, process_group=group)
        k = shuffle_striped_input(to_send=k, dim=1, granularity=granularity, process_group=group)
        v = shuffle_striped_input(to_send=v, dim=1, granularity=granularity, process_group=group)
        k, v = k.contiguous(), v.contiguous()

        # ----------------------------------------------
        # Compute
        out, softmax_lse = stripe_flash_attn_forward(
            group,
            q, k, v,
            layer_idx,
            softmax_scale=softmax_scale,
            granularity=granularity,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )

        # ----------------------------------------------
        # Recover outputs
        recovered_out = recover_striped_output(out, dim=1, granularity=granularity, process_group=group)
        if return_softmax:
            recovered_softmax_lse = recover_striped_output(softmax_lse, dim=2, granularity=granularity, process_group=group)

        # ----------------------------------------------
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.granularity = granularity
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.layer_idx = layer_idx
        ctx.return_softmax = return_softmax
        ctx.group = group

        # ----------------------------------------------
        # Output and return
        if return_softmax:
            return (recovered_out, recovered_softmax_lse, None)
        return recovered_out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, return_softmax, group = (
            ctx.dropout_p, ctx.softmax_scale, ctx.causal, ctx.window_size,
            ctx.alibi_slopes, ctx.deterministic, ctx.return_softmax,
            ctx.group
        )


        # ----------------------------------------------
        # Shuffle
        dout = shuffle_striped_input(
            to_send=dout, dim=1, granularity=ctx.granularity, 
            process_group=ctx.group
        )

        # ----------------------------------------------
        # Compute
        dq, dk, dv = stripe_flash_attn_backward(
            ctx.group,
            dout,
            q, k, v, out, softmax_lse,
            layer_idx=layer_idx,
            softmax_scale=ctx.softmax_scale,
            granularity=ctx.granularity,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        

        # ----------------------------------------------
        # Recover
        dq = recover_striped_output(dq, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        dk = recover_striped_output(dk, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        dv = recover_striped_output(dv, dim=1, granularity=ctx.granularity, process_group=ctx.group)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


def stripe_flash_attn_qkvpacked_func(
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
    return StripeFlashAttnFunc.apply(
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


def stripe_flash_attn_kvpacked_func(
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
    return StripeFlashAttnFunc.apply(
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


def stripe_flash_attn_func(
    q,
    k,
    v,
    layer_idx: int,
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
    return StripeFlashAttnFunc.apply(
        q,
        k,
        v,
        layer_idx,
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
