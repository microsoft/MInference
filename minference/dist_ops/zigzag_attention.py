#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# Credits: This logger implementation is inspired by project https://github.com/zhuzilin/ring-flash-attention
import os
import copy
import torch
import torch.distributed as dist

from time import perf_counter
from typing import List, Tuple, Dict
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward

from .utils import (
    RingComm, update_out_and_lse, shuffle_zigzag_input, 
    recover_zigzag_output, get_default_args
)

def zigzag_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor, # [B, S, H, D]
    k: torch.Tensor,
    v: torch.Tensor,
    layer_idx: int,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group, zigzag=True)

    bsz, seq_len, num_heads, head_dim = q.shape
    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    lse = None
    next_k, next_v = None, None

    def forward(q, k, v, causal):
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
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

        if step == 0:
            # Do softmax(QK^T / sqrt(d_k))V on the currently hold K and V
            # and record the output and the LSE
            block_out, block_lse = forward(q, k, v, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.revert_rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_lse = forward(q, k0, v0, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, causal=False)
            out, lse = update_out_and_lse(
                out, lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
            
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

def zigzag_ring_flash_attn_backward(
    process_group,
    dout,
    q, k, v, out,
    layer_idx: int,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    bsz, seq_len, num_heads, head_dim = q.shape

    kv_comm = RingComm(process_group, zigzag=True)
    d_kv_comm = RingComm(process_group, zigzag=True)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def backward(dout_, q_, k_, v_, out_, softmax_lse_, causal_):
        seqlen_q = q_.shape[1]
        seqlen_kv = k_.shape[1]
        params = get_default_args(_flash_attn_backward).copy()
        params.update(
            {
                "dout": dout_,
                "q": q_,
                "k": k_,
                "v": v_,
                "out": out_,
                "softmax_lse": softmax_lse_,
                "dq": dq_buffer[:, :seqlen_q],
                "dk": dk_buffer[:, :seqlen_kv],
                "dv": dv_buffer[:, :seqlen_kv],
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal_,
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

        # -----------------------------------------------------------
        if step == 0:
            backward(dout, q, k, v, out, softmax_lse, causal_=True)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.revert_rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                backward(dout, q, k0, v0, out, softmax_lse, causal_=False)
            else:
                backward(dout1, q1, k, v, out1, softmax_lse1, causal_=False)

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.revert_rank:
                dq += dq_buffer
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]
                dk += dk_buffer
                dv += dv_buffer

        # -----------------------------------------------------------
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer,
        )
        
    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

'''
In nnscaler, sequence are stored in the initial order, e.g., [0 1 2 3 4 5 6 7].
However, zigzag ring flash attention requires the sequence to be in the order of [0 7 2 5 3 4 1 6].
As a result:
- in forward, we need to shuffle q, k, v and recover the out
- in backward, we need to shuffle dout and recover the dq, dk, dv
'''
class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        layer_idx, 
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        assert alibi_slopes is None
        if softmax_scale is None: softmax_scale = q.shape[-1] ** (-0.5)

        # ----------------------------------------------
        # Shuffle
        q = shuffle_zigzag_input(to_send=q, dim=1, process_group=group)
        k = shuffle_zigzag_input(to_send=k, dim=1, process_group=group)
        v = shuffle_zigzag_input(to_send=v, dim=1, process_group=group)
        k, v = k.contiguous(), v.contiguous()

        # ----------------------------------------------
        # Compute
        out, softmax_lse = zigzag_ring_flash_attn_forward(
            group,
            q, k, v,
            layer_idx,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        
        # ----------------------------------------------
        # Recover outputs
        recovered_out = recover_zigzag_output(out, dim=1, process_group=group)
        if return_softmax:
            recovered_softmax_lse = recover_zigzag_output(softmax_lse, dim=2, process_group=group)

        # ------------------------------
        # Saving tensors
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.layer_idx = layer_idx 
        ctx.return_softmax = return_softmax

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
        dout = shuffle_zigzag_input(to_send=dout, dim=1, process_group=group) 

        # ----------------------------------------------
        # Compute
        dq, dk, dv = zigzag_ring_flash_attn_backward(
            group,
            dout,
            q, k, v, out,
            layer_idx,
            softmax_lse,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
        

        # ----------------------------------------------
        # Recover
        dq = recover_zigzag_output(dq, dim=1, process_group=group)
        dk = recover_zigzag_output(dk, dim=1, process_group=group)
        dv = recover_zigzag_output(dv, dim=1, process_group=group)

        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_qkvpacked_func(
    qkv,
    layer_idx,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        layer_idx,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def zigzag_ring_flash_attn_kvpacked_func(
    q,
    kv,
    layer_idx,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        layer_idx,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def zigzag_ring_flash_attn_func(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor,
    layer_idx,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q, k, v,
        layer_idx,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
