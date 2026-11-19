# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import torch
import triton
import triton.language as tl

from flash_attn import flash_attn_func
import pdb

@triton.jit
def _triton_attn_fwd_inner(
    q, acc, l_i, m_i,
    k_ptrs, v_ptrs, stride_kn, stride_vn,
    lo, hi,
    offs_m, offs_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, INV_CAUSAL: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if CAUSAL:
            qk = tl.where(cols[None, :] <= offs_m[:, None], qk, float("-inf"))
        if INV_CAUSAL:
            qk = tl.where(cols[None, :] > offs_m[:, None], qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(q.type.element_ty), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    return acc, l_i, m_i


@triton.jit
def _triton_attn_fwd_kernel(
    Q, K, V,
    M, L,
    sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_mh, stride_lh,
    Z, num_heads, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = N_CTX - (tl.program_id(0) + 1) * BLOCK_M
    off_hz = tl.program_id(1)

    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // num_heads) * stride_qz + (off_hz % num_heads) * stride_qh
    kv_offset = (off_hz // num_heads) * stride_kz + (off_hz % num_heads) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    m_ptrs = M + off_hz * stride_mh + offs_m
    l_ptrs = L + off_hz * stride_lh + offs_m

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(Out.type.element_ty)

    # loop over k, v and update accumulator
    acc, l_i, m_i = _triton_attn_fwd_inner(
        q, acc, l_i, m_i,
        k_ptrs, v_ptrs, stride_kn, stride_vn,
        0, start_m,
        offs_m, offs_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=False, INV_CAUSAL=False,
    )
    acc, l_i, m_i = _triton_attn_fwd_inner(
        q, acc, l_i, m_i,
        k_ptrs, v_ptrs, stride_kn, stride_vn,
        start_m, start_m + BLOCK_M,
        offs_m, offs_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=True, INV_CAUSAL=False,
    )

    # write back M, L
    tl.store(m_ptrs, m_i)
    tl.store(l_ptrs, l_i)

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(Out.type.element_ty))


@triton.jit
def _triton_tri_shape_attn_fwd_kernel(
    Q, K, V,
    M, L,
    sm_scale,
    sink_tokens, local_window, last_tokens, chunk_size,  # mod 64 == 0
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_mh, stride_lh,
    Z, num_heads, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)

    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)

    if start_m < N_CTX:
        offs_q = offs_m
    else:
        offs_q = N_CTX - last_tokens + (start_m - (N_CTX - last_tokens)) % last_tokens + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // num_heads) * stride_qz + (off_hz % num_heads) * stride_qh
    kv_offset = (off_hz // num_heads) * stride_kz + (off_hz % num_heads) * stride_kh

    q_ptrs = Q + qo_offset + offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    m_ptrs = M + off_hz * stride_mh + offs_m
    l_ptrs = L + off_hz * stride_lh + offs_m

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(Out.type.element_ty)

    # loop over k, v and update accumulator
    if start_m < sink_tokens + local_window or start_m >= N_CTX - last_tokens:
        if start_m < sink_tokens + local_window:
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                0, start_m,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=False, INV_CAUSAL=False,
            )
        elif start_m < N_CTX:
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                (tl.num_programs(0) * BLOCK_M - N_CTX) // last_tokens * chunk_size, start_m,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=False, INV_CAUSAL=False,
            )
        else:
            chunk_idx = (start_m - N_CTX) // last_tokens
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=False, INV_CAUSAL=False,
            )
        if start_m < N_CTX:
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                start_m, start_m + BLOCK_M,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=True, INV_CAUSAL=False,
            )
    else:
        acc, l_i, m_i = _triton_attn_fwd_inner(
            q, acc, l_i, m_i,
            k_ptrs, v_ptrs, stride_kn, stride_vn,
            0, sink_tokens,
            offs_m, offs_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=False, INV_CAUSAL=False,
        )
        if local_window > 0:
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                start_m - local_window, start_m - local_window + BLOCK_M,
                offs_m - local_window, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=False, INV_CAUSAL=True,
            )
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                start_m - local_window + BLOCK_M, start_m,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=False, INV_CAUSAL=False,
            )
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                start_m, start_m + BLOCK_M,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=True, INV_CAUSAL=False,
            )

    # write back M, L
    tl.store(m_ptrs, m_i)
    tl.store(l_ptrs, l_i)

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(Out.type.element_ty))


@triton.jit
def _triton_cross_attn_fwd_inner(
    q, acc, l_i, m_i,
    k_ptrs, v_ptrs, stride_kn, stride_vn,
    vis_stride,
    phase_start, phase_end,
    block_start, block_end,
    offs_m, offs_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, SUB_CAUSAL: tl.constexpr,
):
    num_phases = phase_end - phase_start
    num_blocks = block_end - block_start
    for block_idx in range(num_phases * num_blocks):
        phase_idx_n = phase_start + block_idx // num_blocks
        phase_off_n = (block_start + block_idx % num_blocks) * BLOCK_N
        cols = (phase_off_n + offs_n) * vis_stride + phase_idx_n
        # cols = (phase_off_n + offs_n) + phase_idx_n * phase_size
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if SUB_CAUSAL:
            causal_mask = phase_off_n + offs_n[None, :] < offs_m[:, None]
            qk = tl.where(causal_mask, qk, float("-inf"))
        elif CAUSAL:
            causal_mask = phase_off_n + offs_n[None, :] <= offs_m[:, None]
            qk = tl.where(causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(q.type.element_ty), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    return acc, l_i, m_i


@triton.jit
def _triton_cross_attn_fwd_kernel(
    Q, K, V,
    M, L,
    sm_scale,
    vis_stride,
    vis_start_q, vis_end_q, vis_phase_q,
    vis_start_k, vis_end_k, vis_phase_k,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_mh, stride_lh,
    Z, num_heads, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    off_hz = tl.program_id(2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    phase_size = (vis_end_q - vis_start_q) // vis_stride
    phase_idx_m = tl.program_id(1)
    phase_off_m = tl.program_id(0) * BLOCK_M
    if phase_size % BLOCK_M > 0:
        phase_off_m -= (BLOCK_M - phase_size % BLOCK_M)
    rows = vis_start_q + (phase_off_m % phase_size + offs_m) * vis_stride + phase_idx_m
    qo_offset = (off_hz // num_heads) * stride_qz + (off_hz % num_heads) * stride_qh
    kv_offset = (off_hz // num_heads) * stride_kz + (off_hz % num_heads) * stride_kh
    q_ptrs = Q + qo_offset + rows[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + vis_start_k * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + vis_start_k * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + rows[:, None] * stride_om + offs_d[None, :] * stride_ok
    m_ptrs = M + off_hz * stride_mh + rows
    l_ptrs = L + off_hz * stride_lh + rows
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(Out.type.element_ty)
    # loop over k, v and update accumulator
    causal_start = max(phase_off_m // BLOCK_N, 0)
    causal_end = causal_start + BLOCK_M // BLOCK_N
    if phase_idx_m == vis_phase_q:
        acc, l_i, m_i = _triton_cross_attn_fwd_inner(
            q, acc, l_i, m_i,
            k_ptrs, v_ptrs, stride_kn, stride_vn,
            vis_stride,
            0, vis_stride,
            0, causal_start,
            offs_m, offs_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=False, SUB_CAUSAL=False,
        )
        acc, l_i, m_i = _triton_cross_attn_fwd_inner(
            q, acc, l_i, m_i,
            k_ptrs, v_ptrs, stride_kn, stride_vn,
            vis_stride,
            0, vis_phase_q + 1,
            causal_start, causal_end,
            offs_m + phase_off_m, offs_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=True, SUB_CAUSAL=False,
        )
        acc, l_i, m_i = _triton_cross_attn_fwd_inner(
            q, acc, l_i, m_i,
            k_ptrs, v_ptrs, stride_kn, stride_vn,
            vis_stride,
            vis_phase_q + 1, vis_stride,
            causal_start, causal_end,
            offs_m + phase_off_m, offs_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=False, SUB_CAUSAL=True,
        )
    else:
        if phase_idx_m < vis_phase_k:
            acc, l_i_tmp, m_i_tmp = _triton_cross_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                vis_stride,
                vis_phase_k, vis_phase_k + 1,
                causal_start, causal_end,
                offs_m + phase_off_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=False, SUB_CAUSAL=True,
            )
            m_mask = offs_m > -min(phase_off_m, 0)  # TODO: double-check
            l_i = tl.where(m_mask, l_i_tmp, l_i)
            m_i = tl.where(m_mask, m_i_tmp, m_i)
            acc = tl.where(m_mask[:, None], acc, 0)
        else:
            acc, l_i, m_i = _triton_cross_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, stride_kn, stride_vn,
                vis_stride,
                vis_phase_k, vis_phase_k + 1,
                causal_start, causal_end,
                offs_m + phase_off_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                CAUSAL=True, SUB_CAUSAL=False,
            )
        acc, l_i, m_i = _triton_cross_attn_fwd_inner(
            q, acc, l_i, m_i,
            k_ptrs, v_ptrs, stride_kn, stride_vn,
            vis_stride,
            vis_phase_k, vis_phase_k + 1,
            0, causal_start,
            offs_m, offs_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=False, SUB_CAUSAL=False,
        )
    # write back O
    m_0 = tl.load(m_ptrs)
    m = tl.maximum(m_i, m_0)
    l_0 = tl.load(l_ptrs)
    l = tl.math.exp2(m_0 - m) * l_0 + tl.math.exp2(m_i - m) * l_i
    alpha_0 = tl.math.exp2(m_0 - m) * (l_0 / l)
    alpha_i = tl.math.exp2(m_i - m) / l
    acc = tl.load(o_ptrs).to(tl.float32) * alpha_0[:, None] + acc * alpha_i[:, None]
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=(phase_off_m + offs_m >= 0)[:, None])

def _triton_tri_cross_attention(
    q: torch.Tensor,        # [BATCH, N_CTX, N_HEADS, D_HEAD]
    k: torch.Tensor,        # [BATCH, N_CTX, N_HEADS, D_HEAD]
    v: torch.Tensor,        # [BATCH, N_CTX, N_HEADS, D_HEAD]
    sm_scale: float,
    sink_tokens: int,
    local_window: int,
    last_tokens: int,
    vis_stride: int,
    vis_start_q: int,
    vis_end_q: int,
    vis_phase_q: int,
    vis_start_k: int,
    vis_end_k: int,
    vis_phase_k: int,
) -> torch.Tensor:
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    batch_size, num_tokens, num_heads = q.shape[:3]
    # print(num_tokens, sink_tokens, local_window, last_tokens)

    o = torch.empty_like(q)
    m = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=q.device)
    l = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=q.device)

    block_M = 128 if num_tokens > 131072 else 64
    block_N = 64
    num_warps = 4
    num_stages = 2

    num_chunks = 1
    chunk_size = num_tokens
    num_last_blocks = triton.cdiv(last_tokens, block_M)
    num_extra_tokens = 0
    if last_tokens > 0:
        num_chunks = max(min(1024 // num_last_blocks, num_tokens // 4096), 1)
        chunk_size = (num_tokens // num_chunks) & -block_N
        num_extra_tokens = (num_chunks - 1) * last_tokens
        o = torch.nn.functional.pad(o, [0, 0, 0, 0, 0, num_extra_tokens, 0, 0])
        m = torch.nn.functional.pad(m, [0, num_extra_tokens, 0, 0, 0, 0])
        l = torch.nn.functional.pad(l, [0, num_extra_tokens, 0, 0, 0, 0])
    # print(num_last_blocks, num_chunks, chunk_size, num_extra_tokens)

    grid = (triton.cdiv(num_tokens + num_extra_tokens, block_M), batch_size * num_heads, 1)
    _triton_tri_shape_attn_fwd_kernel[grid](
        q, k, v,
        m, l,
        sm_scale,
        sink_tokens, local_window, last_tokens, chunk_size,
        o,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        m.stride(1), l.stride(1),
        batch_size, num_heads, num_tokens,
        BLOCK_M=block_M, BLOCK_N=block_N,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps, num_stages=num_stages,
    )

    if num_extra_tokens > 0:
        m_list = m[:, :, num_tokens-last_tokens:].reshape((batch_size, num_heads, num_chunks, last_tokens))
        l_list = l[:, :, num_tokens-last_tokens:].reshape((batch_size, num_heads, num_chunks, last_tokens))
        o_list = o[:, num_tokens-last_tokens:].reshape((batch_size, num_chunks, last_tokens, num_heads, Lq))
        m_merged = m_list.max(dim=2, keepdim=True).values
        alpha = torch.exp2(m_list - m_merged)
        l_merged = (l_list * alpha).sum(dim=2, keepdim=True)
        beta = l_list / l_merged
        o_merged = (o_list * (alpha * beta).permute(0, 2, 3, 1).unsqueeze(-1)).sum(dim=1)
        o[:, num_tokens-last_tokens:num_tokens] = o_merged

    assert not torch.any(torch.isnan(o))
    num_vis_tokens = vis_end_q - vis_start_q
    phase_size = num_vis_tokens // vis_stride
    # print(num_vis_tokens, phase_size, phase_size % block_M)
    grid = (triton.cdiv(phase_size, block_M), vis_stride, batch_size * num_heads)
    _triton_cross_attn_fwd_kernel[grid](
        q, k, v,
        m, l,
        sm_scale,
        vis_stride,
        vis_start_q, vis_end_q, vis_phase_q,
        vis_start_k, vis_end_k, vis_phase_k,
        o,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        m.stride(1), l.stride(1),
        batch_size, num_heads, num_tokens,
        BLOCK_M=block_M, BLOCK_N=block_N,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps, num_stages=num_stages,
    )
    # import ipdb; ipdb.set_trace()
    assert not torch.any(torch.isnan(o))

    return o


def multimodal_grid_attention(
    query: torch.Tensor,  # [BATCH, N_CTX, N_HEADS, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_CTX, N_HEADS, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_CTX, N_HEADS, D_HEAD]
    sink_tokens: int,
    local_window: int,
    vis_start: int,
    vis_end: int,
    vis_stride: int,
    block_size_M: int = 128,
    block_size_N: int = 64,
):
    vis_chunk = vis_stride * block_size_N
    if sink_tokens < vis_start:
        sink_tokens = (vis_start + block_size_N - 1) // block_size_N * block_size_N
    else:
        sink_tokens = (sink_tokens + block_size_N - 1) // block_size_N * block_size_N
    local_window = (local_window + block_size_N - 1) // block_size_N * block_size_N
    if vis_start + vis_chunk + local_window > vis_end:
        return flash_attn_func(query, key, value, causal=True).transpose(1, 2)
    vis_tokens = (vis_end - sink_tokens - local_window) // vis_chunk * vis_chunk

    batch_size, context_size, num_heads, head_dim = query.shape
    sm_scale = head_dim ** -0.5

    seq_pad = ((context_size + block_size_M - 1) // block_size_M) * block_size_M - context_size
    dim_pad = 2 ** math.ceil(math.log2(head_dim)) - head_dim
    query = torch.nn.functional.pad(query, [0, dim_pad, 0, 0, 0, seq_pad, 0, 0])
    key = torch.nn.functional.pad(key, [0, dim_pad, 0, 0, 0, seq_pad, 0, 0])
    value = torch.nn.functional.pad(value, [0, dim_pad, 0, 0, 0, seq_pad, 0, 0])

    last_tokens = context_size + seq_pad - (sink_tokens + local_window + vis_tokens)
    vis_start_q = sink_tokens + local_window
    vis_end_q = vis_start_q + vis_tokens
    vis_start_k = sink_tokens
    vis_end_k = vis_start_k + vis_tokens
    vis_phase_q = (vis_start - vis_start_q) % vis_stride
    vis_phase_k = (vis_start - vis_start_k) % vis_stride
    # vis_phase_q = (vis_start + vis_stride - 1 - vis_start_q) % vis_stride
    # vis_phase_k = (vis_start + vis_stride - 1 - vis_start_k) % vis_stride

    (
        sink_tokens, local_window, last_tokens, vis_stride,
        vis_start_q, vis_end_q, vis_phase_q,
        vis_start_k, vis_end_k, vis_phase_k,
    ) = [
        int(i) for i in (
            sink_tokens, local_window, last_tokens, vis_stride,
            vis_start_q, vis_end_q, vis_phase_q,
            vis_start_k, vis_end_k, vis_phase_k,
        )
    ]

    out = _triton_tri_cross_attention(
        query, key, value,
        sm_scale,
        sink_tokens, local_window, last_tokens, vis_stride,
        vis_start_q, vis_end_q, vis_phase_q,
        vis_start_k, vis_end_k, vis_phase_k,
    )

    return out[:, :context_size, :, :head_dim].transpose(1, 2)


def _ref_attention(
    query: torch.Tensor,  # [BATCH, N_CTX, N_HEADS, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_CTX, N_HEADS, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_CTX, N_HEADS, D_HEAD]
    sink_tokens: int,
    local_window: int,
    vis_start: int,
    vis_end: int,
    vis_stride: int,
    block_size_M: int = 128,
    block_size_N: int = 64,
    plot_mask: bool = False,
):
    batch_size, num_tokens, num_heads, head_dim = query.shape

    vis_chunk = vis_stride * block_size_N
    if sink_tokens < vis_start:
        sink_tokens = (vis_start + block_size_N - 1) // block_size_N * block_size_N
    else:
        sink_tokens = (sink_tokens + block_size_N - 1) // block_size_N * block_size_N
    local_window = (local_window + block_size_N - 1) // block_size_N * block_size_N
    if vis_start + vis_chunk + local_window > vis_end:
        return flash_attn_func(query, key, value, causal=True)
    vis_tokens = (vis_end - sink_tokens - local_window) // vis_chunk * vis_chunk

    last_tokens = num_tokens - (sink_tokens + local_window + vis_tokens)
    vis_start_q = sink_tokens + local_window
    vis_end_q = vis_start_q + vis_tokens
    vis_start_k = sink_tokens
    vis_end_k = vis_start_k + vis_tokens
    vis_phase_q = (vis_start + vis_stride - 1 - vis_start_q) % vis_stride
    vis_phase_k = (vis_start + vis_stride - 1 - vis_start_k) % vis_stride

    arange = torch.arange(num_tokens, dtype=torch.int32, device=query.device)
    mask = arange[None, None, :, None] - local_window < arange[None, None, None, :]
    mask[:, :, -last_tokens:, :] = True
    mask[:, :, :, :sink_tokens] = True
    mask[:, :, vis_start_q+vis_phase_q:vis_end_q:vis_stride, :] = True
    mask[:, :, :, vis_start_k+vis_phase_k:vis_end_k:vis_stride] = True
    mask &= arange[None, None, :, None] >= arange[None, None, None, :]

    if plot_mask:
        print(f'tri_shape = ({sink_tokens, local_window, last_tokens})')
        print(f'vis_tokens_q = ({vis_start_q}, {vis_end_q} | {vis_tokens} | {vis_phase_q})')
        print(f'vis_tokens_k = ({vis_start_k}, {vis_end_k} | {vis_tokens} | {vis_phase_k})')
        _plot_mask(mask[0, 0], path='mask.png')
        mask1 = mask & (arange[None, None, :, None] - local_window >= arange[None, None, None, :])
        shfl_idx_q = torch.arange(num_tokens, dtype=torch.int64, device=mask1.device)
        vis_idx_q = torch.arange(vis_start_q, vis_end_q, dtype=torch.int64, device=mask1.device)
        shfl_idx_q[vis_start_q:vis_end_q] = vis_idx_q.reshape((-1, vis_stride)).T.flatten()
        shfl_idx_k = torch.arange(num_tokens, dtype=torch.int64, device=mask1.device)
        vis_idx_k = torch.arange(vis_start_k, vis_end_k, dtype=torch.int64, device=mask1.device)
        shfl_idx_k[vis_start_k:vis_end_k] = vis_idx_k.reshape((-1, vis_stride)).T.flatten()
        mask1 = torch.gather(input=mask1, dim=2, index=shfl_idx_q[None, None, :, None].expand(mask1.shape))
        mask1 = torch.gather(input=mask1, dim=3, index=shfl_idx_k[None, None, None, :].expand(mask1.shape))
        _plot_mask(mask1[0, 0], path='mask-permuted.png')

    qk = torch.einsum('bmhd,bnhd->bhmn', query, key).where(mask, -torch.inf) * (head_dim ** -0.5)
    out = torch.einsum('bhmn,bnhd->bmhd', torch.softmax(qk, dim=-1), value)

    return out


def _plot_mask(mask: torch.Tensor, path: str = 'mask.png'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(16, 16))
    sns.heatmap(mask.cpu().numpy(), cbar=False)
    plt.savefig(path)


def test_cross_attn(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    sink_tokens: int,
    local_window: int,
    vis_start: int,
    vis_end: int,
    vis_stride: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = 'cuda',
    torch_check: bool = False,
    profile: bool = False,
):
    print(f'[B={batch_size}, N={num_tokens}, H={num_heads}, D={head_dim}]')
    print(f'[Streaming=({sink_tokens}, {local_window}), Grid=({vis_start}:{vis_end}:{vis_stride})]')

    query = torch.randn((batch_size, num_tokens, num_heads, head_dim), dtype=dtype, device=device)
    key = torch.randn((batch_size, num_tokens, num_heads, head_dim), dtype=dtype, device=device)
    value = torch.randn((batch_size, num_tokens, num_heads, head_dim), dtype=dtype, device=device)

    out = multimodal_grid_attention(query, key, value, sink_tokens, local_window, vis_start, vis_end, vis_stride)
    torch.cuda.synchronize()

    if torch_check:
        ref = _ref_attention(query, key, value, sink_tokens, local_window, vis_start, vis_end, vis_stride)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        print('Correctness check passed.')

    if profile:
        def call_cross_attn():
            return multimodal_grid_attention(query, key, value, sink_tokens, local_window, vis_start, vis_end, vis_stride)
        def call_flash_attn():
            return flash_attn_func(query, key, value, causal=True)
        print(f'Flash: {triton.testing.do_bench(call_flash_attn, warmup=50, rep=50):.3f} ms')
        print(f'Cross: {triton.testing.do_bench(call_cross_attn, warmup=50, rep=50):.3f} ms')
        # torch.testing.assert_close(call_cross_attn(), call_flash_attn(), rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    # test_cross_attn(1, 4321, 1, 128, 53, 176, 35, 35 + 14 * 280, 14, torch_check=True, profile=False)
    # test_cross_attn(1, 5060, 1, 128, 47, 81, 69, 35 + 14 * 340, 14, torch_check=True, profile=False)
    test_cross_attn(1, 16384, 1, 128, 1024, 1024, 812, 16310, 14, torch_check=False, profile=True)
    test_cross_attn(1, 32768, 1, 128, 1024, 1024, 812, 32704, 14, torch_check=False, profile=True)
    test_cross_attn(1, 65536, 1, 128, 1024, 1024, 812, 65464, 14, torch_check=False, profile=True)
    test_cross_attn(1, 131072, 1, 128, 1024, 1024, 812, 131012, 14, torch_check=False, profile=True)
    test_cross_attn(1, 262144, 1, 128, 1024, 1024, 812, 262010, 14, torch_check=False, profile=True)
    test_cross_attn(1, 524288, 1, 128, 1024, 1024, 812, 524202, 14, torch_check=False, profile=True)
    test_cross_attn(1, 1048576, 1, 128, 1024, 1024, 812, 1048460, 14, torch_check=False, profile=True)
