# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
import pycuda.autoprimaryctx
import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func
from pycuda.compiler import SourceModule


# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=['N_CTX'],
# )
@triton.jit
def _triton_block_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW

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
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # if start_n + BLOCK_N < seqlen:
        #     qk = tl.where(m_mask, qk, float("-inf"))
        # else:
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_block_sparse_attention(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens,           # [BATCH, ]
    block_index,       # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), MAX_BLOCKS_PRE_ROW]
    sm_scale,
    block_size_M=64,
    block_size_N=64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_block_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_index.shape[-2], block_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o


def _build_block_index(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,       # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    query_pool = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)
    arange_M = torch.arange(query_pool.shape[-2], dtype=torch.int32, device=query.device) * block_size_M
    arange_N = torch.arange(key_pool.shape[-2], dtype=torch.int32, device=key.device) * block_size_N
    p_pool = torch.einsum(f'bhmk, bhnk -> bhmn', query_pool, key_pool)
    p_pool = p_pool.where(arange_M[None, None, :, None] >= arange_N[None, None, None, :], -torch.inf)
    top_k = min(top_k, context_size // block_size_N)
    return torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values


def block_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = block_size_M - (query.shape[2] & (block_size_M - 1))
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    block_index = _build_block_index(query, key, top_k, block_size_N, block_size_N)
    out = _triton_block_sparse_attention(query, key, value, seqlens, block_index, sm_scale, block_size_M, block_size_N)
    return out[..., :context_size, :]
