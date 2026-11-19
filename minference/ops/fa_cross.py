# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import torch
import triton
import triton.language as tl

from flash_attn import flash_attn_func


_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in ([1, 2, 3, 4])
    for w in [4, 8]
]


def _keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(_keep, _configs)), key=["N_CTX"])
@triton.jit
def _triton_streaming_attn_fwd_kernel(
    Q, K, V,
    M, L,
    seqlens, sm_scale,
    sink_tokens: tl.constexpr,
    sliding_window: tl.constexpr,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, num_heads, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // num_heads)
    if start_m >= seqlen:
        return

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

    m_ptrs = M + off_hz * N_CTX + offs_m
    l_ptrs = L + off_hz * N_CTX + offs_m

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32)# - float("inf")
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

    for start_n in range(0, sink_tokens, BLOCK_N):
        cols = start_n + offs_n
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        pattern_mask = (cols[None, :] < sink_tokens) & (cols[None, :] + sliding_window <= offs_m[:, None])
        qk = tl.where(pattern_mask, qk, float("-inf"))
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

    for start_n in range(max(start_m - sliding_window, 0) & -BLOCK_N, start_m + BLOCK_M, BLOCK_N):
        cols = start_n + offs_n
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        pattern_mask = (cols[None, :] <= offs_m[:, None]) & (cols[None, :] + sliding_window > offs_m[:, None])
        qk = tl.where(pattern_mask, qk, float("-inf"))
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

    # write back M, L
    tl.store(m_ptrs, m_i)
    tl.store(l_ptrs, l_i)

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype))


# @triton.autotune(list(filter(_keep, _configs)), key=["N_CTX"])
@triton.jit
def _triton_cross_attn_fwd_kernel(
    Q, K, V,
    M, L,
    seqlens, sm_scale,
    sink_tokens: tl.constexpr,
    sliding_window: tl.constexpr,
    row_cnt, row_idx,
    col_cnt, col_idx,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, num_heads, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    if start_m < row_cnt:
        m_mask = start_m + offs_m < row_cnt
    else:
        start_m = row_cnt + start_m - ((row_cnt + BLOCK_M - 1) & -BLOCK_M)
        seqlen = tl.load(seqlens + off_hz // num_heads)
        if start_m >= seqlen:
            return
        m_mask = start_m + offs_m < seqlen
    rows = tl.load(row_idx + start_m + offs_m, mask=m_mask, other=0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // num_heads) * stride_qz + (off_hz % num_heads) * stride_qh
    kv_offset = (off_hz // num_heads) * stride_kz + (off_hz % num_heads) * stride_kh

    q_ptrs = Q + qo_offset + rows[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + rows[:, None] * stride_om + offs_d[None, :] * stride_ok

    m_ptrs = M + off_hz * N_CTX + rows
    l_ptrs = L + off_hz * N_CTX + rows

    # initialize pointer to m and l
    m_i = tl.load(m_ptrs)
    l_i = tl.load(l_ptrs)
    acc = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32) * l_i[:, None]
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator

    if start_m < row_cnt:
        for start_n in range(sink_tokens, N_CTX - sliding_window, BLOCK_N):
            cols = start_n + offs_n
            causal_mask = cols[None, :] + sliding_window <= rows[:, None]
            # -- load k, v --
            k = tl.load(k_ptrs + cols[None, :] * stride_kn)
            v = tl.load(v_ptrs + cols[:, None] * stride_vn)
            # -- compute qk --
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.where(causal_mask, qk, float("-inf"))
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
    else:
        for start_n in range(0, col_cnt, BLOCK_N):
            n_mask = start_n + offs_n < col_cnt
            cols = tl.load(col_idx + start_n + offs_n)#, mask=n_mask, other=N_CTX-1)
            causal_mask = (cols[None, :] + sliding_window <= rows[:, None]) & n_mask[None, :]
            # -- load k, v --
            k = tl.load(k_ptrs + cols[None, :] * stride_kn)
            v = tl.load(v_ptrs + cols[:, None] * stride_vn)
            # -- compute qk --
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.where(causal_mask, qk, float("-inf"))
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
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask[:, None])


def _triton_streaming_cross_attention(
    q: torch.Tensor,        # [BATCH, N_CTX, N_HEADS, D_HEAD]
    k: torch.Tensor,        # [BATCH, N_CTX, N_HEADS, D_HEAD]
    v: torch.Tensor,        # [BATCH, N_CTX, N_HEADS, D_HEAD]
    seqlens: torch.Tensor,  # [BATCH, ]
    sm_scale: float,
    sink_tokens: int,
    sliding_window: int,
    row_cnt: int,
    row_idx: torch.Tensor,  # [num_rows, ]
    col_cnt: int,
    col_idx: torch.Tensor,  # [num_cols, ]
    block_size_M: int = 128,
    block_size_N: int = 64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    batch_size, num_tokens, num_heads = q.shape[:3]
    o = torch.zeros_like(q)
    m = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=q.device)
    l = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=q.device)
    # grid = lambda args: (triton.cdiv(num_tokens, args['BLOCK_M']), batch_size * num_heads, 1)
    grid = (triton.cdiv(num_tokens, block_size_M), batch_size * num_heads, 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_streaming_attn_fwd_kernel[grid](
        q, k, v,
        m, l,
        seqlens, sm_scale,
        sink_tokens, sliding_window,
        o,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        batch_size, num_heads, num_tokens,
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=8, num_stages=2,
    )
    # grid = lambda args: (triton.cdiv(num_tokens, args['BLOCK_M']) + 1, batch_size * num_heads, 1)
    grid = (triton.cdiv(num_tokens, block_size_M) + 1, batch_size * num_heads, 1)
    _triton_cross_attn_fwd_kernel[grid](
        q, k, v,
        m, l,
        seqlens, sm_scale,
        sink_tokens, sliding_window,
        row_cnt, row_idx,
        col_cnt, col_idx,
        o,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        batch_size, num_heads, num_tokens,
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=8, num_stages=2,
    )
    return o


def streaming_cross_attention(
    query: torch.Tensor,  # [BATCH, N_CTX, N_HEADS, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_CTX, N_HEADS, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_CTX, N_HEADS, D_HEAD]
    sink_tokens: int,
    sliding_window: int,
    row_idx: list[int],   # [num_rows, ]
    col_idx: list[int],   # [num_cols, ]
    block_size_M: int = 128,
    block_size_N: int = 64,
):
    batch_size, context_size, num_heads, head_dim = query.shape
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5

    seq_pad = ((context_size + block_size_M - 1) // block_size_M) * block_size_M - context_size
    dim_pad = 2 ** math.ceil(math.log2(head_dim)) - head_dim
    query = torch.nn.functional.pad(query, [0, dim_pad, 0, 0, 0, seq_pad, 0, 0])
    key = torch.nn.functional.pad(key, [0, dim_pad, 0, 0, 0, seq_pad, 0, 0])
    value = torch.nn.functional.pad(value, [0, dim_pad, 0, 0, 0, seq_pad, 0, 0])

    if isinstance(row_idx, torch.Tensor):
        row_cnt = row_idx.shape[0]
        row_idx = row_idx.to(torch.int32).to(query.device)
    else:
        row_cnt = len(row_idx)
        row_idx = torch.tensor(row_idx, dtype=torch.int32, device=query.device)
    if isinstance(col_idx, torch.Tensor):
        col_cnt = col_idx.shape[0]
        col_idx = col_idx.to(torch.int32).to(query.device)
    else:
        col_cnt = len(col_idx)
        col_idx = torch.tensor(col_idx, dtype=torch.int32, device=query.device)

    uniques, counts = torch.cat((
        row_idx,
        torch.arange(context_size, dtype=row_idx.dtype, device=row_idx.device)
    )).unique(return_counts=True)
    row_idx = torch.cat((row_idx, uniques[counts == 1])).contiguous()
    uniques, counts = torch.cat((
        col_idx,
        torch.arange(context_size, dtype=col_idx.dtype, device=col_idx.device)
    )).unique(return_counts=True)
    col_idx = torch.cat((col_idx, uniques[counts == 1])).contiguous()

    out = _triton_streaming_cross_attention(
        query, key, value,
        seqlens, sm_scale,
        sink_tokens, sliding_window,
        row_cnt, row_idx, col_cnt, col_idx,
        block_size_M, block_size_N,
    )

    return out[:, :context_size, :, :head_dim].transpose(1, 2)


def _ref_attention(
    query: torch.Tensor,  # [BATCH, N_CTX, N_HEADS, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_CTX, N_HEADS, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_CTX, N_HEADS, D_HEAD]
    sink_tokens: int,
    sliding_window: int,
    row_idx: list[int],   # [num_rows, ]
    col_idx: list[int],   # [num_cols, ]
    plot_mask: bool = False,
):
    batch_size, num_tokens, num_heads, head_dim = query.shape

    arange = torch.arange(num_tokens, dtype=torch.int32, device=query.device)
    mask = arange[None, None, :, None] - sliding_window < arange[None, None, None, :]
    mask |= arange[None, None, None, :] < sink_tokens
    mask[:, :, row_idx, :] = True
    mask[:, :, :, col_idx] = True
    mask &= arange[None, None, :, None] >= arange[None, None, None, :]

    if plot_mask:
        _plot_mask(mask[0, 0])

    qk = torch.einsum('bmhd,bnhd->bhmn', query, key).where(mask, -torch.inf) * (head_dim ** -0.5)
    out = torch.einsum('bhmn,bnhd->bmhd', torch.softmax(qk, dim=-1), value)

    return out


def _plot_mask(mask: torch.Tensor):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(mask.cpu().numpy())
    plt.savefig('mask.png')


def test_cross_attn(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    sink_tokens: int,
    sliding_window: int,
    num_rows: int,
    num_cols: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = 'cuda',
    torch_check: bool = False,
    plot_mask: bool = False,
    profile: bool = False,
):
    print(f'[B={batch_size}, N={num_tokens}, H={num_heads}, D={head_dim}]')
    print(f'[Streaming=({sink_tokens}, {sliding_window}), Cross=({num_rows}, {num_cols})]')

    row_idx = torch.randperm(num_tokens - sink_tokens - sliding_window)[:num_rows] + sink_tokens
    col_idx = torch.randperm(num_tokens - sink_tokens - sliding_window)[:num_cols] + sink_tokens

    query = torch.randn((batch_size, num_tokens, num_heads, head_dim), dtype=dtype, device=device)
    key = torch.randn((batch_size, num_tokens, num_heads, head_dim), dtype=dtype, device=device)
    value = torch.randn((batch_size, num_tokens, num_heads, head_dim), dtype=dtype, device=device)

    out = streaming_cross_attention(query, key, value, sink_tokens, sliding_window, row_idx, col_idx)
    torch.cuda.synchronize()

    if torch_check:
        ref = _ref_attention(query, key, value, sink_tokens, sliding_window, row_idx, col_idx, plot_mask)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        print('Correctness check passed.')

    if profile:
        def call_cross_attn():
            out = streaming_cross_attention(query, key, value, sink_tokens, sliding_window, row_idx, col_idx)
        def call_flash_attn():
            out = flash_attn_func(query, key, value, causal=True)
        print(f'Flash: {triton.testing.do_bench(call_flash_attn, warmup=1000, rep=1000):.3f} ms')
        print(f'Cross: {triton.testing.do_bench(call_cross_attn, warmup=1000, rep=1000):.3f} ms')


if __name__ == '__main__':
    test_cross_attn(1, 4321, 1, 128, 123, 456, 123, 456, torch_check=True, profile=False)
    test_cross_attn(1, 131072, 32, 128, 1024, 1024, 1024, 1024, torch_check=False, profile=True)
    test_cross_attn(1, 128745, 32, 128, 1234, 4321, 2345, 5432, torch_check=False, profile=True)
