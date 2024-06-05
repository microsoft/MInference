import torch
import numpy as np

import triton
import triton.language as tl
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule

from flash_attn import flash_attn_varlen_func


@triton.jit
def triton_block_sparse_attn_kernel(
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


def triton_block_sparse_forward(
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
    triton_block_sparse_attn_kernel[grid](
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


def torch_build_index(
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


def make_causal_mask(seqlens, device, context_size):
    batch_size = seqlens.shape[0]
    arange = torch.arange(context_size, dtype=torch.int32, device=device)
    causal_mask = arange[None, None, :, None] >= arange[None, None, None, :]
    causal_mask = causal_mask.repeat((batch_size, 1, 1, 1))
    for b, seqlen in enumerate(seqlens):
        causal_mask[b, :, seqlen:, :] = False
        causal_mask[b, :, :, seqlen:] = False
    return causal_mask


def make_block_mask(block_index, causal_mask, device, block_size_M=64, block_size_N=64):
    batch_size, num_heads, num_rows, max_blocks_per_row = block_index.shape
    context_size = causal_mask.shape[-1]
    block_mask = torch.zeros((batch_size, num_heads, context_size, context_size), dtype=torch.bool, device=device)
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(num_rows):
                start_m = i * block_size_M
                end_m = start_m + block_size_M
                for j in range(max_blocks_per_row):
                    real_j = block_index[b, h, i, j]
                    start_n = real_j * block_size_N
                    end_n = start_n + block_size_N
                    block_mask[b, h, start_m:end_m, start_n:end_n] = True
    block_mask.logical_and_(causal_mask)
    return block_mask


def plot_mask(mask, name, batch=0, head=0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(16, 12))
    plt.clf()
    mask = mask[batch, head].cpu().numpy()
    sns.heatmap(mask)
    plt.savefig(name)


@triton.jit
def triton_dense_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(dtype)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    m_mask = offs_m[:, None] < seqlen

    for start_n in range(lo, hi, BLOCK_N):
        n_mask = (start_n + offs_n[None, :]) <= offs_m[:, None]
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
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
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back O
    acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    O_block_ptr = tl.make_block_ptr(
        base=Out + qo_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(dtype))


def triton_dense_forward(q, k, v, seqlens, sm_scale, block_size_M=128, block_size_N=64) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    num_warps = 4 if Lk <= 64 else 8  # 4
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    triton_dense_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=num_warps, num_stages=4,
    )

    return o


def flash_attn_forward(q, k, v, seqlens, sm_scale, context_size) -> torch.Tensor:
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=seqlens,
        cu_seqlens_k=seqlens,
        max_seqlen_q=context_size,
        max_seqlen_k=context_size,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=True,
    )


def torch_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    p = torch.einsum(f'bhmk, bhnk -> bhmn', query, key) * sm_scale
    p = p.where(mask, -torch.inf)
    p_max = p.max(-1, keepdim=True).values
    p_max = torch.where(p_max < 0, 0.0, p_max)
    p_exp = torch.exp(p - p_max)
    s = p_exp / (p_exp.sum(-1, keepdim=True) + 1e-6)
    out = torch.einsum(f'bhmn, bhnk -> bhmk', s, value)
    return out


def profile(fn, total_flops, tag, warmup=25, rep=100):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    gflops = total_flops / ms * 1e-9
    print(f'{tag}: {ms:.3f} ms | {gflops:.3f} GFLOP/s')


def test_flash_attention(
    seqlens=None,
    dtype=torch.float16,
    device="cuda",
    torch_test=True,
    batch_size=4,
    num_heads=32,
    context_size=1024,
    head_dim=128,
    top_k=5,
    block_size_M=64,
    block_size_N=64,
):
    print('========================================')
    print(f'BATCH={batch_size}, N_CTX={context_size}, N_HEADS={num_heads}, D_HEAD={head_dim}')
    q = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
    if seqlens is None:
        seqlens = torch.randint(context_size // 2, context_size, (batch_size, ), dtype=torch.int32, device=device)
    else:
        seqlens = torch.tensor(seqlens, dtype=torch.int32, device=device)
    dense_mask_nnz = seqlens.to(torch.float32).square().sum().item() * num_heads / 2
    sm_scale = head_dim ** -0.5

    causal_mask = make_causal_mask(seqlens, device, context_size)
    if torch_test:
        ref_o_dense = torch_forward(q, k, v, causal_mask, sm_scale)

    block_index = torch_build_index(q, k, top_k, block_size_M, block_size_N)
    arange_M = torch.arange(block_index.shape[-2], device=device)
    block_index_mask = arange_M[None, None, :, None] * block_size_M >= block_index * block_size_N
    sparse_mask_nnz = block_index_mask.to(torch.float32).sum().item() * block_size_M * block_size_N
    print(f'block mask sparsity: {1 - sparse_mask_nnz / dense_mask_nnz}')
    torch_build_index_fn = lambda: torch_build_index(q, k, top_k, block_size_M, block_size_N)
    profile(torch_build_index_fn, 0., 'torch-index')

    if torch_test:
        block_mask = make_block_mask(block_index, causal_mask, device, block_size_M, block_size_N)
        ref_o_sparse = torch_forward(q, k, v, block_mask, sm_scale)

    triton_dense_fn = lambda: triton_dense_forward(q, k, v, seqlens, sm_scale)
    output = triton_dense_fn()
    if torch_test:
        torch.testing.assert_close(output, ref_o_dense, atol=1e-2, rtol=0)
    profile(triton_dense_fn, 2. * head_dim * dense_mask_nnz, 'triton-dense')

    triton_sparse_fn = lambda: triton_block_sparse_forward(q, k, v, seqlens, block_index, sm_scale, block_size_M, block_size_N)
    output = triton_sparse_fn()
    if torch_test:
        torch.testing.assert_close(output, ref_o_sparse, atol=1e-2, rtol=0)
    profile(triton_sparse_fn, 2. * head_dim * sparse_mask_nnz, 'triton-sparse')

    q = q.swapaxes(1, 2).contiguous()
    k = k.swapaxes(1, 2).contiguous()
    v = v.swapaxes(1, 2).contiguous()
    q = torch.concatenate([q[i, :seqlen, :, :] for i, seqlen in enumerate(seqlens)])
    k = torch.concatenate([k[i, :seqlen, :, :] for i, seqlen in enumerate(seqlens)])
    v = torch.concatenate([v[i, :seqlen, :, :] for i, seqlen in enumerate(seqlens)])
    seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

    flash_fn = lambda: flash_attn_forward(q, k, v, seqlens, sm_scale, context_size)
    output = flash_fn()
    output = torch.stack([
        torch.nn.functional.pad(
            output[seqlens[i]:seqlens[i + 1], :, :],
            (0, 0, 0, 0, 0, context_size + seqlens[i] - seqlens[i + 1])
        )
        for i in range(batch_size)
    ]).swapaxes(1, 2).contiguous()
    if torch_test:
        torch.testing.assert_close(output, ref_o_dense, atol=1e-2, rtol=0)
    profile(flash_fn, 2. * head_dim * dense_mask_nnz, 'flash-dense')
    print('========================================\n')


def block_sparse_flash_attention_forward(
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
    block_index = torch_build_index(query, key, top_k, block_size_N, block_size_N)
    out = triton_block_sparse_forward(query, key, value, seqlens, block_index, sm_scale, block_size_M, block_size_N)
    return out[..., :context_size, :]
