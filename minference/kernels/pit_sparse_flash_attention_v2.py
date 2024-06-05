import torch
import numpy as np
import math

import triton
import triton.language as tl
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule

from flash_attn import flash_attn_varlen_func


@triton.jit
def triton_sparse_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count, block_offset, column_count, column_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, NNZ_S, NNZ_V,
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

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

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

    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        cols = start_n + offs_n
        n_mask = cols < seqlen
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
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

    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
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

    # write back O
    acc /= l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def triton_sparse_forward(
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens: torch.Tensor,    # [BATCH, ]
    block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    triton_sparse_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o


def torch_build_index(seqlens, vertical_indexes, slash_indexes, context_size, block_size_M=64, block_size_N=64):
    device = seqlens.device
    batch_size, num_heads, NNZ_S = slash_indexes.shape
    NNZ_V = vertical_indexes.shape[-1]
    num_rows = triton.cdiv(context_size, block_size_M)
    block_count = torch.zeros((batch_size, num_heads, num_rows), dtype=torch.int32)
    block_offset = torch.zeros((batch_size, num_heads, num_rows, NNZ_S), dtype=torch.int32)
    column_count = torch.zeros((batch_size, num_heads, num_rows), dtype=torch.int32)
    column_index = torch.zeros((batch_size, num_heads, num_rows, NNZ_V), dtype=torch.int32)

    for b in range(batch_size):
        seqlen = seqlens[b]
        for h in range(num_heads):
            for m, start_m in enumerate(range(0, seqlen, block_size_M)):
                end_m = start_m + block_size_M
                s = 0
                while slash_indexes[b, h, s] >= end_m:
                    s += 1
                s_idx = max(end_m - slash_indexes[b, h, s], block_size_M)
                s += 1
                range_start = s_idx - block_size_M
                range_end = s_idx
                tmp_blocks = []
                while s < NNZ_S:
                    s_idx = max(end_m - slash_indexes[b, h, s], block_size_M)
                    if s_idx > range_end + block_size_M:
                        tmp_blocks += list(range(range_start, range_end, block_size_N))
                        range_start = s_idx - block_size_M
                        range_end = s_idx
                    elif s_idx > range_end:
                        range_end += block_size_M
                    s += 1
                tmp_blocks += list(range(range_start, range_end, block_size_N))
                block_count[b, h, m] = len(tmp_blocks)
                block_offset[b, h, m, :len(tmp_blocks)] = torch.tensor(tmp_blocks, dtype=block_offset.dtype)
                tmp_columns = vertical_indexes[b, h].cpu().numpy().tolist()
                tmp_columns = [col for col in tmp_columns if col < range_end]
                for range_start in tmp_blocks:
                    range_end = range_start + block_size_N
                    tmp_columns = [col for col in tmp_columns if col < range_start or col >= range_end]
                column_count[b, h, m] = len(tmp_columns)
                column_index[b, h, m, :len(tmp_columns)] = torch.tensor(tmp_columns, dtype=block_offset.dtype)

    return block_count.to(device), block_offset.to(device), column_count.to(device), column_index.to(device)


PYCUDA_BUILD_INDEX_KERNEL_CODE = '''\
__device__ int min(int x, int y) {
    return x < y ? x : y;
}

__device__ int max(int x, int y) {
    return x > y ? x : y;
}

__device__ void save_blocks(int* block_offset, int range_start, int range_end, int block_size, int& block_count) {
    for (int idx = range_start; idx < range_end; idx += block_size) {
        block_offset[block_count++] = idx;
    }
}

__global__ void PYCUDA_BUILD_INDEX_KERNEL(
    const int* seqlens,           // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* block_count,             // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int N_HEADS,
    int N_ROWS,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int NNZ_V,
    int NNZ_S
) {
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int group_idx = blockIdx.z;

    int seqlen = seqlens[batch_idx];
    int block_idx_m = group_idx * blockDim.x + threadIdx.x;
    int start_m = block_idx_m * BLOCK_SIZE_M;
    if (start_m >= seqlen) {
        return;
    }
    int end_m = start_m + BLOCK_SIZE_M;
    vertical_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_V;
    slash_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_S;
    int row_offset = (batch_idx * N_HEADS + head_idx) * N_ROWS + block_idx_m;
    block_count += row_offset;
    block_offset += row_offset * NNZ_S;
    column_count += row_offset;
    column_index += row_offset * NNZ_V;

    int tmp_col_cnt = 0, tmp_blk_cnt = 0;
    int s = 0, v = 0;
    int v_idx = vertical_indexes[v++];
    int s_idx = slash_indexes[s++];
    while (s_idx >= end_m) {
        s_idx = slash_indexes[s++];
    }
    s_idx = max(end_m - s_idx, BLOCK_SIZE_M);
    int range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
    while (1) {
        if (v_idx < range_end) {
            if (v_idx < range_start) {
                column_index[tmp_col_cnt++] = v_idx;
            }
            if (v < NNZ_V) {
                v_idx = vertical_indexes[v++];
            } else {
                v_idx = end_m + BLOCK_SIZE_M;
            }
        } else {
            if (s < NNZ_S) {
                s_idx = max(end_m - slash_indexes[s++], BLOCK_SIZE_M);
            } else {
                save_blocks(block_offset, range_start, range_end, BLOCK_SIZE_N, tmp_blk_cnt);
                break;
            }
            if (s_idx > range_end + BLOCK_SIZE_M) {
                save_blocks(block_offset, range_start, range_end, BLOCK_SIZE_N, tmp_blk_cnt);
                range_start = s_idx - BLOCK_SIZE_M;
                range_end = s_idx;
            } else if (s_idx > range_end) {
                range_end += BLOCK_SIZE_M;
            }
        }
    }

    block_count[0] = tmp_blk_cnt;
    column_count[0] = tmp_col_cnt;
}
'''
PYCUDA_BUILD_INDEX_KERNEL = SourceModule(
    PYCUDA_BUILD_INDEX_KERNEL_CODE,
    options=['-std=c++14', '-O3'],
).get_function(f'PYCUDA_BUILD_INDEX_KERNEL')


def pycuda_build_index(seqlens, vertical_indexes, slash_indexes, context_size, block_size_M=64, block_size_N=64):
    batch_size, num_heads, NNZ_S = slash_indexes.shape
    NNZ_V = vertical_indexes.shape[-1]
    num_rows = triton.cdiv(context_size, block_size_M)
    block_count = torch.zeros((batch_size, num_heads, num_rows), dtype=torch.int32, device=seqlens.device)
    block_offset = torch.zeros((batch_size, num_heads, num_rows, NNZ_S), dtype=torch.int32, device=seqlens.device)
    column_count = torch.zeros((batch_size, num_heads, num_rows), dtype=torch.int32, device=seqlens.device)
    column_index = torch.zeros((batch_size, num_heads, num_rows, NNZ_V), dtype=torch.int32, device=seqlens.device)
    num_threads = 64
    # import ipdb; ipdb.set_trace()
    PYCUDA_BUILD_INDEX_KERNEL(
        seqlens, vertical_indexes, slash_indexes,
        block_count, block_offset, column_count, column_index,
        np.int32(num_heads), np.int32(num_rows),
        np.int32(block_size_M), np.int32(block_size_N),
        np.int32(NNZ_V), np.int32(NNZ_S),
        # grid=(triton.cdiv(num_rows, num_threads), N_HEADS, BATCH),
        grid=(num_heads, batch_size, triton.cdiv(num_rows, num_threads)),
        block=(num_threads, 1, 1),
    )
    return block_count, block_offset, column_count, column_index


def make_causal_mask(seqlens, device, context_size):
    batch_size = seqlens.shape[0]
    arange = torch.arange(context_size, dtype=torch.int32, device=device)
    causal_mask = arange[None, None, :, None] >= arange[None, None, None, :]
    causal_mask = causal_mask.repeat((batch_size, 1, 1, 1))
    for b, seqlen in enumerate(seqlens):
        causal_mask[b, :, seqlen:, :] = False
        causal_mask[b, :, :, seqlen:] = False
    return causal_mask


def make_finegrained_mask(vertical_indexes, slash_indexes, causal_mask, device):
    batch_size, num_heads, _ = vertical_indexes.shape
    context_size = causal_mask.shape[-1]
    arange = torch.arange(context_size, dtype=torch.int32, device=device)
    sparse_mask = torch.zeros((batch_size, num_heads, context_size, context_size), dtype=torch.bool, device=device)
    for b in range(batch_size):
        for h in range(num_heads):
            for vertical_index in vertical_indexes[b, h]:
                sparse_mask[b, h, :, vertical_index] = True
            for slash_index in slash_indexes[b, h]:
                sparse_mask[b, h].logical_or_(arange[:, None] - arange[None, :] == slash_index)
    sparse_mask.logical_and_(causal_mask)
    return sparse_mask


def make_block_mask(
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    seqlens: torch.Tensor,
    causal_mask: torch.Tensor,
    device: torch.device,
    block_size_M: int = 64,
    block_size_N: int = 64.
):
    batch_size, num_heads, _ = block_count.shape
    context_size = causal_mask.shape[-1]
    block_mask = torch.zeros((batch_size, num_heads, context_size, context_size), dtype=torch.bool, device=device)
    for b in range(batch_size):
        for h in range(num_heads):
            for m, start_m in enumerate(range(0, seqlens[b], block_size_M)):
                end_m = start_m + block_size_M
                for col_idx in range(column_count[b, h, m]):
                    block_mask[b, h, start_m:end_m, column_index[b, h, m, col_idx]] = True
                for blk_idx in range(block_count[b, h, m]):
                    blk_start = block_offset[b, h, m, blk_idx].item()
                    blk_end = blk_start + block_size_N
                    block_mask[b, h, start_m:end_m, blk_start:blk_end] = True
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
    query=None,
    key=None,
    value=None,
    seqlens=None,
    vertical_indexes=None,
    slash_indexes=None,
    dtype=torch.float16,
    device="cuda",
    torch_test=True,
    batch_size=4,
    num_heads=32,
    context_size=2048,
    head_dim=128,
    nnz_v=100,
    nnz_s=10,
    block_size_M=64,
    block_size_N=64,
):
    print('========================================')
    if query is None and key is None and value is None:
        q = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
        k = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
        v = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=dtype, device=device)
    else:
        q = torch.tensor(query, dtype=dtype, device=device)
        k = torch.tensor(key, dtype=dtype, device=device)
        v = torch.tensor(value, dtype=dtype, device=device)
        batch_size, num_heads, context_size, head_dim = q.shape
    print(f'BATCH={batch_size}, N_CTX={context_size}, N_HEADS={num_heads}, D_HEAD={head_dim}')
    if seqlens is None:
        seqlens = torch.randint(context_size // 2, context_size, (batch_size, ), dtype=torch.int32, device=device)
    else:
        seqlens = torch.tensor(seqlens, dtype=torch.int32, device=device)
    print(seqlens)
    dense_mask_nnz = seqlens.to(torch.float32).square().sum().item() * num_heads / 2
    sm_scale = head_dim ** -0.5

    if torch_test:
        causal_mask = make_causal_mask(seqlens, device, context_size)
        ref_o_dense = torch_forward(q, k, v, causal_mask, sm_scale)

    if vertical_indexes is None or slash_indexes is None:
        vertical_indexes = torch.stack([
            torch.stack([
                torch.randperm(seqlen, dtype=torch.int32, device=device)[:nnz_v].sort(descending=False)[0]
                for _ in range(num_heads)
            ])
            for seqlen in seqlens
        ])
        slash_indexes = torch.concatenate([
            torch.stack([
                torch.stack([
                    torch.randperm(seqlen - 1, dtype=torch.int32, device=device)[:nnz_s - 1].sort(descending=True)[0] + 1
                    for _ in range(num_heads)
                ])
                for seqlen in seqlens
            ]),
            torch.zeros((batch_size, num_heads, 1), dtype=torch.int32, device=device)
        ], dim=-1)
    pycuda_build_index_fn = lambda: pycuda_build_index(
        seqlens, vertical_indexes, slash_indexes, context_size, block_size_M, block_size_N
    )
    indexes = pycuda_build_index_fn()
    block_count, block_offset, column_count, column_index = indexes
    if torch_test:
        block_count_ref, block_offset_ref, column_count_ref, column_index_ref = torch_build_index(
            seqlens, vertical_indexes, slash_indexes, context_size, block_size_M, block_size_N
        )
        torch.testing.assert_close(block_count_ref, block_count)
        torch.testing.assert_close(block_offset_ref, block_offset)
        torch.testing.assert_close(column_count_ref, column_count)
        torch.testing.assert_close(column_index_ref, column_index)
    sparse_mask_nnz = column_count.to(torch.float64).sum().item() * block_size_M + \
        block_count.to(torch.float64).sum().item() * block_size_M * block_size_N
    print(f'block mask sparsity: {1 - sparse_mask_nnz / dense_mask_nnz}')

    pycuda_build_index_fn = lambda: pycuda_build_index(
        seqlens, vertical_indexes, slash_indexes, context_size, block_size_M, block_size_N
    )
    profile(pycuda_build_index_fn, 0., 'pycuda-index')

    if torch_test:
        finegrained_mask = make_finegrained_mask(vertical_indexes, slash_indexes, causal_mask, device)
        block_mask = make_block_mask(*indexes, seqlens, causal_mask, device, block_size_M, block_size_N)
        plot_mask(finegrained_mask, 'mask.png', 0, 0)
        plot_mask(block_mask, 'mask-1.png', 0, 0)
        ref_o_sparse = torch_forward(q, k, v, block_mask, sm_scale)

    triton_dense_fn = lambda: triton_dense_forward(q, k, v, seqlens, sm_scale)
    output_triton_dense = triton_dense_fn()
    if torch_test:
        # Note: not correct for context_size % block_size_M != 0
        torch.testing.assert_close(output_triton_dense, ref_o_dense, atol=1e-2, rtol=0)
    profile(triton_dense_fn, 2. * head_dim * dense_mask_nnz, 'triton-dense')

    triton_sparse_fn = lambda: triton_sparse_forward(q, k, v, seqlens, *indexes, sm_scale, block_size_M, block_size_N)
    output_triton_sparse = triton_sparse_fn()
    if torch_test:
        torch.testing.assert_close(output_triton_sparse, ref_o_sparse, atol=1e-2, rtol=0)
    profile(triton_sparse_fn, 2. * head_dim * sparse_mask_nnz, 'triton-sparse')

    q = q.swapaxes(1, 2).contiguous()
    k = k.swapaxes(1, 2).contiguous()
    v = v.swapaxes(1, 2).contiguous()
    q = torch.concatenate([q[i, :seqlen, :, :] for i, seqlen in enumerate(seqlens)])
    k = torch.concatenate([k[i, :seqlen, :, :] for i, seqlen in enumerate(seqlens)])
    v = torch.concatenate([v[i, :seqlen, :, :] for i, seqlen in enumerate(seqlens)])
    seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

    flash_fn = lambda: flash_attn_forward(q, k, v, seqlens, sm_scale, context_size)
    output_flash = flash_fn()
    output_flash = torch.stack([
        torch.nn.functional.pad(
            output_flash[seqlens[i]:seqlens[i + 1], :, :],
            (0, 0, 0, 0, 0, context_size + seqlens[i] - seqlens[i + 1])
        )
        for i in range(batch_size)
    ]).swapaxes(1, 2).contiguous()
    if torch_test:
        torch.testing.assert_close(output_flash, ref_o_dense, atol=1e-2, rtol=0)
    profile(flash_fn, 2. * head_dim * dense_mask_nnz, 'flash-dense')
    print('========================================\n')

    if torch_test and sparse_mask_nnz >= dense_mask_nnz:
        torch.testing.assert_close(output_flash, output_triton_sparse, atol=1e-2, rtol=0)


def pit_sparse_flash_attention_forward(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = block_size_M - (context_size & (block_size_M - 1))
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    block_count, block_offset, column_count, column_index = pycuda_build_index(
        seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N,
    )
    # if context_size > 700000:
    #     import ipdb; ipdb.set_trace()
    # dense_mask_nnz = seqlens.to(torch.float32).square().sum().item() * num_heads / 2
    # sparse_mask_nnz = column_count.to(torch.float64).sum().item() * block_size_M + \
    #     block_count.to(torch.float64).sum().item() * block_size_M * block_size_N
    # print(f'block mask sparsity: {1 - sparse_mask_nnz / dense_mask_nnz}')
    out = triton_sparse_forward(
        query, key, value, seqlens,
        block_count, block_offset, column_count, column_index,
        sm_scale, block_size_M, block_size_N,
    )
    return out[..., :context_size, :head_dim]
