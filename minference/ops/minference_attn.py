import os
import sys
import math

import torch
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl

from typing import List, Tuple

# Save current flags
if torch.version.hip is None:
    original_flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
        import block_sparse_attn_cuda
        from block_sparse_attn.block_sparse_attn_interface import convert_blockmask_row_reverse, convert_blockmask_col_reverse
        # NOTE: Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_blockmask.h: add head_idx to blockmask_ptr
    finally:
        # Restore original flags for future imports
        sys.setdlopenflags(original_flags)
    # NOTE: Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_blockmask.h: add head_idx to blockmask_ptr

from .utils import build_index_local


def block_attn_fwd(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    softmax_scale: float,
    block_mask: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, num_blocks]
    granularity: int,
    causal: bool,
    step_idx: int=-1,
):
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    cu_seqlens = torch.arange(0, (batch_size + 1) * num_tokens, step=num_tokens, dtype=torch.int32, device=q.device)
    head_mask_type = torch.ones((num_qo_heads, ), dtype=torch.int32, device=q.device)  # Block-Sparse
    streaming_info = torch.zeros((num_qo_heads * 2), dtype=torch.int32, device=q.device)
    row_blockmask = convert_blockmask_row_reverse(block_mask, causal=True)

    p_dropout = 0.0
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = block_sparse_attn_cuda.fwd_block(
        q.reshape((-1, num_qo_heads, head_dim)),
        k.reshape((-1, num_kv_heads, head_dim)),
        v.reshape((-1, num_kv_heads, head_dim)),
        cu_seqlens, cu_seqlens,
        granularity, granularity,
        head_mask_type,
        streaming_info,
        row_blockmask,
        num_tokens, num_tokens,
        p_dropout,
        softmax_scale,
        causal,  # is_causal
        False,  # exact_streaming
        False,  # return_softmax
        -1,  # window_size_left
        -1,  # window_size_right
        None
    )
    out = out.reshape((batch_size, num_tokens, num_qo_heads, head_dim))
    return out, softmax_lse


def block_attn_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    o: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    softmax_lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    softmax_scale: float,
    block_mask: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, num_blocks]
    granularity: int,
    deterministic: bool,
    causal: bool,
    converted: bool = False,
):
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    cu_seqlens = torch.arange(0, (batch_size + 1) * num_tokens, step=num_tokens, dtype=torch.int32, device=q.device)
    head_mask_type = torch.ones((num_qo_heads, ), dtype=torch.int32, device=q.device)  # Block-Sparse
    streaming_info = torch.zeros((num_qo_heads * 2), dtype=torch.int32, device=q.device)
    if converted:
        col_blockmask = block_mask
    else:
        col_blockmask = convert_blockmask_col_reverse(block_mask, causal=True)
    p_dropout = 0.0
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dq, dk, dv, softmax_d = block_sparse_attn_cuda.bwd_block(
        grad.reshape((-1, num_qo_heads, head_dim)),
        q.reshape((-1, num_qo_heads, head_dim)),
        k.reshape((-1, num_kv_heads, head_dim)),
        v.reshape((-1, num_kv_heads, head_dim)),
        o.reshape((-1, num_qo_heads, head_dim)),
        softmax_lse,
        dq.reshape((-1, num_qo_heads, head_dim)),
        dk.reshape((-1, num_kv_heads, head_dim)),
        dv.reshape((-1, num_kv_heads, head_dim)),
        cu_seqlens, cu_seqlens,
        granularity, granularity,
        head_mask_type,
        streaming_info,
        col_blockmask,
        num_tokens, num_tokens,
        p_dropout,
        softmax_scale,
        True,  # zero_tensors
        causal,  # is_causal
        -1,  # window_size_left
        -1,  # window_size_right
        deterministic,
        None, None
    )
    dq = dq.reshape((batch_size, num_tokens, num_qo_heads, head_dim))
    dk = dk.reshape((batch_size, num_tokens, num_kv_heads, head_dim))
    dv = dv.reshape((batch_size, num_tokens, num_kv_heads, head_dim))
    return dq, dk, dv


@triton.jit
def _triton_bar_attn_fwd_kernel(
    Q, K, V, sm_scale,
    bar_cnt, # [BATCH, N_Q_HEADS, NUM_ROWS, WORLD_SIZE + 1]
    bar_idx, # [BATCH, N_Q_HEADS, NUM_ROWS, NNZ_V]
    Out, # [BATCH, N_Q_HEADS, N_CTX, D_HEAD]
    softmax_lse, # [BATCH, N_Q_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_cz, stride_ch, stride_cm, stride_cr,
    stride_iz, stride_ih, stride_im, stride_in,
    stride_sz, stride_sh, stride_sm,
    step, num_qo_heads, num_kv_heads, num_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    qo_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    kv_head_idx = qo_head_idx // (num_qo_heads // num_kv_heads)

    if start_m * BLOCK_M >= num_tokens:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    m_mask = offs_m < num_tokens

    qo_offset = batch_idx * stride_qz + qo_head_idx * stride_qh
    kv_offset = batch_idx * stride_kz + kv_head_idx * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kd
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vd
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    lse_ptrs = softmax_lse + batch_idx * stride_sz + qo_head_idx * stride_sh + offs_m * stride_sm

    bar_l = tl.load(bar_cnt + batch_idx * stride_cz + qo_head_idx * stride_ch + start_m * stride_cm + step * stride_cr)
    bar_r = tl.load(bar_cnt + batch_idx * stride_cz + qo_head_idx * stride_ch + start_m * stride_cm + (step + 1) * stride_cr)
    bar_idx_ptr = bar_idx + batch_idx * stride_iz + qo_head_idx * stride_ih + start_m * stride_im

    if bar_l >= bar_r:
        return

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # 1/ln2 = lne/ln2 = log2(e) => 2^(x / ln2) = 2^(x * log2(e)) = (2^(log2(e)))^x = e^x
    qk_scale = sm_scale * 1.44269504

    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0)
    q = (q * qk_scale).to(Q.type.element_ty)

    # loop over k, v and update accumulator
    for start_n in range(bar_l, bar_r, BLOCK_N):
        n_mask = start_n + offs_n < bar_r
        cols = tl.load(bar_idx_ptr + (start_n + offs_n) * stride_in, mask=n_mask, other=0)

        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask[:, None] & n_mask[None, :], qk, float("-inf"))
        qk = qk + tl.dot(q, k)

        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc = acc * acc_scale[:, None]
        acc = acc + tl.dot(p.to(Q.type.element_ty), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O and LSE
    acc_1 = acc / l_i[:, None]
    s_1 = m_i * 0.69314718 + tl.math.log(l_i)
    acc_0 = tl.load(o_ptrs, mask=m_mask[:, None], other=0.).to(tl.float32)
    s_0 = tl.load(lse_ptrs, mask=m_mask, other=float("-inf"))

    overflow_mask = (s_0 - s_1) < 88.0

    theta = tl.math.exp(s_0 - s_1)
    alpha_0 = 1 / (1 + 1 / theta)
    alpha_1 = 1 / (1 + theta)
    acc = alpha_0[:, None] * acc_0 + alpha_1[:, None] * acc_1
    s = s_1 - tl.math.log(alpha_1)

    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=m_mask[:, None])
    tl.store(lse_ptrs, s, mask=(m_mask & overflow_mask))

def stable_sigmoid(x: torch.Tensor):
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),
        torch.exp(x) / (1 + torch.exp(x))
    )

def naive_bar_attn_fwd(
        q, k, v, 
        sm_scale, 
        bar_cnt, bar_idx, 
        out, softmax_lse,
        step, granularity, BLOCK_N=64
    ):
    """
    Naive PyTorch implementation of the Triton bar attention forward kernel.
    
    Args:
      q: Query tensor of shape [B, num_qo_heads, num_tokens, head_dim]
      k: Key tensor of shape [B, num_kv_heads, num_tokens,  head_dim]
      v: Value tensor of shape [B, num_kv_heads, num_tokens, head_dim]
      sm_scale: A scalar (float) softmax scale.
      bar_cnt: Tensor of shape [B, num_qo_heads, num_blocks, world_size+1]
               where each block (row) holds bar boundary indices.
      bar_idx: Tensor of shape [B, num_qo_heads, num_blocks, nnz_v]
               containing indices of keys (columns) for each block.
      out: Output tensor of shape [B, num_qo_heads, num_tokens, head_dim].
           This is assumed to have a previous value to merge with.
      softmax_lse: Tensor of shape [B, num_qo_heads, num_tokens] containing
                   the previous log-sum-exp values.
      step: integer step indicating which pair of boundaries to use in bar_cnt.
      granularity: BLOCK_M, i.e. the number of query tokens processed per block.
      BLOCK_N: Block size for the key dimension (default: 64)
      
    This function updates `out` and `softmax_lse` in-place.
    """
    # Get dimensions from q.
    B, num_tokens, num_qo_heads, head_dim = q.shape

    # Determine number of query blocks (each corresponding to a row in bar_cnt/bar_idx).
    num_blocks = math.ceil(num_tokens / granularity)

    # Compute the ratio for mapping query-head to key/value head.
    head_ratio = num_qo_heads // k.shape[2]  # since k.shape[1] is num_kv_heads

    # Precompute scale for q: note that 1.44269504 = log2(e)
    qk_scale = sm_scale * 1.44269504

    ln2 = 0.69314718  # constant for converting exp2 to exp

    # Loop over batch and query-head
    for b in range(B):
        for qh in range(num_qo_heads):
            # corresponding key/value head index
            kvh = qh // head_ratio

            # Loop over query blocks (rows)
            for block in range(num_blocks):
                start_m = block * granularity
                end_m = min(start_m + granularity, num_tokens)
                block_size = end_m - start_m

                # Get bar boundaries for this block & step:
                # bar_cnt is assumed to store cumulative indices per block.
                bar_l = bar_cnt[b, qh, block, step].item()   # starting index (inclusive)
                bar_r = bar_cnt[b, qh, block, step + 1].item() # ending index (exclusive)
                if bar_l >= bar_r:
                    continue  # nothing to do in this block

                # Initialize accumulators per query token in the block.
                # m_i tracks the running maximum (in "log2" domain).
                m_i = torch.full((block_size,), -float('inf'), device=q.device, dtype=torch.float32)
                # l_i tracks the running sum-of-weights.
                l_i = torch.zeros(block_size, device=q.device, dtype=torch.float32)
                # acc accumulates the weighted sum of values.
                acc = torch.zeros((block_size, head_dim), device=q.device, dtype=torch.float32)

                # Load and scale the q block.
                # Shape: [block_size, head_dim]
                q_block = q[b, start_m:end_m, qh, :] * qk_scale

                # Loop over key indices in steps of BLOCK_N
                for n_start in range(bar_l, bar_r, BLOCK_N):
                    n_end = min(n_start + BLOCK_N, bar_r)
                    
                    # Load column indices from bar_idx.
                    # bar_idx shape: [nnz_v] for this block.
                    cols = bar_idx[b, qh, block, n_start:n_end]
                    cols = cols.long()
                    
                    k_selected = k[b, cols, kvh, :]  # shape: [n_valid, head_dim]
                    v_selected = v[b, cols, kvh, :]  # shape: [n_valid, head_dim]

                    # Compute scaled dot product: [block_size, head_dim] x [head_dim, n_valid]
                    # Result: [block_size, n_valid]
                    qk = torch.matmul(q_block, k_selected.T)

                    # Numerically stable softmax update in the log2 domain.
                    # m_i_new = max(m_i, max(qk, dim=1))
                    cur_max, _ = qk.max(dim=1)
                    m_i_new = torch.max(m_i, cur_max)

                    alpha = torch.exp2((m_i - m_i_new))
                    p = torch.exp2((qk - m_i_new.unsqueeze(1)))
                    
                    # Update acc and l_i.
                    # Scale previous acc by alpha.
                    acc = acc * alpha.unsqueeze(1) + torch.matmul(p.to(q.dtype), v_selected)

                    l_i = l_i * alpha + p.sum(dim=1)

                    # Update m_i to the new maximum.
                    m_i = m_i_new

                # check zeros in l_i, if any, print out the indices
                if (l_i == 0).any():
                    zero_indices = torch.nonzero(l_i == 0).squeeze()
                    print(f"Rank {dist.get_rank()} | Zeros in l_i (step = {step}, batch_idx={b}, head_idx={qh}, block={block}): {zero_indices}")

                # Finalize the block output.
                # Compute weighted output.
                acc_1 = acc / l_i.unsqueeze(1)
                s_1 = m_i * ln2 + torch.log(l_i)
                # check positive infinity in s_1, if any, print out the indices
                if torch.isinf(s_1).any() and ( torch.isinf(s_1) & (s_1 > 0) ).any():
                    mask = torch.isinf(s_1) & (s_1 > 0)
                    print(f"Rank {dist.get_rank()} | Positive infinity in s_1 (step = {step}, batch_idx={b}, head_idx={qh}, block={block}): {torch.nonzero(mask).squeeze()}")
                # check negative infinity in s_1, if any, print out the indices
                if torch.isinf(s_1).any() and ( torch.isinf(s_1) & (s_1 < 0) ).any():
                    mask = torch.isinf(s_1) & (s_1 < 0)
                    print(f"Rank {dist.get_rank()} | Negative infinity in s_1 (step = {step}, batch_idx={b}, head_idx={qh}, block={block}): {torch.nonzero(mask).squeeze()}")

                # Load previous stored values (accumulated output and LSE).
                old_out = out[b, start_m:end_m, qh, :].to(acc_1.dtype)
                old_lse = softmax_lse[b, qh, start_m:end_m]
                # check positive infinity in old_lse, if any, print out the indices
                if torch.isinf(old_lse).any() and ( torch.isinf(old_lse) & (old_lse > 0) ).any():
                    mask = torch.isinf(old_lse) & (old_lse > 0)
                    print(f"Rank {dist.get_rank()} | Positive infinity in old_lse (step = {step}, batch_idx={b}, head_idx={qh}, block={block}): {torch.nonzero(mask).squeeze()}")

                # -------------------------------------------------
                # Logsigmoid solution
                # out - old_out, block_out - acc1, lse - old_lse, block_lse - s_1
                new_out = old_out - F.sigmoid(s_1 - old_lse).unsqueeze(1) * (old_out - acc_1)
                new_lse = s_1 - F.logsigmoid(s_1 - old_lse)
                if torch.isinf(new_lse).any() and ( torch.isinf(new_lse) & (new_lse > 0) ).any():
                    mask = torch.isinf(new_lse) & (new_lse > 0)
                    print(f"Rank {dist.get_rank()} | Positive infinity in new_lse (step = {step}, batch_idx={b}, head_idx={qh}, block={block}): {torch.nonzero(mask).squeeze()}")

                    pos_inf_indices = torch.nonzero(mask).squeeze()
                    print(f"Rank {dist.get_rank()} | Values of (old_lse - s_1) resulting in pos-inf in theta (step = {step}, batch_idx={b}, head_idx={qh}, block={block}): {(old_lse - s_1)[pos_inf_indices]}")
                    print(f"Rank {dist.get_rank()} | Values of (old_lse) resulting in pos-inf in theta (step = {step}, batch_idx={b}, head_idx={qh}, block={block}): {(old_lse)[pos_inf_indices]}")
                    print(f"Rank {dist.get_rank()} | Values of (s_1) resulting in pos-inf in theta (step = {step}, batch_idx={b}, head_idx={qh}, block={block}): {(s_1)[pos_inf_indices]}")

                # Store back into out and softmax_lse.
                out[b, start_m:end_m, qh, :] = new_out.to(out.dtype)
                softmax_lse[b, qh, start_m:end_m] = new_lse
    return out, softmax_lse


def bar_attn_fwd(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    o: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    softmax_scale: float,
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int,
    step: int = 0,
):
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_blocks = bar_idx.shape[2]
    _triton_bar_attn_fwd_kernel[(num_blocks, num_qo_heads, batch_size)](
        q, k, v, softmax_scale, bar_cnt, bar_idx, o, lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        bar_cnt.stride(0), bar_cnt.stride(1), bar_cnt.stride(2), bar_cnt.stride(3),
        bar_idx.stride(0), bar_idx.stride(1), bar_idx.stride(2), bar_idx.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        step, num_qo_heads, num_kv_heads, num_tokens,
        BLOCK_M=granularity, BLOCK_N=64, BLOCK_DMODEL=head_dim,
        num_warps=4, num_stages=2,
    )
    return o, lse


@triton.jit
def _triton_bar_attn_bwd_kernel(
    Q, K, V, O,
    DQ, DK, DV, DO,
    sm_scale,
    bar_cnt, # [BATCH, N_Q_HEADS, NUM_ROWS, WORLD_SIZE + 1]
    bar_idx, # [BATCH, N_Q_HEADS, NUM_ROWS, NNZ_V]
    softmax_lse, # [BATCH, N_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_cz, stride_ch, stride_cm, stride_cr,
    stride_iz, stride_ih, stride_im, stride_in,
    stride_sz, stride_sh, stride_sm,
    step, num_qo_heads, num_kv_heads, num_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    qo_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    kv_head_idx = qo_head_idx // (num_qo_heads // num_kv_heads)

    if start_m * BLOCK_M >= num_tokens:
        return

    qk_scale = sm_scale * 1.44269504

    # offset pointers for batch/head
    Q += batch_idx * stride_qz + qo_head_idx * stride_qh
    K += batch_idx * stride_kz + kv_head_idx * stride_kh
    V += batch_idx * stride_vz + kv_head_idx * stride_vh
    O += batch_idx * stride_oz + qo_head_idx * stride_oh
    DQ += batch_idx * stride_dqz + qo_head_idx * stride_dqh
    DK += batch_idx * stride_dkz + kv_head_idx * stride_dkh
    DV += batch_idx * stride_dvz + kv_head_idx * stride_dvh
    DO += batch_idx * stride_doz + qo_head_idx * stride_doh

    # loop over rows
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens

    # initialize pointers to value-like data
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + offs_d[None, :] * stride_kd
    v_ptrs = V + offs_d[None, :] * stride_vd
    o_ptrs = O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    dq_ptrs = DQ + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    dk_ptrs = DK + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + offs_d[None, :] * stride_dvd
    do_ptrs = DO + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod

    l_ptrs = softmax_lse + batch_idx * stride_sz + qo_head_idx * stride_sh + offs_m * stride_sm

    bar_l = tl.load(bar_cnt + batch_idx * stride_cz + qo_head_idx * stride_ch + start_m * stride_cm + step * stride_cr)
    bar_r = tl.load(bar_cnt + batch_idx * stride_cz + qo_head_idx * stride_ch + start_m * stride_cm + (step + 1) * stride_cr)
    bar_idx_ptr = bar_idx + batch_idx * stride_iz + qo_head_idx * stride_ih + start_m * stride_im

    if bar_l >= bar_r:
        return

    o = tl.load(o_ptrs, mask=m_mask[:, None], other=0.).to(tl.float32)
    do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.).to(tl.float32)
    d_i = tl.sum(o * do, axis=1)

    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.)
    do = do.to(DO.dtype.element_ty)
    l_i = tl.load(l_ptrs, mask=m_mask, other=0.) * 1.44269504

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(bar_l, bar_r, BLOCK_N):
        n_mask = start_n + offs_n < bar_r
        cols = tl.load(bar_idx_ptr + (start_n + offs_n) * stride_in, mask=n_mask, other=0)

        # -- load k, v --
        k = tl.load(k_ptrs + cols[:, None] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)

        # Computer qk
        qk = tl.where(m_mask[:, None] & n_mask[None, :], float(0.), float("-inf"))
        qk = qk + tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do).to(tl.float32)
        tl.atomic_add(dv_ptrs + cols[:, None] * stride_dvn, dv_vals, mask=n_mask[:, None], sem="relaxed")

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - d_i[:, None]
        dp = dp + tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q).to(tl.float32)
        tl.atomic_add(dk_ptrs + cols[:, None] * stride_dkn, dk_vals, mask=n_mask[:, None], sem="relaxed")

        # compute dq
        dq = dq + tl.dot(ds.to(Q.dtype.element_ty), k)

    dq_old = tl.load(dq_ptrs, mask=m_mask[:, None], other=0.).to(tl.float32)
    tl.store(dq_ptrs, (dq_old + dq).to(DQ.dtype.element_ty), mask=m_mask[:, None])


def bar_attn_bwd(
    grad: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    o: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    dq: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    dk: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    dv: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    softmax_lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    softmax_scale: float,
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    granularity: int,
    deterministic: bool,
    step: int = 0,
):
    assert not deterministic
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_blocks = bar_idx.shape[2]
    dq = torch.zeros_like(q, dtype=torch.float32) if dq is None else dq.to(torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32) if dk is None else dk.to(torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32) if dv is None else dv.to(torch.float32)
    _triton_bar_attn_bwd_kernel[(num_blocks, num_qo_heads, batch_size)](
        q, k, v, o, dq, dk, dv, grad, softmax_scale,
        bar_cnt, bar_idx, softmax_lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3),
        dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3),
        dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3),
        grad.stride(0), grad.stride(2), grad.stride(1), grad.stride(3),
        bar_cnt.stride(0), bar_cnt.stride(1), bar_cnt.stride(2), bar_cnt.stride(3),
        bar_idx.stride(0), bar_idx.stride(1), bar_idx.stride(2), bar_idx.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        step, num_qo_heads, num_kv_heads, num_tokens,
        BLOCK_M=granularity, BLOCK_N=64, BLOCK_DMODEL=head_dim,
        num_warps=4, num_stages=2,
    )
    return dq, dk.to(dq.dtype), dv.to(dq.dtype)



class MInferenceAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        v_size,
        s_size,
        softmax_scale,
        granularity,
        return_softmax,
        deterministic,
    ):
        batch_size, num_tokens, num_qo_heads, head_dim = q.shape
        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        block_mask, bar_idx, bar_cnt = build_index_local(q, k, v_size, s_size, num_tokens, granularity)

        # Block Mask
        out, softmax_lse = block_attn_fwd(
            q, k, v, softmax_scale,
            block_mask,
            granularity=granularity,
            causal=True,
        )
        # Bar Mask
        out, softmax_lse = bar_attn_fwd(
            q, k, v, out, softmax_lse, softmax_scale,
            bar_idx, bar_cnt,
            granularity=granularity,
            step=0,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse, block_mask, bar_idx, bar_cnt)
        ctx.granularity = granularity
        ctx.deterministic = deterministic
        ctx.softmax_scale = softmax_scale
        return (out, softmax_lse, None) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, block_mask, bar_idx, bar_cnt = ctx.saved_tensors
        # Block Mask
        dq, dk, dv = block_attn_bwd(
            dout, q, k, v, out,
            softmax_lse, ctx.softmax_scale,
            block_mask,
            granularity=ctx.granularity,
            deterministic=ctx.deterministic,
            causal=True,
        )

        # Bar Mask
        dq, dk, dv = bar_attn_bwd(
            dout, q, k, v, out, dq, dk, dv,
            softmax_lse, ctx.softmax_scale,
            bar_idx, bar_cnt,
            granularity=ctx.granularity,
            deterministic=ctx.deterministic,
            step=0,
        )
        return dq, dk, dv, None, None, None, None, None, None


def minference_flash_attn_qkvpacked_func(
    qkv: torch.Tensor,  # [batch_size, num_tokens, 3, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    assert dropout_p == 0
    assert causal
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    return MInferenceAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        v_size,
        s_size,
        softmax_scale,
        granularity,
        return_attn_probs,
        deterministic,
    )


def minference_flash_attn_kvpacked_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    kv: torch.Tensor,  # [batch_size, num_tokens, 2, num_kv_heads, head_dim]
    v_size: List[int],  # [num_qo_heads]
    s_size: List[int],  # [num_qo_heads]
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    assert dropout_p == 0
    assert causal
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    return MInferenceAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        v_size,
        s_size,
        softmax_scale,
        granularity,
        return_attn_probs,
        deterministic,
    )


def minference_flash_attn_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v_size: List[int],  # [num_qo_heads]
    s_size: List[int],  # [num_qo_heads]
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    assert dropout_p == 0
    assert causal
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    return MInferenceAttnFunc.apply(
        q,
        k,
        v,
        v_size,
        s_size,
        softmax_scale,
        granularity,
        return_attn_probs,
        deterministic,
    )


def _build_mask_local(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v_size: List[int],
    s_size: List[int],
    num_tokens: int,
    granularity: int,
    world_size: int = 1,
    rank: int = 0,
):
    with torch.no_grad():
        block_mask, bar_idx, bar_cnt = build_index_local(q, k, v_size, s_size, num_tokens, granularity, world_size, rank)
        batch_size, num_tokens, num_heads, head_dim = q.shape
        num_blocks = block_mask.shape[-1]
        num_tokens_pad = num_blocks * granularity
        # Block Mask
        mask = block_mask.unsqueeze(3).unsqueeze(5).repeat((1, 1, 1, granularity, 1, granularity))
        mask = mask.reshape((batch_size, num_heads, num_tokens_pad, num_tokens_pad))
        # Bar Mask
        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                for row_idx in range(num_blocks):
                    row_u = row_idx * granularity
                    row_d = row_u + granularity
                    bar_l = bar_cnt[batch_idx, head_idx, row_idx, rank]
                    bar_r = bar_cnt[batch_idx, head_idx, row_idx, rank + 1]
                    for col_idx in bar_idx[batch_idx, head_idx, row_idx, bar_l:bar_r]:
                        mask[batch_idx, head_idx, row_u:row_d, col_idx] = True
        # Causal Mask
        arange = torch.arange(0, num_tokens_pad, dtype=torch.int32, device=q.device)
        mask.masked_fill_(arange[None, None, :, None] < arange[None, None, None, :], False)
    return mask[:, :, :num_tokens, :num_tokens]


def _torch_sparse_attn_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v_size: List[int],  # [num_qo_heads]
    s_size: List[int],  # [num_qo_heads]
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    assert dropout_p == 0
    assert causal
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    group_size = num_qo_heads // num_kv_heads
    softmax_scale = head_dim ** -0.5 if softmax_scale is None else softmax_scale
    mask = _build_mask_local(q, k, v_size, s_size, num_tokens, granularity)
    k = k.repeat_interleave(group_size, dim=2)
    v = v.repeat_interleave(group_size, dim=2)
    p = torch.einsum('bmhd, bnhd -> bhmn', q * softmax_scale, k)
    p = torch.where(mask, p, -torch.inf).to(torch.float32)
    m = torch.max(p, dim=-1, keepdim=True).values.to(torch.float32)
    p = torch.exp(p - m)
    l = torch.sum(p, dim=-1, keepdim=True)
    p = (p / l).to(q.dtype)
    o = torch.einsum('bhmn, bnhd -> bmhd', p, v)
    o = o.reshape((batch_size, num_tokens, num_qo_heads, head_dim))
    if return_attn_probs:
        lse = m + l.log()
        return o, lse.squeeze(-1), None
    return o


def _torch_sparse_attn_qkvpacked_func(
    qkv: torch.Tensor,  # [batch_size, num_tokens, 3, num_heads, head_dim]
    v_size: List[int],  # [num_heads]
    s_size: List[int],  # [num_heads]
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    return _torch_sparse_attn_func(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        v_size,
        s_size,
        dropout_p,
        softmax_scale,
        granularity,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def _torch_sparse_attn_kvpacked_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    kv: torch.Tensor,  # [batch_size, num_tokens, 2, num_kv_heads, head_dim]
    v_size: List[int],  # [num_qo_heads]
    s_size: List[int],  # [num_qo_heads]
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    granularity: int = 128,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    return _torch_sparse_attn_func(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        v_size,
        s_size,
        dropout_p,
        softmax_scale,
        granularity,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )
