import torch
import triton
import triton.language as tl
from typing import List, Tuple

from .op_utils.vertical_slash_utils import (
    build_index_local, _build_mask_local, convert_blockmask,
)


@triton.jit
def _triton_block_attn_fwd_kernel(
    Q, K, V, sm_scale,
    block_cnt,  # [BATCH, N_Q_HEADS, NUM_ROWS]
    block_idx,  # [BATCH, N_Q_HEADS, NUM_ROWS, NUM_COLS]
    Out, # [BATCH, N_Q_HEADS, N_CTX, D_HEAD]
    softmax_lse, # [BATCH, N_Q_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_2cz, stride_2ch, stride_2cm,
    stride_2iz, stride_2ih, stride_2im, stride_2in,
    stride_sz, stride_sh, stride_sm,
    num_qo_heads, num_kv_heads, num_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
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

    qo_offset = batch_idx * stride_qz + qo_head_idx * stride_qh
    kv_offset = batch_idx * stride_kz + kv_head_idx * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kd
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vd
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    lse_ptrs = softmax_lse + batch_idx * stride_sz + qo_head_idx * stride_sh + offs_m * stride_sm

    block_num = tl.load(block_cnt + batch_idx * stride_2cz + qo_head_idx * stride_2ch + start_m * stride_2cm)
    if block_num <= 0:
        return

    block_idx_ptr = block_idx + batch_idx * stride_2iz + qo_head_idx * stride_2ih + start_m * stride_2im

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
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(Q.type.element_ty)
    
    if CAUSAL:
        block_split = block_num - 2
    else:
        block_split = block_num

    # Block
    for start_n in range(0, block_split):
        block_off = tl.load(block_idx_ptr + start_n * stride_2in) * BLOCK_N

        # -- load k, v --
        k = tl.load(k_ptrs + block_off * stride_kn + offs_n[None, :] * stride_kn)
        v = tl.load(v_ptrs + block_off * stride_vn + offs_n[:, None] * stride_vn)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
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

    # Block (Causal)
    for start_n in range(max(block_split, 0), block_num):
        block_off = tl.load(block_idx_ptr + start_n * stride_2in) * BLOCK_N

        # -- load k, v --
        k = tl.load(k_ptrs + block_off * stride_kn + offs_n[None, :] * stride_kn)
        v = tl.load(v_ptrs + block_off * stride_vn + offs_n[:, None] * stride_vn)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(offs_m[:, None] >= offs_n[None, :] + block_off, qk, float("-inf"))
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
    acc_0 = tl.load(o_ptrs).to(tl.float32)
    s_0 = tl.load(lse_ptrs)

    overflow_mask = (s_0 - s_1) < 88.0

    theta = tl.math.exp(s_0 - s_1)
    alpha_0 = 1 / (1 + 1 / theta)
    alpha_1 = 1 / (1 + theta)
    acc = alpha_0[:, None] * acc_0 + alpha_1[:, None] * acc_1
    s = s_1 - tl.math.log(alpha_1)

    tl.store(o_ptrs, acc.to(Out.type.element_ty))
    tl.store(lse_ptrs, s, mask=overflow_mask)

def triton_block_attn_fwd(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    softmax_scale: float,
    block_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks]
    granularity: int,
    step: int = 0,
    causal: bool = True,
):
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_blocks = block_idx.shape[2]
    
    o = torch.zeros_like(q)
    lse = torch.zeros((batch_size, num_qo_heads, num_tokens), dtype=torch.float32, device=q.device) - torch.inf

    _triton_block_attn_fwd_kernel[(num_blocks, num_qo_heads, batch_size)](
        q, k, v, softmax_scale, 
        block_cnt, block_idx, 
        o, lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        block_cnt.stride(0), block_cnt.stride(1), block_cnt.stride(2),
        block_idx.stride(0), block_idx.stride(1), block_idx.stride(2), block_idx.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        num_qo_heads, num_kv_heads, num_tokens,
        BLOCK_M=granularity, BLOCK_N=64, BLOCK_DMODEL=head_dim, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o, lse

@triton.jit
def _triton_block_attn_bwd_kernel(
    Q, K, V, O,
    DQ, DK, DV, DO,
    sm_scale,
    block_cnt,  # [BATCH, N_Q_HEADS, NUM_ROWS]
    block_idx,  # [BATCH, N_Q_HEADS, NUM_ROWS, NUM_COLS]
    softmax_lse, # [BATCH, N_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_2cz, stride_2ch, stride_2cm,
    stride_2iz, stride_2ih, stride_2im, stride_2in,
    stride_sz, stride_sh, stride_sm,
    num_qo_heads, num_kv_heads, num_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
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

    block_num = tl.load(block_cnt + batch_idx * stride_2cz + qo_head_idx * stride_2ch + start_m * stride_2cm)
    block_idx_ptr = block_idx + batch_idx * stride_2iz + qo_head_idx * stride_2ih + start_m * stride_2im

    o = tl.load(o_ptrs).to(tl.float32)
    do = tl.load(do_ptrs).to(tl.float32)
    d_i = tl.sum(o * do, axis=1)

    q = tl.load(q_ptrs)
    do = do.to(DO.dtype.element_ty)
    l_i = tl.load(l_ptrs) * 1.44269504

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    if CAUSAL:
        block_split = block_num - 2
    else:
        block_split = block_num

    # Block
    for start_n in range(0, block_split):
        block_off = tl.load(block_idx_ptr + start_n * stride_2in) * BLOCK_N

        # -- load k, v --
        k = tl.load(k_ptrs + block_off * stride_kn + offs_n[:, None] * stride_kn)
        v = tl.load(v_ptrs + block_off * stride_vn + offs_n[:, None] * stride_vn)

        # Computer qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = qk + tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do).to(tl.float32)
        tl.atomic_add(dv_ptrs + block_off * stride_dvn + offs_n[:, None] * stride_dvn, dv_vals, sem="relaxed")

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - d_i[:, None]
        dp = dp + tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q).to(tl.float32)
        tl.atomic_add(dk_ptrs + block_off * stride_dkn + offs_n[:, None] * stride_dkn, dk_vals, sem="relaxed")

        # compute dq
        dq = dq + tl.dot(ds.to(Q.dtype.element_ty), k)

    # Block (Causal)
    for start_n in range(max(block_split, 0), block_num):
        block_off = tl.load(block_idx_ptr + start_n * stride_2in) * BLOCK_N

        # -- load k, v --
        k = tl.load(k_ptrs + block_off * stride_kn + offs_n[:, None] * stride_kn)
        v = tl.load(v_ptrs + block_off * stride_vn + offs_n[:, None] * stride_vn)

        # Computer qk
        qk = tl.where(offs_m[:, None] >= offs_n[None, :] + block_off, float(0.), float("-inf"))
        qk = qk + tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do).to(tl.float32)
        tl.atomic_add(dv_ptrs + block_off * stride_dvn + offs_n[:, None] * stride_dvn, dv_vals, sem="relaxed")

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - d_i[:, None]
        dp = dp + tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q).to(tl.float32)
        tl.atomic_add(dk_ptrs + block_off * stride_dkn + offs_n[:, None] * stride_dkn, dk_vals, sem="relaxed")

        # compute dq
        dq = dq + tl.dot(ds.to(Q.dtype.element_ty), k)

    dq_old = tl.load(dq_ptrs).to(tl.float32)
    tl.store(dq_ptrs, (dq_old + dq).to(DQ.dtype.element_ty))


def triton_block_attn_bwd(
    grad: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    o: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    softmax_lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    softmax_scale: float,
    block_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks]
    granularity: int,
    deterministic: bool,
    step: int = 0,
    causal: bool = True,
):
    assert not deterministic
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_blocks = block_idx.shape[2]

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    _triton_block_attn_bwd_kernel[(num_blocks, num_qo_heads, batch_size)](
        q, k, v, o, dq, dk, dv, grad, softmax_scale,
        block_cnt, block_idx, softmax_lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3),
        dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3),
        dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3),
        grad.stride(0), grad.stride(2), grad.stride(1), grad.stride(3),
        block_cnt.stride(0), block_cnt.stride(1), block_cnt.stride(2),
        block_idx.stride(0), block_idx.stride(1), block_idx.stride(2), block_idx.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        num_qo_heads, num_kv_heads, num_tokens,
        BLOCK_M=granularity, BLOCK_N=64, BLOCK_DMODEL=head_dim, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return dq, dk.to(dq.dtype), dv.to(dq.dtype)


@triton.jit
def _triton_block_bar_attn_fwd_kernel(
    Q, K, V, sm_scale,
    bar_cnt, # [BATCH, N_Q_HEADS, NUM_ROWS, WORLD_SIZE + 1]
    bar_idx, # [BATCH, N_Q_HEADS, NUM_ROWS, NNZ_V]
    block_cnt,  # [BATCH, N_Q_HEADS, NUM_ROWS]
    block_idx,  # [BATCH, N_Q_HEADS, NUM_ROWS, NUM_COLS]
    Out, # [BATCH, N_Q_HEADS, N_CTX, D_HEAD]
    softmax_lse, # [BATCH, N_Q_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_1cz, stride_1ch, stride_1cm, stride_1cr,
    stride_1iz, stride_1ih, stride_1im, stride_1in,
    stride_2cz, stride_2ch, stride_2cm,
    stride_2iz, stride_2ih, stride_2im, stride_2in,
    stride_sz, stride_sh, stride_sm,
    step, num_qo_heads, num_kv_heads, num_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
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

    qo_offset = batch_idx * stride_qz + qo_head_idx * stride_qh
    kv_offset = batch_idx * stride_kz + kv_head_idx * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kd
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vd
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    lse_ptrs = softmax_lse + batch_idx * stride_sz + qo_head_idx * stride_sh + offs_m * stride_sm

    bar_l = tl.load(bar_cnt + batch_idx * stride_1cz + qo_head_idx * stride_1ch + start_m * stride_1cm + step * stride_1cr)
    bar_r = tl.load(bar_cnt + batch_idx * stride_1cz + qo_head_idx * stride_1ch + start_m * stride_1cm + (step + 1) * stride_1cr)
    bar_idx_ptr = bar_idx + batch_idx * stride_1iz + qo_head_idx * stride_1ih + start_m * stride_1im

    block_num = tl.load(block_cnt + batch_idx * stride_2cz + qo_head_idx * stride_2ch + start_m * stride_2cm)
    block_idx_ptr = block_idx + batch_idx * stride_2iz + qo_head_idx * stride_2ih + start_m * stride_2im

    if (bar_l >= bar_r) and (block_num <= 0):
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
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(Q.type.element_ty)
    
    if CAUSAL:
        block_split = block_num - 2
    else:
        block_split = block_num

    # Block
    for start_n in range(0, block_split):
        block_off = tl.load(block_idx_ptr + start_n * stride_2in) * BLOCK_N

        # -- load k, v --
        k = tl.load(k_ptrs + block_off * stride_kn + offs_n[None, :] * stride_kn)
        v = tl.load(v_ptrs + block_off * stride_vn + offs_n[:, None] * stride_vn)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
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

    # Block (Causal)
    for start_n in range(max(block_split, 0), block_num):
        block_off = tl.load(block_idx_ptr + start_n * stride_2in) * BLOCK_N

        # -- load k, v --
        k = tl.load(k_ptrs + block_off * stride_kn + offs_n[None, :] * stride_kn)
        v = tl.load(v_ptrs + block_off * stride_vn + offs_n[:, None] * stride_vn)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(offs_m[:, None] >= offs_n[None, :] + block_off, qk, float("-inf"))
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

    # Bar
    for start_n in range(bar_l, bar_r, BLOCK_N):
        n_mask = start_n + offs_n < bar_r
        cols = tl.load(bar_idx_ptr + (start_n + offs_n) * stride_1in, mask=n_mask, other=0)

        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(n_mask[None, :], qk, float("-inf"))
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
    acc_0 = tl.load(o_ptrs).to(tl.float32)
    s_0 = tl.load(lse_ptrs)

    overflow_mask = (s_0 - s_1) < 88.0

    theta = tl.math.exp(s_0 - s_1)
    alpha_0 = 1 / (1 + 1 / theta)
    alpha_1 = 1 / (1 + theta)
    acc = alpha_0[:, None] * acc_0 + alpha_1[:, None] * acc_1
    s = s_1 - tl.math.log(alpha_1)

    tl.store(o_ptrs, acc.to(Out.type.element_ty))
    tl.store(lse_ptrs, s, mask=overflow_mask)


def block_bar_attn_fwd(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    o: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    softmax_scale: float,
    bar_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, max_v_size]
    bar_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, world_size + 1]
    block_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks]
    granularity: int,
    step: int = 0,
    causal: bool = True,
):
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_blocks = bar_idx.shape[2]
    if o is None:
        o = torch.zeros_like(q)
        lse = torch.zeros((batch_size, num_qo_heads, num_tokens), dtype=torch.float32, device=q.device) - torch.inf
    _triton_block_bar_attn_fwd_kernel[(num_blocks, num_qo_heads, batch_size)](
        q, k, v, softmax_scale, bar_cnt, bar_idx, block_cnt, block_idx, o, lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        bar_cnt.stride(0), bar_cnt.stride(1), bar_cnt.stride(2), bar_cnt.stride(3),
        bar_idx.stride(0), bar_idx.stride(1), bar_idx.stride(2), bar_idx.stride(3),
        block_cnt.stride(0), block_cnt.stride(1), block_cnt.stride(2),
        block_idx.stride(0), block_idx.stride(1), block_idx.stride(2), block_idx.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        step, num_qo_heads, num_kv_heads, num_tokens,
        BLOCK_M=granularity, BLOCK_N=64, BLOCK_DMODEL=head_dim, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o, lse


@triton.jit
def _triton_block_bar_attn_bwd_kernel(
    Q, K, V, O,
    DQ, DK, DV, DO,
    sm_scale,
    bar_cnt, # [BATCH, N_Q_HEADS, NUM_ROWS, WORLD_SIZE + 1]
    bar_idx, # [BATCH, N_Q_HEADS, NUM_ROWS, NNZ_V]
    block_cnt,  # [BATCH, N_Q_HEADS, NUM_ROWS]
    block_idx,  # [BATCH, N_Q_HEADS, NUM_ROWS, NUM_COLS]
    softmax_lse, # [BATCH, N_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_1cz, stride_1ch, stride_1cm, stride_1cr,
    stride_1iz, stride_1ih, stride_1im, stride_1in,
    stride_2cz, stride_2ch, stride_2cm,
    stride_2iz, stride_2ih, stride_2im, stride_2in,
    stride_sz, stride_sh, stride_sm,
    step, num_qo_heads, num_kv_heads, num_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
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

    bar_l = tl.load(bar_cnt + batch_idx * stride_1cz + qo_head_idx * stride_1ch + start_m * stride_1cm + step * stride_1cr)
    bar_r = tl.load(bar_cnt + batch_idx * stride_1cz + qo_head_idx * stride_1ch + start_m * stride_1cm + (step + 1) * stride_1cr)
    bar_idx_ptr = bar_idx + batch_idx * stride_1iz + qo_head_idx * stride_1ih + start_m * stride_1im

    block_num = tl.load(block_cnt + batch_idx * stride_2cz + qo_head_idx * stride_2ch + start_m * stride_2cm)
    block_idx_ptr = block_idx + batch_idx * stride_2iz + qo_head_idx * stride_2ih + start_m * stride_2im

    if (bar_l >= bar_r) and (block_num <= 0):
        return

    o = tl.load(o_ptrs).to(tl.float32)
    do = tl.load(do_ptrs).to(tl.float32)
    d_i = tl.sum(o * do, axis=1)

    q = tl.load(q_ptrs)
    do = do.to(DO.dtype.element_ty)
    l_i = tl.load(l_ptrs) * 1.44269504

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    if CAUSAL:
        block_split = block_num - 2
    else:
        block_split = block_num

    # Block
    for start_n in range(0, block_split):
        block_off = tl.load(block_idx_ptr + start_n * stride_2in) * BLOCK_N

        # -- load k, v --
        k = tl.load(k_ptrs + block_off * stride_kn + offs_n[:, None] * stride_kn)
        v = tl.load(v_ptrs + block_off * stride_vn + offs_n[:, None] * stride_vn)

        # Computer qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = qk + tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do).to(tl.float32)
        tl.atomic_add(dv_ptrs + block_off * stride_dvn + offs_n[:, None] * stride_dvn, dv_vals, sem="relaxed")

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - d_i[:, None]
        dp = dp + tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q).to(tl.float32)
        tl.atomic_add(dk_ptrs + block_off * stride_dkn + offs_n[:, None] * stride_dkn, dk_vals, sem="relaxed")

        # compute dq
        dq = dq + tl.dot(ds.to(Q.dtype.element_ty), k)

    # Block (Causal)
    for start_n in range(max(block_split, 0), block_num):
        block_off = tl.load(block_idx_ptr + start_n * stride_2in) * BLOCK_N

        # -- load k, v --
        k = tl.load(k_ptrs + block_off * stride_kn + offs_n[:, None] * stride_kn)
        v = tl.load(v_ptrs + block_off * stride_vn + offs_n[:, None] * stride_vn)

        # Computer qk
        qk = tl.where(offs_m[:, None] >= offs_n[None, :] + block_off, float(0.), float("-inf"))
        qk = qk + tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do).to(tl.float32)
        tl.atomic_add(dv_ptrs + block_off * stride_dvn + offs_n[:, None] * stride_dvn, dv_vals, sem="relaxed")

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - d_i[:, None]
        dp = dp + tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q).to(tl.float32)
        tl.atomic_add(dk_ptrs + block_off * stride_dkn + offs_n[:, None] * stride_dkn, dk_vals, sem="relaxed")

        # compute dq
        dq = dq + tl.dot(ds.to(Q.dtype.element_ty), k)

    # Bar
    for start_n in range(bar_l, bar_r, BLOCK_N):
        n_mask = start_n + offs_n < bar_r
        cols = tl.load(bar_idx_ptr + (start_n + offs_n) * stride_1in, mask=n_mask, other=0)

        # -- load k, v --
        k = tl.load(k_ptrs + cols[:, None] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)

        # Computer qk
        qk = tl.where(n_mask[None, :], float(0.), float("-inf"))
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

    dq_old = tl.load(dq_ptrs).to(tl.float32)
    tl.store(dq_ptrs, (dq_old + dq).to(DQ.dtype.element_ty))


def block_bar_attn_bwd(
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
    block_idx: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks, num_blocks]
    block_cnt: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks]
    granularity: int,
    deterministic: bool,
    step: int = 0,
    causal: bool = True,
):
    assert not deterministic
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_blocks = bar_idx.shape[2]
    dq = torch.zeros_like(q) if dq is None else dq
    dk = torch.zeros_like(k, dtype=torch.float32) if dk is None else dk.to(torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32) if dv is None else dv.to(torch.float32)
    _triton_block_bar_attn_bwd_kernel[(num_blocks, num_qo_heads, batch_size)](
        q, k, v, o, dq, dk, dv, grad, softmax_scale,
        bar_cnt, bar_idx, block_cnt, block_idx, softmax_lse,
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
        block_cnt.stride(0), block_cnt.stride(1), block_cnt.stride(2),
        block_idx.stride(0), block_idx.stride(1), block_idx.stride(2), block_idx.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        step, num_qo_heads, num_kv_heads, num_tokens,
        BLOCK_M=granularity, BLOCK_N=64, BLOCK_DMODEL=head_dim, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return dq, dk.to(dq.dtype), dv.to(dq.dtype)


class MInferenceAttnTritonFunc(torch.autograd.Function):
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
        block_idx, block_cnt = convert_blockmask(block_mask, block_size_M=granularity, block_size_N=64)

        out, softmax_lse = block_bar_attn_fwd(
            q, k, v, None, None, softmax_scale,
            bar_idx, bar_cnt, block_idx, block_cnt,
            granularity=granularity,
            step=0,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse, block_idx, block_cnt, bar_idx, bar_cnt)
        ctx.granularity = granularity
        ctx.deterministic = deterministic
        ctx.softmax_scale = softmax_scale
        return (out, softmax_lse, None) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, block_idx, block_cnt, bar_idx, bar_cnt = ctx.saved_tensors

        # Bar Mask
        dq, dk, dv = block_bar_attn_bwd(
            dout, q, k, v, out, None, None, None,
            softmax_lse, ctx.softmax_scale,
            bar_idx, bar_cnt, block_idx, block_cnt,
            granularity=ctx.granularity,
            deterministic=ctx.deterministic,
            step=0,
        )

        return dq, dk, dv, None, None, None, None, None, None

def minference_flash_attn_triton_qkvpacked_func(
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
    return MInferenceAttnTritonFunc.apply(
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


def minference_flash_attn_triton_kvpacked_func(
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
    return MInferenceAttnTritonFunc.apply(
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


def minference_flash_attn_triton_func(
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
    return MInferenceAttnTritonFunc.apply(
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

