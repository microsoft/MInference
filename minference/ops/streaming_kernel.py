# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import math

import torch
import triton
import triton.language as tl

_BLOCK_N=64
_BLOCK_M=64

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, qk_scale, N_CTX,
                    sliding_window_offset, sliding_window_size,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
                    IS_EVEN_M: tl.constexpr, IS_EVEN_N: tl.constexpr, COMPLEMENT_SLIDING_WINDOW: tl.constexpr
                ):
    # range of values handled by this stage
    if SLIDING_WINDOW and not COMPLEMENT_SLIDING_WINDOW:
        if COMPLEMENT_SLIDING_WINDOW:
            lo = 0
            hi = (((start_m + 1) * BLOCK_M + sliding_window_offset - sliding_window_size + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        else:
            lo = ((start_m * BLOCK_M + sliding_window_offset - sliding_window_size + 1) // BLOCK_N) * BLOCK_N
            hi = ((((start_m + 1) * BLOCK_M - 1) + sliding_window_offset + BLOCK_N) // BLOCK_N) * BLOCK_N
            if lo < 0:
                lo = 0
            if hi > N_CTX:
                hi = N_CTX

            # lo = 0
            # hi = N_CTX
            lo = tl.multiple_of(lo, BLOCK_N)
            K_block_ptr = tl.advance(K_block_ptr, (0, lo))
            V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    else:
        lo, hi = 0, N_CTX

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_EVEN_N:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale

        if SLIDING_WINDOW:
            dist = tl.arange(0, BLOCK_M)[:, None] - tl.arange(0, BLOCK_N)[None, :] \
                   + start_m * BLOCK_M - start_n + sliding_window_offset

            if COMPLEMENT_SLIDING_WINDOW:
                mask = (dist >= sliding_window_size)
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)

            qk = tl.where(mask, qk, float("-inf"))

        if not IS_EVEN_N:
            qk = tl.where(((tl.arange(0, BLOCK_N) + start_n) < N_CTX)[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)

        if not IS_EVEN_N:
            p = tl.where(((tl.arange(0, BLOCK_N) + start_n) < N_CTX)[None, :], p, 0)

        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        tmp = m_i - m_ij
        alpha_mask = (tmp != tmp) # check nan
        alpha = tl.math.exp2(tmp)
        alpha = tl.where(alpha_mask, 1., alpha)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if IS_EVEN_N:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        acc += tl.dot(p.to(v.dtype), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.heuristics(
    {
        "IS_EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "IS_EVEN_N": lambda args: args["NKV_CTX"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, L,#
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, H_KV, #
              N_CTX,  #
              ROUND_CTX,
              NKV_CTX,
              sliding_window_offset,
              sliding_window_size,
              IS_EVEN_M: tl.constexpr,
              IS_EVEN_N: tl.constexpr,
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              END: tl.constexpr,
              INIT: tl.constexpr,
              SLIDING_WINDOW: tl.constexpr,
              COMPLEMENT_SLIDING_WINDOW: tl.constexpr
            ):

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H//H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_hkv.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(NKV_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, NKV_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(ROUND_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # initialize pointer to m and l
    m_ptrs = M + off_hz * ROUND_CTX + offs_m
    l_ptrs = L + off_hz * ROUND_CTX + offs_m
    if INIT:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    else:
        # don't have to check boundary for q len
        m_i = tl.load(m_ptrs).to(tl.float32)
        l_i = tl.load(l_ptrs).to(tl.float32)
        acc = tl.load(O_block_ptr).to(tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634   # 1/log(2)
    # load q: it will stay in SRAM throughout
    if IS_EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, #
                                    start_m, qk_scale, NKV_CTX, #
                                    sliding_window_offset, sliding_window_size,
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW, IS_EVEN_M, IS_EVEN_N,
                                    COMPLEMENT_SLIDING_WINDOW)
    # epilogue
    if (END):
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
    else:
        tl.store(l_ptrs, l_i)

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.heuristics(
    {
        "IS_EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "IS_EVEN_N": lambda args: args["NKV_CTX"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _score_kernel(
    Q, K, M, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,  #
    stride_kz, stride_kh, stride_kn, stride_kk,  #
    stride_oz, stride_oh, stride_on,
    Z, H, H_KV, #
    N_CTX,  #
    ROUND_CTX,
    NKV_CTX,
    sliding_window_offset,
    sliding_window_size,
    SLIDING_WINDOW: tl.constexpr,
    COMPLEMENT_SLIDING_WINDOW: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H//H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
    m_ptrs = M + off_hz * ROUND_CTX + tl.arange(0, BLOCK_M)
    o = tl.zeros([BLOCK_M], dtype=tl.float32)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, NKV_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )

    if IS_EVEN_N:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")


    lo = 0
    hi = ROUND_CTX
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634   # 1/log(2)

    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        if IS_EVEN_M:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

        m = tl.load(m_ptrs)

        # calc qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale

        if SLIDING_WINDOW:
            # dist = tl.arange(start_m, start_m + BLOCK_M)[:, None] \
            #     - tl.arange(start_n * BLOCK_N, (start_n + 1) + BLOCK_N)[None, :] + sliding_window_offset
            dist = tl.arange(0, BLOCK_M)[:, None] - tl.arange(0, BLOCK_N)[None, :] \
                 + start_m - start_n * BLOCK_N + sliding_window_offset

            if COMPLEMENT_SLIDING_WINDOW:
                mask = (dist >= sliding_window_size)
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)

        qk = qk - m[:, None]
        p = tl.math.exp2(qk) # (BLOCK_M, BLOCK_N)

        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)

        if not IS_EVEN_N:
            p = tl.where(
                ((tl.arange(0, BLOCK_M) + start_m) < N_CTX)[:, None],
                p, 0
            )

        o += tl.sum(p, axis=0)


        Q_block_ptr = tl.advance(Q_block_ptr, offsets=(BLOCK_M, 0))
        m_ptrs = m_ptrs + BLOCK_M

    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    o_range = tl.arange(0, BLOCK_N) + start_n * BLOCK_N # orange
    o_ptrs = Out + o_offset + o_range
    tl.store(o_ptrs, o.to(Out.type.element_ty), mask = o_range < NKV_CTX)

def get_score(q, k, m, sliding_window, complement_sliding_window):
    assert q.dim() == 4
    assert k.dim() == 4
    assert m.dim() == 3
    assert q.shape[:2] == m.shape[:2]
    N_CTX = q.size(-2)
    NKV_CTX = k.size(-2)
    ROUND_CTX = m.size(-1)
    ret = torch.zeros(
        (q.size(0), q.size(1), k.size(2)),
        dtype=k.dtype, device=k.device
    )
    if sliding_window is not None:
        sliding_window_offset, sliding_window_size = sliding_window
    else:
        sliding_window_offset, sliding_window_size = None, None


    grid = lambda META: (
        triton.cdiv(k.shape[2], META["BLOCK_N"]),
        q.shape[0] * q.shape[1]
    )
    sm_scale = 1 / math.sqrt(q.size(-1))

    global _BLOCK_N
    global _BLOCK_M

    try:
        _score_kernel[grid](
            q, k, m, sm_scale, ret,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            ret.stride(0), ret.stride(1), ret.stride(2),
            q.size(0), q.size(1), k.size(1),
            N_CTX, ROUND_CTX, NKV_CTX,
            sliding_window_offset,
            sliding_window_size,
            SLIDING_WINDOW=(sliding_window is not None),
            COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_DMODEL=q.size(-1)
        )
    except triton.OutOfResources as E:
        from warnings import warn
        _BLOCK_N = _BLOCK_N // 2
        _BLOCK_M = _BLOCK_M // 2
        warn(f"Triton Attention Output Resources. {E}\nUse smaller block size {_BLOCK_N}.")
        _score_kernel[grid](
            q, k, m, sm_scale, ret,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            ret.stride(0), ret.stride(1), ret.stride(2),
            q.size(0), q.size(1), k.size(1),
            N_CTX, ROUND_CTX, NKV_CTX,
            sliding_window_offset,
            sliding_window_size,
            SLIDING_WINDOW=(sliding_window is not None),
            COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_DMODEL=q.size(-1)
        )

    return ret

def _forward(
    q, k, v, sm_scale,
    o = None, m = None, l = None, end = False,
    sliding_window=None, init=False,
    complement_sliding_window=False
):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    q_round_len = math.ceil(q.shape[2] / 64) * 64

    if sliding_window is not None:
        sliding_window_offset, sliding_window_size = sliding_window
    else:
        sliding_window_offset, sliding_window_size = None, None

    grid = lambda META: (
        triton.cdiv(q.shape[2], META["BLOCK_M"]),
        q.shape[0] * q.shape[1],
    )

    global _BLOCK_N
    global _BLOCK_M

    try:
        with torch.cuda.device(q.device):
            _attn_fwd[grid](
                q, k, v, sm_scale, m, o, l, #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                q.shape[0], q.shape[1], k.shape[1], #
                q.shape[2],  #
                q_round_len,
                k.shape[2],
                sliding_window_offset,
                sliding_window_size,
                BLOCK_DMODEL=Lk,  #
                END=end,
                INIT=init,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                SLIDING_WINDOW=(sliding_window is not None),
                COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
                num_warps=4,
                num_stages=4
            )
    except triton.OutOfResources as E:
        _BLOCK_N = _BLOCK_N // 2
        _BLOCK_M = _BLOCK_M // 2
        from warnings import warn
        warn(f"Triton Attention Output Resources. {E}\nUse smaller block size {_BLOCK_N}.")
        with torch.cuda.device(q.device):
            _attn_fwd[grid](
                q, k, v, sm_scale, m, o, l, #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                q.shape[0], q.shape[1], k.shape[1], #
                q.shape[2],  #
                q_round_len,
                k.shape[2],
                sliding_window_offset,
                sliding_window_size,
                BLOCK_DMODEL=Lk,  #
                END=end,
                INIT=init,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                SLIDING_WINDOW=(sliding_window is not None),
                COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
                num_warps=4,
                num_stages=4
            )


    if end:
        o = o[:, :, :q.shape[2], :].contiguous().to(q.dtype)

    return o, m, l

class MultiStageDotProductionAttention:
    def __init__(
        self,
        q_shape,
        dtype,
        device,
    ):
        self.q_shape = q_shape
        self.dtype = dtype
        self.device = device
        self.end = False
        self.ret = torch.zeros(
            q_shape, dtype=dtype, device=device
        )
        self.score_list = []

    def append(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        sliding_window=None, complement_sliding_window: bool = False,
        end=False, get_score=False,
        *args, **kwargs
    ):
        raise NotImplementedError


    def get_result(self):
        return self.ret, self.score_list


class TritonMultiStageDotProductionAttention(MultiStageDotProductionAttention):
    def __init__(self, q_shape, dtype, device):
        self.q_shape = q_shape
        self.dtype = dtype
        self.device = device
        q_round_len = math.ceil(q_shape[2] / 64) * 64
        o_shape = (q_shape[0], q_shape[1], q_round_len, q_shape[3])
        m_shape = (q_shape[0], q_shape[1], q_round_len)
        l_shape = (q_shape[0], q_shape[1], q_round_len)

        self.o = torch.empty(o_shape, device=device, dtype=torch.float32)
        self.m = torch.empty(m_shape, device=device, dtype=torch.float32)
        self.l = torch.empty(l_shape, device=device, dtype=torch.float32)
        self.q_list = []
        self.k_list = []
        self.sliding_window_list = []
        self.complement_sliding_window_list = []
        self.score_list = []
        self.end = False
        self.init = False

    def finalize(self):
        self.end = True
        for q, k, sliding_window, comp in zip(self.q_list, self.k_list, self.sliding_window_list, self.complement_sliding_window_list):
            if q is not None:
                score = get_score(q, k, self.m, sliding_window, comp)
                self.score_list.append(score)
            else:
                self.score_list.append(None)

        self.ret = self.o

    def append(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, end=False, get_score=False, sliding_window = None, complement_sliding_window: bool = False):
        assert q.shape == self.q_shape

        if isinstance(sliding_window, int):
            sliding_window = (
                k.shape[2] - q.shape[2], sliding_window
            )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        sm_scale = 1 / math.sqrt(q.shape[-1])
        o, m, l = _forward(
            q, k, v, sm_scale, self.o, self.m, self.l,
            sliding_window=sliding_window, end=end, init=not self.init,
            complement_sliding_window=complement_sliding_window
        )
        self.init = True
        self.o = o
        self.m = m
        self.l = l
        if get_score:
            self.q_list.append(q)
            self.k_list.append(k)
            self.sliding_window_list.append(sliding_window)
            self.complement_sliding_window_list.append(complement_sliding_window)
        else:
            self.q_list.append(None)
            self.k_list.append(None)
            self.sliding_window_list.append(None)
            self.complement_sliding_window_list.append(None)

        if end:
            assert not self.end
            self.finalize()

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def streaming_forward(
    q, k, v,
    n_init, n_local,
):
    # q,k,v should be tensors already equipped with RoPE
    # k,v should already repeated to align with q.shape

    assert q.dim() == 4 # (bsz, num_heads, seqlen, head_dim)
    assert q.shape == k.shape == v.shape

    head_dim = q.shape[-1]
    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        q = torch.nn.functional.pad(q, [0, target_dim, 0, 0, 0, 0, 0, 0])
        k = torch.nn.functional.pad(k, [0, target_dim, 0, 0, 0, 0, 0, 0])
        v = torch.nn.functional.pad(v, [0, target_dim, 0, 0, 0, 0, 0, 0])

    q_len = q.size(2)
    k_len = k.size(2)

    attn = TritonMultiStageDotProductionAttention(q.shape, q.dtype, q.device)

    if k_len > n_local:
        init_k = k[:, :, :n_init, :].contiguous()
        init_v = v[:, :, :n_init, :].contiguous()

        attn.append(q, k, v, sliding_window=n_local)
        attn.append(
            q, init_k, init_v, end=True,
            sliding_window=(k_len - q_len, n_local), complement_sliding_window=True
        )
    else:
        attn.append(q, k, v, sliding_window=n_local, end=True)

    score, _ = attn.get_result()
    return score[...,:head_dim]

def streaming_forward2(
    q, k, v,
    n_init, n_local,
):
    q_len = q.size(2)
    k_len = k.size(2)

    attn = TritonMultiStageDotProductionAttention(q.shape, q.dtype, q.device)

    if k_len > n_local:
        init_k = k[:, :, :n_init, :].contiguous()
        init_v = v[:, :, :n_init, :].contiguous()

    else:
        init_k = torch.empty(
            (k.size(0), k.size(1), 0, k.size(3)),
            dtype=k.dtype, device=k.device
        )
        init_v = torch.empty(
            (v.size(0), v.size(1), 0, v.size(3)),
            dtype=v.dtype, device=v.device
        )

    attn.append(q, k, v, sliding_window=n_local)
    attn.append(
        q, init_k, init_v, end=True,
        sliding_window=(k_len - q_len, n_local), complement_sliding_window=True
    )

    score, _ = attn.get_result()
    return score

def stream_llm_forward(n_local, n_init, *args, **kwargs):
    Attn = TritonMultiStageDotProductionAttention
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : torch.Tensor,
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out,
                    dim_head, num_heads, num_heads_kv
    ):

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)

        h_q = h_q.contiguous()      # (batch * num_heads, len_q, dim_head)
        h_k = h_k.contiguous()      # (batch * num_heads, len_k, dim_head)
        h_v = h_v.contiguous()      # (batch * num_heads, len_k, dim_head)

        if past_key_value is not None:
            h_k = torch.cat([past_key_value[0], h_k], dim=-2)
            h_v = torch.cat([past_key_value[1], h_v], dim=-2)

            len_k += past_key_value[2]

        if use_cache:
            if len_k <= n_local + n_init:
                h_k_cache = h_k
                h_v_cache = h_v
            else:
                h_k_cache = torch.cat([h_k[:,:, :n_init, :], h_k[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                h_v_cache = torch.cat([h_v[:,:, :n_init, :], h_v[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)

            current_key_value = (h_k_cache, h_v_cache, len_k)

        else:
            current_key_value = None

        h_q_ = h_q
        h_k_ = h_k
        h_v_ = h_v

        if len_q + n_local < h_k_.size(-2):
            h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local:, :].contiguous().clone()
            h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local:, :].contiguous().clone()

        local_h_q, local_h_k = position_bias(h_q_, h_k_)
        local_h_v = h_v_

        if len_k > n_local:
            init_h_q = position_bias.apply_rotary_pos_emb_one_angle(
                h_q, n_local + n_init
            )
            init_h_k = position_bias.apply_rotary_pos_emb(
                h_k[:, :, :n_init, :].contiguous(),
                n_init, n_init, position_bias._cos_cached, position_bias._sin_cached
            )
            init_h_v = h_v[:, :, :n_init, :].contiguous()

        else:
            init_h_q = h_q
            init_h_k = torch.empty(
                (batch_size, num_heads_kv, 0, dim_head),
                device=h_k.device,
                dtype=h_k.dtype
            )
            init_h_v = torch.empty(
                (batch_size, num_heads_kv, 0, dim_head),
                device=h_v.device,
                dtype=h_v.dtype
            )

        attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local)
        attn.append(
            init_h_q, init_h_k, init_h_v, end=True,
            sliding_window=(len_k - len_q, n_local),
            complement_sliding_window=True
        )
        score, _ = attn.get_result()

        score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3).contiguous() # (batch, len_q, num_heads, dim_head)
        score = score.reshape(batch_size, len_q, num_heads * dim_head) # (batch, len_q, num_heads * dim_head)

        score = attention_out(score)

        if use_cache:
            return score, current_key_value
        else:
            return score

    return forward
