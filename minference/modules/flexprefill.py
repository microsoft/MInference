# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Refer to the code in https://openreview.net/forum?id=OfjIlbelrT

import math
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from einops import rearrange
from flash_attn import flash_attn_func


def torch_block_wise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: torch.Tensor,
    block_size: int,
    grid_offset: int = 0,
):
    b, n, h, d = q.shape
    assert k.shape == q.shape
    assert v.shape == k.shape
    num_block = math.ceil(grid_offset / block_size) + math.ceil(
        (n - grid_offset) / block_size
    )
    # get topk block idx and build mask
    mask = torch.zeros(b, h, num_block, num_block, dtype=torch.bool, device=q.device)
    mask[
        torch.arange(b).view(b, 1, 1).expand(b, h, block_idx.shape[-1]),
        torch.arange(h).view(1, h, 1).expand(b, h, block_idx.shape[-1]),
        block_idx // num_block,
        block_idx % num_block,
    ] = 1
    act_blocks_per_row = torch.tril(mask).sum(-1)
    mask = mask.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
    mask = mask[..., grid_offset : grid_offset + n, grid_offset : grid_offset + n]
    mask = torch.tril(mask)
    attn_weight = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(d)
    attn_weight.masked_fill_(~mask, float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    o = torch.einsum("bhij,bjhd->bhid", attn_weight, v)
    o = o.transpose(1, 2)
    return o


@triton.jit
def block_wise_decode_attention_kernel(
    q_ptr,  # shape: [batch_size, seq_len, num_heads, head_dim]
    k_ptr,
    v_ptr,
    o_ptr,
    block_idx_ptr,  # shape: [batch_size, num_heads, num_activated_block]
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    GQA_GROUPS,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCK,
    # softmax_scale
    softmax_scale,
    # gqa
    gqa_interleave: tl.constexpr,
    # stride
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    stride_bb,
    stride_bh,
    stride_bt,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // GQA_GROUPS
    # get column block index ptr
    block_idx_ptr = block_idx_ptr + pid_b * stride_bb + pid_h * stride_bh
    # init qkv ptrs
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(1, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)
    # flash attention
    for i in range(0, NUM_BLOCK):
        # get current block start index
        c = tl.load(block_idx_ptr).to(tl.int32) * BLOCK_SIZE_K
        block_idx_ptr = block_idx_ptr + stride_bt
        # load k
        k = tl.load(
            tl.advance(k_ptrs, (0, c)), boundary_check=(1,), padding_option="zero"
        )
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where((off_n < K_LEN - c)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * softmax_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(
            tl.advance(v_ptrs, (c, 0)), boundary_check=(0,), padding_option="zero"
        )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
    # final scale
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(1, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0,))


def triton_block_wise_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: torch.Tensor,
    block_size: int,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """Block wise sparse attention (causal attention) implemented by openai triton (ver 3.0.0).

    Args:
        q (torch.Tensor): Query states, shape [batch_size, 1, num_heads, head_dim]
        k (torch.Tensor): Key states, shape [batch_size, seq_len, num_heads, head_dim]
        v (torch.Tensor): Value states, same as key
        block_idx (torch.Tensor): Index of activated blocks, shape [batch_size, num_heads, activated_block_num]
        block_size (int): Block size, only support 16, 32, 64 and 128.
        softmax_scale (Optional[float], optional): Softmax scale. Defaults to 1/math.sqrt(head_dim)
        gqa_interleave (bool): use interleave mode of gqa, default to False.

    Returns:
        torch.Tensor: Attention output, shape [batch_size, 1, num_heads, head_dim]
    """
    batch_size, q_len, num_q_heads, head_dim = q.shape
    assert q_len == 1
    batch_size, k_len, num_kv_heads, head_dim = k.shape
    batch_size, num_q_heads, num_blocks = block_idx.shape
    assert q.dtype == torch.bfloat16
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    assert block_size in {
        16,
        32,
        64,
        128,
    }, "only support block size in {16, 32, 64, 128}"
    assert num_blocks <= triton.cdiv(k_len, block_size)
    # gqa
    assert num_q_heads % num_kv_heads == 0
    gqa_groups = num_q_heads // num_kv_heads
    # softmax_scale
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)
    # sort idx and get block index bins
    block_idx = block_idx.sort(-1).values
    # launch attention kernel
    o = torch.empty_like(q)
    num_warps = 8
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = block_size
    block_wise_decode_attention_kernel[(batch_size, num_q_heads)](
        q,
        k,
        v,
        o,
        block_idx,
        batch_size,
        num_q_heads,
        num_q_heads,
        num_kv_heads,
        gqa_groups,
        k_len,
        head_dim,
        num_blocks,
        softmax_scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return o


@triton.jit
def count_kernel(
    x_ptr,
    y_ptr,
    k,
    r,
    stride_xb,
    stride_xh,
    stride_xk,
    stride_yb,
    stride_yh,
    stride_yr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    # load x
    x_ptr = x_ptr + pid_b * stride_xb + pid_h * stride_xh
    off_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + off_k * stride_xk
    y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)
    for i in range(0, k, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, off_k < k - i, -1)
        x = x // r
        x = tl.where(off_k < k - i, x, -1)
        # count
        # maybe triton bug: when BLOCK_SIZE_R == r, the count of values ​​in bin [r-1, r) will be wrong
        y += tl.histogram(x, BLOCK_SIZE_R)
        # move ptr
        x_ptrs = x_ptrs + BLOCK_SIZE_K * stride_xk
    # cumsum
    y = tl.cumsum(y, axis=0)
    # store result
    y_ptr = y_ptr + pid_b * stride_yb + pid_h * stride_yh + stride_yr
    off_r = tl.arange(0, BLOCK_SIZE_R)
    tl.store(y_ptr + off_r * stride_yr, y, off_r < r)


def triton_column_count_cumsum(x: torch.Tensor, num_columns: int) -> torch.Tensor:
    """count columns of each row for a given index tensor, then do cumsum

    Args:
        x (torch.Tensor): block index in a flatten 2d grid, shape [batch_size, num_heads, activated_block_num]
        num_colums (int): number of columns in the grid

    Returns:
        torch.Tensor: cumsum of columns num in each row, shape [batch_size, num_heads, num_rows + 1 ]
            For example, in a 4x4 block grid, activated blocks have index [0, 5, 8, 9, 13, 14], number of blocks in each row is [1, 1, 2, 2],
            this function will return cumsum tensor [0, 1, 2, 4, 6]
    """
    x = x.to(torch.int32)
    b, h, k = x.shape
    r = num_columns
    # torch implementation:
    # y = torch.zeros(b,h,r*r,dtype=x.dtype,device=x.device)
    # y[torch.arange(b,device=x.device)[:,None,None],torch.arange(h,device=x.device)[None,:,None],torch.where(x<r*r,x,0)]=1
    # y = torch.nn.functional.pad(torch.cumsum(y.view(b,h,r,r).sum(-1),-1),(1,0),value=0).to(torch.int32)
    block_size_k = min(triton.next_power_of_2(k), 4096)
    # plus r by 1 to avoid tl.histogram bug
    block_size_r = triton.next_power_of_2(r + 2)
    y = torch.zeros(b, h, r + 1, device=x.device, dtype=torch.int32)
    count_kernel[(b, h)](
        x,
        y,
        k,
        r,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        block_size_k,
        block_size_r,
    )
    return y


@triton.jit
def block_wise_prefill_attention_kernel(
    q_ptr,  # shape: [batch_size, seq_len, num_heads, head_dim]
    k_ptr,
    v_ptr,
    o_ptr,
    block_idx_ptr,  # shape: [batch_size, num_heads, num_all_block]
    idx_bin_ptr,  # shape: [batch_size, num_heads, seq_len / block_size + 1]
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    GQA_GROUPS,
    Q_LEN,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCK,
    grid_offset,
    # softmax_scale
    softmax_scale,
    # gqa
    gqa_interleave: tl.constexpr,
    # stride
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    stride_bb,
    stride_bh,
    stride_bt,
    stride_ib,
    stride_ih,
    stride_it,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
):
    tl.static_assert(BLOCK_SIZE_Q == BLOCK_SIZE_K)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // GQA_GROUPS
    pid_q = tl.program_id(2)
    # get column index bin
    idx_bin_ptr = idx_bin_ptr + pid_b * stride_ib + pid_h * stride_ih
    bin_start = tl.load(idx_bin_ptr + pid_q * stride_it)
    bin_end = tl.load(idx_bin_ptr + (pid_q + 1) * stride_it)
    num_active_block = bin_end - bin_start
    # get column block index ptr
    block_idx_ptr = (
        block_idx_ptr + pid_b * stride_bb + pid_h * stride_bh + bin_start * stride_bt
    )
    # init qkv ptrs
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_m = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q - grid_offset
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)
    # flash attention
    for i in tl.range(0, num_active_block):
        # get current block start index
        c = tl.load(block_idx_ptr).to(tl.int32) % NUM_BLOCK * BLOCK_SIZE_K - grid_offset
        block_idx_ptr = block_idx_ptr + stride_bt
        # load k
        k = tl.load(
            tl.advance(k_ptrs, (0, c)), boundary_check=(1,), padding_option="zero"
        )
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where((c + off_n)[None, :] >= 0, 0, float("-inf"))
        qk += tl.where(off_m[:, None] >= (c + off_n)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * softmax_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(
            tl.advance(v_ptrs, (c, 0)), boundary_check=(0,), padding_option="zero"
        )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
    # final scale
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0,))


def triton_block_wise_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: Union[torch.Tensor, List[List[torch.Tensor]]],
    block_size: int,
    grid_offset: int = 0,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """Block wise sparse attention (causal attention) implemented by openai triton (ver 3.0.0).

    Args:
        q (torch.Tensor): Query states, shape [batch_size, seq_lens, num_heads, head_dim]
        k (torch.Tensor): Key states, same as query
        v (torch.Tensor): Value states, same as query
        block_idx (torch.Tensor): Index of activated blocks, shape [batch_size, num_heads, activated_block_num], which is the index of the flattened block grid.
            For example, in a 4x4 block grid, if you want to activate 5 blocks: (0,0), (1,1), (2,0), (3,1), (3,2), the index will be: [0, 5, 8, 13, 14]
        block_size (int): Block size, only support 16, 32, 64 and 128.
        grid_offset (int): Move the grid that divides the block to the lower left corner by grid_offset, default to 0.
        softmax_scale (Optional[float], optional): Softmax scale. Defaults to 1/math.sqrt(head_dim)
        gqa_interleave (bool): use interleave mode of gqa, default to False.

    Returns:
        torch.Tensor: Attention output, shape [batch_size, seq_lens, num_heads, head_dim]
    """
    batch_size, q_len, num_q_heads, head_dim = q.shape
    batch_size, k_len, num_kv_heads, head_dim = k.shape
    assert q.dtype == torch.bfloat16
    assert q_len == k_len
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    assert block_size in {
        32,
        64,
        128,
    }, "only support block size in {16, 32, 64, 128}"
    total_q_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
        q_len - grid_offset, block_size
    )
    total_k_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
        k_len - grid_offset, block_size
    )
    # pad block_idx if get list[list[tensor]]
    if not isinstance(block_idx, torch.Tensor):
        assert (
            isinstance(block_idx, list)
            and isinstance(block_idx[0], list)
            and isinstance(block_idx[0][0], torch.Tensor)
        )
        assert len(block_idx) == batch_size and len(block_idx[0]) == num_q_heads
        block_idx = [item.view(-1, 1) for sublist in block_idx for item in sublist]
        block_idx = torch.nn.utils.rnn.pad_sequence(
            block_idx,
            batch_first=True,
            padding_value=total_k_blocks * (total_k_blocks + 1),
        )
        block_idx = block_idx.view(batch_size, num_q_heads, -1)
    batch_size, num_q_heads, num_block = block_idx.shape
    assert q_len == k_len
    assert num_block <= total_q_blocks * (total_q_blocks + 1) // 2
    # gqa
    assert num_q_heads % num_kv_heads == 0
    gqa_groups = num_q_heads // num_kv_heads
    # softmax_scale
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)
    # sort idx and get block index bins
    block_idx = block_idx.sort(-1).values
    idx_bins = triton_column_count_cumsum(block_idx, total_k_blocks)
    # launch attention kernel
    o = torch.empty_like(q)
    num_warps = 8
    num_stages = 3 if block_size >= 128 else 5
    block_wise_prefill_attention_kernel[(batch_size, num_q_heads, total_q_blocks)](
        q,
        k,
        v,
        o,
        block_idx,
        idx_bins,
        batch_size,
        num_q_heads,
        num_kv_heads,
        gqa_groups,
        q_len,
        k_len,
        head_dim,
        total_q_blocks,
        grid_offset,
        softmax_scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        idx_bins.stride(0),
        idx_bins.stride(1),
        idx_bins.stride(2),
        BLOCK_SIZE_Q=block_size,
        BLOCK_SIZE_K=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def triton_block_wise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: torch.Tensor,
    block_size: int,
    grid_offset: int = 0,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """Block wise sparse attention (causal attention) implemented by openai triton (ver 3.0.0).

    Args:
        q (torch.Tensor): Query states, shape [batch_size, seq_lens, num_heads, head_dim]
        k (torch.Tensor): Key states, same as query
        v (torch.Tensor): Value states, same as query
        block_idx (torch.Tensor): Index of activated blocks, shape [batch_size, num_heads, activated_block_num], which is the index of the flattened block grid.
            For example, in a 4x4 block grid, if you want to activate 5 blocks: (0,0), (1,1), (2,0), (3,1), (3,2), the index will be: [0, 5, 8, 13, 14]
        block_size (int): Block size, only support 16, 32, 64 and 128.
        grid_offset (int): Move the grid that divides the block to the lower left corner by grid_offset, default to 0.
        softmax_scale (Optional[float], optional): Softmax scale. Defaults to 1/math.sqrt(head_dim)
        gqa_interleave (bool): use interleave mode of gqa, default to False.

    Returns:
        torch.Tensor: Attention output, shape [batch_size, seq_lens, num_heads, head_dim]
    """
    if q.shape[1] > 1:
        return triton_block_wise_prefill_attention(
            q,
            k,
            v,
            block_idx,
            block_size,
            grid_offset,
            softmax_scale,
            gqa_interleave,
        )
    else:
        return triton_block_wise_decode_attention(
            q, k, v, block_idx, block_size, softmax_scale, gqa_interleave
        )


@triton.jit
def bnhd_pool_kernel(
    x_ptr,
    y_ptr,
    # pool type. avg: 0, max: 1, min: 2, max abs: 3, sum: 4
    pool_type: tl.constexpr,
    # shape
    batch_size,
    seq_len,
    num_heads,
    head_dim: tl.constexpr,
    # stride
    stride_xb,
    stride_xn,
    stride_xh,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yh,
    stride_yd,
    # META parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,  # {16, 32, 64, 128, 256, 512}
    BLOCK_SIZE_D: tl.constexpr,  # {16, 32, 64, 128, 256, 512}
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)

    x_ptr = (
        x_ptr
        + pid_b * stride_xb
        + pid_n * BLOCK_SIZE_N * stride_xn
        + pid_h * BLOCK_SIZE_H * stride_xh
    )

    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    cur_block_size_n = min(seq_len - pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)

    x_mask = (
        (off_n < seq_len - pid_n * BLOCK_SIZE_N)[:, None, None]
        & (off_h < num_heads - pid_h * BLOCK_SIZE_H)[None, :, None]
        & (off_d < head_dim)[None, None, :]
    )
    x = tl.load(
        x_ptr
        + off_n[:, None, None] * stride_xn
        + off_h[None, :, None] * stride_xh
        + off_d[None, None, :] * stride_xd,
        mask=x_mask,
        other=0,
    )
    if pool_type == 0:
        y = tl.sum(x, axis=0) / cur_block_size_n
    elif pool_type == 1:
        y = tl.max(x, axis=0)
    elif pool_type == 2:
        y = tl.min(x, axis=0)
    elif pool_type == 3:
        y = tl.max(tl.abs(x), axis=0)
    elif pool_type == 4:
        y = tl.sum(x, axis=0)
    else:
        y = tl.sum(x, axis=0) / cur_block_size_n
    y_ptr = (
        y_ptr + pid_b * stride_yb + pid_n * stride_yn + pid_h * BLOCK_SIZE_H * stride_yh
    )
    y_mask = (off_h < num_heads - pid_h * BLOCK_SIZE_H)[:, None] & (off_d < head_dim)[
        None, :
    ]
    tl.store(
        y_ptr + off_h[:, None] * stride_yh + off_d[None, :] * stride_yd, y, mask=y_mask
    )


def triton_bnhd_pool(x: torch.Tensor, kernel_size: int, pool_type: str = "avg"):
    x = x.to("cuda")
    b, n, h, d = x.shape
    assert d in {16, 32, 64, 128}
    assert kernel_size in {16, 32, 64, 128, 256, 512}
    m = triton.cdiv(n, kernel_size)
    y = torch.zeros(b, m, h, d, device=x.device, dtype=x.dtype)

    if pool_type == "last":
        if n % kernel_size == 0:
            return x[:, kernel_size - 1 :: kernel_size, ...]
        else:
            return torch.cat(
                (x[:, kernel_size - 1 :: kernel_size, ...], x[:, -1:, ...]), dim=1
            )

    block_size_h = triton.next_power_of_2(h)
    while kernel_size * block_size_h * d > 128 * 128 * 128:
        block_size_h = block_size_h // 2

    block_size_d = triton.next_power_of_2(d)
    pool_str_to_type = {"avg": 0, "max": 1, "min": 2, "maxabs": 3, "sum": 4}
    pool_type = pool_str_to_type[pool_type]

    grid = lambda META: (
        b,
        triton.cdiv(n, META["BLOCK_SIZE_N"]),
        triton.cdiv(h, META["BLOCK_SIZE_H"]),
    )
    bnhd_pool_kernel[grid](
        x,
        y,
        pool_type,
        b,
        n,
        h,
        d,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        BLOCK_SIZE_N=kernel_size,
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_D=block_size_d,
    )
    return y


@triton.jit
def bhn_sumpool_kernel(
    x_ptr,
    y_ptr,
    # shape
    batch_size,
    num_heads,
    seq_len,
    # stride
    stride_xb,
    stride_xh,
    stride_xn,
    stride_yb,
    stride_yh,
    stride_yn,
    # META parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,  # {16, 32, 64, 128, 256, 512}
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    x_ptr = (
        x_ptr
        + pid_b * stride_xb
        + pid_h * BLOCK_SIZE_H * stride_xh
        + pid_n * BLOCK_SIZE_N * stride_xn
    )
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    x_mask = (off_n < seq_len - pid_n * BLOCK_SIZE_N)[None, :] & (
        off_h < num_heads - pid_h * BLOCK_SIZE_H
    )[:, None]
    x = tl.load(
        x_ptr + off_h[:, None] * stride_xh + off_n[None, :] * stride_xn,
        mask=x_mask,
        other=0,
    )
    y = tl.sum(x, axis=1)
    y_ptr = (
        y_ptr + pid_b * stride_yb + pid_h * BLOCK_SIZE_H * stride_yh + pid_n * stride_yn
    )
    y_mask = off_h < num_heads - pid_h * BLOCK_SIZE_H
    tl.store(y_ptr + off_h * stride_yh, y, mask=y_mask)


def triton_bhn_sumpool(x: torch.Tensor, kernel_size: int):
    b, h, n = x.shape
    assert kernel_size in {16, 32, 64, 128, 256, 512}
    m = triton.cdiv(n, kernel_size)
    y = torch.empty(b, h, m, device=x.device, dtype=x.dtype)
    block_size_h = triton.next_power_of_2(h)
    grid = lambda META: (
        b,
        triton.cdiv(h, META["BLOCK_SIZE_H"]),
        triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    bhn_sumpool_kernel[grid](
        x,
        y,
        b,
        h,
        n,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        BLOCK_SIZE_N=kernel_size,
        BLOCK_SIZE_H=block_size_h,
    )
    return y


def torch_bhn_sumpool(x: torch.Tensor, kernel_size: int):
    b, h, n = x.shape
    x = torch.nn.functional.pad(
        x,
        (
            0,
            math.ceil(n / kernel_size) * kernel_size - n,
        ),
        value=0,
    )
    x = x.view(b, h, -1, kernel_size).sum(-1)
    return x


def score_cover_topk(x: torch.Tensor, score: float):
    cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1)
    topk = torch.sum(cumsum_x <= score, dim=-1) + 1
    return topk


def score_cover_idx(x: torch.Tensor, score: float, padding_value=0):
    x, idx = torch.sort(x, dim=-1, descending=True)
    cumsum_x = torch.cumsum(x, dim=-1)
    idx[cumsum_x > score] = padding_value
    return idx


def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    mat_padded = torch.nn.functional.pad(mat, (n - 1, 0), value=0)
    mat_strided = mat_padded.as_strided(
        (b, h, m, n), (h * n * (n + m - 1), n * (n + m - 1), 1, n + m)
    )
    sum_diags = torch.sum(mat_strided, -1)
    return sum_diags


def transform_veritcal_slash_idx(v_idx, s_idx, num_blocks):
    batch_size, num_heads, _ = v_idx.shape
    range_blocks = torch.arange(num_blocks, device=s_idx.device)[None, None, :, None]
    # vertical
    v_idx = (
        torch.arange(0, num_blocks, device=v_idx.device)[None, None, :, None]
        * num_blocks
        + v_idx[:, :, None, :]
    ).view(batch_size, num_heads, -1)
    v_idx[v_idx // num_blocks < v_idx % num_blocks] = 0
    # slash
    s_idx = (
        range_blocks * num_blocks + range_blocks + s_idx[:, :, None, :] * num_blocks
    ).view(batch_size, num_heads, -1)
    s_idx[s_idx >= num_blocks * num_blocks] = 0
    # union
    vs_idx = torch.cat((s_idx, v_idx), dim=-1)
    block_idx = [
        [torch.unique(vs_idx[b, h]) for h in range(num_heads)]
        for b in range(batch_size)
    ]
    return block_idx


causal_mask = None


def get_block_vertical_slash_from_qk(
    qk: torch.Tensor,
    block_size: int,
):
    batch_size, num_heads, last_q_len, seq_len = qk.shape
    # slash shape: [batch_size, num_heads, seq_len] -> [batch_size, num_heads, num_blocks]
    slash = sum_all_diagonal_matrix(qk)
    slash = torch_bhn_sumpool(slash, block_size)
    slash = slash / last_q_len
    # vertical shape: [batch_size, num_heads, seq_len] -> [batch_size, num_heads, num_blocks]
    vertical = qk.sum(-2)
    vertical = torch_bhn_sumpool(vertical, block_size)
    vertical = vertical / last_q_len
    return vertical, slash


def square_root_js_divergence(p: torch.Tensor, q: torch.Tensor):
    m = (p + q) / 2
    return torch.sqrt(
        0.5 * (p * torch.log(p / m)).sum(-1) + 0.5 * (q * torch.log(q / m)).sum(-1)
    )


def get_active_blocks(
    q,
    k,
    v,
    block_size,
    gamma,
    min_budget,
    max_budget,
    tau=0,
    gqa_interleave=False,
):
    batch_size, seq_len, num_heads, head_dim = q.shape
    gqa_groups = num_heads // k.shape[2]
    num_blocks = math.ceil(seq_len / block_size)
    max_budget = min(max_budget, num_blocks)
    # last qk attention, qk shape: [batch_size, num_heads, block_size, seq_len]
    last_q = q[:, -block_size:, :, :] / math.sqrt(head_dim)
    if not gqa_interleave:
        qk = torch.einsum(
            "bihgd, bjhgd -> bhgij",
            last_q.view(last_q.shape[0], last_q.shape[1], -1, gqa_groups, head_dim),
            k.view(k.shape[0], k.shape[1], -1, 1, head_dim),
        )
    else:
        qk = torch.einsum(
            "bihgd, bjhgd -> bhgij",
            last_q.view(last_q.shape[0], last_q.shape[1], gqa_groups, -1, head_dim),
            k.view(k.shape[0], k.shape[1], 1, -1, head_dim),
        )
    global causal_mask
    if causal_mask is None:
        causal_mask = torch.arange(0, block_size, device=last_q.device)
        causal_mask = causal_mask[:, None] >= causal_mask[None, :]
        causal_mask = causal_mask[None, None, None, ...]
    qk[..., -block_size:].masked_fill_(
        ~causal_mask[..., :block_size, :block_size].to(qk.device), float("-inf")
    )
    # softmax
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    qk = rearrange(qk, "b h g i j -> b (h g) i j")
    slash = sum_all_diagonal_matrix(qk) / qk.shape[-2]
    vertical = qk.mean(-2)
    # get vertical slash size to make sure attention score >= gamma. shape: [batch_size, num_heads]
    num_vertical_blocks = score_cover_topk(vertical, gamma) // 128 + 1
    num_slash_blocks = score_cover_topk(slash, gamma) // 128 + 1
    num_vertical_blocks[num_vertical_blocks < min_budget] = min_budget
    num_vertical_blocks[num_vertical_blocks > max_budget] = max_budget
    num_slash_blocks[num_slash_blocks < min_budget] = min_budget
    num_slash_blocks[num_slash_blocks > max_budget] = max_budget
    # block avg pool
    vertical = torch_bhn_sumpool(vertical, block_size)
    slash = torch_bhn_sumpool(slash, block_size)
    # get block sparse mask
    if not gqa_interleave:
        avg_k = triton_bnhd_pool(k, block_size).repeat_interleave(gqa_groups, 2)
    else:
        avg_k = triton_bnhd_pool(k, block_size).repeat(1, 1, gqa_groups, 1)
    avg_qk = torch.einsum(
        "bihd, bjhd -> bhij", last_q.mean(1, keepdim=True), avg_k
    ).squeeze(2)
    avg_qk = torch.softmax(avg_qk, dim=-1, dtype=torch.float32)
    kl_div = square_root_js_divergence(avg_qk, vertical)
    block_sparse_mask = kl_div < tau
    num_vertical_blocks[block_sparse_mask] = min_budget
    num_slash_blocks[block_sparse_mask] = min_budget
    # keep first vertical and slash block
    vertical[..., :1] = torch.inf
    slash[..., -1:] = torch.inf
    # get slash topk
    num_slash_blocks = num_slash_blocks.view(batch_size * num_heads)
    slash = slash.view(batch_size * num_heads, -1)
    slash_topk = (num_blocks - 1) - slash.topk(
        min(num_slash_blocks.max().item(), num_blocks), -1
    ).indices
    slash_topk[
        torch.arange(slash_topk.shape[-1], device=num_slash_blocks.device)[None, :]
        >= num_slash_blocks[:, None]
    ] = 0
    slash_topk = slash_topk.view(batch_size, num_heads, -1)
    # get vertical topk
    num_vertical_blocks = num_vertical_blocks.view(batch_size * num_heads)
    vertical = vertical.view(batch_size * num_heads, -1)
    vertical_topk = vertical.topk(
        min(num_vertical_blocks.max().item(), num_blocks), -1
    ).indices
    vertical_topk[
        torch.arange(vertical_topk.shape[-1], device=num_vertical_blocks.device)[
            None, :
        ]
        >= num_vertical_blocks[:, None]
    ] = 0
    vertical_topk = vertical_topk.view(batch_size, num_heads, -1)
    # transform vertical slash index
    block_idx = transform_veritcal_slash_idx(vertical_topk, slash_topk, num_blocks)
    # get block sparse topk
    block_causal_mask = None
    for b, h in block_sparse_mask.nonzero():
        if block_causal_mask is None:
            block_causal_mask = torch.tril(
                torch.ones(num_blocks, num_blocks, device=q.device, dtype=torch.bool)
            )
        pad_q = math.ceil(seq_len / block_size) * block_size - seq_len
        avg_q = (
            torch.nn.functional.pad(q[b, :, h, :], (0, 0, 0, pad_q), value=0)
            .view(num_blocks, block_size, head_dim)
            .mean(1)
        )
        avg_q[-1, :] = avg_q[-1, :] * block_size / (block_size - pad_q)
        attn = torch.einsum(
            "id, jd -> ij", avg_q / math.sqrt(head_dim), avg_k[b, :, h, :]
        ).masked_fill_(~block_causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).view(-1)
        block_topk = score_cover_idx(attn, gamma * num_blocks)
        block_idx[b][h] = torch.unique(torch.cat((block_idx[b][h], block_topk), dim=-1))
    return block_idx


def flex_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: float,
    tau: float = 0,
    min_budget: int = None,
    max_budget: int = None,
    gqa_interleave: bool = False,
    softmax_scale: Optional[float] = None,
    block_size: int = 128,
):
    batch_size, seq_len, num_heads, head_dim = q.shape
    assert q.shape[1] == k.shape[1]
    assert head_dim in {16, 32, 64, 128}
    assert block_size in {16, 32, 64, 128}
    num_blocks = math.ceil(seq_len / block_size)
    min_budget = 1 if min_budget is None else min_budget
    max_budget = 2147483647 if max_budget is None else max_budget
    if seq_len <= max(2 * block_size, math.ceil(min_budget / block_size) * block_size):
        return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=True)
    # get vertical slash index
    block_idx = get_active_blocks(
        q,
        k,
        v,
        block_size,
        gamma,
        math.ceil(min_budget / block_size),
        math.ceil(max_budget / block_size),
        tau,
        gqa_interleave,
    )
    attn_out = triton_block_wise_attention(
        q,
        k,
        v,
        block_idx,
        block_size,
        softmax_scale=softmax_scale,
        gqa_interleave=gqa_interleave,
    )
    return attn_out


def flexprefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config,
):
    gamma = config["attn_forward_config"].get("gamma", 0.9)
    tau = config["attn_forward_config"].get("tau", 0.1)
    min_budget = config["attn_forward_config"].get("min_budget", None)
    max_budget = config["attn_forward_config"].get("max_budget", None)
    block_size = config["attn_forward_config"].get("block_size", 128)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    out = flex_prefill_attention(
        q, k, v, gamma, tau, min_budget, max_budget, block_size
    )
    return out.transpose(1, 2)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, H, D = 1, 64000, 32, 64
    gamma = 0.9
    tau = 0.1

    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

    flex_prefill_output = flex_prefill_attention(q, k, v, gamma, tau)
