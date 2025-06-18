import os
from typing import List

import torch
import torch.distributed as dist

import triton
import triton.language as tl


@triton.jit
def _triton_extract_kv_kernel(
    local_k, local_v, bar_k, bar_v, v_idx, v_cnt,
    stride_lz, stride_ln, stride_lh, stride_ld,
    stride_bz, stride_bn, stride_bh, stride_bd,
    stride_iz, stride_ih, stride_in,
    stride_cz, stride_ch, stride_cr,
    step, num_tokens, num_qo_heads, num_kv_heads,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0)
    qo_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    kv_head_idx = qo_head_idx // (num_qo_heads // num_kv_heads)

    v_cnt_ptr = v_cnt + batch_idx * stride_cz + qo_head_idx * stride_ch
    min_n = tl.load(v_cnt_ptr + step * stride_cr)
    max_n = tl.load(v_cnt_ptr + (step + 1) * stride_cr)
    start_n = start_n * BLOCK_N
    end_n = start_n + BLOCK_N
    if start_n >= max_n or end_n <= min_n:
        return

    offs_d = tl.arange(0, BLOCK_D)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = (offs_n >= min_n) & (offs_n < max_n)

    v_idx_ptr = v_idx + batch_idx * stride_iz + qo_head_idx * stride_ih
    local_k_ptr = local_k + batch_idx * stride_lz + kv_head_idx * stride_lh + offs_d[None, :] * stride_ld
    local_v_ptr = local_v + batch_idx * stride_lz + kv_head_idx * stride_lh + offs_d[None, :] * stride_ld
    bar_k_ptr = bar_k + batch_idx * stride_bz + qo_head_idx * stride_bh + offs_d[None, :] * stride_bd
    bar_v_ptr = bar_v + batch_idx * stride_bz + qo_head_idx * stride_bh + offs_d[None, :] * stride_bd

    # idx = tl.load(v_idx_ptr + offs_n * stride_in, mask=mask_n, other=0) - step * num_tokens
    idx = tl.load(v_idx_ptr + offs_n * stride_in, mask=mask_n, other=0) % num_tokens
    k = tl.load(local_k_ptr + idx[:, None] * stride_ln, mask=mask_n[:, None], other=0.)
    v = tl.load(local_v_ptr + idx[:, None] * stride_ln, mask=mask_n[:, None], other=0.)
    tl.store(bar_k_ptr + offs_n[:, None] * stride_bn, k, mask=mask_n[:, None])
    tl.store(bar_v_ptr + offs_n[:, None] * stride_bn, v, mask=mask_n[:, None])


def extract_kv(
    local_k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    local_v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    bar_k: torch.Tensor,  # [batch_size, max_v_size, num_qo_heads, head_dim]
    bar_v: torch.Tensor,  # [batch_size, max_v_size, num_qo_heads, head_dim]
    v_idx: torch.Tensor,  # [batch_size, num_qo_heads, max_v_size]
    v_cnt: torch.Tensor,  # [batch_size, num_qo_heads, world_size + 1]
    step: int,
):
    batch_size, max_v_size, num_qo_heads, head_dim = bar_k.shape
    _, num_tokens, num_kv_heads, _ = local_k.shape
    block_N = 128
    block_D = head_dim
    _triton_extract_kv_kernel[(triton.cdiv(max_v_size, block_N), num_qo_heads, batch_size)](
        local_k, local_v, bar_k, bar_v, v_idx, v_cnt,
        local_k.stride(0), local_k.stride(1), local_k.stride(2), local_k.stride(3),
        bar_k.stride(0), bar_k.stride(1), bar_k.stride(2), bar_k.stride(3),
        v_idx.stride(0), v_idx.stride(1), v_idx.stride(2),
        v_cnt.stride(0), v_cnt.stride(1), v_cnt.stride(2),
        step, num_tokens, num_qo_heads, num_kv_heads,
        BLOCK_N=block_N, BLOCK_D=block_D,
        num_warps=4, num_stages=1,
    )


@triton.jit
def _triton_merge_kv_kernel(
    local_k, local_v, bar_k, bar_v, v_idx, v_cnt,
    stride_lz, stride_ln, stride_lh, stride_ld,
    stride_bz, stride_bn, stride_bh, stride_bd,
    stride_iz, stride_ih, stride_in,
    stride_cz, stride_ch, stride_cr,
    step, num_tokens, num_qo_heads, num_kv_heads,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0)
    qo_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    kv_head_idx = qo_head_idx // (num_qo_heads // num_kv_heads)

    v_cnt_ptr = v_cnt + batch_idx * stride_cz + qo_head_idx * stride_ch
    min_n = tl.load(v_cnt_ptr + step * stride_cr)
    max_n = tl.load(v_cnt_ptr + (step + 1) * stride_cr)
    start_n = start_n * BLOCK_N
    end_n = start_n + BLOCK_N
    if start_n >= max_n or end_n <= min_n:
        return

    offs_d = tl.arange(0, BLOCK_D)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = (offs_n >= min_n) & (offs_n < max_n)

    v_idx_ptr = v_idx + batch_idx * stride_iz + qo_head_idx * stride_ih
    local_k_ptr = local_k + batch_idx * stride_lz + kv_head_idx * stride_lh + offs_d[None, :] * stride_ld
    local_v_ptr = local_v + batch_idx * stride_lz + kv_head_idx * stride_lh + offs_d[None, :] * stride_ld
    bar_k_ptr = bar_k + batch_idx * stride_bz + qo_head_idx * stride_bh + offs_d[None, :] * stride_bd
    bar_v_ptr = bar_v + batch_idx * stride_bz + qo_head_idx * stride_bh + offs_d[None, :] * stride_bd

    # idx = tl.load(v_idx_ptr + offs_n * stride_in, mask=mask_n, other=0) - step * num_tokens
    idx = tl.load(v_idx_ptr + offs_n * stride_in, mask=mask_n, other=0) % num_tokens
    k = tl.load(bar_k_ptr + offs_n[:, None] * stride_bn, mask=mask_n[:, None], other=0.).to(local_k.type.element_ty)
    v = tl.load(bar_v_ptr + offs_n[:, None] * stride_bn, mask=mask_n[:, None], other=0.).to(local_v.type.element_ty)
    tl.atomic_add(local_k_ptr + idx[:, None] * stride_ln, k, mask=mask_n[:, None], sem="relaxed")
    tl.atomic_add(local_v_ptr + idx[:, None] * stride_ln, v, mask=mask_n[:, None], sem="relaxed")


def merge_kv(
    local_k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    local_v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    bar_k: torch.Tensor,  # [batch_size, max_v_size, num_qo_heads, head_dim]
    bar_v: torch.Tensor,  # [batch_size, max_v_size, num_qo_heads, head_dim]
    v_idx: torch.Tensor,  # [batch_size, num_qo_heads, max_v_size]
    v_cnt: torch.Tensor,  # [batch_size, num_qo_heads, world_size + 1]
    step: int,
):
    batch_size, max_v_size, num_qo_heads, head_dim = bar_k.shape
    _, num_tokens, num_kv_heads, _ = local_k.shape
    block_N = 128
    block_D = head_dim
    _triton_merge_kv_kernel[(triton.cdiv(max_v_size, block_N), num_qo_heads, batch_size)](
        local_k, local_v, bar_k, bar_v, v_idx, v_cnt,
        local_k.stride(0), local_k.stride(1), local_k.stride(2), local_k.stride(3),
        bar_k.stride(0), bar_k.stride(1), bar_k.stride(2), bar_k.stride(3),
        v_idx.stride(0), v_idx.stride(1), v_idx.stride(2),
        v_cnt.stride(0), v_cnt.stride(1), v_cnt.stride(2),
        step, num_tokens, num_qo_heads, num_kv_heads,
        BLOCK_N=block_N, BLOCK_D=block_D,
        num_warps=4, num_stages=1,
    )


# triton.cdiv(world_size * num_blocks, BLOCK_N), num_heads, batch_size
# block_mask: [batch_size, num_heads, num_blocks_global]
@triton.jit
def _calc_block_mask_kernel(
    s_idx, block_mask,
    stride_sz, stride_sh, stride_sk,
    stride_bz, stride_bh, stride_bn,
    max_s_size, num_tokens, granularity,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    head_idx = tl.program_id(1)
    group_idx = tl.program_id(0)

    block_offs = tl.arange(0, BLOCK_N)
    slash_offs = tl.arange(0, BLOCK_K)

    s_idx_ptr = s_idx + batch_idx * stride_sz + head_idx * stride_sh
    block_mask_ptr = block_mask + batch_idx * stride_bz + head_idx * stride_bh
    block_idx = group_idx * BLOCK_N + block_offs
 
    blocks = tl.zeros([BLOCK_N], dtype=tl.uint8)
    for s_off in range(0, max_s_size, BLOCK_K):
        s = tl.load(s_idx_ptr + (s_off + slash_offs) * stride_sk)
        left = (num_tokens - granularity - s) // granularity
        right = (num_tokens - 1 - s) // granularity

        # mask is generated by checking if a block's index falls between the calculated ranges
        blocks |= tl.max((block_idx[None, :] >= left[:, None]) & (block_idx[None, :] <= right[:, None]), 0).to(tl.uint8)

    b_mask = (group_idx * BLOCK_N + block_offs) * granularity < num_tokens
    tl.store(block_mask_ptr + (group_idx * BLOCK_N + block_offs) * stride_bn, blocks, mask=b_mask)


@triton.jit
def _striped_convert_indices_kernel(
    last_row_mask, v_idx, v_cnt,
    block_mask, bar_idx, bar_pos, bar_cnt,
    stride_rz, stride_rh, stride_rn,
    stride_vz, stride_vh, stride_vk,
    stride_nz, stride_nh, stride_nt,
    stride_bt, stride_bz, stride_bh, stride_bm, stride_bn,
    stride_iz, stride_ih, stride_im, stride_ik,
    stride_cz, stride_ch, stride_cm, stride_ct,
    max_v_size, num_blocks, granularity, world_size, rank,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    head_idx = tl.program_id(1)
    block_idx_q_local = tl.program_id(0)

    block_idx_q_global = block_idx_q_local * world_size + rank

    num_tokens_local = num_blocks * granularity
    num_blocks_global = world_size * num_blocks
    shift = num_blocks_global - 1 - block_idx_q_global

    block_offs = tl.arange(0, BLOCK_N)

    last_row_mask_ptr = last_row_mask + batch_idx * stride_rz + head_idx * stride_rh
    v_idx_ptr = v_idx + batch_idx * stride_vz + head_idx * stride_vh
    v_cnt_ptr = v_cnt + batch_idx * stride_nz + head_idx * stride_nh
    block_mask_ptr = block_mask + batch_idx * stride_bz + head_idx * stride_bh + block_idx_q_local * stride_bm
    bar_idx_ptr = bar_idx + batch_idx * stride_iz + head_idx * stride_ih + block_idx_q_local * stride_im
    bar_pos_ptr = bar_pos + batch_idx * stride_iz + head_idx * stride_ih + block_idx_q_local * stride_im
    bar_cnt_ptr = bar_cnt + batch_idx * stride_cz + head_idx * stride_ch + block_idx_q_local * stride_cm

    cnt_valid = 0
    cnt_all = 0
    v_off = 0
    v = tl.load(v_idx_ptr + cnt_all * stride_vk)
    cnt_all += 1

    tl.store(bar_cnt_ptr, cnt_valid)
    bar_cnt_ptr += stride_ct
    if block_idx_q_local == tl.num_programs(0) - 1:
        tl.store(v_cnt_ptr, cnt_all - 1)
        v_cnt_ptr += stride_nt

    for step in range(world_size):
        for block_off_k in range(0, num_blocks, BLOCK_N):
            block_idx_k_local = block_off_k + block_offs
            block_idx_k_global = (block_off_k + block_offs) * world_size + step
            mask_local = tl.load(
                last_row_mask_ptr + (block_idx_k_global + shift) * stride_rn,
                mask=(block_idx_k_global + shift < num_blocks_global),
                other=0,
            )
            tl.store(
                block_mask_ptr + block_idx_k_local * stride_bn,
                mask_local,
                mask=(block_idx_k_local < num_blocks),
            )
            block_left = v_off + block_idx_k_local * granularity
            block_right = block_left + granularity
            max_blocks = block_idx_q_local + 1 if step <= rank else block_idx_q_local
            v_max = v_off + min(block_off_k + BLOCK_N, max_blocks) * granularity
            while v < v_max and cnt_all < max_v_size:
                if tl.max(((v >= block_left) & (v < block_right)) & (~mask_local), 0):
                    tl.store(bar_idx_ptr + cnt_valid * stride_ik, v - v_off)
                    tl.store(bar_pos_ptr + cnt_valid * stride_ik, cnt_all - 1)
                    cnt_valid += 1
                v = tl.load(v_idx_ptr + cnt_all * stride_vk)
                cnt_all += 1
        block_mask_ptr += stride_bt
        tl.store(bar_cnt_ptr, cnt_valid)
        bar_cnt_ptr += stride_ct
        v_off += num_tokens_local
        if block_idx_q_local == tl.num_programs(0) - 1:
            tl.store(v_cnt_ptr, cnt_all - 1)
            v_cnt_ptr += stride_nt


@triton.jit
def _zigzag_convert_indices_kernel(
    last_row_mask, v_idx, v_cnt,
    block_mask, bar_idx, bar_pos, bar_cnt,
    stride_rz, stride_rh, stride_rn,
    stride_vz, stride_vh, stride_vk,
    stride_nz, stride_nh, stride_nt,
    stride_bt, stride_bz, stride_bh, stride_bm, stride_bn,
    stride_iz, stride_ih, stride_im, stride_ik,
    stride_cz, stride_ch, stride_cm, stride_ct,
    max_v_size, num_blocks, granularity, world_size, rank,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    head_idx = tl.program_id(1)
    block_idx_q_local = tl.program_id(0)

    if rank < world_size // 2:
        revert_rank = rank * 2
    else:
        revert_rank = (world_size - 1 - rank) * 2 + 1
    if block_idx_q_local < num_blocks // 2:
        block_idx_q_global = revert_rank * (num_blocks // 2) + block_idx_q_local
    else:
        block_idx_q_global = (world_size * 2 - 1 - revert_rank) * (num_blocks // 2) + block_idx_q_local - (num_blocks // 2)

    num_blocks_global = world_size * num_blocks
    shift = num_blocks_global - 1 - block_idx_q_global

    block_offs = tl.arange(0, BLOCK_N)

    last_row_mask_ptr = last_row_mask + batch_idx * stride_rz + head_idx * stride_rh
    v_idx_ptr = v_idx + batch_idx * stride_vz + head_idx * stride_vh
    v_cnt_ptr = v_cnt + batch_idx * stride_nz + head_idx * stride_nh
    block_mask_ptr = block_mask + batch_idx * stride_bz + head_idx * stride_bh + block_idx_q_local * stride_bm
    bar_idx_ptr = bar_idx + batch_idx * stride_iz + head_idx * stride_ih + block_idx_q_local * stride_im
    bar_pos_ptr = bar_pos + batch_idx * stride_iz + head_idx * stride_ih + block_idx_q_local * stride_im
    bar_cnt_ptr = bar_cnt + batch_idx * stride_cz + head_idx * stride_ch + block_idx_q_local * stride_cm

    cnt_valid = 0
    cnt_all = 0
    v = tl.load(v_idx_ptr + cnt_all * stride_vk)
    cnt_all += 1

    tl.store(bar_cnt_ptr, cnt_valid)
    bar_cnt_ptr += stride_ct
    if block_idx_q_local == tl.num_programs(0) - 1:
        tl.store(v_cnt_ptr, cnt_all - 1)
        v_cnt_ptr += stride_nt

    for step in range(world_size):
        v_off = step * num_blocks * granularity
        v_end = v_off + num_blocks * granularity
        for block_off_k in range(0, num_blocks, BLOCK_N):
            block_idx_k_local = block_off_k + block_offs
            # assert BLOCK_N <= num_blocks // 2
            if block_off_k < num_blocks // 2:
                v_off_global = step * (num_blocks // 2) * granularity
                block_idx_k_global = step * (num_blocks // 2) + block_idx_k_local
            else:
                v_off_global = (world_size * 2 - 2 - step) * (num_blocks // 2) * granularity
                block_idx_k_global = (world_size * 2 - 1 - step) * (num_blocks // 2) + block_idx_k_local - (num_blocks // 2)
            mask_local = tl.load(
                last_row_mask_ptr + (block_idx_k_global + shift) * stride_rn,
                mask=(block_idx_k_global + shift < num_blocks_global),
                other=0,
            )
            tl.store(
                block_mask_ptr + block_idx_k_local * stride_bn,
                mask_local,
                mask=(block_idx_k_local < num_blocks),
            )
            # block_left = block_idx_k_global * granularity - v_off_global + v_off
            # block_right = block_left + granularity
            block_left = v_off + block_idx_k_local * granularity
            block_right = block_left + granularity
            v_max = (block_idx_q_global + 1) * granularity - v_off_global + v_off
            while v < v_end and cnt_all <= max_v_size:
                if v < v_max:
                    if tl.max(((v >= block_left) & (v < block_right)) & (~mask_local), 0):
                        tl.store(bar_idx_ptr + cnt_valid * stride_ik, v - v_off)
                        tl.store(bar_pos_ptr + cnt_valid * stride_ik, cnt_all - 1)
                        cnt_valid += 1
                v = tl.load(v_idx_ptr + cnt_all * stride_vk)
                cnt_all += 1
        block_mask_ptr += stride_bt
        tl.store(bar_cnt_ptr, cnt_valid)
        bar_cnt_ptr += stride_ct
        if block_idx_q_local == tl.num_programs(0) - 1:
            tl.store(v_cnt_ptr, cnt_all - 1)
            v_cnt_ptr += stride_nt


def convert_indices(
    v_idx: torch.Tensor,  # [batch_size, num_heads, max_v_size]
    s_idx: torch.Tensor,  # [batch_size, num_heads, max_s_size]
    world_size: int,
    rank: int,
    num_blocks: int,
    granularity: int,
    num_tokens: int = None,
    stripe_transform: bool = False,
    zigzag_transform: bool = False,
):
    num_blocks_global = world_size * num_blocks
    if num_tokens is None:
        # Note that for each invokation of `convert_indices`, `num_tokens` is None and becomes the **global number of tokens**
        num_tokens = num_blocks_global * granularity
    batch_size, num_heads, max_v_size = v_idx.shape
    batch_size, num_heads, max_s_size = s_idx.shape
    last_row_mask = torch.zeros((batch_size, num_heads, num_blocks_global), dtype=torch.bool, device=s_idx.device)

    BLOCK_N, BLOCK_K = 128, 128
    assert max_s_size <= BLOCK_K * BLOCK_K, f"max_s_size={max_s_size} > BLOCK_K * BLOCK_K={BLOCK_K * BLOCK_K}"
    _calc_block_mask_kernel[(triton.cdiv(num_blocks_global, BLOCK_N), num_heads, batch_size)](
        s_idx, last_row_mask,
        s_idx.stride(0), s_idx.stride(1), s_idx.stride(2),
        last_row_mask.stride(0), last_row_mask.stride(1), last_row_mask.stride(2),
        max_s_size, num_tokens, granularity,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    block_mask = torch.zeros((world_size, batch_size, num_heads, num_blocks, num_blocks), dtype=torch.bool, device=v_idx.device)
    bar_idx = torch.zeros((batch_size, num_heads, num_blocks, max_v_size), dtype=torch.int32, device=v_idx.device)
    bar_cnt = torch.empty((batch_size, num_heads, num_blocks, world_size + 1), dtype=torch.int32, device=v_idx.device)
    v_cnt = torch.empty((batch_size, num_heads, world_size + 1), dtype=torch.int32, device=v_idx.device)
    bar_pos = torch.zeros_like(bar_idx)
    if zigzag_transform:
        convert_indices_kernel = _zigzag_convert_indices_kernel
        assert num_blocks % 2 == 0
        BLOCK_N = max(num_blocks // 2, 128)
    else:
        convert_indices_kernel = _striped_convert_indices_kernel
        BLOCK_N = 128
    convert_indices_kernel[(num_blocks, num_heads, batch_size)](
        last_row_mask, v_idx, v_cnt, block_mask, bar_idx, bar_pos, bar_cnt,
        last_row_mask.stride(0), last_row_mask.stride(1), last_row_mask.stride(2),
        v_idx.stride(0), v_idx.stride(1), v_idx.stride(2),
        v_cnt.stride(0), v_cnt.stride(1), v_cnt.stride(2),
        block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3), block_mask.stride(4),
        bar_idx.stride(0), bar_idx.stride(1), bar_idx.stride(2), bar_idx.stride(3),
        bar_cnt.stride(0), bar_cnt.stride(1), bar_cnt.stride(2), bar_cnt.stride(3),
        max_v_size, num_blocks, granularity, world_size, rank, BLOCK_N=BLOCK_N,
        num_warps=1, num_stages=1,
    )
    
    return block_mask, bar_idx, bar_cnt, bar_pos, v_cnt


def _torch_convert_indices(
    v_idx: torch.Tensor,  # [batch_size, num_heads, max_v_size]
    s_idx: torch.Tensor,  # [batch_size, num_heads, max_s_size]
    world_size: int,
    rank: int,
    num_blocks: int,
    granularity: int,
):
    batch_size, num_heads, max_v_size = v_idx.shape
    num_tokens = world_size * num_blocks * granularity
    block_mask = torch.zeros((world_size, batch_size, num_heads, num_blocks, num_blocks), dtype=torch.bool, device=v_idx.device)
    bar_idx = torch.zeros((batch_size, num_heads, num_blocks, max_v_size), dtype=torch.int32, device=v_idx.device)
    bar_cnt = torch.zeros((batch_size, num_heads, num_blocks, world_size + 1), dtype=torch.int32, device=v_idx.device)
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            for block_idx_q in range(num_blocks):
                block_idx_q_global = block_idx_q * world_size + rank
                cnt_all, cnt_valid = 0, 0
                for step in range(world_size):
                    for block_idx_k in range(block_idx_q + 1):
                        block_idx_k_global = block_idx_k * world_size + step
                        s_min = max((block_idx_q_global - block_idx_k_global - 1) * granularity, 0)
                        s_max = (block_idx_q_global - block_idx_k_global + 1) * granularity
                        flag = torch.any((s_idx[batch_idx, head_idx] > s_min) & (s_idx[batch_idx, head_idx] < s_max))
                        block_mask[step, batch_idx, head_idx, block_idx_q, block_idx_k] = flag
                        v_min = (step * num_blocks + block_idx_k) * granularity
                        max_blocks = block_idx_q + 1 if step <= rank else block_idx_q
                        v_max = (step * num_blocks + min(block_idx_k + 1, max_blocks)) * granularity
                        while cnt_all < max_v_size and v_idx[batch_idx, head_idx, cnt_all] < v_min:
                            cnt_all += 1
                        while cnt_all < max_v_size and v_idx[batch_idx, head_idx, cnt_all] < v_max:
                            if not flag:
                                bar_idx[batch_idx, head_idx, block_idx_q, cnt_valid] = \
                                    v_idx[batch_idx, head_idx, cnt_all] - step * num_blocks * granularity
                                cnt_valid += 1
                            cnt_all += 1
                    bar_cnt[batch_idx, head_idx, block_idx_q, step + 1] = cnt_valid
    return block_mask, bar_idx, bar_cnt



def sum_all_diagonal_matrix(mat: torch.Tensor):
    b, h, m, n = mat.shape

    # Pads the matrix on left and right (on the last dimension)
    mat_padded = torch.nn.functional.pad(mat, (m, m), "constant", 0.) # shape: [b, h, m, 2 * m + n]
    # Change the strides
    mat_strided = mat_padded.as_strided((b, h, m, m + n), (m * (2 * m + n) * h, m * (2 * m + n), 2 * m + n + 1, 1))
    # Sums the resulting matrix's columns
    sum_diags = torch.sum(mat_strided, 2) # shape: [b, h, m + n]
    return sum_diags[:, :, 1:].contiguous()

def calc_index(
    q: torch.Tensor,
    k: torch.Tensor,
    v_size: List[int],
    s_size: List[int],
    last_q_size: int = 64,
    sink_tokens: int = 30,
    sliding_window: int = 100,
    group: dist.group = None,
    stripe_transform: bool = False,
    zigzag_transform: bool = False,
    granularity: int = 128,
):
    # TODO: adapt naturely striped inputs
    # TODO: flex-prefill (top-P)
    # TODO: reduce bubble
    # TODO: support total_num_tokens % world_size != 0
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if all([type(x) is list for x in v_size]) and all([type(x) is list for x in s_size]):
        flex_prefill = True
        v_p = [x[0] for x in v_size]
        v_size = [x[1] for x in v_size]
        s_p = [x[0] for x in s_size]
        s_size = [x[1] for x in s_size]
    else:
        flex_prefill = False
        assert all([type(x) is int for x in v_size]) and all([type(x) is int for x in s_size])

    max_v_size = min(triton.cdiv(max(v_size), 128), num_tokens // 128) * 128
    max_s_size = min(triton.cdiv(max(s_size), 128), num_tokens // 128) * 128

    last_rank = world_size - 1
    if rank == last_rank:
        last_q = q[:, -last_q_size:, :, :].detach().clone().reshape((batch_size, last_q_size, num_kv_heads, -1, head_dim))
    else:
        last_q = torch.zeros((batch_size, last_q_size, num_kv_heads, num_qo_heads // num_kv_heads, head_dim), device=q.device, dtype=q.dtype)

    dist.broadcast(last_q, src=last_rank, group=group, async_op=False)

    qk = torch.einsum('bmghd, bngd -> bghmn', last_q, k) * (k.shape[-1] ** -0.5)
    qk = qk.reshape((batch_size, num_qo_heads, last_q_size, num_tokens))

    if rank == last_rank:
        # Causal Mask, requires num_tokens // world_size >= last_q
        arange = torch.arange(last_q_size, device=k.device)
        mask = arange[None, None, :, None] >= arange[None, None, None, :]
        qk[:, :, :, -last_q_size:] = torch.where(mask, qk[:, :, :, -last_q_size:], -torch.inf)
    if flex_prefill:  # qk = torch.softmax(qk, dim=-1) / last_q_size
        qk_max = torch.max(qk, dim=-1, keepdim=True).values
        qk_max_list = [torch.empty_like(qk_max) for _ in range(world_size)]
        dist.all_gather(qk_max_list, qk_max, group=group, async_op=False)
        qk_max = torch.max(torch.stack(qk_max_list), dim=0).values
        qk = torch.exp(qk - qk_max)
        qk_sum = torch.sum(qk, dim=-1, keepdim=True)
        qk_sum_list = [torch.empty_like(qk_sum) for _ in range(world_size)]
        dist.all_gather(qk_sum_list, qk_sum, group=group, async_op=False)
        qk_sum = torch.sum(torch.stack(qk_sum_list), dim=0)
        qk /= (qk_sum * last_q_size)

    v_gather_rank = 0
    vertical = qk.sum(-2, keepdim=False)  # [B, H, N_LOCAL]
    if rank == 0 and not flex_prefill:
        vertical[..., :sink_tokens] = torch.inf
    if rank == v_gather_rank:
        gathered_vertical = [torch.empty_like(vertical) for _ in range(world_size)]
    else:
        gathered_vertical = None
    dist.gather(vertical, gathered_vertical, dst=v_gather_rank, group=group, async_op=False)

    if rank == v_gather_rank:
        vertical: torch.Tensor = torch.cat(gathered_vertical, dim=-1)
        if stripe_transform:
            vertical = vertical.reshape((batch_size, num_qo_heads, -1, world_size, granularity))
            vertical = vertical.swapaxes(2, 3)
            vertical = vertical.reshape((batch_size, num_qo_heads, -1))
        elif zigzag_transform:
            vertical = vertical.reshape((batch_size, num_qo_heads, 2, world_size, -1))
            chunks = []
            for step in range(world_size):
                chunks.append(vertical[:, :, 0, step])
                chunks.append(vertical[:, :, 1, world_size - 1 - step])
            vertical = torch.concat(chunks, dim=2).reshape((batch_size, num_qo_heads, -1))

        v_topk = torch.topk(vertical, max_v_size, -1, sorted=True)
        v_indices = v_topk.indices.to(torch.int32)
        if flex_prefill:
            v_cumsum = v_topk.values.cumsum_(dim=-1)
            v_size = (v_cumsum < torch.tensor(v_p, device=k.device)[None, :, None]).sum(dim=-1, keepdim=True)
        else:
            v_size = torch.tensor(v_size, device=k.device)[None, :, None]
        v_arange = torch.arange(max_v_size, device=k.device)
        v_indices.masked_fill_(v_arange[None, None, :] >= v_size, num_tokens * world_size)
        v_indices = v_indices.sort(dim=-1, descending=False).values
    else:
        v_indices = torch.empty((batch_size, num_qo_heads, max_v_size), dtype=torch.int32, device=k.device)
    dist.broadcast(v_indices, src=v_gather_rank, group=group, async_op=False)  # async

    s_gather_rank = 0
    slash = sum_all_diagonal_matrix(qk) # shape: [B, H, N_LOCAL + LAST_Q_SIZE - 1]
    if rank == world_size - 1 and not flex_prefill:
        # -> index starting from the left bottom corner to right upper corner
        # (sliding_window) from -(last_q_size-1) is the sliding window close to the main diagonal
        slash[..., -(last_q_size - 1 + sliding_window):] = torch.inf


    if rank == s_gather_rank:
        gathered_slash = [torch.empty_like(slash) for _ in range(world_size)]
    else:
        gathered_slash = None
    dist.gather(slash, gathered_slash, dst=s_gather_rank, group=group, async_op=False)

    if rank == s_gather_rank:
        slash = gathered_slash[0]
        for next_slash in gathered_slash[1:]:
            slash[..., -last_q_size + 1:] += next_slash[..., :last_q_size - 1]
            slash = torch.cat((slash, next_slash[..., last_q_size - 1:]), dim=-1)

        # slash presents the sum of attention from 0-th to (num_tokens_global - last_q_size - 1), where 0 represents the diagonal at bottom left corner
        slash = slash[..., :-last_q_size + 1]
        s_topk = torch.topk(slash, max_s_size, -1, sorted=True)

        # s_indices contain indices starting from the right upper corner to left bottom corner
        s_indices = (num_tokens * world_size - 1) - s_topk.indices.to(torch.int32)
        if flex_prefill:
            s_cumsum = s_topk.values.cumsum_(dim=-1)
            s_size = (s_cumsum < torch.tensor(s_p, device=k.device)[None, :, None]).sum(dim=-1, keepdim=True)
        else:
            s_size = torch.tensor(s_size, device=k.device)[None, :, None]
        s_arange = torch.arange(max_s_size, device=k.device)
        s_indices.masked_fill_(s_arange[None, None, :] >= s_size, -1)
        s_indices = s_indices.sort(dim=-1, descending=True).values
    else:
        s_indices = torch.empty((batch_size, num_qo_heads, max_s_size), dtype=torch.int32, device=k.device)
    dist.broadcast(s_indices, src=s_gather_rank, group=group, async_op=False)

    return v_indices.to(torch.int32), s_indices.to(torch.int32)

def calc_index_local(
    q: torch.Tensor,
    k: torch.Tensor,
    v_size: List[int],
    s_size: List[int],
    last_q_size: int = 64,
    sink_tokens: int = 30,
    sliding_window: int = 100,
    group: dist.group = None,
    stripe_transform: bool = False,
    zigzag_transform: bool = False,
    granularity: int = 128,
):
    batch_size, num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]

    if all([type(x) is list for x in v_size]) and all([type(x) is list for x in s_size]):
        flex_prefill = True
        v_p = [x[0] for x in v_size]
        v_size = [x[1] for x in v_size]
        s_p = [x[0] for x in s_size]
        s_size = [x[1] for x in s_size]
    else:
        flex_prefill = False
        assert all([type(x) is int for x in v_size]) and all([type(x) is int for x in s_size])

    qk = torch.einsum(
        f'bmghd, bngd -> bghmn',
        q[:, -last_q_size:, :, :].reshape((batch_size, last_q_size, num_kv_heads, -1, head_dim)),
        k,
    ).reshape((batch_size, num_qo_heads, last_q_size, num_tokens)) * (head_dim ** -0.5)

    arange = torch.arange(last_q_size, device=k.device)
    mask = arange[None, None, :, None] >= arange[None, None, None, :]
    qk[:, :, :, -last_q_size:] = torch.where(mask, qk[:, :, :, -last_q_size:], -torch.inf)
    if flex_prefill:
        qk = torch.softmax(qk, dim=-1) / last_q_size

    max_v_size = min(max(v_size), num_tokens)
    max_v_size = triton.cdiv(max_v_size, 128) * 128
    vertical = qk.sum(-2, keepdim=False)
    if not flex_prefill:
        vertical[..., :sink_tokens] = torch.inf
    if stripe_transform:
        vertical = vertical.reshape((batch_size, num_qo_heads, -1, dist.get_world_size(group), granularity))
        vertical = vertical.swapaxes(2, 3)
        vertical = vertical.reshape((batch_size, num_qo_heads, -1))
    elif zigzag_transform:
        vertical = vertical.reshape((batch_size, num_qo_heads, 2, dist.get_world_size(group), -1))
        chunks = []
        for step in range(dist.get_world_size(group)):
            chunks.append(vertical[:, :, 0, step])
            chunks.append(vertical[:, :, 1, dist.get_world_size(group) - 1 - step])
        vertical = torch.concat(chunks, dim=2).reshape((batch_size, num_qo_heads, -1))
    v_topk = torch.topk(vertical, max_v_size, -1, sorted=True)
    v_indices = v_topk.indices
    if flex_prefill:
        v_cumsum = v_topk.values.cumsum_(dim=-1)
        v_size = (v_cumsum < torch.tensor(v_p, device=k.device)[None, :, None]).sum(dim=-1, keepdim=True)
    else:
        v_size = torch.tensor(v_size, device=k.device)[None, :, None]

    max_s_size = min(max(s_size), num_tokens)
    max_s_size = triton.cdiv(max_s_size, 128) * 128
    slash = sum_all_diagonal_matrix(qk)[..., :-last_q_size + 1]
    if not flex_prefill:
        slash[..., -sliding_window:] = torch.inf
    s_topk = torch.topk(slash, max_s_size, -1, sorted=True)
    s_indices = (num_tokens - 1) - s_topk.indices
    if flex_prefill:
        s_cumsum = s_topk.values.cumsum_(dim=-1)
        s_size = (s_cumsum < torch.tensor(s_p, device=k.device)[None, :, None]).sum(dim=-1, keepdim=True)
    else:
        s_size = torch.tensor(s_size, device=k.device)[None, :, None]

    v_arange = torch.arange(max_v_size, device=k.device)
    v_idx = v_indices.to(torch.int32).reshape((batch_size, num_qo_heads, -1))
    v_idx.masked_fill_(v_arange[None, None, :] >= v_size, 2147483647)
    v_idx = v_idx.sort(dim=-1, descending=False).values

    s_arange = torch.arange(max_s_size, device=k.device)
    s_idx = s_indices.to(torch.int32).reshape((batch_size, num_qo_heads, -1))
    s_idx.masked_fill_(s_arange[None, None, :] >= s_size, -1)
    s_idx = s_idx.sort(dim=-1, descending=True).values

    return v_idx, s_idx

def build_index_local(
    q: torch.Tensor,
    k: torch.Tensor,
    v_size: List[int],
    s_size: List[int],
    num_tokens: int,
    granularity: int,
    world_size: int = 1,
    rank: int = 0,
):
    if type(v_size) is list:
        assert len(v_size) == q.shape[2]
        assert len(s_size) == q.shape[2]
        v_idx, s_idx = calc_index_local(q, k, v_size, s_size, last_q_size=64)
    else:
        v_idx, s_idx = v_size, s_size

    num_blocks = triton.cdiv(num_tokens, granularity)
    block_mask, bar_idx, bar_cnt, _, _ = convert_indices(v_idx, s_idx, world_size, rank, num_blocks, granularity)
    block_mask = block_mask[rank]
    return block_mask, bar_idx, bar_cnt

def build_index(
    q: torch.Tensor,
    k: torch.Tensor,
    v_size: List[int],
    s_size: List[int],
    num_tokens: int, # num_tokens_local
    granularity: int,
    stripe_transform: bool = True,
    zigzag_transform: bool = False,
    group: dist.group = None,
):
    """
    Input: (all inputs correspond to the local part for each rank)
        q: shape [batch_size, num_tokens_local, num_qo_heads, head_dim]
        k: shape [batch_size, num_tokens_local, num_kv_heads, head_dim]
        v_size: shape [num_qo_heads]
        s_size: shape [num_qo_heads]
        num_tokens: number of tokens in the local part of QK
    Returns:
        block_mask: shape [world_size, batch_size, num_heads, num_blocks, num_blocks]
        bar_idx: shape [batch_size, num_heads, num_blocks, max_v_size]
        bar_cnt: shape [batch_size, num_heads, num_blocks, world_size + 1], each entry is the cumulative number of selected bars corresponding a rank
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if isinstance(v_size, list):
        v_idx, s_idx = calc_index(
            q, k, v_size, s_size, last_q_size=64, group=group,
            stripe_transform=stripe_transform,
            zigzag_transform=zigzag_transform,
            granularity=granularity
        )
    else:
        v_idx, s_idx = v_size, s_size

    num_blocks = triton.cdiv(num_tokens, granularity) # num_blocks_local

    # Note that block_mask is a 5D tensor with shape [world_size, batch_size, num_heads, num_blocks, num_blocks]
    # with each block_mask[i] is to a mask corresponding the num_tokens_local x num_tokens_local matmul for each step
    block_mask, bar_idx, bar_cnt, bar_pos, v_cnt = convert_indices(
        v_idx, s_idx, world_size, rank, num_blocks, granularity,
        stripe_transform=stripe_transform,
        zigzag_transform=zigzag_transform,
    )
    return block_mask, bar_idx, bar_cnt, bar_pos, v_idx, v_cnt


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


def convert_blockmask(
    blockmask: torch.Tensor,  # [world_size, batch_size, num_heads, num_blocks, num_blocks]
    block_size_M: int,
    block_size_N: int,
):
    ratio = block_size_M // block_size_N
    original_shape = blockmask.shape
    blockmask = blockmask.to(dtype=torch.uint8)
    blockmask = blockmask.unsqueeze(-1).tile([1] * len(original_shape) + [ratio]).reshape((*original_shape[:-1], -1))

    # now block_mask is [world_size, batch_size, num_heads, num_blocks, num_blocks * ratio]
    nonzero_val, nonzero_idx = blockmask.sort(dim=-1, stable=True, descending=True)

    nonzero_rowcnt = blockmask.sum(dim=-1, dtype=torch.int32)
    return nonzero_idx.contiguous().to(dtype=torch.int32), nonzero_rowcnt.contiguous()

