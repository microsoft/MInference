# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import itertools

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from tilelang.autotuner import *

torch.random.manual_seed(0)

def get_configs():
    block_N = [64, 128]
    block_H = [64, 32]
    num_split = [2, 4, 8, 16]
    num_stages = [1, 2, 3]
    threads = [128]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

    configs = [{
        'block_N': c[0],
        'block_H': c[1],
        'num_split': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs

def leank_flashattn(batch, heads, heads1, heads2, heads3, heads4, groups, groups1, groups2, groups3, groups4, seqlen_kv, true_seq_len, seqlen_fullkv, true_full_len, dim, dim1, dim2, dim3, dim4, ndim, dtype, tune=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q1 = [batch, heads1, dim1]
    shape_k1 = [batch, groups1, true_seq_len, dim1]
    shape_v1 = [batch, groups1, true_seq_len, dim]

    shape_q2 = [batch, heads2, dim2]
    shape_k2 = [batch, groups2, true_seq_len, dim2]
    shape_v2 = [batch, groups2, true_seq_len, dim]

    shape_q3 = [batch, heads3, dim3]
    shape_k3 = [batch, groups3, true_seq_len, dim3]
    shape_v3 = [batch, groups3, true_seq_len, dim]

    shape_q4 = [batch, heads4, dim]
    shape_k4 = [batch, groups4, true_seq_len, dim4]
    shape_v4 = [batch, groups4, true_seq_len, dim]

    shape_full_q = [batch, heads, dim]
    shape_full_k = [batch, groups, true_full_len, dim]
    shape_full_v = [batch, groups, true_full_len, dim]

    shape_o = [batch, heads, dim]

    accum_dtype = "float"
    kv_group_num = heads // groups

    def kernel_func(block_N, block_H, num_split, num_stages, threads):
        part_shape = [batch, heads, num_split + 1, dim]
        part_shape1 = [batch, heads1, num_split + 1, dim]
        part_shape2 = [batch, heads2, num_split + 1, dim]
        part_shape3 = [batch, heads3, num_split + 1, dim]
        part_shape4 = [batch, heads4, num_split + 1, dim]
        valid_block_H = min(block_H, kv_group_num)

        @T.macro
        def flash_attn_split1(
                Q: T.Tensor(shape_q1, dtype),
                K: T.Tensor(shape_k1, dtype),
                V: T.Tensor(shape_v1, dtype),
                mask_mid: T.Tensor([batch, true_seq_len], "uint8"),
                glse1: T.Tensor([batch, heads1, num_split + 1], dtype),
                Output_partial1: T.Tensor(part_shape1, dtype),
        ):
            with T.Kernel(
                    batch, heads1 // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim1], dtype)
                K_shared = T.alloc_shared([block_N, dim1], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)

                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H: hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                per_block_len = T.ceildiv(seqlen_kv, num_split)
                this_block_end = T.min(per_block_len * (sid + 1), true_seq_len)
                this_block_begin = per_block_len * sid

                if this_block_begin < true_seq_len:
                    loop_range = T.ceildiv(per_block_len, block_N)
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        if per_block_len * sid + k * block_N < true_seq_len:
                            T.copy(
                                K[bid, cur_kv_head, per_block_len * sid +
                                k * block_N: per_block_len * sid + (k + 1) * block_N, :], K_shared)
                            for i in T.Parallel(block_N):
                                if per_block_len * sid + k * block_N + i < this_block_end:
                                    mask_local[i] = mask_mid[bid, per_block_len * sid +
                                                                    k * block_N + i]
                                else:
                                    mask_local[i] = 0
                            T.clear(acc_s)
                            T.gemm(
                                Q_shared,
                                K_shared,
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow, )

                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                            -T.infinity(accum_dtype))

                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Parallel(block_H):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Parallel(block_H):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            T.copy(acc_s, acc_s_cast)
                            for i, j in T.Parallel(block_H, dim):
                                acc_o[i, j] *= scores_scale[i]
                            T.copy(
                                V[bid, cur_kv_head, per_block_len * sid +
                                k * block_N:per_block_len * sid + (k + 1) * block_N, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                    for i in T.Parallel(block_H):
                        if i < valid_block_H:
                            glse1[bid, hid * valid_block_H + i, sid] = logsum[i]
                    T.copy(acc_o[:valid_block_H, :], O_shared)
                    T.copy(O_shared, Output_partial1[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                    sid, :])

        @T.macro
        def flash_attn_split2(
                Q2: T.Tensor(shape_q2, dtype),
                K2: T.Tensor(shape_k2, dtype),
                V2: T.Tensor(shape_v2, dtype),
                mask_mid: T.Tensor([batch, true_seq_len], "uint8"),
                glse2: T.Tensor([batch, heads2, num_split + 1], dtype),
                Output_partial2: T.Tensor(part_shape2, dtype),
        ):
            with T.Kernel(
                    batch, heads1 // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim2], dtype)
                K_shared = T.alloc_shared([block_N, dim2], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)

                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q2[bid, hid * valid_block_H: hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                per_block_len = T.ceildiv(seqlen_kv, num_split)
                this_block_end = T.min(per_block_len * (sid + 1), true_seq_len)
                this_block_begin = per_block_len * sid

                if this_block_begin < true_seq_len:
                    loop_range = T.ceildiv(per_block_len, block_N)
                    for k in T.Pipelined(loop_range, num_stages=2):
                        if per_block_len * sid + k * block_N < true_seq_len:
                            T.copy(
                                K2[bid, cur_kv_head, per_block_len * sid +
                                k * block_N: per_block_len * sid + (k + 1) * block_N, :], K_shared)
                            for i in T.Parallel(block_N):
                                if per_block_len * sid + k * block_N + i < this_block_end:
                                    mask_local[i] = mask_mid[bid, per_block_len * sid +
                                                                    k * block_N + i]
                                else:
                                    mask_local[i] = 0
                            T.clear(acc_s)
                            T.gemm(
                                Q_shared,
                                K_shared,
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow, )

                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                            -T.infinity(accum_dtype))

                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Parallel(block_H):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Parallel(block_H):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            T.copy(acc_s, acc_s_cast)
                            for i, j in T.Parallel(block_H, dim):
                                acc_o[i, j] *= scores_scale[i]
                            T.copy(
                                V2[bid, cur_kv_head, per_block_len * sid +
                                k * block_N:per_block_len * sid + (k + 1) * block_N, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                    for i in T.Parallel(block_H):
                        if i < valid_block_H:
                            glse2[bid, hid * valid_block_H + i, sid] = logsum[i]
                    T.copy(acc_o[:valid_block_H, :], O_shared)
                    T.copy(O_shared, Output_partial2[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                    sid, :])

        @T.macro
        def flash_attn_split3(
                Q3: T.Tensor(shape_q3, dtype),
                K3: T.Tensor(shape_k3, dtype),
                V3: T.Tensor(shape_v3, dtype),
                mask_mid: T.Tensor([batch, true_seq_len], "uint8"),
                glse3: T.Tensor([batch, heads3, num_split + 1], dtype),
                Output_partial3: T.Tensor(part_shape3, dtype),
        ):
            with T.Kernel(
                    batch, heads1 // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim3], dtype)
                K_shared = T.alloc_shared([block_N, dim3], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)

                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q3[bid, hid * valid_block_H: hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                per_block_len = T.ceildiv(seqlen_kv, num_split)
                this_block_end = T.min(per_block_len * (sid + 1), true_seq_len)
                this_block_begin = per_block_len * sid

                if this_block_begin < true_seq_len:
                    loop_range = T.ceildiv(per_block_len, block_N)
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        if per_block_len * sid + k * block_N < true_seq_len:
                            T.copy(
                                K3[bid, cur_kv_head, per_block_len * sid +
                                k * block_N: per_block_len * sid + (k + 1) * block_N, :], K_shared)
                            for i in T.Parallel(block_N):
                                if per_block_len * sid + k * block_N + i < this_block_end:
                                    mask_local[i] = mask_mid[bid, per_block_len * sid +
                                                                    k * block_N + i]
                                else:
                                    mask_local[i] = 0
                            T.clear(acc_s)
                            T.gemm(
                                Q_shared,
                                K_shared,
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow, )

                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                            -T.infinity(accum_dtype))

                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Parallel(block_H):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Parallel(block_H):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            T.copy(acc_s, acc_s_cast)
                            for i, j in T.Parallel(block_H, dim):
                                acc_o[i, j] *= scores_scale[i]
                            T.copy(
                                V3[bid, cur_kv_head, per_block_len * sid +
                                k * block_N: per_block_len * sid + (k + 1) * block_N, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                    for i in T.Parallel(block_H):
                        if i < valid_block_H:
                            glse3[bid, hid * valid_block_H + i, sid] = logsum[i]
                    T.copy(acc_o[:valid_block_H, :], O_shared)
                    T.copy(O_shared, Output_partial3[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                    sid, :])

        @T.macro
        def flash_attn_split4(
                Q4: T.Tensor(shape_q4, dtype),
                K4: T.Tensor(shape_k4, dtype),
                V4: T.Tensor(shape_v4, dtype),
                mask_mid: T.Tensor([batch, true_seq_len], "uint8"),
                glse4: T.Tensor([batch, heads4, num_split + 1], dtype),
                Output_partial4: T.Tensor(part_shape4, dtype),
        ):
            with T.Kernel(
                    batch, heads1 // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim4], dtype)
                K_shared = T.alloc_shared([block_N, dim4], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)

                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q4[bid, hid * valid_block_H: hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                per_block_len = T.ceildiv(seqlen_kv, num_split)
                this_block_end = T.min(per_block_len * (sid + 1), true_seq_len)
                this_block_begin = per_block_len * sid

                if this_block_begin < true_seq_len:
                    loop_range = T.ceildiv(per_block_len, block_N)
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        if per_block_len * sid + k * block_N < true_seq_len:
                            T.copy(
                                K4[bid, cur_kv_head, per_block_len * sid +
                                k * block_N: per_block_len * sid + (k + 1) * block_N, :], K_shared)
                            for i in T.Parallel(block_N):
                                if per_block_len * sid + k * block_N + i < this_block_end:
                                    mask_local[i] = mask_mid[bid, per_block_len * sid +
                                                                    k * block_N + i]
                                else:
                                    mask_local[i] = 0
                            T.clear(acc_s)
                            T.gemm(
                                Q_shared,
                                K_shared,
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow, )

                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                            -T.infinity(accum_dtype))

                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Parallel(block_H):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Parallel(block_H):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            T.copy(acc_s, acc_s_cast)
                            for i, j in T.Parallel(block_H, dim):
                                acc_o[i, j] *= scores_scale[i]
                            T.copy(
                                V4[bid, cur_kv_head, per_block_len * sid +
                                k * block_N: per_block_len * sid + (k + 1) * block_N, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                    for i in T.Parallel(block_H):
                        if i < valid_block_H:
                            glse4[bid, hid * valid_block_H + i, sid] = logsum[i]
                    T.copy(acc_o[:valid_block_H, :], O_shared)
                    T.copy(O_shared, Output_partial4[bid, hid * valid_block_H: (hid + 1) * valid_block_H,
                                                    sid, :])

        @T.macro
        def flash_attn_split_full(
                Q_full: T.Tensor(shape_full_q, dtype),
                K_full: T.Tensor(shape_full_k, dtype),
                V_full: T.Tensor(shape_full_v, dtype),
                mask: T.Tensor([batch, true_full_len], "uint8"),
                glse: T.Tensor([batch, heads, num_split + 1], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, threads=threads) as (bx, by):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)

                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = num_split
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q_full[bid, hid * valid_block_H: hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(seqlen_fullkv, block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if k * block_N < true_full_len:
                        T.copy(
                            K_full[bid, cur_kv_head,
                            k * block_N: (k + 1) * block_N, :], K_shared)
                        for i in T.Parallel(block_N):
                            if k * block_N + i < true_full_len:
                                mask_local[i] = mask[bid, k * block_N + i]
                            else:
                                mask_local[i] = 0
                        T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow, )

                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                        -T.infinity(accum_dtype))

                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_H):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_H):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_H, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(
                            V_full[bid, cur_kv_head,
                            k * block_N: (k + 1) * block_N, :], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid, :])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split + 1], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split + 1, 128], dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    lse_local:
                        T.Fragment(lse_local.shape, forward_fn=lambda i, j: (j, i)),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                for k, j in T.Parallel(num_split + 1, 128):
                    lse_local[k, j] = glse[bz, by, k]
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
                for k in T.Pipelined(num_split + 1, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split + 1):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split[0] = glse[bz, by, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def flashattn_gqa_decode_split_stream(
                Q_full: T.Tensor(shape_full_q, dtype),
                K_full: T.Tensor(shape_full_k, dtype),
                V_full: T.Tensor(shape_full_v, dtype),
                mask: T.Tensor([batch, true_full_len], "uint8"),
                glse: T.Tensor([batch, heads, num_split + 1], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split_full(Q_full, K_full, V_full, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def flashattn_gqa_decode_split_1group(
                Q_full: T.Tensor(shape_full_q, dtype),
                K_full: T.Tensor(shape_full_k, dtype),
                V_full: T.Tensor(shape_full_v, dtype),
                mask: T.Tensor([batch, true_full_len], "uint8"),
                glse: T.Tensor([batch, heads, num_split + 1], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Q: T.Tensor(shape_q1, dtype),
                K: T.Tensor(shape_k1, dtype),
                V: T.Tensor(shape_v1, dtype),
                glse1: T.Tensor([batch, heads1, num_split + 1], dtype),
                Output_partial1: T.Tensor(part_shape1, dtype),
                mask_mid: T.Tensor([batch, true_seq_len], "uint8"),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split1(Q, K, V, mask_mid, glse1, Output_partial1)
            flash_attn_split_full(Q_full, K_full, V_full, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def flashattn_gqa_decode_split_2groups(
                Q_full: T.Tensor(shape_full_q, dtype),
                K_full: T.Tensor(shape_full_k, dtype),
                V_full: T.Tensor(shape_full_v, dtype),
                mask: T.Tensor([batch, true_full_len], "uint8"),
                glse: T.Tensor([batch, heads, num_split + 1], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Q: T.Tensor(shape_q1, dtype),
                K: T.Tensor(shape_k1, dtype),
                V: T.Tensor(shape_v1, dtype),
                glse1: T.Tensor([batch, heads1, num_split + 1], dtype),
                Output_partial1: T.Tensor(part_shape1, dtype),
                Q2: T.Tensor(shape_q2, dtype),
                K2: T.Tensor(shape_k2, dtype),
                V2: T.Tensor(shape_v2, dtype),
                glse2: T.Tensor([batch, heads2, num_split + 1], dtype),
                Output_partial2: T.Tensor(part_shape2, dtype),
                mask_mid: T.Tensor([batch, true_seq_len], "uint8"),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split1(Q, K, V, mask_mid, glse1, Output_partial1)
            flash_attn_split2(Q2, K2, V2, mask_mid, glse2, Output_partial2)
            flash_attn_split_full(Q_full, K_full, V_full, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def flashattn_gqa_decode_split_3groups(
                Q_full: T.Tensor(shape_full_q, dtype),
                K_full: T.Tensor(shape_full_k, dtype),
                V_full: T.Tensor(shape_full_v, dtype),
                mask: T.Tensor([batch, true_full_len], "uint8"),
                glse: T.Tensor([batch, heads, num_split + 1], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Q: T.Tensor(shape_q1, dtype),
                K: T.Tensor(shape_k1, dtype),
                V: T.Tensor(shape_v1, dtype),
                glse1: T.Tensor([batch, heads1, num_split + 1], dtype),
                Output_partial1: T.Tensor(part_shape1, dtype),
                Q2: T.Tensor(shape_q2, dtype),
                K2: T.Tensor(shape_k2, dtype),
                V2: T.Tensor(shape_v2, dtype),
                glse2: T.Tensor([batch, heads2, num_split + 1], dtype),
                Output_partial2: T.Tensor(part_shape2, dtype),
                Q3: T.Tensor(shape_q3, dtype),
                K3: T.Tensor(shape_k3, dtype),
                V3: T.Tensor(shape_v3, dtype),
                glse3: T.Tensor([batch, heads3, num_split + 1], dtype),
                Output_partial3: T.Tensor(part_shape3, dtype),
                mask_mid: T.Tensor([batch, true_seq_len], "uint8"),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split1(Q, K, V, mask_mid, glse1, Output_partial1)
            flash_attn_split2(Q2, K2, V2, mask_mid, glse2, Output_partial2)
            flash_attn_split3(Q3, K3, V3, mask_mid, glse3, Output_partial3)
            flash_attn_split_full(Q_full, K_full, V_full, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def flashattn_gqa_decode_split_4groups(
                Q_full: T.Tensor(shape_full_q, dtype),
                K_full: T.Tensor(shape_full_k, dtype),
                V_full: T.Tensor(shape_full_v, dtype),
                mask: T.Tensor([batch, true_full_len], "uint8"),
                glse: T.Tensor([batch, heads, num_split + 1], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Q: T.Tensor(shape_q1, dtype),
                K: T.Tensor(shape_k1, dtype),
                V: T.Tensor(shape_v1, dtype),
                glse1: T.Tensor([batch, heads1, num_split + 1], dtype),
                Output_partial1: T.Tensor(part_shape1, dtype),
                Q2: T.Tensor(shape_q2, dtype),
                K2: T.Tensor(shape_k2, dtype),
                V2: T.Tensor(shape_v2, dtype),
                glse2: T.Tensor([batch, heads2, num_split + 1], dtype),
                Output_partial2: T.Tensor(part_shape2, dtype),
                Q3: T.Tensor(shape_q3, dtype),
                K3: T.Tensor(shape_k3, dtype),
                V3: T.Tensor(shape_v3, dtype),
                glse3: T.Tensor([batch, heads3, num_split + 1], dtype),
                Output_partial3: T.Tensor(part_shape3, dtype),
                Q4: T.Tensor(shape_q4, dtype),
                K4: T.Tensor(shape_k4, dtype),
                V4: T.Tensor(shape_v4, dtype),
                glse4: T.Tensor([batch, heads4, num_split + 1], dtype),
                Output_partial4: T.Tensor(part_shape4, dtype),
                mask_mid: T.Tensor([batch, true_seq_len], "uint8"),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split1(Q, K, V, mask_mid, glse1, Output_partial1)
            flash_attn_split2(Q2, K2, V2, mask_mid, glse2, Output_partial2)
            flash_attn_split3(Q3, K3, V3, mask_mid, glse3, Output_partial3)
            flash_attn_split4(Q4, K4, V4, mask_mid, glse4, Output_partial4)
            flash_attn_split_full(Q_full, K_full, V_full, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        if ndim == 0:
            return flashattn_gqa_decode_split_stream
        elif ndim == 1:
            return flashattn_gqa_decode_split_1group
        elif ndim == 2:
            return flashattn_gqa_decode_split_2groups
        elif ndim == 3:
            return flashattn_gqa_decode_split_3groups
        elif ndim == 4:
            return flashattn_gqa_decode_split_4groups


    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit(out_idx=[5 * (ndim + 1) + 1 + (ndim > 0)])
        def kernel(block_N=None, block_H=None, num_split=None, num_stages=None, threads=None):
            return kernel_func(block_N, block_H, num_split, num_stages, threads)

        return kernel()

    else:

        def kernel(block_N, block_H, num_split, num_stages, threads):
            return kernel_func(block_N, block_H, num_split, num_stages, threads)

    return kernel
