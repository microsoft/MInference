# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from tilelang.autotuner import *

torch.random.manual_seed(0)


@tilelang.jit(out_idx=[6])
def flashattn(
    batch,
    heads,
    groups,
    seqlen_kv,
    config_len,
    dim,
    block_N,
    block_H,
    num_split,
    num_stages,
    threads,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, groups, dim]
    shape_v = [batch, seqlen_kv, groups, dim]
    shape_o = [batch, heads, dim]
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // groups

    part_shape = [batch, heads, num_split, dim]
    valid_block_H = min(block_H, kv_group_num)
    valid_block_N = min(block_N, seqlen_kv // num_split)

    @T.macro
    def flash_attn(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (
            bx,
            by,
            bz,
        ):
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
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(
                Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_shared
                )
                T.copy(
                    mask[bid, k * block_N : (k + 1) * block_N, cur_kv_head], mask_local
                )
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        mask_local[j] != 0, acc_s[i, j], -T.infinity(accum_dtype)
                    )
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(
                    V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared
                )
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(
                O_shared,
                Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :],
            )

    @T.macro
    def flash_attn_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor(part_shape, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (
            bx,
            by,
            bz,
        ):
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
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(
                Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)

            per_block_len = T.ceildiv(config_len, num_split)
            this_block_end = T.min(per_block_len * (sid + 1), seqlen_kv)
            this_block_begin = per_block_len * sid

            if this_block_begin < seqlen_kv:
                loop_range = T.ceildiv(per_block_len, block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if per_block_len * sid + k * block_N < seqlen_kv:
                        T.copy(
                            K[
                                bid,
                                per_block_len * sid
                                + k * block_N : per_block_len * sid
                                + (k + 1) * block_N,
                                cur_kv_head,
                                :,
                            ],
                            K_shared,
                        )
                        T.copy(
                            mask[
                                bid,
                                per_block_len * sid
                                + k * block_N : per_block_len * sid
                                + (k + 1) * block_N,
                                cur_kv_head,
                            ],
                            mask_local,
                        )
                        T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.if_then_else(
                                (mask_local[j] != 0) & (j < seqlen_kv // num_split),
                                acc_s[i, j],
                                -T.infinity(accum_dtype),
                            )
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_H):
                            scores_scale[i] = T.exp2(
                                scores_max_prev[i] * scale - scores_max[i] * scale
                            )
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.exp2(
                                acc_s[i, j] * scale - scores_max[i] * scale
                            )
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_H):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_H, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(
                            V[
                                bid,
                                per_block_len * sid
                                + k * block_N : per_block_len * sid
                                + (k + 1) * block_N,
                                cur_kv_head,
                                :,
                            ],
                            V_shared,
                        )
                        T.gemm(
                            acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                        )
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(
                    O_shared,
                    Output_partial[
                        bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid, :
                    ],
                )

    @T.macro
    def combine(
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local = T.alloc_fragment([num_split, 128], dtype)
            lse_local_split = T.alloc_local([1], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_fragment([128], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            T.annotate_layout(
                {
                    lse_logsum_local: T.Fragment(
                        lse_logsum_local.shape, forward_thread_fn=lambda i: i
                    ),
                    lse_max_local: T.Fragment(
                        lse_max_local.shape, forward_thread_fn=lambda i: i
                    ),
                    # lse_local: (local_id, thread_id)
                    lse_local: T.Fragment(
                        lse_local.shape, forward_fn=lambda i, j: (j, i)
                    ),
                }
            )

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            for k, j in T.Parallel(num_split, 128):
                lse_local[k, j] = glse[bz, by, k]
            T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split[0] = glse[bz, by, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, by, k, i]
                lse_local_split[0] = glse[bz, by, k]
                scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dim):
                Output[bz, by, i] = o_accum_local[i]

    @T.prim_func
    def flashattn_gqa_decode_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        flash_attn_split(Q, K, V, mask, glse, Output_partial)
        combine(glse, Output_partial, Output)

    @T.prim_func
    def flashattn_gqa_decode_no_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        flash_attn(Q, K, V, mask, Output)

    if num_split > 1:
        return flashattn_gqa_decode_split
    else:
        return flashattn_gqa_decode_no_split
