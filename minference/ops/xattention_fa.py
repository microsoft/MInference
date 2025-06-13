# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Refer to the code in https://github.com/mit-han-lab/x-attention
import math
import torch
from typing import List, Tuple, Dict, Any

from minference.ops.pit_sparse_flash_attention_v3 import block_attn_fwd, block_attn_bwd
from .op_utils.xattn_utils import (
    LN2, find_blocks_chunked, flat_group_gemm_fuse_reshape, softmax_fuse_block_sum
)

def xattn_estimate(
    query_states: torch.Tensor, # (batch_size, num_q_head, q_len, head_dim)
    key_states: torch.Tensor, # (batch_size, num_kv_head, k_len, head_dim)
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    if num_q_head > num_kv_head:
        key_states = torch.repeat_interleave(key_states.contiguous(), num_q_head // num_kv_head, dim=1)

    assert q_len % chunk_size == 0
    assert k_len % chunk_size == 0

    q_chunk_num = q_len // chunk_size
    q_block_num = q_len // block_size

    # assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    if use_triton and (
        "100" not in torch.cuda.get_device_properties(torch.cuda.current_device()).name
    ):
        use_triton = False
        print(
            "setting use triton to false. Triton kernel not surpported on this device"
        )

    num_strides_in_k = k_len // stride

    num_strides_per_chunk = chunk_size // stride
    num_strides_per_block = block_size // stride
    num_blocks_per_chunk = num_strides_per_chunk // num_strides_per_block

    for chunk_idx in range(q_chunk_num):
        if kdb != 1:
            raise ValueError("use_triton and kdb cannot be used together")

        q_chunk_start = chunk_idx * num_strides_per_chunk * stride
        q_chunk_end =  (chunk_idx + 1) * num_strides_per_chunk * stride

        q_chunk_start_stride = chunk_idx * num_strides_per_chunk
        q_chunk_end_stride = (chunk_idx + 1) * num_strides_per_chunk

        # attn_weights_slice: (batch_size, num_heads, chunk_size // stride, kv_len // stride)
        # (i.e. the attention sum of each SxS stride block)
        # This step is agnostic to block size and just computes the attention sum in each stride block
        attn_weights_slice = flat_group_gemm_fuse_reshape(
            # query_states, key_states, stride, chunk_start, chunk_end, is_causal=True
            query_states[:, :, q_chunk_start : q_chunk_end, :,],
            key_states,
            stride,
            q_chunk_start_stride,
            q_chunk_end_stride,
            is_causal=causal,
        )

        # (batch_size, num_heads, q_block_num, k_block_num),
        attn_sum = softmax_fuse_block_sum(
            attn_weights_slice, # (batch_size, num_heads, chunk_size // stride, kv_len // stride)
            num_strides_per_block,
            min(4096, num_strides_per_block),
            q_chunk_start_stride, q_chunk_end_stride,
            num_strides_in_k,
            1 / LN2 / math.sqrt(head_dim) / stride / norm,
            is_causal=causal,
        )
        
        
        # (batch_size, head_num, num_blocks_per_chunk, block_num)
        simple_mask = find_blocks_chunked(
            attn_sum,
            chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    attn_sums = torch.cat(attn_sum_list, dim=-2)

    #  (batch_size, head_num, num_blocks_per_chunk * q_chunk_num, block_num)
    # i.e. (batch_size, head_num, q_block_num, q_block_num)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num, q_block_num, dtype=bool, device=key_states.device
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
        # print(f"{__name__} | simple_masks[:, :, -q_block_num:, -q_block_num:].shape {simple_masks[:, :, -q_block_num:, -q_block_num:].shape} after torch.where")
    
    
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )

    # simple_masks -> (batch_size, head_num, q_block_num, q_block_num)
    return attn_sums, simple_masks

class XAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_indices,
        xattn_params, # Dict[str, Any] 
        granularity,
        causal,
        softmax_scale,
        return_softmax,
        deterministic,
    ):
        batch_size, num_tokens, num_qo_heads, head_dim = q.shape
        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        q_block_num = (q.shape[1] + granularity - 1) // granularity
        # (batch_size, head_num, q_block_num, q_block_num)
        _, block_mask = xattn_estimate(
            q.transpose(1, 2), k.transpose(1, 2), 
            granularity, 
            **xattn_params
        )
        block_mask = block_mask[:, :, -q_block_num:, -q_block_num:].contiguous()

        # Block Mask
        out, softmax_lse = block_attn_fwd(
            q, k, v, softmax_scale,
            block_mask,
            granularity=granularity,
            causal=causal,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse, block_mask)
        ctx.granularity = granularity
        ctx.deterministic = deterministic
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.head_indices = head_indices

        # print(f"{__name__} | out shape: {out.shape}")
        return (out, softmax_lse, None) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, block_mask = ctx.saved_tensors
        causal = ctx.causal

        # Block Mask
        dq, dk, dv = block_attn_bwd(
            dout, q, k, v, out,
            softmax_lse, ctx.softmax_scale,
            block_mask,
            granularity=ctx.granularity,
            deterministic=ctx.deterministic,
            causal=causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None

def xattn_flash_attn_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    head_indices: List[int], # [num_qo_heads]
    xattn_params: Dict[str, Any], 
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
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

    return XAttnFunc.apply(
        q, k, v,
        head_indices,
        xattn_params,
        granularity,
        causal,
        softmax_scale,
        return_attn_probs,
        deterministic,
    )
