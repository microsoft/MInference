#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# Credits: This logger implementation is inspired by project https://github.com/zhuzilin/ring-flash-attention
import os
import torch
import torch.distributed as dist

from einops import rearrange
from typing import List, Tuple, Dict
from time import perf_counter

from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)


from .utils import (
    RingComm, update_out_and_lse, 
    recover_zigzag_output, get_default_args, 
)
from .op_utils.moba_utils import (
    shuffle_input_all, shuffle_input_only, compute_moba_gate
)


def moba_zigzag_attn_fwd_step(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, # [S, H, D]
    step: int,
    causal: bool, 
    # q_seq_offsets: torch.Tensor,
    num_q_blocks: int,
    k_seq_offsets: torch.Tensor,

    gate_mask: torch.Tensor, # [num_filtered_chunk, num_head, seq_len] 
    cu_chunk: torch.Tensor,
    filtered_chunk_indices: torch.Tensor,
    num_filtered_chunk: int,
    chunk_to_batch: torch.Tensor,
    moba_chunk_size: int,
    moba_topk: int,

    softmax_scale,
    dropout_p=0,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    _, _, seq_len = gate_mask.shape
    q_block_seq_len, num_head, head_dim = q.shape
    k_block_seq_len, k_num_head, _ = k.shape
    if num_head > k_num_head:
        k = torch.repeat_interleave(k, num_head // k_num_head, dim=1)
        v = torch.repeat_interleave(v, num_head // k_num_head, dim=1)

    block_seq_len = q_block_seq_len // num_q_blocks

    # assumption: block_seq_len is divisible by moba_chunk_size
    assert (block_seq_len % moba_chunk_size == 0), "block_seq_len should be divisible by moba_chunk_size"

    kv = torch.stack((k, v), dim=1)
    k_seq_offset_list = [k_seq_offsets[i].detach().cpu().item() for i in range(len(k_seq_offsets))]
    filtered_kv_indices = torch.arange(
        0, min(k_seq_offset_list[0] + block_seq_len, num_filtered_chunk * moba_chunk_size) - k_seq_offset_list[0], 
        device=k.device, dtype=torch.int32
    )
    kv_chunk_indices = torch.arange(
        k_seq_offset_list[0], min(k_seq_offset_list[0] + block_seq_len, num_filtered_chunk * moba_chunk_size), 
        moba_chunk_size, device=k.device, dtype=torch.int32
    )
    if len(k_seq_offset_list) > 1:
        filtered_kv_indices =  torch.cat([
            filtered_kv_indices,
            torch.arange(
                block_seq_len, 
                min(k_seq_offset_list[1] + block_seq_len, num_filtered_chunk * moba_chunk_size) - k_seq_offset_list[1] + block_seq_len, 
                device=k.device, dtype=torch.int32
            )
        ])
        kv_chunk_indices = torch.cat([
            kv_chunk_indices, 
            torch.arange(
                k_seq_offset_list[1], 
                min(k_seq_offset_list[1] + block_seq_len, num_filtered_chunk * moba_chunk_size), 
                moba_chunk_size, 
                device=k.device, dtype=torch.int32
            )
        ])
    filtered_kv = kv.index_select(0, filtered_kv_indices)
    kv_chunk_indices = kv_chunk_indices // moba_chunk_size
    num_filtered_kv_chunks = len(kv_chunk_indices)

    q_indices = torch.arange(
        0 if num_q_blocks == 2 else block_seq_len,  2 * block_seq_len, 
        device=q.device, dtype=torch.int32
    )
    
    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    gate_mask_q = gate_mask.index_select(0, kv_chunk_indices) 
    gate_mask_q = gate_mask_q.index_select(2, q_indices) # we need to know which part(s) of the two query blocks should be activated
   
    moba_q_indices = gate_mask_q.reshape(gate_mask_q.shape[0], -1).nonzero(as_tuple=True)[-1]
    moba_seqlen_q = gate_mask_q.sum(dim=-1).flatten()
    
    # -----------------------------------------------------------
    # select all q that needs moba attn based on the moba_q_indices
    moba_q = rearrange(q, "s h d -> ( h s ) d")
    moba_q = moba_q.index_select(0, moba_q_indices)  # [ selected_HS, D ]
    moba_q = moba_q.unsqueeze(1)

    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % q_block_seq_len * num_head + moba_q_indices // q_block_seq_len

    """ prepare moba kv """
    # Since moba_q is organized as HS * N, we need to reorganize kv to adapt to q

    # cut off zero experts
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]

    # moba cu_seqlen for flash attn
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)

    # -----------------------------------------------------------------------------------
    # here `x` only stands for a dimension (stack dimension for KV)
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d") # [H, K_S, 2, D ]

    moba_kv = moba_kv.split(moba_chunk_size, dim=1) # tuple of (num_selected_chunks) elements with shape [H, chunk_size, 2, D]
    moba_kv = torch.cat(moba_kv, dim=0) # [H x num_selected_chunks, chunk_size, 2, D ] after split

    # The transformation is aimed for masking out by valid_expert_mask where the mask selects elements along (H x num_selected_chunks) dimension
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan

    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2) # [H x num_selected_chunks x chunk_size, 2, 1, D]
    moba_cu_seqlen_kv = (
        torch.arange(
            0, num_filtered_kv_chunks * num_head + 1 - zero_expert_count,
            dtype=torch.int32, device=q.device,
        ) * moba_chunk_size
    )

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

    softmax_scale = softmax_scale = head_dim ** (-0.5)

    self_attn_cu_seqlen = [0] + [moba_chunk_size] * (q_block_seq_len // moba_chunk_size)
    if q_block_seq_len % moba_chunk_size != 0:
        self_attn_cu_seqlen.append(q_block_seq_len % moba_chunk_size)
    self_attn_cu_seqlen = torch.tensor(self_attn_cu_seqlen, device=q.device, dtype=torch.int32)
    self_attn_cu_seqlen = self_attn_cu_seqlen.cumsum(dim=0, dtype=torch.int32)

    # -----------------------------------------------------------------------------------
    # self attn 
    if causal:
        # out, softmax_lse, S_dmask, rng_state
        self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q, k=k, v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=q_block_seq_len,
                max_seqlen_k=k_block_seq_len,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )
    else:
        # self_attn_out_sh, self_attn_lse_hs = None, None
        self_attn_out_sh = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        self_attn_lse_hs = torch.zeros((num_head, q_block_seq_len), device=q.device, dtype=torch.float32) + (-float('inf'))


    # moba attn
    # moba_attn_lse_hs - [1, num_nonzero_elems]
    if moba_q.shape[0] > 0:
        # out, softmax_lse, S_dmask, rng_state
        moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=q_block_seq_len,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )
    else:
        moba_attn_lse_hs = torch.zeros((1, moba_q.shape[0]), device=q.device, dtype=torch.float32) + (-float('inf'))

    # -----------------------------------------------------------------------------------
    # If no queries need to be computed with the current KV chunk and no causal attention is needed, return None to skip the output update
    if not causal and moba_q.shape[0] == 0:
        return None, None, 0, torch.zeros((num_head,), device=q.device, dtype=torch.float32)

    # -----------------------------------------------------------------------------------
    # Processing output and lse
    # output buffer [S, H, D], same shape as q
    output = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    # flatten vS & H for index ops
    output_2d = output.view(-1, q.shape[2])

    # --------------------------------------------------
    moba_attn_lse: torch.Tensor = moba_attn_lse_hs.t().contiguous() # [ num_nonzero_elems, 1 ]
    self_attn_lse_sh = self_attn_lse_hs.t().contiguous() # [q_S, H]

    # calc mixed_lse
    # minus max lse to avoid exp explosion
    max_lse_1d = self_attn_lse_sh.view(-1) # [ vS ]
    max_lse_1d = max_lse_1d.index_reduce( 
        0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
    )
    self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
    moba_attn_lse = (
        moba_attn_lse.view(-1)
        .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
        .reshape_as(moba_attn_lse)
    )

    # --------------------------------------------------
    # Build mixed attn lse
    mixed_attn_se_sh = self_attn_lse_sh.exp() if causal else torch.zeros_like(self_attn_lse_sh)
    moba_attn_se = moba_attn_lse.exp() if moba_q.shape[0] > 0 else torch.zeros_like(moba_attn_lse)

    # index_add_: converting elements from 1D tensor (num_nonzero_elems) to matrices (HS)
    # Now, mixed_attn_se_sh is the sum of LSE of self attn and LSE of moba attn (including multiple LSEs corresponding to the same q token but in different HS positions)
    mixed_attn_se_sh.view(-1).index_add_(
        0, moba_q_sh_indices, moba_attn_se.view(-1)
    )
    mixed_attn_lse_sh = mixed_attn_se_sh.log()

    # ----------------------------------------------------
    # Compute factor of self-attention and add to output
    if causal:
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

    # add moba output
    # ----------------------------------------------------
    # Compute factor of moba-attention and add to output
    if moba_q.shape[0] > 0:
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)

        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out.to(output_2d.dtype))

    output = output.to(q.dtype)

    # add back max lse
    mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
    return output, mixed_attn_lse_sh.t()

def moba_zigzag_attn_fwd(
    process_group,
    q: torch.Tensor, # [S, H, D]
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor, # sequence offsets for Q
    layer_idx: int,

    gate_mask, 
    cu_chunk,
    filtered_chunk_indices,
    num_filtered_chunk,
    chunk_to_batch,
    moba_chunk_size,
    moba_topk,

    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[0] // 2
    seq_len, num_q_heads, head_dim = q.shape

    out, lse = None, None
    next_k, next_v = None, None

    kv_seq_offsets = torch.clone(seq_offsets)
    next_kv_seq_offsets = None

    def fwd_step(
            q_, k_, v_, step_, causal_, 
            # q_seq_offsets, 
            num_q_blocks,
            k_seq_offsets
    ):
        return moba_zigzag_attn_fwd_step(
                q_, k_, v_, 
                step_,
                causal_,
                num_q_blocks,
                k_seq_offsets,

                gate_mask, 
                cu_chunk,
                filtered_chunk_indices,
                num_filtered_chunk,
                chunk_to_batch,
                moba_chunk_size,
                moba_topk,

                softmax_scale,
                dropout_p,
                window_size,
                alibi_slopes,
                deterministic,
            )

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            # when step < N-1, do the ring-communication to get KV to be used in the next round
            next_k, next_v, next_kv_seq_offsets = comm.send_recv_kv_offsets(k, v, kv_seq_offsets)

        if step == 0:
            # Do softmax(QK^T / sqrt(d_k))V on the currently hold K and V
            # and record the output and the LSE
            block_out, block_lse = fwd_step(
                q, k, v, step, causal_=True, 
                num_q_blocks=2,
                k_seq_offsets=kv_seq_offsets,
            )

            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.revert_rank:
            k0 = k[:block_seq_len]
            v0 = v[:block_seq_len]
            block_out, block_lse = fwd_step(
                q, k0, v0, step, causal_=False, 
                num_q_blocks=2,
                k_seq_offsets=kv_seq_offsets[0:1],
            )

            if block_out is not None:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            q1 = q[block_seq_len:]
            block_out, block_lse = fwd_step(
                q1, k, v, step, causal_=False,
                num_q_blocks=1,
                k_seq_offsets=kv_seq_offsets,
            )

            if block_out is not None:
                out, lse = update_out_and_lse(
                    out, lse,
                    block_out,
                    block_lse,
                    slice_=(slice(block_seq_len, None)), 
                )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v, kv_seq_offsets = next_k, next_v, next_kv_seq_offsets

    out = out.to(q.dtype) # [S, H, D]
    lse = lse.squeeze(dim=-1).transpose(0, 1) # [H, S]
    return out, lse

def moba_zigzag_attn_bwd_step(
    step: int,

    dout, # [blk_S, H, D]
    out, # [blk_S, H, D]
    causal: bool,

    q: torch.Tensor, # [blk_S, H, D]
    k: torch.Tensor, # [blk_S, H, D]
    v: torch.Tensor, # [blk_S, H, D]
    # dq: torch.Tensor, 
    # dk: torch.Tensor, 
    # dv: torch.Tensor, # [blk_S, H, D]
    
    softmax_lse: torch.Tensor, # [H, blk_S]
    num_q_blocks: int,
    k_seq_offsets: torch.Tensor,
    layer_idx: int,
    
    gate_mask,
    cu_chunk,
    filtered_chunk_indices,
    num_filtered_chunk,
    chunk_to_batch: torch.Tensor,
    moba_chunk_size: int,
    moba_topk: int,

    softmax_scale,
    dropout_p=0,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    _, _, seq_len = gate_mask.shape
    q_block_seq_len, num_head, head_dim = q.shape
    k_block_seq_len, k_num_head, _ = k.shape
    if num_head > k_num_head:
        k = torch.repeat_interleave(k, num_head // k_num_head, dim=1)
        v = torch.repeat_interleave(v, num_head // k_num_head, dim=1)

    block_seq_len = q_block_seq_len // num_q_blocks

    # assumption: block_seq_len is divisible by moba_chunk_size
    assert (block_seq_len % moba_chunk_size == 0), "block_seq_len should be divisible by moba_chunk_size"

    # -----------------------------------------------------------------------------------
    dq = torch.zeros_like(q, dtype=q.dtype)
    dk = torch.zeros_like(k, dtype=k.dtype)
    dv = torch.zeros_like(v, dtype=v.dtype)

    kv = torch.stack((k, v), dim=1)
    dkv = torch.stack((dk, dv), dim=1)
    # -----------------------------------------------------------------------------------


    k_seq_offset_list = [k_seq_offsets[i].detach().cpu().item() for i in range(len(k_seq_offsets))]
    filtered_kv_indices = torch.arange(
        0, min(k_seq_offset_list[0] + block_seq_len, num_filtered_chunk * moba_chunk_size) - k_seq_offset_list[0], 
        device=k.device, dtype=torch.int32
    )
    kv_chunk_indices = torch.arange(
        k_seq_offset_list[0], min(k_seq_offset_list[0] + block_seq_len, num_filtered_chunk * moba_chunk_size), 
        moba_chunk_size, device=k.device, dtype=torch.int32
    )
    if len(k_seq_offset_list) > 1:
        filtered_kv_indices = torch.cat([
            filtered_kv_indices,
            torch.arange(
                block_seq_len, 
                min(k_seq_offset_list[1] + block_seq_len, num_filtered_chunk * moba_chunk_size) - k_seq_offset_list[1] + block_seq_len, 
                device=k.device, dtype=torch.int32
            )
        ])
        kv_chunk_indices = torch.cat([
            kv_chunk_indices, 
            torch.arange(
                k_seq_offset_list[1], 
                min(k_seq_offset_list[1] + block_seq_len, num_filtered_chunk * moba_chunk_size), 
                moba_chunk_size, 
                device=k.device, dtype=torch.int32
            )
        ])
    filtered_kv = kv.index_select(0, filtered_kv_indices)
    filtered_dkv = dkv.index_select(0, filtered_kv_indices)

    kv_chunk_indices = kv_chunk_indices // moba_chunk_size
    num_filtered_kv_chunks = len(kv_chunk_indices)

    q_indices = torch.arange(
        0 if num_q_blocks == 2 else block_seq_len,  2 * block_seq_len, 
        device=q.device, dtype=torch.int32
    )

    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    gate_mask_q = gate_mask.index_select(0, kv_chunk_indices) 
    gate_mask_q = gate_mask_q.index_select(2, q_indices) # we need to know which part(s) of the two query blocks should be activated

    # equivalent to einops.rearrange(q, "n h s -> n (h s)"). ([s] [s] ... [s] for h times)
    # [HS indices] * N (total size: all non-zero elements in HS dimension, potentially repeat)
    # [num_selected_chunks, HS indices of non-zero elements]
    # gate_mask has been filtered by q_indices. If we still need use gate_mask_q for indexing, it should be offset by block_seq_len if num_q_blocks == 1
    #  + (0 if num_q_blocks == 2 else block_seq_len)  
    moba_q_indices = gate_mask_q.reshape(gate_mask_q.shape[0], -1).nonzero(as_tuple=True)[-1]

    # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    # moba_seqlen_q has shape (num_selecte_chunks * num_heads, ) => varlen_forward computes attention by (num_selecte_chunks * num_heads) times
    moba_seqlen_q = gate_mask_q.sum(dim=-1).flatten()

    # -----------------------------------------------------------
    # select all q that needs moba attn based on the moba_q_indices
    moba_q = rearrange(q, "s h d -> ( h s ) d")
    moba_dq = rearrange(dq, "s h d -> ( h s ) d")

    moba_q = moba_q.index_select(0, moba_q_indices)  # [ selected_HS, D ]
    moba_dq = moba_dq.index_select(0, moba_q_indices)  # [ selected_HS, D ]
    
    # [ selected_S, 1, D ] (pseudo head dim for flash attn)
    moba_q = moba_q.unsqueeze(1)
    moba_dq = moba_dq.unsqueeze(1)

    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    # note that original q has shape (S, H, D) while moba_q_indices is based on (H S)
    # Ignoring D, q has the flattend form like [H] [H] ... [H] for S times 
    # => moba_q_sh_indices is the index of each token in the original q tensor
    moba_q_sh_indices = moba_q_indices % q_block_seq_len * num_head + moba_q_indices // q_block_seq_len


    """ prepare moba kv """
    # Since moba_q is organized as HS * N, we need to reorganize kv to adapt to q
    # cut off zero experts
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]

    # moba cu_seqlen for flash attn
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)


    # ------------------------------
    # Select dout and output
    d_moba_out = (
        # [num_non-zero_elements, D]
        dout.view(-1, head_dim).index_select(0, moba_q_sh_indices).unsqueeze(1)
    )
    moba_out = (
        # [num_non-zero_elements, D]
        out.view(-1, head_dim).index_select(0, moba_q_sh_indices).unsqueeze(1)
    )
    


    # -----------------------------------------------------------------------------------
    # here `x` only stands for a dimension (stack dimension for KV)
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d") # [H, K_S, 2, D ]
    moba_dkv = rearrange(filtered_dkv, "s x h d -> h s x d") # [H, K_S, 2, D ]

    moba_kv = moba_kv.split(moba_chunk_size, dim=1) # tuple of (num_selected_chunks) elements with shape [H, chunk_size, 2, D]
    moba_kv = torch.cat(moba_kv, dim=0) # [H x num_selected_chunks, chunk_size, 2, D ] after split
    moba_dkv = torch.cat(moba_dkv.split(moba_chunk_size, dim=1), dim=0) # [H x num_selected_chunks, chunk_size, 2, D ] after split

    # The transformation is aimed for masking out by valid_expert_mask where the mask selects elements along (H x num_selected_chunks) dimension
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count

        # cut off zero Q expert from kv , or the grad may be nan
        moba_kv = moba_kv[valid_expert_mask]  
        moba_dkv = moba_dkv[valid_expert_mask]

    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2) # [H x num_selected_chunks x chunk_size, 2, 1, D]
    moba_dkv = moba_dkv.flatten(start_dim=0, end_dim=1).unsqueeze(2) # [H x num_selected_chunks x chunk_size, 2, 1, D]

    moba_cu_seqlen_kv = (
        torch.arange(
            0, num_filtered_kv_chunks * num_head + 1 - zero_expert_count,
            dtype=torch.int32, device=q.device,
        ) * moba_chunk_size
    )

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"


    self_attn_cu_seqlen = [0] + [moba_chunk_size] * (q_block_seq_len // moba_chunk_size)
    if q_block_seq_len % moba_chunk_size != 0:
        self_attn_cu_seqlen.append(q_block_seq_len % moba_chunk_size)
    self_attn_cu_seqlen = torch.tensor(self_attn_cu_seqlen, device=q.device, dtype=torch.int32)
    self_attn_cu_seqlen = self_attn_cu_seqlen.cumsum(dim=0, dtype=torch.int32)

    # -----------------------------------------------------------------------------------
    # self attn
    if causal:
        dq_, dk_, dv_ = torch.empty_like(dq), torch.empty_like(dkv[:, 0]), torch.empty_like(dkv[:, 1])
        _flash_attn_varlen_backward(
            dout=dout, out=out, 
            q=q, k=k, v=v,
            dq=dq_, dk=dk_, dv=dv_,
            softmax_lse=softmax_lse.contiguous(),

            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=q_block_seq_len,
            max_seqlen_k=k_block_seq_len,

            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            softcap=0.0,
        )
        dq, dkv[:, 0], dkv[:, 1] = dq + dq_, dk_ + dkv[:, 0], dv_ + dkv[:, 1]

    if moba_q.shape[0] > 0:
        softmax_lse_sh = rearrange(softmax_lse.contiguous(), "h s -> (s h)")
        moba_attn_lse = (
            # [1, num_non-zero_elements]
            softmax_lse_sh.index_select(0, moba_q_sh_indices).view(1, -1)
        )

        moba_dq_, moba_dk_, moba_dv_ = torch.empty_like(moba_q), torch.empty_like(moba_kv[:, 0]), torch.empty_like(moba_kv[:, 1])
        _flash_attn_varlen_backward(
            dout=d_moba_out, out=moba_out, 
            q=moba_q, k=moba_kv[:, 0], v=moba_kv[:, 1],
            dq=moba_dq_, dk=moba_dk_, dv=moba_dv_,
            softmax_lse=moba_attn_lse,

            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,

            max_seqlen_q=q_block_seq_len,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,

            causal=False,
            dropout_p=0.0,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            softcap=0.0,
        )

        dq.view(-1, q.shape[-1]).index_add_(
            0, moba_q_sh_indices, moba_dq.view(-1, head_dim).to(dq.dtype)
        )
        moba_dkv[:, 0] = moba_dkv[:, 0] + moba_dk_
        moba_dkv[:, 1] = moba_dkv[:, 1] + moba_dv_

        # ------------------------------------------------------------------------------------
        # Backpropagate moba_dkv to dk and dv
        moba_dkv = moba_dkv.squeeze(2) # [H x num_selected_chunks x chunk_size, 2, D]
        moba_dkv = moba_dkv.unflatten(0, (-1, moba_chunk_size)) # [H x num_selected_chunks, chunk_size, 2, D]

        if zero_expert_count > 0:
            full_moba_dkv = torch.zeros(
                (moba_dkv.shape[0] + zero_expert_count, moba_chunk_size, 2, head_dim),
                dtype=moba_dkv.dtype, device=moba_dkv.device
            )
            full_moba_dkv[valid_expert_mask] = moba_dkv
            moba_dkv = full_moba_dkv # [H x num_selected_chunks, chunk_size, 2, D]
        moba_dkv = moba_dkv.split(num_head, dim=0) # [H, num_selected_chunks, chunk_size, 2, D]
        moba_dkv = torch.cat(moba_dkv, dim=1) # [H, num_selected_chunks x chunk_size, 2, D]

        filtered_dkv = rearrange(moba_dkv, "h s x d -> s x h d")
        dkv.index_add_(
            0, filtered_kv_indices, filtered_dkv # [K_S, 2, H, D]
        )
    
    if num_head > k_num_head:
        num_kv_replicas = num_head // k_num_head
        dkv_reshaped = dkv.view(-1, 2, k_num_head, num_kv_replicas, head_dim)
        dkv = dkv_reshaped.sum(dim=3)

    return dq, dkv[:, 0], dkv[:, 1]

def moba_zigzag_attn_bwd(
    process_group,
    dout, # [blk_S, H, D]
    q: torch.Tensor, # [blk_S, H, D]
    k: torch.Tensor, # [blk_S, H, D]
    v: torch.Tensor, # [blk_S, H, D]
    out, # [blk_S, H, D]
    softmax_lse, # [H, blk_S]

    seq_offsets: torch.Tensor, # sequence offsets for Q
    layer_idx: int,

    gate_mask, 
    cu_chunk,
    filtered_chunk_indices,
    num_filtered_chunk,
    chunk_to_batch,
    moba_chunk_size,
    moba_topk,

    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"

    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)

    kv_seq_offsets = torch.clone(seq_offsets)
    seq_len, num_q_heads, head_dim = q.shape
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = dout.chunk(2, dim=0)[1]
    q1 = q.chunk(2, dim=0)[1]
    out1 = out.chunk(2, dim=0)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=1)[1].contiguous()
    block_seq_len = q.shape[0] // 2

    def backward(
            step,
            dout_, q_, k_, v_, out_, 
            k_seq_offsets,
            softmax_lse_, 
            causal
        ):
        seqlen_q = q_.shape[0]
        seqlen_kv = k_.shape[0]

        params = get_default_args(moba_zigzag_attn_bwd_step).copy()
        params.update(
            {
                "step": step,
                "causal": causal,
                "dout": dout_,
                "out": out_,

                "q": q_,
                "k": k_,
                "v": v_,
                "softmax_lse": softmax_lse_,

                "num_q_blocks": 1 if seqlen_q == block_seq_len else 2,
                "k_seq_offsets": k_seq_offsets,
                "layer_idx": layer_idx,

                "gate_mask": gate_mask,
                "cu_chunk": cu_chunk,
                "filtered_chunk_indices": filtered_chunk_indices,
                "num_filtered_chunk": num_filtered_chunk,
                "chunk_to_batch": chunk_to_batch,
                "moba_chunk_size": moba_chunk_size,
                "moba_topk": moba_topk,

                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        return moba_zigzag_attn_bwd_step(**params)

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            # next_k, next_v = kv_comm.send_recv_kv(k, v)
            next_k, next_v, next_kv_seq_offsets = kv_comm.send_recv_kv_offsets(k, v, kv_seq_offsets)

        if step == 0:
            dq_buffer, dk_buffer, dv_buffer = backward(
                step,
                dout, q, k, v, out, 
                kv_seq_offsets, softmax_lse, causal=True
            )
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.revert_rank:
                k0 = k[:block_seq_len]
                v0 = v[:block_seq_len]
                dq_buffer, dk_buffer, dv_buffer = backward(
                    step,
                    dout, q, k0, v0, out, 
                    kv_seq_offsets[0:1], softmax_lse, causal=False
                )
                dq += dq_buffer
            else:
                dq_buffer, dk_buffer, dv_buffer = backward(
                    step,
                    dout1, q1, k, v, out1, 
                    kv_seq_offsets, softmax_lse1, causal=False
                )

                # use the first half in dq_buffer.
                dq[block_seq_len:] += dq_buffer

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.revert_rank:
                dk[:block_seq_len] += dk_buffer
                dv[:block_seq_len] += dv_buffer
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v, kv_seq_offsets = next_k, next_v, next_kv_seq_offsets

        # the finally received dk and dv will be the same as the first dk and dv (corresponding to local Q)
        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )
    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


'''
In nnscaler, sequence are stored in the initial order, e.g., [0 1 2 3 4 5 6 7].
However, zigzag ring flash attention requires the sequence to be in the order of [0 7 2 5 3 4 1 6].
As a result:
- in forward, we need to shuffle q, k, v and recover the out
- in backward, we need to shuffle dout and recover the dq, dk, dv
'''
class MoBAZigzagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor, # [batch * seq_block_len, n_heads, head_dim]
        k: torch.Tensor, 
        v: torch.Tensor,
        seq_offset: torch.Tensor,
        layer_idx,
        dropout_p,
        softmax_scale,
        cu_seqlens,
        moba_chunk_size,
        moba_topk,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        # print(f"Rank {dist.get_rank()} | forward | q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        assert alibi_slopes is None

        (
            gate_mask, cu_chunk,
            filtered_chunk_indices,
            num_filtered_chunk,
            chunk_to_batch
        ) = compute_moba_gate(
            q, k, v,
            seq_offset,
            cu_seqlens,
            moba_chunk_size,
            moba_topk,
        )

        # gate_mask needs to be shuffled as it is coupled with q
        q, seq_offsets, gate_mask = shuffle_input_all(
            to_send=q, gate_mask=gate_mask, seq_offset=seq_offset, 
            process_group=group
        )
        k = shuffle_input_only(to_send=k, process_group=group)
        v = shuffle_input_only(to_send=v, process_group=group)

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = moba_zigzag_attn_fwd(
                group,
                q, k, v,
                seq_offsets, # sequence offsets for Q
                layer_idx,

                gate_mask, cu_chunk,
                filtered_chunk_indices,
                num_filtered_chunk,
                chunk_to_batch,
                moba_chunk_size,
                moba_topk,
                
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=False,
            )
        
        # this should be out_padded
        ctx.save_for_backward(
            q, k, v, out, softmax_lse, seq_offsets, 
            gate_mask, cu_chunk, filtered_chunk_indices,
            chunk_to_batch
        )
        ctx.num_filtered_chunk = num_filtered_chunk
        ctx.moba_chunk_size = moba_chunk_size
        ctx.moba_topk = moba_topk

        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.layer_idx = layer_idx


        out = recover_zigzag_output(out, process_group=group)
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        dout = shuffle_input_only(to_send=dout, process_group=ctx.group)
        (
            q, k, v, out, 
            softmax_lse, # [n_heads, seq_block_len]
            seq_offsets, 
            gate_mask, cu_chunk, filtered_chunk_indices,
            chunk_to_batch
        ) = ctx.saved_tensors

        num_filtered_chunk = ctx.num_filtered_chunk
        moba_chunk_size = ctx.moba_chunk_size
        moba_topk = ctx.moba_topk
        
        dq, dk, dv = moba_zigzag_attn_bwd(
            ctx.group,
            dout,
            q, k, v,
            out,
            softmax_lse,
            
            seq_offsets, 
            ctx.layer_idx,
            gate_mask,
            cu_chunk,
            filtered_chunk_indices,
            num_filtered_chunk,
            chunk_to_batch,
            moba_chunk_size,
            moba_topk,

            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        
        dq = recover_zigzag_output(dq, ctx.group)
        dk = recover_zigzag_output(dk, ctx.group)
        dv = recover_zigzag_output(dv, ctx.group)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None


def moba_zigzag_qkvpacked_func(
    qkv,
    seq_offset: torch.Tensor,
    layer_idx: int,
    cu_seqlens,
    moba_chunk_size,
    moba_topk,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return MoBAZigzagRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        seq_offset,
        layer_idx,
        dropout_p,
        softmax_scale,
        cu_seqlens,
        moba_chunk_size,
        moba_topk,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def moba_zigzag_kvpacked_func(
    q,
    kv,
    seq_offset: torch.Tensor,
    layer_idx: int,
    cu_seqlens,
    moba_chunk_size,
    moba_topk,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return MoBAZigzagRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        seq_offset,
        layer_idx,
        dropout_p,
        softmax_scale,
        cu_seqlens,
        moba_chunk_size,
        moba_topk,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def moba_zigzag_func(
    q, k, v,
    seq_offset: torch.Tensor,
    layer_idx: int,
    cu_seqlens,
    moba_chunk_size,
    moba_topk,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return MoBAZigzagRingFlashAttnFunc.apply(
        q, k, v,
        seq_offset,
        layer_idx,
        dropout_p,
        softmax_scale,
        cu_seqlens,
        moba_chunk_size,
        moba_topk,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
