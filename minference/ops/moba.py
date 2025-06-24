"""A clean version of moba implementation for educational purposes"""
import math
import torch

from einops import rearrange
from typing import Union, Tuple, Callable, Optional
from flash_attn import flash_attn_varlen_func, flash_attn_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)

from .op_utils.moba_utils import calc_chunks

def hf_to_fa(x: torch.Tensor):
    """
    Args:
        x (torch.Tensor): [batch, heads, seqlen, head_dim]

    Returns:
        torch.Tensor: [batch * seqlen, heads, head_dim]
    """
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])


def moba_attn_varlen_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """Implement the moba brute-force setting for reference

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        max_seqlen (int): the max sequence length of the batch, same definition in flash attn

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """

    # qkv shape = [ S, H, D ]
    batch = cu_seqlens.numel() - 1
    softmax_scale = q.shape[-1] ** (-0.5)

    o = torch.zeros_like(q)
    for batch_idx in range(batch):
        batch_start = cu_seqlens[batch_idx].item()
        batch_end = cu_seqlens[batch_idx + 1].item()
        # get qkv of this batch
        q_ = q[batch_start:batch_end]
        k_ = k[batch_start:batch_end]
        v_ = v[batch_start:batch_end]
        o_ = o[batch_start:batch_end]
        # calc key gate weight
        key_gate_weight = []
        batch_size = batch_end - batch_start
        num_block = math.ceil(batch_size / moba_chunk_size)
        for block_idx in range(0, num_block):
            block_start = block_idx * moba_chunk_size
            block_end = min(batch_size, block_start + moba_chunk_size)
            key_gate_weight.append(k_[block_start:block_end].mean(dim=0, keepdim=True))
        key_gate_weight = torch.cat(key_gate_weight, dim=0)  # [ N, H, D ]
        # calc & mask gate
        # use fp32 to avoid precision issue in bf16
        q_ = q_.type(torch.float32)
        key_gate_weight = key_gate_weight.type(torch.float32)
        gate = torch.einsum("shd,nhd->hsn", q_, key_gate_weight)  # [ H, S, N ]
        key_gate_weight = key_gate_weight.type_as(k)
        q_ = q_.type_as(k)
        for i in range(num_block):
            # select the future Qs that can attend to KV chunk i
            gate[:, : (i + 1) * moba_chunk_size, i] = float("-inf")
            gate[:, i * moba_chunk_size : (i + 1) * moba_chunk_size, i] = float("inf")
        # gate_top_k_idx = gate_top_k_val = [ H S K ]
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=min(moba_topk, num_block), dim=-1, largest=True, sorted=False
        )
        gate_top_k_val, _ = gate_top_k_val.min(dim=-1)  # [ H, S ]
        need_attend = gate >= gate_top_k_val.unsqueeze(-1)
        # add gate_idx_mask in case of there is cornercases of same topk val been selected
        gate_idx_mask = torch.zeros(
            need_attend.shape, dtype=torch.bool, device=q.device
        )
        gate_idx_mask = gate_idx_mask.scatter_(dim=-1, index=gate_top_k_idx, value=True)
        need_attend = torch.logical_and(need_attend, gate_idx_mask)
        gate[need_attend] = 0
        gate[~need_attend] = -float("inf")
        gate = gate.repeat_interleave(moba_chunk_size, dim=-1)[
            :, :, :batch_size
        ]  # [ H, S, S ]
        gate.masked_fill_(
            torch.ones_like(gate, dtype=torch.bool).tril().logical_not(), -float("inf")
        )
        # print(f"moba_naive | gate ({gate.shape}): {gate}")

        # calc qk = qk^t
        q_ = q_.type(torch.float32)
        k_ = k_.type(torch.float32)
        v_ = v_.type(torch.float32)
        qk = torch.einsum("xhd,yhd->hxy", q_, k_)
        # mask
        qk += gate
        qk *= softmax_scale
        # calc o
        p = qk.softmax(dim=-1)
        o_ += torch.einsum("hxy,yhd->xhd", p, v_)
        o = o.type_as(q)

    return o




class MixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
        return_lse,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn
        self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )

        moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        # convert lse shape hs -> sh ( follow the legacy mix attn logic )
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        # output buffer [S, H, D], same shape as q
        output = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # flatten vS & H for index ops
        output_2d = output.view(-1, q.shape[2])

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # add attn output
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # add moba output
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)


        # add back max lse
        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)


        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )
        ctx.return_lse = return_lse

        if return_lse:
            return output, mixed_attn_lse_sh
        else:
            return output

    @staticmethod
    def backward(ctx, d_output, *args):

        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        (
            output,
            mixed_attn_vlse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()

        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        dmq = torch.zeros_like(moba_q, dtype=moba_q.dtype, device=moba_q.device)
        dmk = torch.zeros_like(moba_kv[:, 0], dtype=moba_kv.dtype, device=moba_kv.device)
        dmv = torch.zeros_like(moba_kv[:, 1], dtype=moba_kv.dtype, device=moba_kv.device)
        _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=dmq,
            dk=dmk,
            dv=dmv,
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None, None


def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
    return_lse: bool=False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """An efficient version of moba implementation with triton kernels and flash-attn, the core logic:
    1. Calculate the chunks and the number of chunks, n = floor(data_size / chunk_size)
       - tokens in the tail chunk are reserved for self attn
       - tokens in other chunks will be processed in later steps
    2. K in each chunk will calculate mean value as the representative k, and Q will attend to these representative
    k to get the gate logit, which will be used to select topk chunks
    3. Select the topk chunks and get the dense q for each kv chunk pair and do the varlen attention
    4. Combine the varlen attn and self attn results via online softmax to get the final result

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        max_seqlen (int): the max sequence length of the batch, same definition in flash attn

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """
    head_group_size = q.shape[1] // k.shape[1]
    if head_group_size > 1:
        k = torch.repeat_interleave(k, head_group_size, dim=1)
        v = torch.repeat_interleave(v, head_group_size, dim=1)

    # ---------------------------------------------------------------------------------------------
    kv = torch.stack((k, v), dim=1) # stack along a new dimension -> [S, 2, H, D]

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    self_attn_cu_seqlen = cu_chunk

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]

    # select the elements of KV corresponding to all chunks that are not filtered out
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1)) 

    """ calc key_gate_weight and gate """
    # key_gate_weight [ F_N_CHUNK, HEAD, HEAD_DIM ]
    key_gate_weight = (
        filtered_kv[:, 0] # K
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1) # mean pooling along chunk size
        .float()
    )
    q = q.type(torch.float32)  # float logit on the fly for better gate logit perception
    key_gate_weight = key_gate_weight.type(
        torch.float32
    )  # float logit for better gate logit perception
    gate = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    )  # gate [ F_N_CHUNK, HEAD, SEQ ]
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # pose process gate, masking unchosen batch and apply causal mask to current chunk
    gate_seq_idx = torch.arange(0, seqlen, device=q.device, dtype=torch.int32)[
        None, :
    ].repeat(num_filtered_chunk, 1)
    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    """ find moba q that needs moba attn """
    # find topk chunks
    # gate_mask [ N_CHUNK, HEAD, SEQ ], true indicates that needs attention
    _, gate_top_k_idx = torch.topk(gate, k=moba_topk, dim=0, largest=True, sorted=False)

    # apply causal mask
    gate_mask = torch.logical_not(gate.isinf())

    # select topk chunks
    gate_idx_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=0, index=gate_top_k_idx, value=True)

    # gate_mask has the shape [ N_CHUNK, HEAD, SEQ ]. 
    # For each chunk, the sequence-dimension indices will be True if it belongs to the top-K chunks
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask)
    # ---------------------------------------------------------------------------------------------


    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    # torch.nonzero (as_tuple=True): Returns a tuple of 1-D tensors, one for each dimension in input, each containing the indices (in that dimension) of all non-zero elements of input .
    # if input has n-dimension, the resulting tuple will have n tensors of size z, where z is the number of non-zero elements in input.
    # (i-th values of all n tuple elements represent the indices of the i-th non-zero element in each dimension)
    # using index [-1] => indices of HS (combined sequence) dimension that contains non-zero elements
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1) # [ N, HS ]

    moba_q_indices = moba_q_indices.nonzero(as_tuple=True)[-1]  # [HS indices] * N (total size: all non-zero elements in HS dimension)


    # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()

    # select all q that needs moba attn based on the moba_q_indices
    moba_q = rearrange(q, "s h d -> ( h s ) d").index_select(
        0, moba_q_indices
    )  # [ selected_S, D ]
    moba_q = moba_q.unsqueeze(1)

    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen

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

    # -----------------------------------------------
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d") # here `x` only stands for a dimension (stack dimension for KV)

    moba_kv = moba_kv.split(moba_chunk_size, dim=1) 
    moba_kv = torch.cat(moba_kv, dim=0) # [num_selected_chunks, H x S // moba_chunk_size, D]

    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan

    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)

    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

    # Wrapping up the flash attn call and online softmax dlse inside MixedAttention class
    return MixedAttention.apply(
        q, k, v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
        return_lse
    )


def moba_attn_func(
    q: torch.Tensor, # [batch, q_len, q_heads, head_dim]
    k: torch.Tensor,
    v: torch.Tensor,
    global_seq_len: int,
    moba_chunk_size: int,
    moba_topk: int,
    **kwargs,
):
    batch_size = q.shape[0]
    cu_seqlens = torch.cumsum(
        torch.tensor([0] + [global_seq_len] * batch_size, device=q.device),
        dim=0,
        dtype=torch.int32,
    )

    q_3d, k_3d, v_3d = \
        q.reshape(-1, q.shape[2], q.shape[3]), \
        k.reshape(-1, k.shape[2], k.shape[3]), \
        v.reshape(-1, v.shape[2], v.shape[3])

    # output: [batch_size, global_seq_len, q_heads, head_dim]
    return moba_attn_varlen(
        q_3d, k_3d, v_3d,
        cu_seqlens,
        global_seq_len,
        moba_chunk_size,
        moba_topk,
    ).view(q.shape)
    

def moba_layer(
    moba_impl: Callable,
    moba_chunk_size: int,
    moba_topk: int,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Args:
        query (torch.Tensor): [batch, q_heads, q_len, head_dim]
        key (torch.Tensor): [batch, kv_heads, kv_len, head_dim]
        value (torch.Tensor): [batch, kv_heads, kv_len, head_dim]

    Returns:
        attn_output (torch.Tensor): [batch, q_len, q_heads, head_dim]
        attn_weights (None): not needed
    """
    assert module.is_causal
    batch, q_heads, q_len, head_dim = query.shape
    _, kv_heads, kv_len, _ = key.shape
    if q_len == kv_len:
        # prefill phase
        query = hf_to_fa(query)
        key = hf_to_fa(key)
        value = hf_to_fa(value)
        kv_replicas = q_heads // kv_heads
        key = torch.repeat_interleave(key, kv_replicas, dim=1)
        value = torch.repeat_interleave(value, kv_replicas, dim=1)
        cu_seqlens_k = torch.cumsum(
            torch.tensor([0] + [kv_len] * batch, device=query.device),
            dim=0,
            dtype=torch.int32,
        )
        out = moba_impl(
            q=query,
            k=key,
            v=value,
            cu_seqlens=cu_seqlens_k,
            max_seqlen=kv_len,
            moba_chunk_size=moba_chunk_size,
            moba_topk=moba_topk,
        )
    else:
        # decode phase
        # TODO release paged attn implementation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = flash_attn_func(query, key, value, dropout, scaling, True)
    return out, None
