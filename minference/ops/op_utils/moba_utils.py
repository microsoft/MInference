#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import torch
import torch.distributed as dist

from functools import lru_cache
from dataclasses import dataclass

@dataclass
class MoBAConfig:
    moba_chunk_size: int
    moba_topk: int

def shuffle_input_all(
        to_send: torch.Tensor, # [S, H, D]
        seq_offset: torch.Tensor, # [2]
        gate_mask: torch.Tensor = None, # [num_chunks, H, S]
        process_group: dist.ProcessGroup = None
    ):
    orig_ndim = to_send.ndim
    if orig_ndim == 3: to_send = to_send.unsqueeze(0)

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    
    if not to_send.is_contiguous():
        to_send = to_send.contiguous()
    block_seq_len = to_send.shape[1] // 2

    seq_offset_val = seq_offset.detach().cpu().item()
    seq_offsets = torch.Tensor([seq_offset_val, seq_offset_val + block_seq_len]).to(to_send.device)

    # We must use outplace, otherwise it will raise error at backward due to inplace operations.
    # We can not change to_send directly and create a new tensor to store the result.
    to_send_f = torch.zeros_like(to_send)
    to_send_gate_mask = torch.zeros_like(gate_mask)
    to_send_offset = seq_offsets[1]

    # assume the input sequence length is 8, and computation runs on 4 GPUs
    # the seq is represented as [0 1 2 3 4 5 6 7], world size is 4
    # the input status before `shuffle_input` is
    # - gpu A: [0 1]
    # - gpu B: [2 3]
    # - gpu C: [4 5]
    # - gpu D: [6 7]
    # the value of `to_send_slice` is
    # - gpu A: [1]
    # - gpu B: [3]
    # - gpu C: [5]
    # - gpu D: [7]
    to_send_slice = to_send[:, block_seq_len:].contiguous()
    to_send_gate_mask_slice = gate_mask[..., block_seq_len:].contiguous()

    res = torch.zeros_like(to_send_slice)
    res_gate_mask = torch.zeros_like(to_send_gate_mask_slice)
    res_offset= torch.zeros_like(to_send_offset)

    _ops = []
    offset = ((dist.get_rank() // world_size) * world_size)
    # rank  src_rank
    # 0     3
    # 1     2
    # 2     1
    # 3     0
    src_rank = (world_size - rank - 1) % world_size + offset
    send_op = dist.P2POp(
        dist.isend, to_send_slice, src_rank, group=process_group
    )
    send_gate_mask_op = dist.P2POp(
        dist.isend, to_send_gate_mask_slice, src_rank, group=process_group
    )
    send_offset_op = dist.P2POp(
        dist.isend, to_send_offset, src_rank, group=process_group
    )
    _ops.append(send_op)
    _ops.append(send_gate_mask_op)
    _ops.append(send_offset_op)

    recv_op = dist.P2POp(
        dist.irecv, res, src_rank, group=process_group)
    recv_gate_mask_op = dist.P2POp(
        dist.irecv, res_gate_mask, src_rank, group=process_group
    )
    recv_offset_op = dist.P2POp(
        dist.irecv, res_offset, src_rank, group=process_group
    )
    _ops.append(recv_op)
    _ops.append(recv_gate_mask_op)
    _ops.append(recv_offset_op)
    
    # response = dist.dist.batch_isend_irecv(_ops)
    response = dist.batch_isend_irecv(_ops)
    for resp in response:
        resp.wait()

    if rank >= world_size // 2: # D: 6 7, -> 1 6
        to_send_f[:, block_seq_len:] = to_send[:, :block_seq_len]
        to_send_f[:, :block_seq_len, ...] = res
        
        to_send_gate_mask[..., block_seq_len:] = gate_mask[..., :block_seq_len]
        to_send_gate_mask[..., :block_seq_len] = res_gate_mask

        seq_offsets[1] = seq_offsets[0]
        seq_offsets[0] = res_offset
    else:                       # A: 0 1, -> 0 7
        to_send_f[:, :block_seq_len] = to_send[:, :block_seq_len]
        to_send_f[:, block_seq_len:, ...] = res

        to_send_gate_mask[..., :block_seq_len] = gate_mask[..., :block_seq_len]
        to_send_gate_mask[..., block_seq_len:] = res_gate_mask

        seq_offsets[1] = res_offset

    # after shuffle, the status of `to_send_f`
    # GPU A: [0 7]
    # GPU B: [2 5]
    # GPU C: [3 4]
    # GPU D: [1 6]
    return (
        to_send_f if orig_ndim != 3 else to_send_f.squeeze(0), 
        seq_offsets,
        to_send_gate_mask,
    )


def shuffle_input_only(
        to_send: torch.Tensor, # [S, H, D]
        process_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
    orig_ndim = to_send.ndim
    if orig_ndim == 3: to_send = to_send.unsqueeze(0)

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    
    if not to_send.is_contiguous():
        to_send = to_send.contiguous()
    block_seq_len = to_send.shape[1] // 2

    # We must use outplace, otherwise it will raise error at backward due to inplace operations.
    # We can not change to_send directly and create a new tensor to store the result.
    to_send_f = torch.zeros_like(to_send)

    to_send_slice = to_send[:, block_seq_len:].contiguous()
    res = torch.zeros_like(to_send_slice)
    
    _ops = []
    offset = ((dist.get_rank() // world_size) * world_size)
    
    src_rank = (world_size - rank - 1) % world_size + offset
    send_op = dist.P2POp(
        dist.isend, to_send_slice, src_rank, group=process_group
    )
    _ops.append(send_op)

    recv_op = dist.P2POp(
        dist.irecv, res, src_rank, group=process_group)
    _ops.append(recv_op)
    
    # response = dist.dist.batch_isend_irecv(_ops)
    response = dist.batch_isend_irecv(_ops)
    for resp in response:
        resp.wait()

    if rank >= world_size // 2: # D: 6 7, -> 1 6
        to_send_f[:, block_seq_len:] = to_send[:, :block_seq_len]
        to_send_f[:, :block_seq_len, ...] = res
    else:                       # A: 0 1, -> 0 7
        to_send_f[:, :block_seq_len] = to_send[:, :block_seq_len]
        to_send_f[:, block_seq_len:, ...] = res
    return to_send_f if orig_ndim != 3 else to_send_f.squeeze(0)

@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """calc chunks that needs moba attention"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    # example: [seq_len]
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    # example: [number of all chunks with chunk size equal to moba_chunk_size + 1 (the one with smaller size)]
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size

    # cu_num_chunk[batch_idx] = first chunk id of this batch
    # example: [1, 1]
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    # example: [1, 1 + num of chunks]
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    
    # total chunk ( for all batch )
    # example: 1 + num of chunks
    num_chunk = cu_num_chunk[-1]

    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # example chunk_sizes: [0, moba_chunk_size, ..., moba_chunk_size, batch_last_chunk_size]


    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    # example: [0, moba_chunk_size, ..., seq_len]
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)


    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    # example: [0, 0, 0, ...., 0]
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )

    # example: [0, 0, 0, ... , 0] (if there are multiple samples in the batch, the index of the starting chunk of each batch from the 1st sample will be 1)
    # but if there is only one batch, cu_num_chunk[1:-1] will be empty and no element will be assigned with 1 (all correspond to 0-th sample)
    chunk_to_batch[cu_num_chunk[1:-1]] = 1

    # example: [0, 0, 0, ..., 0]
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """
    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    chunk_to_remove = cu_num_chunk[1:] - 1 # example: number of chunks (num_chunk - 1)
    # print(f"calc_chunks | chunk_to_remove: {chunk_to_remove}")

    chunk_to_remain = torch.ones(
        (num_chunk, ), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False # example: 
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )


def compute_moba_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offset: torch.Tensor,
    cu_seqlens: torch.Tensor,
    moba_chunk_size: int,
    moba_topk: int,
):
    seq_offset: int = seq_offset.detach().cpu().item()
    seqlen_block, num_head, head_dim = q.shape
    _, k_num_head, _ = k.shape
    if num_head > k_num_head:
        k = torch.repeat_interleave(k, num_head // k_num_head, dim=1)
        v = torch.repeat_interleave(v, num_head // k_num_head, dim=1)

    # ---------------------------------------------------------------------------------------------
    kv = torch.stack((k, v), dim=1) # [ blk_S, 2, H, D ]
    
    world_size = dist.get_world_size()
    kv_list = [torch.zeros_like(kv, dtype=q.dtype, device=q.device) for _ in range(world_size)]
    dist.all_gather(kv_list, kv)
    kv_gathered = torch.cat(kv_list, dim=0) # [ S, 2, H, D ]


    """ some basic variables """
    # qkv shape = [ S, H, D ]
    block_size = q.shape[0]
    seqlen, _, num_head, head_dim = kv_gathered.shape

    """ prepare chunk meta """
    (
        cu_chunk,                          # example: [0, moba_chunk_size, ..., seq_len]
        filtered_chunk_indices,            # example: [0, 1, 2, ..., num_filtered_chunk-1] (i.e. except the last chunk)
        num_filtered_chunk,                # example: num_filtered_chunk
        chunk_to_batch,                    # example: [0, 0, ... ,0] (for batch_size=1) with size  1 + real num of chunks
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    assert moba_topk > 0, "moba_topk should be greater than 0"

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]

    # select the elements of KV corresponding to all chunks that are not filtered out
    filtered_kv = kv_gathered.index_select(0, filtered_kv_indices.view(-1)) 

    """ calc key_gate_weight and gate """
    # key_gate_weight [ F_N_CHUNK, HEAD, HEAD_DIM ]
    key_gate_weight = (
        filtered_kv[:, 0] # K
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1) # mean pooling along chunk size
        .float()
    )
    # print(f"Rank {dist.get_rank()} | compute_moba_gate | key_gate_weight shape: {key_gate_weight.shape}")

    q = q.type(torch.float32)  # float logit on the fly for better gate logit perception
    key_gate_weight = key_gate_weight.type(
        torch.float32
    )  # float logit for better gate logit perception
    gate = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    )  # gate [ F_N_CHUNK, HEAD, SEQ_BLOCK]
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # pose process gate, masking unchosen batch and apply causal mask to current chunk
    gate_seq_idx = torch.arange(
        seq_offset, min(seq_offset + block_size, seqlen), device=q.device, dtype=torch.int32
    )[None, :].repeat(num_filtered_chunk, 1)
    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))
    # print(f"Rank {dist.get_rank()} | compute_moba_gate | gate shape before topK: {gate.shape}")

    """ find moba q that needs moba attn """
    # find topk chunks
    # gate_top_k_idx with shape [TOP_K, HEAD, SEQ_BLOCK]
    _, gate_top_k_idx = torch.topk(gate, k=moba_topk, dim=0, largest=True, sorted=False)
    # apply causal mask
    gate_mask = torch.logical_not(gate.isinf())
    
    # select topk chunks
    gate_idx_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=0, index=gate_top_k_idx, value=True)

    # [ F_N_CHUNK, HEAD, SEQ_BLOCK]
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask).contiguous()

    return (
        # gate_mask does not need to be gathered because 
        # each device only needs the gate_mask corresponding to the current query block
        gate_mask, 
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch
    )
