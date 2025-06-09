import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

from minference.dist_ops.utils import RingComm
from minference.ops.xattention_fa import (
    softmax_fuse_block_sum,
    flat_group_gemm_fuse_reshape,
)


LN2 = 1 / 1.4426950408889634
def create_causal_mask(batch_size, head_num, block_size, block_num, divide_block_num):
    """
        Creates a causal attention mask used in transformer-based models.

        Parameters:
        - batch_size (int): The number of sequences in the batch.
        - head_num (int): The number of attention heads.
        - block_size (int): The size of each block in the sequence.
        - block_num (int): The total number of blocks in the sequence.
        - divide_block_num (int): The block index at which causality is applied.

        Returns:
        - torch.Tensor: A mask tensor of shape (batch_size, head_num, block_size, total_size)
        where total_size = block_size * block_num. The mask enforces causal attention by 
        setting certain positions to `-inf` to prevent information leakage from future tokens.
    """
    divide_block_num += 1
    if divide_block_num < 1 or divide_block_num > block_num:
        raise ValueError(
            f"divide_block_num ({divide_block_num}) must be between 1 and block_num ({block_num})."
        )

    total_size = block_size * block_num
    device = "cuda"
    mask = torch.zeros(block_size, total_size, device=device)
    if divide_block_num < block_num:
        mask[:, divide_block_num * block_size :] = float("-inf")

    if divide_block_num - 1 < block_num:
        start_col = (divide_block_num - 1) * block_size
        end_col = start_col + block_size
        upper_tri_mask = torch.triu(
            torch.full((block_size, block_size), float("-inf"), device=device),
            diagonal=1,
        )
        mask[:, start_col:end_col] = upper_tri_mask

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, head_num, block_size, total_size)
    return mask

def find_blocks_chunked(
    input_tensor: torch.Tensor, # (batch_size, num_heads, num_block_q, num_block_k)
    current_index, # 
    threshold, num_to_choose, decoding: bool, mode: str = "both", causal=True
):
    """
        Finds and selects relevant blocks of attention for transformer-based models based on a 
        threshold or a predefined number of blocks.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, num_block_q, num_block_k).
        - current_index (int): The current index in the sequence processing.
        - threshold (float or None): A threshold value used to determine the minimum attention weight sum.
        - num_to_choose (int or None): The number of blocks to be selected, ensuring sufficient information retrieval.
        - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
        - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
        - causal (bool): If True, applies causal masking to prevent future information leakage.

        Returns:
        - torch.Tensor: A boolean mask of shape (batch_size, head_num, num_block_q, num_block_k),
        indicating which blocks should be attended to.
    """
    assert threshold is None or num_to_choose is None
    batch_size, head_num, num_block_q, num_block_k = input_tensor.shape
    input_tensor = input_tensor.to(float)
    
    total_sum = input_tensor.sum(dim=-1, keepdim=True)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(float)
        required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1
        ).expand((batch_size, head_num, num_block_q, 1)).to(input_tensor.device)
    else:
        required_sum = total_sum * threshold


    mask = torch.zeros_like(input_tensor, dtype=torch.bool)
    mask[:, :, :, 0] = 1
    mask[:, :, :, current_index : current_index + num_block_q] = (
        torch.eye(num_block_q, device=mask.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, head_num, num_block_q, num_block_q)
    )
    # Note that other_values only contains the values of the current block 
    # (the sink blocks and diagonal are filled with 0)
    other_values = input_tensor.masked_fill(mask, 0)


    # Get sorted values
    sorted_values, _ = torch.sort(other_values, dim=-1, descending=True)
    sorted_values = sorted_values.to(input_tensor.device)
    sorted_values = torch.cat(
        [
            torch.zeros(
                (batch_size, head_num, num_block_q, 1), device=input_tensor.device
            ),
            torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True), # shape: (batch_size, head_num, num_block_q, 1)
            sorted_values[:, :, :, :-2], # :-2 excludes the first and diagonal (which are marked 0 in other_values)
        ],
        dim=-1,
    )

    # Get sorted indices
    # index will select the already-masked (sink and diagonal) at the beginning
    _, index = torch.sort(
        torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
        dim=-1,
        descending=True,
    )

    # [batch_size, head_num, num_block_q, num_block_k]
    cumulative_sum_without_self = torch.cat(
        [
            torch.zeros(
                (batch_size, head_num, num_block_q, 1), device=input_tensor.device
            ),
            sorted_values[:, :, :, 0:-1],
        ],
        dim=-1,
    ).cumsum(dim=-1)

    # Mask for indices where cumulative sum is below the required threshold.
    index_mask = cumulative_sum_without_self < required_sum
    index = torch.where(index_mask, index, 0)

    mask = mask.view(batch_size, head_num * num_block_q, num_block_k)
    index = index.view(batch_size, head_num * num_block_q, num_block_k)
    mask[:, torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1), index] = True
    mask = mask.view(batch_size, head_num, num_block_q, num_block_k)


    assert bool((torch.where(mask,input_tensor,0).sum(dim=-1, keepdim=True) >= required_sum * 0.99).all()), \
        f"mask sum {torch.where(mask,input_tensor,0).sum(dim=-1, keepdim=True)} < required_sum {required_sum}"
    
    try:
        if causal:
            assert (~mask[:, :, :, current_index + num_block_q :]).all()
    except:
        mask[:, :, :, current_index + num_block_q :] = False

    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = torch.zeros_like(input_tensor,dtype=bool,device=input_tensor.device)
            lambda_mask[:,:,:,0] = 1
            lambda_mask[:,:,:,current_index:current_index+num_block_q] = torch.eye(num_block_q, device=lambda_mask.device).unsqueeze(0).unsqueeze(0).expand(1,head_num,num_block_q,num_block_q)
            assert(torch.where(lambda_mask,mask,True).all())

    return mask


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

def check_device(use_triton: bool):
    avail = use_triton and (
        "100" not in torch.cuda.get_device_properties(torch.cuda.current_device()).name
    )
    if not avail:
        print("Setting use triton to false. Triton kernel not surpported on this device")
    return avail



def xattn_zigzag_estimate(
    query_states: torch.Tensor, # (batch_size, num_q_head, q_len, head_dim)
    key_states: torch.Tensor, # (batch_size, num_kv_head, k_len, head_dim)
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    group: dist.group = None,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len_local, head_dim = key_states.shape
    batch_size, num_q_head, q_len_local, head_dim = query_states.shape

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    k_gather_list = [torch.empty_like(key_states) for _ in range(world_size)]   
    dist.all_gather(k_gather_list, key_states.contiguous(), group=group)
    k_gathered = torch.cat(k_gather_list, dim=2)
    k_len = k_gathered.shape[2]

    if num_q_head > num_kv_head:
        k_gathered = torch.repeat_interleave(k_gathered.contiguous(), num_q_head // num_kv_head, dim=1)

    chunk_size = q_len_local // 2
    q_chunk_num = 2
    q_block_num = q_len_local // block_size
    q_block_num_per_chunk = chunk_size // block_size

    # assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    num_strides_in_k = k_len // stride
    num_strides_per_chunk = chunk_size // stride
    num_strides_per_block = block_size // stride
    num_blocks_per_chunk = num_strides_per_chunk // num_strides_per_block

    attn_weight_slices = [None, None]
    for chunk_idx in range(q_chunk_num):
        global_chunk_idx = rank * 2 + chunk_idx

        # Local start index
        q_chunk_start = chunk_idx * chunk_size
        q_chunk_end =  (chunk_idx + 1) * chunk_size

        # Global start index (stride-level)
        q_chunk_start_stride_global = global_chunk_idx * num_strides_per_chunk
        q_chunk_end_stride_global = (global_chunk_idx + 1) * num_strides_per_chunk

        # attn_weights_slice: (batch_size, num_heads, chunk_size // stride, kv_len // stride)
        # (i.e. the attention sum of each SxS stride block)
        # This step is agnostic to block size and just computes the attention sum in each stride block
        attn_weight_slice = flat_group_gemm_fuse_reshape(
            # query_states, key_states, stride, chunk_start, chunk_end, is_causal=True
            query_states[:, :, q_chunk_start : q_chunk_end, :,],
            k_gathered,
            stride,
            q_chunk_start_stride_global, q_chunk_end_stride_global,
            is_causal=causal,
        )
        attn_weight_slices[chunk_idx] = attn_weight_slice
    del k_gathered, k_gather_list

    for chunk_idx in range(q_chunk_num):
        global_chunk_idx = rank * 2 + chunk_idx
        
        # Local start index
        q_chunk_start = chunk_idx * chunk_size
        q_chunk_end =  (chunk_idx + 1) * chunk_size

        # Global start index (block-level)
        q_block_start = global_chunk_idx * q_block_num_per_chunk
        q_block_end   = (global_chunk_idx + 1) * q_block_num_per_chunk

        # Global start index (stride-level)
        q_chunk_start_stride_global = global_chunk_idx * num_strides_per_chunk
        q_chunk_end_stride_global = (global_chunk_idx + 1) * num_strides_per_chunk

        attn_weight_slice = attn_weight_slices[chunk_idx]

        # (batch_size, num_heads, q_block_num, k_block_num),
        attn_sum = softmax_fuse_block_sum(
            attn_weight_slice, # (batch_size, num_heads, chunk_size // stride, kv_len // stride)
            num_strides_per_block,
            min(4096, num_strides_per_block),
            q_chunk_start_stride_global, q_chunk_end_stride_global,
            num_strides_in_k,
            1 / LN2 / math.sqrt(head_dim) / stride / norm,
            is_causal=causal,
        )
        
        # (batch_size, head_num, num_blocks_per_chunk, block_num)
        simple_mask = find_blocks_chunked(
            attn_sum,
            global_chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        del attn_weight_slice
        if causal:
            simple_mask[:, :, :, q_block_start:q_block_end] = torch.where(
                torch.tril(
                    torch.ones(
                        q_block_num_per_chunk, q_block_num_per_chunk, 
                        dtype=bool, device=key_states.device
                    ),
                    diagonal=0,
                ),
                simple_mask[:, :, :, q_block_start:q_block_end],
                False,
            )
            simple_mask[:, :, :, q_block_end:] = 0
        if keep_sink:
            simple_mask[:, :, 0, :] = True
        if keep_recent:
            eye_matrix = torch.eye(q_block_num_per_chunk, device=simple_mask.device, dtype=bool)
            eye_matrix_expanded = (
                eye_matrix.unsqueeze(0)
                .unsqueeze(0)
                .expand(1, num_kv_head, q_block_num_per_chunk, q_block_num_per_chunk)
            )
            simple_mask[:, :, :, q_block_start:q_block_end] = torch.where(
                eye_matrix_expanded, True, simple_mask[:, :, :, q_block_start:q_block_end]
            )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2) # (batch_size, head_num, q_local_block_num, k_global_block_num)
    return attn_sums, simple_masks


def shuffle_zigzag_masks(
        block_masks: torch.Tensor, # [batch_size, num_qo_heads, num_blocks_local, num_blocks]
        process_group: dist.ProcessGroup = None
    ):
    dim = len(block_masks.shape) - 1
    if not block_masks.is_contiguous():
        block_masks = block_masks.contiguous()

    # We must use outplace, otherwise it will raise error at backward due to inplace operations.
    # We can not change to_send directly and create a new tensor to store the result.
    to_send_f = torch.zeros_like(block_masks)

    # assume the input sequence length is 8, and computation runs on 4 GPUs
    # the seq is represented as [0 1 2 3 4 5 6 7], world size is 4
    # the input status before `shuffle_zigzag_input` is
    # - gpu A: [0 1]
    # - gpu B: [2 3]
    # - gpu C: [4 5]
    # - gpu D: [6 7]
    # the value of `to_send_slice` is
    # - gpu A: [1]
    # - gpu B: [3]
    # - gpu C: [5]
    # - gpu D: [7]
    block_seq_len = block_masks.shape[dim] // 2
    left_slicer = [slice(None)] * dim + [slice(None, block_seq_len)]
    right_slicer = [slice(None)] * dim + [slice(block_seq_len, None)]
    to_send_slice = block_masks[right_slicer].contiguous()

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    res = torch.zeros_like(to_send_slice)

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
    recv_op = dist.P2POp(
        dist.irecv, res, src_rank, group=process_group)

    _ops.append(send_op)
    _ops.append(recv_op)
    
    response = dist.batch_isend_irecv(_ops)
    for resp in response:
        resp.wait()

    if rank >= world_size // 2: # D: 6 7, -> 1 6
        to_send_f[right_slicer] = block_masks[left_slicer]
        to_send_f[left_slicer] = res
    else:                       # A: 0 1, -> 0 7
        to_send_f[left_slicer] = block_masks[left_slicer]
        to_send_f[right_slicer] = res
    # after shuffle, the status of `to_send_f`
    # GPU A: [0 7]
    # GPU B: [2 5]
    # GPU C: [3 4]
    # GPU D: [1 6]

    return to_send_f
