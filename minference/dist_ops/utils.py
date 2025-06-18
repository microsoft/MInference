#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# Credits: This logger implementation is inspired by project https://github.com/zhuzilin/ring-flash-attention
import os
import math
import torch
import inspect
import operator
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl

from functools import reduce, cache
from typing import Optional, Tuple, List, Dict
from torch.distributed.distributed_c10d import P2POp

PROCESS_GROUPS: Dict[str, dist.ProcessGroup] = {}


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


# copy from megatron/core/utils.py
class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)

@triton.jit
def _update_out_and_lse_kernel(
    Out0, Lse0, Out1, Lse1,
    stride_oz0, stride_om0, stride_oh0, stride_od0,
    stride_lz0, stride_lm0, stride_lh0,
    stride_oz1, stride_om1, stride_oh1, stride_od1,
    stride_lz1, stride_lm1, stride_lh1,
    num_tokens,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    if start_m * BLOCK_M >= num_tokens:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < num_tokens

    o0_ptrs = Out0 + batch_idx * stride_oz0 + head_idx * stride_oh0 + offs_m[:, None] * stride_om0 + offs_d[None, :] * stride_od0
    o1_ptrs = Out1 + batch_idx * stride_oz1 + head_idx * stride_oh1 + offs_m[:, None] * stride_om1 + offs_d[None, :] * stride_od1
    lse0_ptrs = Lse0 + batch_idx * stride_lz0 + head_idx * stride_lh0 + offs_m * stride_lm0
    lse1_ptrs = Lse1 + batch_idx * stride_lz1 + head_idx * stride_lh1 + offs_m * stride_lm1

    lse0 = tl.load(lse0_ptrs, mask=m_mask, other=float("-inf"))
    lse1 = tl.load(lse1_ptrs, mask=m_mask, other=float("-inf"))
    o0 = tl.load(o0_ptrs, mask=m_mask[:, None], other=0.).to(tl.float32)
    o1 = tl.load(o1_ptrs, mask=m_mask[:, None], other=0.).to(tl.float32)

    m_mask &= (lse0 - lse1) < 88.0

    theta = tl.math.exp(lse0 - lse1)
    alpha0 = 1 / (1 + 1 / theta)
    alpha1 = 1 / (1 + theta)
    o = alpha0[:, None] * o0 + alpha1[:, None] * o1
    lse = lse1 - tl.math.log(alpha1)

    tl.store(o0_ptrs, o.to(Out0.type.element_ty), mask=m_mask[:, None])
    tl.store(lse0_ptrs, lse, mask=m_mask)


def _update_out_and_lse_triton(
    out: torch.Tensor,        # [batch_size, num_tokens, num_heads, head_dim]
    lse: torch.Tensor,        # [batch_size, num_tokens, num_heads, 1]
    block_out: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    block_lse: torch.Tensor,  # [batch_size, num_heads, num_tokens] => [batch_size, num_tokens, num_heads, 1]
    step_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    batch_size, num_tokens, num_heads, head_dim = out.shape
    block_M = 128
    block_D = head_dim
    _update_out_and_lse_kernel[(triton.cdiv(num_tokens, block_M), num_heads, batch_size)](
        out, lse, block_out, block_lse,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        block_out.stride(0), block_out.stride(1), block_out.stride(2), block_out.stride(3),
        block_lse.stride(0), block_lse.stride(1), block_lse.stride(2),
        num_tokens, BLOCK_M=block_M, BLOCK_D=block_D,
        num_warps=4, num_stages=1,
    )
    return out, lse


@torch.jit.script
def _update_out_and_lse_torch(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    step_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
    step_idx: Optional[int] = None,
    use_triton_kernel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_triton_kernel:
        _update_out_and_lse = _update_out_and_lse_triton
    else:
        _update_out_and_lse = _update_out_and_lse_torch
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse,
            step_idx=step_idx
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(
            out, lse, block_out, block_lse, 
            step_idx=step_idx
        )

    return out, lse


class RingComm:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        zigzag: bool = False,
        ring_list: Optional[list] = None,
    ):
        self._process_group = process_group
        self._ops: List[P2POp] = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        if ring_list is not None:
            curr_idx = ring_list.index(self.rank)
            self.send_rank = ring_list[(curr_idx + 1) % len(ring_list)]
            self.recv_rank = ring_list[(curr_idx - 1 + len(ring_list)) % len(ring_list)]
        elif zigzag:
            parts = self.world_size // 2
            self.ring_list = []
            for i in range(parts):
                self.ring_list.extend([i, self.world_size - i - 1])
            self.revert_rank = self.ring_list.index(self.rank)
            offset = ((dist.get_rank() // self.world_size) * self.world_size)
            self.send_rank = self.ring_list[(self.revert_rank + 1) % self.world_size] + offset
            self.recv_rank = self.ring_list[(self.revert_rank - 1) % self.world_size] + offset
        else:
            self.send_rank = (self.rank + 1) % self.world_size
            self.recv_rank = (self.rank - 1) % self.world_size

    def send_recv(
        self, 
        to_send: torch.Tensor, 
        recv_tensor: Optional[torch.Tensor] = None, 
        step_idx: int = 0,
        fwd: int = 1,
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group,
            tag=2 * (step_idx * (self.rank + 1)) + fwd,
        )
        recv_op = dist.P2POp(
            dist.irecv, res, self.recv_rank, group=self._process_group,
            tag=2 * (step_idx * (self.rank + 1)) + fwd,
        )

        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")        
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

            
    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v

    def send_recv_kv_offsets(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_seq_offsets: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None, 
        v_buffer: Optional[torch.Tensor] = None,
        kv_seq_offsets_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        next_kv_seq_offsets = self.send_recv(kv_seq_offsets, kv_seq_offsets_buffer)
        
        self.commit()
        return next_k, next_v, next_kv_seq_offsets

def shuffle_zigzag_input(to_send: torch.Tensor,
                  dim: int = 1,
                  process_group: dist.ProcessGroup = None):
    dim %= len(to_send.shape)

    if not to_send.is_contiguous():
        to_send = to_send.contiguous()

    # We must use outplace, otherwise it will raise error at backward due to inplace operations.
    # We can not change to_send directly and create a new tensor to store the result.
    to_send_f = torch.zeros_like(to_send)

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
    block_seq_len = to_send.shape[dim] // 2
    left_slicer = [slice(None)] * dim + [slice(None, block_seq_len)]
    right_slicer = [slice(None)] * dim + [slice(block_seq_len, None)]
    to_send_slice = to_send[right_slicer].contiguous()

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
        to_send_f[right_slicer] = to_send[left_slicer]
        to_send_f[left_slicer] = res
    else:                       # A: 0 1, -> 0 7
        to_send_f[left_slicer] = to_send[left_slicer]
        to_send_f[right_slicer] = res
    # after shuffle, the status of `to_send_f`
    # GPU A: [0 7]
    # GPU B: [2 5]
    # GPU C: [3 4]
    # GPU D: [1 6]

    return to_send_f


def recover_zigzag_output(to_send: torch.Tensor,
                   dim: int = 1,
                   process_group: dist.ProcessGroup = None):
    dim %= len(to_send.shape)

    if not to_send.is_contiguous():
        to_send = to_send.contiguous()

    to_send_f = torch.zeros_like(to_send)

    block_seq_len = to_send.shape[dim] // 2
    left_slicer = [slice(None)] * dim + [slice(None, block_seq_len)]
    right_slicer = [slice(None)] * dim + [slice(block_seq_len, None)]

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    if rank >= world_size // 2:
        to_send_slice = to_send[left_slicer].contiguous()
    else:
        to_send_slice = to_send[right_slicer].contiguous()
    res = torch.zeros_like(to_send_slice)
    
    assert to_send_slice.is_contiguous()
    assert res.is_contiguous()

    _ops = []
    offset = ((dist.get_rank() // world_size) * world_size)
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

    if rank >= world_size // 2:
        to_send_f[left_slicer] = to_send[right_slicer]
        to_send_f[right_slicer] = res
    else:
        to_send_f[left_slicer] = to_send[left_slicer]
        to_send_f[right_slicer] = res

    return to_send_f.contiguous()


def shuffle_block_mask_zigzag(
    block_mask: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    num_blocks_per_chunk: int,
    group: dist.ProcessGroup,
):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    
    # ---------------------------------------
    # Shuffle Query chunks
    block_mask = shuffle_zigzag_input(to_send=block_mask, dim=-2, process_group=group) # [batch_size, num_qo_heads, num_blocks_local, num_blocks]

    # ---------------------------------------
    # Shuffle Key chunks
    ring_list = RingComm(group, zigzag=True).ring_list
    ring_index = ring_list.index(rank)

    shuffled_block_mask_list = []
    for i in range(world_size): 
        rank_src = ring_list[(ring_index - i) % world_size]

        curr_chunk_index = 2 * rank_src
        rev_chunk_index = (2 * world_size - 1 - curr_chunk_index)
        if curr_chunk_index > rev_chunk_index:
            curr_chunk_index, rev_chunk_index = rev_chunk_index, curr_chunk_index

        shuffled_block_mask_list.append(
            torch.cat(
                [
                    block_mask[..., curr_chunk_index * num_blocks_per_chunk : (curr_chunk_index + 1) * num_blocks_per_chunk],
                    block_mask[..., rev_chunk_index * num_blocks_per_chunk  : (rev_chunk_index + 1) * num_blocks_per_chunk]
                ], dim=-1
            )
        )
    block_mask = torch.stack(shuffled_block_mask_list, dim=0).contiguous() # [world_size, batch_size, num_qo_heads, num_blocks_local, num_blocks_local]
    return block_mask


def shuffle_striped_input(to_send: torch.Tensor,  # [B, N / W, H, D]
                          granularity: int = 1,
                          dim: int = 1,
                          process_group: dist.ProcessGroup = None):
    # 00, 01, 02, 03, 04, 05, 06, 07  =>  00, 04, 08, 12, 16, 20, 24, 28
    # 08, 09, 10, 11, 12, 13, 14, 15  =>  01, 05, 09, 13, 17, 21, 25, 29
    # 16, 17, 18, 19, 20, 21, 22, 23  =>  02, 06, 10, 14, 18, 22, 26, 30
    # 24, 25, 26, 27, 28, 39, 30, 31  =>  03, 07, 11, 15, 19, 23, 27, 31
    shape = to_send.shape
    dim %= len(shape)
    world_size = dist.get_world_size(process_group)
    input_reshape = to_send.reshape((*shape[:dim], -1, world_size * granularity, *shape[dim+1:]))
    input_list = [x.contiguous() for x in input_reshape.split(granularity, dim=dim+1)]  # [N / W / (W * G), W*, G]
    output_list = [torch.empty_like(x) for x in input_list]                             # [W*, N / W / (W * G), G]


    dist.all_to_all(output_list, input_list, group=process_group)
    return torch.stack(output_list, dim=dim).reshape(shape).contiguous()


def recover_striped_output(to_send: torch.Tensor,  # [B, N / W, H, D]
                           granularity: int = 1,
                           dim: int = 1,
                           process_group: dist.ProcessGroup = None):
    # 00, 04, 08, 12, 16, 20, 24, 28  =>  00, 01, 02, 03, 04, 05, 06, 07
    # 01, 05, 09, 13, 17, 21, 25, 29  =>  08, 09, 10, 11, 12, 13, 14, 15
    # 02, 06, 10, 14, 18, 22, 26, 30  =>  16, 17, 18, 19, 20, 21, 22, 23
    # 03, 07, 11, 15, 19, 23, 27, 31  =>  24, 25, 26, 27, 28, 39, 30, 31
    shape = to_send.shape
    dim %= len(shape)
    world_size = dist.get_world_size(process_group)
    
    input_reshape = to_send.reshape((*shape[:dim], world_size, -1, granularity, *shape[dim+1:]))
    input_list = [x.squeeze(dim).contiguous() for x in input_reshape.split(1, dim=dim)]  # [W*, N / W / (W * G), G]
    output_list = [torch.empty_like(x) for x in input_list]                              # [N / W / (W * G), W*, G]

    dist.all_to_all(output_list, input_list, group=process_group)
    return torch.stack(output_list, dim=dim+1).reshape(shape).contiguous()

# --------------------------------------------------------------------
# Double-Ring Related
def get_inner_ring(group: dist.ProcessGroup):
    rank = dist.get_rank(group)
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE"))
    assert rank % local_world_size == local_rank
    return [i + (rank - local_rank) for i in range(local_world_size)]


def get_outer_ring(group: dist.ProcessGroup): 
    rank = dist.get_rank(group)
    world_size  = dist.get_world_size(group)
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE"))
    assert rank % local_world_size == local_rank
    return [i * local_world_size + local_rank for i in range(world_size // local_world_size)]


