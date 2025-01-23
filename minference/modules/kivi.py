# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Refer to the code in https://github.com/jy-yuan/KIVI/tree/main/quant

import math

import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import Cache, repeat_kv
from transformers.utils.import_utils import _is_package_available

if _is_package_available("kivi_gemv"):
    import kivi_gemv


def quant_and_pack_kcache(k: torch.FloatTensor, group_size: int, bits: int):
    assert len(k.shape) == 4
    shape = k.shape
    B, nh, T, D = shape
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B, nh, num_groups, group_size, D)
    # Quantize
    max_int = 2**bits - 1
    data = k.view(new_shape)
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(shape)
    code = pack_tensor(data, bits, pack_dim=2)
    return code, scale, mn


def quant_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
    shape = v.shape
    assert len(shape) == 4
    assert v.shape[-1] % group_size == 0
    num_groups = shape[-1] // group_size
    new_shape = shape[:-1] + (num_groups, group_size)
    # Quantize
    max_int = 2**bits - 1
    data = v.view(new_shape)
    mn = torch.min(data, dim=-1, keepdim=True)[0]
    mx = torch.max(data, dim=-1, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(shape)
    # Pack
    code = pack_tensor(data, bits, pack_dim=3)
    return code, scale, mn


def unpack_and_dequant_kcache(
    k_code: torch.FloatTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
):
    pack_dim = 2
    assert bits in [2, 4, 8]
    assert len(k_code.shape) == 4
    data = unpack_tensor(k_code, bits, pack_dim=pack_dim)
    shape = data.shape
    num_groups = shape[pack_dim] // group_size
    data = data.view(
        shape[:pack_dim]
        + (
            num_groups,
            group_size,
        )
        + shape[pack_dim + 1 :]
    )
    data = data.to(torch.float16)
    data = data * scale + mn
    return data.view(shape)


def unpack_and_dequant_vcache(
    v_code: torch.FloatTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
):
    assert bits in [2, 4, 8]
    assert len(v_code.shape) == 4
    data = unpack_tensor(v_code, bits, pack_dim=3)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(
        shape[:-1]
        + (
            num_groups,
            group_size,
        )
    )
    data = data.to(torch.float16)
    data = data * scale + mn
    return data.view(shape)


def pack_tensor(data, bits, pack_dim):
    # Pack
    shape = data.shape
    feat_per_int = 32 // bits
    assert bits in [2, 4, 8], "Only 2, 4, 8 bits are supported"
    assert (
        shape[pack_dim] % feat_per_int == 0
    ), "Dimension length must be divisible by number of features per int"
    # BS, nh, T, nd // 16 # 16 is for 2bit
    code = torch.zeros(
        shape[:pack_dim] + (shape[pack_dim] // feat_per_int,) + shape[pack_dim + 1 :],
        dtype=torch.int32,
        device=data.device,
    )
    i = 0
    row = 0
    unpacked_indices = [slice(None)] * len(data.shape)
    packed_indices = [slice(None)] * len(data.shape)
    while row < code.shape[pack_dim]:
        packed_indices[pack_dim] = row
        for j in range(i, i + (32 // bits)):
            unpacked_indices[pack_dim] = j
            code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
        i += 32 // bits
        row += 1
    return code


def unpack_tensor(v_code: torch.FloatTensor, bits: int, pack_dim: int):
    assert bits in [2, 4, 8]
    shape = v_code.shape
    feat_per_int = 32 // bits
    new_shape = (
        shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim + 1 :]
    )
    unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
    i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
    j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
    num = 0xFF >> (8 - bits)
    packed_indices = [slice(None)] * len(new_shape)
    packed_indices[pack_dim] = i
    if pack_dim == 2:
        unpacked_v_code = (
            (v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)
        ) & num
    elif pack_dim == 3:
        unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
    else:
        raise NotImplementedError
    return unpacked_v_code


@triton.jit
def _pack_along_last_dim(
    bits: tl.constexpr,
    intensor_ptr,
    code_ptr,
    N,
    num_feats: tl.constexpr,
    feat_per_int: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    num_int_per_y_dim = num_feats // feat_per_int
    bid = tl.program_id(axis=0)
    yid = tl.program_id(axis=1)
    offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_start = (
        intensor_ptr + offs_N * num_feats + yid * feat_per_int
    )  # offset of the first element at current tile
    packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    for i in range(feat_per_int):
        ptr = block_start + i
        element = tl.load(ptr, mask=offs_N < N, other=0.0)
        element = element << (i * bits)
        # Combine the value using bitwise OR
        packed = packed | element
    tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)


@triton.jit
def _minmax_along_last_dim(
    x_ptr,
    mn_ptr,
    mx_ptr,
    total_elements: tl.constexpr,
    N: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
    mask = offsets < total_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mx_val = tl.max(x, axis=1)
    mn_val = tl.min(x, axis=1)
    # tl.device_print('shape', mn_val[:, None].shape)
    tl.store(mn_ptr + offsets_b, mn_val, mask=offsets_b < N * num_groups)
    tl.store(mx_ptr + offsets_b, mx_val, mask=offsets_b < N * num_groups)


def triton_quantize_and_pack_along_last_dim(
    data: torch.Tensor, group_size: int, bit: int
):
    assert len(data.shape) == 4
    shape = data.shape
    B, nh, D, T = shape
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B * nh * D, num_groups, group_size)
    scale_mn_shape = B, nh, D, num_groups
    # Quantize
    data = data.reshape(new_shape)
    mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(data.shape[0] * data.shape[1], BLOCK_SIZE_N),)
    with torch.cuda.device(data.device):
        _minmax_along_last_dim[grid](
            data,
            mn,
            mx,
            data.numel(),
            data.shape[0],
            num_groups,
            group_size,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=8,
        )
    # mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
    # mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
    scale = (mx - mn) / (2**bit - 1)
    data = data - mn.unsqueeze(-1)
    data.div_(scale.unsqueeze(-1))
    data = data.clamp_(0, 2**bit - 1).round_().to(torch.int32)
    data = data.view(-1, T)
    feat_per_int = 32 // bit
    packshape = (
        np.prod(shape[:-1]),
        shape[-1] // feat_per_int,
    )
    code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
    grid = lambda meta: (
        triton.cdiv(data.shape[0], BLOCK_SIZE_N),
        data.shape[1] // feat_per_int,
    )
    with torch.cuda.device(data.device):
        _pack_along_last_dim[grid](
            bit,
            data,
            code,
            data.shape[0],
            data.shape[1],
            feat_per_int,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=8,
        )
    return (
        code.view(B, nh, D, -1),
        scale.reshape(scale_mn_shape),
        mn.reshape(scale_mn_shape),
    )


@triton.jit
def qbvm_kernel(
    bits,
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    M,
    N,
    K,
    stride_abatch,
    stride_am,
    stride_ak,
    stride_bbatch,
    stride_bk,
    stride_bn,
    stride_cbatch,
    stride_cm,
    stride_cn,
    stride_scales_b,
    stride_scales_k,
    stride_scales_g,
    stride_zeros_b,
    stride_zeros_k,
    stride_zeros_g,
    groupsize,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute the batch matrix multiplication C = A x B.
    A is of shape (B, 1, K) float16
    B is of shape (B, K, N//feat_per_int) int32
    C is of shape (B, 1, N) float16
    scales is of shape (B, K, G) float16
    zeros is of shape (B, K, G) float16
    groupsize is an int specifying the size of groups for scales and zeros.
    G is N // groupsize.
    Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.

    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    feat_per_int = 32 // bits
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_n = pid % num_pid_n
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_batch_offset = pid_batch * stride_abatch
    b_batch_offset = pid_batch * stride_bbatch
    c_batch_offset = pid_batch * stride_cbatch
    a_ptr = a_ptr + a_batch_offset
    b_ptr = b_ptr + b_batch_offset
    c_ptr = c_ptr + c_batch_offset
    a_ptrs = a_ptr + (offs_k[:, None] * stride_ak)  # (BLOCK_SIZE_K, 1)
    # a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the N axis feat_per_int times
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + (offs_bn[None, :] // feat_per_int) * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # shifter is used to extract the # bits bits of each element in the 32-bit word from B
    shifter = (offs_bn % feat_per_int) * bits
    scales_ptr = (
        scales_ptr
        + pid_batch * stride_scales_b
        + ((offs_bn[None, :] // groupsize)) * stride_scales_g
    )  # (BLOCK_SIZE_N,)
    zeros_ptr = (
        zeros_ptr
        + pid_batch * stride_zeros_b
        + ((offs_bn[None, :] // groupsize)) * stride_zeros_g
    )  # (BLOCK_SIZE_N,)

    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    num = 0xFF >> (8 - bits)
    for pid_k in range(0, num_pid_k):
        offs_bk = offs_k[:, None] + pid_k * BLOCK_SIZE_K
        # offs_k[None, :] < K - pid_k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_bk < K, other=0.0)  # (1, BLOCK_SIZE_K)
        b = tl.load(b_ptrs, mask=offs_bk < K, other=0.0)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        ptr = scales_ptr + offs_bk * stride_scales_k
        scales = tl.load(
            ptr, mask=offs_bk < K, other=0.0
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        ptr = zeros_ptr + offs_bk * stride_zeros_k
        zeros = tl.load(
            ptr, mask=offs_bk < K, other=0.0
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # Now we need to unpack b into 32-bit values
        # tl.device_print("scale ",scales.dtype)
        # tl.device_print("zeros ",zeros.dtype)
        b = (b >> shifter[None, :]) & num  # For 4-bit values, bit_op_num is 0xF
        b = b * scales + zeros  # Scale and shift
        accumulator += tl.sum(a * b, 0)  # tl.dot(a, b)
        # if pid_m == 0 and pid_n == 0:
        #     tl.device_print("hello ", tl.dot(a, b).shape)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator  # .to(tl.float16)
    # c = accumulator
    # Store the result
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cn * offs_cn
    c_mask = offs_cn < N
    tl.store(c_ptrs, c, mask=c_mask)


def understand_code():
    M, N, K = 512, 256, 256
    BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M = 64, 64, 4
    total_program_id = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    for pid in range(0, total_program_id):
        num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        print(f"pid={pid}, pid_m={pid_m}, pid_n={pid_n}")


def triton_bmm_fA_qB_outer(
    group_size: int,
    fA: torch.FloatTensor,
    qB: torch.IntTensor,
    scales: torch.FloatTensor,
    zeros: torch.FloatTensor,
    bits: int,
) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = query x key.
    Where key is quantized into 2-bit values.

    fA is of shape (B, nh, M, K) float16
    qB is of shape (B, nh, K, N // feat_per_int) int32
    scales is of shape (B, nh, K, G) float16
    zeros is of shape (B, nh, K, G) float16

    groupsize is the number of outer dimensions in each group.
    G = N // groupsize

    Returns C of shape (B, nh, M, N) float16
    """
    assert len(fA.shape) == 4 and len(qB.shape) == 4
    B, nh, M, K = fA.shape
    feat_per_int = 32 // bits
    # flatten to a 3D tensor
    fA = fA.view(-1, M, K)
    N = qB.shape[-1] * feat_per_int
    qB = qB.reshape(-1, K, qB.shape[-1])
    # This is based on the possible BLOCK_SIZE_Ks
    # assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
    # This is based on the possible BLOCK_SIZE_Ns
    assert (
        N % 16 == 0 and N % 32 == 0 and N % 64 == 0
    ), "N must be a multiple of 16, 32, 64, 128, and 256"
    # This is based on the possible BLOCK_SIZE_Ks
    assert group_size % 64 == 0, "groupsize must be a multiple of 64, and 128"
    flatten_B = B * nh
    c = torch.empty((flatten_B, M, N), device="cuda", dtype=torch.float16)
    # print(f'M {M} N {N} K {K}')
    grid = lambda META: (
        flatten_B,
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1])
    zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1])
    if N > K:
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        num_warps = 4  #
    else:
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 128
        num_warps = 2
    num_stages = 7 if K > 64 else 3  #
    qbvm_kernel[grid](
        bits,
        fA,
        qB,
        c,
        scales,
        zeros,
        M,
        N,
        K,
        fA.stride(0),
        fA.stride(1),
        fA.stride(2),
        qB.stride(0),
        qB.stride(1),
        qB.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        zeros.stride(0),
        zeros.stride(1),
        scales.stride(2),
        group_size,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c.view(B, nh, c.shape[-2], c.shape[-1])


def cuda_bmm_fA_qB_outer(
    group_size: int,
    fA: torch.FloatTensor,
    qB: torch.IntTensor,
    scales: torch.FloatTensor,
    zeros: torch.FloatTensor,
    bits: int,
    mqa: bool = False,
) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = query x key.
    Where key is quantized into 2-bit values.

    fA is of shape (B, nh, M, K) float16
    qB is of shape (B, nh, K, N // feat_per_int) int32
    scales is of shape (B, nh, K, G) float16
    zeros is of shape (B, nh, K, G) float16

    groupsize is the number of outer dimensions in each group.
    G = N // groupsize

    Returns C of shape (B, nh, M, N) float16
    """
    assert len(fA.shape) == 4 and len(qB.shape) == 4
    B, nh, M, K = fA.shape
    feat_per_int = 32 // bits
    # flatten to a 3D tensor
    fA = fA.view(-1, M, K).contiguous()
    N = qB.shape[-1] * feat_per_int
    qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
    # This is based on the possible BLOCK_SIZE_Ks
    # assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
    # This is based on the possible BLOCK_SIZE_Ns
    # assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
    # This is based on the possible BLOCK_SIZE_Ks
    # assert group_size % 64 == 0, "groupsize must be a multiple of 64, and 128"
    flatten_B = B * nh
    if mqa:
        flatten_B = B
    scales = (
        scales.view(flatten_B, scales.shape[-2], scales.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )
    zeros = (
        zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )
    assert bits in [2, 4]
    try:
        c = kivi_gemv.gemv_forward_cuda_outer_dim(
            fA, qB, scales, zeros, bits, group_size, nh, mqa
        )
    except:
        assert False, "Please install kivi. Refer to https://github.com/jy-yuan/KIVI"
    c = c.view(B, nh, c.shape[-2], c.shape[-1])
    return c


class KiviCache(Cache):
    def __init__(self, config):
        super().__init__()
        self.group_size = config.attn_kwargs.get("group_size", 32)
        self.bits = config.attn_kwargs.get("bits", 2)
        self.residual_length = config.attn_kwargs.get("residual_length", 32)
        self.group_size = config.attn_kwargs.get("group_size", 32)

        self.k_bits = self.bits
        self.v_bits = self.bits

        self.kv_cache = []
        self._seen_tokens = 0

        self.temp_key_cache = []
        self.temp_value_cache = []

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        query_states = cache_kwargs["query_states"]

        if layer_idx == 0:
            self._seen_tokens += key_states.size(-2)

        prefilling = False
        if len(self.kv_cache) == layer_idx:
            prefilling = True

            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[
                        :, :, : -(key_states.shape[-2] % self.residual_length), :
                    ].contiguous()
                    key_states_full = key_states[
                        :, :, -(key_states.shape[-2] % self.residual_length) :, :
                    ].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None

            if key_states_quant is not None:
                (
                    key_states_quant_trans,
                    key_scale_trans,
                    key_mn_trans,
                ) = triton_quantize_and_pack_along_last_dim(
                    key_states_quant.transpose(2, 3).contiguous(),
                    self.group_size,
                    self.k_bits,
                )
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None

            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[
                    :, :, : -self.residual_length, :
                ].contiguous()
                value_states_full = value_states[
                    :, :, -self.residual_length :, :
                ].contiguous()
                (
                    value_states_quant,
                    value_scale,
                    value_mn,
                ) = triton_quantize_and_pack_along_last_dim(
                    value_states_quant, self.group_size, self.v_bits
                )

            self.kv_cache.append(
                (
                    key_states_quant_trans,
                    key_states_full,
                    key_scale_trans,
                    key_mn_trans,
                    value_states_quant,
                    value_states_full,
                    value_scale,
                    value_mn,
                )
            )

            return (
                repeat_kv(key_states, query_states.size(1) // key_states.size(1)),
                repeat_kv(value_states, query_states.size(1) // value_states.size(1)),
            )

        else:  # decoding
            key_states_quant_trans = self.kv_cache[layer_idx][0]
            key_states_full = self.kv_cache[layer_idx][1]
            key_scale_trans = self.kv_cache[layer_idx][2]
            key_mn_trans = self.kv_cache[layer_idx][3]

            value_states_quant = self.kv_cache[layer_idx][4]
            value_states_full = self.kv_cache[layer_idx][5]
            value_scale = self.kv_cache[layer_idx][6]
            value_mn = self.kv_cache[layer_idx][7]

            if not update_global_past_kv:
                if len(self.temp_key_cache) == layer_idx:
                    self.temp_key_cache.append(key_states)
                    self.temp_value_cache.append(value_states)
                else:
                    self.temp_key_cache[layer_idx] = torch.cat(
                        [self.temp_key_cache[layer_idx], key_states], dim=-2
                    )
                    self.temp_value_cache[layer_idx] = torch.cat(
                        [self.temp_value_cache[layer_idx], value_states], dim=-2
                    )

                if key_states_full is not None:
                    key_states_full_rt = torch.cat(
                        [key_states_full, self.temp_key_cache[layer_idx]], dim=-2
                    )
                else:
                    key_states_full_rt = self.temp_key_cache[layer_idx]
                value_states_full_rt = torch.cat(
                    [value_states_full, self.temp_value_cache[layer_idx]], dim=-2
                )
            else:
                if key_states_full is not None:
                    key_states_full = torch.cat([key_states_full, key_states], dim=2)
                else:
                    key_states_full = key_states

                value_states_full = torch.cat([value_states_full, value_states], dim=2)
                key_states_full_rt = key_states_full
                value_states_full_rt = value_states_full

            self.kv_cache[layer_idx] = (
                key_states_quant_trans,
                key_states_full,
                key_scale_trans,
                key_mn_trans,
                value_states_quant,
                value_states_full,
                value_scale,
                value_mn,
            )

            key_out = (
                key_states_quant_trans,
                key_states_full_rt,  # key_states_full
                key_scale_trans,
                key_mn_trans,
            )
            value_out = (
                value_states_quant,
                value_states_full_rt,
                value_scale,
                value_mn,
            )
            return key_out, value_out

    def legacy_update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        query_states = cache_kwargs["query_states"]

        if layer_idx == 0:
            self._seen_tokens += key_states.size(-2)

        prefilling = False
        if len(self.kv_cache) == layer_idx:
            prefilling = True

            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[
                        :, :, : -(key_states.shape[-2] % self.residual_length), :
                    ].contiguous()
                    key_states_full = key_states[
                        :, :, -(key_states.shape[-2] % self.residual_length) :, :
                    ].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None

            if key_states_quant is not None:
                (
                    key_states_quant_trans,
                    key_scale_trans,
                    key_mn_trans,
                ) = triton_quantize_and_pack_along_last_dim(
                    key_states_quant.transpose(2, 3).contiguous(),
                    self.group_size,
                    self.k_bits,
                )
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None

            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[
                    :, :, : -self.residual_length, :
                ].contiguous()
                value_states_full = value_states[
                    :, :, -self.residual_length :, :
                ].contiguous()
                (
                    value_states_quant,
                    value_scale,
                    value_mn,
                ) = triton_quantize_and_pack_along_last_dim(
                    value_states_quant, self.group_size, self.v_bits
                )

            self.kv_cache.append(
                (
                    key_states_quant_trans,
                    key_states_full,
                    key_scale_trans,
                    key_mn_trans,
                    value_states_quant,
                    value_states_full,
                    value_scale,
                    value_mn,
                )
            )

            return (
                repeat_kv(key_states, query_states.size(-1) // key_states.size(-1)),
                repeat_kv(value_states, query_states.size(-1) // value_states.size(-1)),
            )

        else:  # decoding
            key_states_quant_trans = self.kv_cache[layer_idx][0]
            key_states_full = self.kv_cache[layer_idx][1]
            key_scale_trans = self.kv_cache[layer_idx][2]
            key_mn_trans = self.kv_cache[layer_idx][3]

            value_states_quant = self.kv_cache[layer_idx][4]
            value_states_full = self.kv_cache[layer_idx][5]
            value_scale = self.kv_cache[layer_idx][6]
            value_mn = self.kv_cache[layer_idx][7]

            if not update_global_past_kv:
                if len(self.temp_key_cache) == layer_idx:
                    self.temp_key_cache.append(key_states)
                    self.temp_value_cache.append(value_states)
                else:
                    self.temp_key_cache[layer_idx] = torch.cat(
                        [self.temp_key_cache[layer_idx], key_states], dim=-2
                    )
                    self.temp_value_cache[layer_idx] = torch.cat(
                        [self.temp_value_cache[layer_idx], value_states], dim=-2
                    )

                key_states_full_rt = torch.cat(
                    [key_states_full, self.temp_key_cache[layer_idx]], dim=-2
                )
                value_states_full_rt = torch.cat(
                    [value_states_full, self.temp_value_cache[layer_idx]], dim=-2
                )

            else:
                if key_states_full is not None:
                    key_states_full = torch.cat([key_states_full, key_states], dim=2)
                else:
                    key_states_full = key_states

                if key_states_full.shape[-2] > self.residual_length:
                    key_states_quant = key_states_full[
                        :, :, : -(key_states_full.shape[-2] % self.residual_length), :
                    ].contiguous()
                    key_states_full = key_states_full[
                        :, :, -(key_states_full.shape[-2] % self.residual_length) :, :
                    ].contiguous()

                    (
                        key_states_quant_trans_new,
                        key_scale_trans_new,
                        key_mn_trans_new,
                    ) = triton_quantize_and_pack_along_last_dim(
                        key_states_quant.transpose(2, 3).contiguous(),
                        self.group_size,
                        self.k_bits,
                    )
                    if key_states_quant_trans is not None:
                        key_states_quant_trans = torch.cat(
                            [key_states_quant_trans, key_states_quant_trans_new], dim=3
                        )
                        key_scale_trans = torch.cat(
                            [key_scale_trans, key_scale_trans_new], dim=3
                        )
                        key_mn_trans = torch.cat(
                            [key_mn_trans, key_mn_trans_new], dim=3
                        )
                    else:
                        key_states_quant_trans = key_states_quant_trans_new
                        key_scale_trans = key_scale_trans_new
                        key_mn_trans = key_mn_trans_new

                value_states_full = torch.cat([value_states_full, value_states], dim=2)
                value_full_length = value_states_full.shape[-2]

                if value_full_length > self.residual_length:
                    value_states_to_quant = value_states_full[
                        :, :, : -(value_full_length % self.residual_length), :
                    ].contiguous()
                    value_states_full = value_states_full[
                        :, :, -(value_full_length % self.residual_length) :, :
                    ].contiguous()

                    (
                        value_states_quant_new,
                        scale,
                        mn,
                    ) = triton_quantize_and_pack_along_last_dim(
                        value_states_to_quant.contiguous(),
                        self.group_size,
                        self.v_bits,
                    )
                    if value_states_quant is not None:
                        value_states_quant = torch.cat(
                            [value_states_quant, value_states_quant_new], dim=2
                        )
                        value_scale = torch.cat([value_scale, scale], dim=2)
                        value_mn = torch.cat([value_mn, mn], dim=2)
                    else:
                        value_states_quant = value_states_quant_new
                        value_scale = scale
                        value_mn = mn
                key_states_full_rt = key_states_full
                value_states_full_rt = value_states_full

            self.kv_cache[layer_idx] = (
                key_states_quant_trans,
                key_states_full,
                key_scale_trans,
                key_mn_trans,
                value_states_quant,
                value_states_full,
                value_scale,
                value_mn,
            )

            key_out = (
                key_states_quant_trans,
                key_states_full_rt,  # key_states_full
                key_scale_trans,
                key_mn_trans,
            )
            value_out = (
                value_states_quant,
                value_states_full_rt,
                value_scale,
                value_mn,
            )
            return key_out, value_out

    def get_seq_length(self, layer_idx=0):
        if len(self.kv_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[-1].shape[
                -2
            ]  # seq_len of temp_kv_cache
        self.temp_key_cache = []
        self.temp_value_cache = []


def kivi_forward(query_states, key_states, value_states, decoding_kwargs):
    group_size = decoding_kwargs["attn_forward_config"].get("group_size", 32)
    k_bits = decoding_kwargs["attn_forward_config"].get("bits", 2)
    v_bits = decoding_kwargs["attn_forward_config"].get("bits", 2)
    head_dim = query_states.size(-1)
    q_len = query_states.shape[-2]

    key_states_quant_trans = key_states[0]
    key_states_full = key_states[1]
    key_scale_trans = key_states[2].to(torch.float16)
    key_mn_trans = key_states[3].to(torch.float16)

    value_states_quant = value_states[0]
    value_states_full = value_states[1]
    value_scale = value_states[2].to(torch.float16)
    value_mn = value_states[3].to(torch.float16)

    num_key_value_groups = query_states.size(1) // key_states_quant_trans.size(1)
    key_states_quant_trans = repeat_kv(key_states_quant_trans, num_key_value_groups)
    key_scale_trans = repeat_kv(key_scale_trans, num_key_value_groups)
    key_mn_trans = repeat_kv(key_mn_trans, num_key_value_groups)
    key_states_full = repeat_kv(key_states_full, num_key_value_groups)

    value_states_quant = repeat_kv(value_states_quant, num_key_value_groups)
    value_scale = repeat_kv(value_scale, num_key_value_groups)
    value_mn = repeat_kv(value_mn, num_key_value_groups)
    value_states_full = repeat_kv(value_states_full, num_key_value_groups)

    if key_states_quant_trans is not None:
        if q_len == 1:
            att_qkquant = cuda_bmm_fA_qB_outer(
                group_size,
                query_states.to(torch.float16),
                key_states_quant_trans,
                key_scale_trans,
                key_mn_trans,
                k_bits,
            )
        else:  # cuda_bmm_fA_qB_outer will lead to nan when query_states.shape[-2] > 1, need a fix in the kernel side
            attn_qkquants_t = []
            for i in range(q_len):
                att_qkquant_i = cuda_bmm_fA_qB_outer(
                    group_size,
                    query_states[:, :, i : i + 1, :].to(torch.float16),
                    key_states_quant_trans,
                    key_scale_trans,
                    key_mn_trans,
                    k_bits,
                )
                attn_qkquants_t.append(att_qkquant_i)
            att_qkquant = torch.cat(attn_qkquants_t, dim=2)
    else:
        att_qkquant = None

    if torch.isnan(att_qkquant).any():
        print("NaN in att_qkquant")
    att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))

    if att_qkquant is not None:
        attn_weights = torch.cat(
            [att_qkquant.to(query_states.dtype), att_qkfull], dim=-1
        ) / math.sqrt(head_dim)
    else:
        attn_weights = att_qkfull / math.sqrt(head_dim)

    value_full_length = value_states_full.shape[-2]
    if q_len == 1:
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
    else:
        attn_weights += (
            torch.triu(
                torch.ones_like(attn_weights), diagonal=value_full_length - q_len + 1
            )
            * torch.finfo(attn_weights.dtype).min
        )
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

    value_full_length = value_states_full.shape[-2]
    if value_states_quant is not None:
        if q_len == 1:
            attn_output = cuda_bmm_fA_qB_outer(
                group_size,
                attn_weights[:, :, :, :-value_full_length].to(torch.float16),
                value_states_quant,
                value_scale,
                value_mn,
                v_bits,
            ).to(query_states.dtype)
            attn_output += torch.matmul(
                attn_weights[:, :, :, -value_full_length:], value_states_full
            )
        else:
            # if q_len > 1:
            attn_outputs = []
            for i in range(q_len):
                attn_output_i = cuda_bmm_fA_qB_outer(
                    group_size,
                    attn_weights[:, :, i : i + 1, :-value_full_length].to(
                        torch.float16
                    ),
                    value_states_quant,
                    value_scale,
                    value_mn,
                    v_bits,
                ).to(query_states.dtype)
                attn_outputs.append(attn_output_i)
            attn_output = torch.cat(attn_outputs, dim=2)
            attn_output += torch.matmul(
                attn_weights[:, :, :, -value_full_length:], value_states_full
            )
            # torch.testing.assert_close(attn_output, attn_output_t)
    else:
        attn_output = torch.matmul(attn_weights, value_states_full)

    if torch.isnan(attn_output).any():
        print("NaN in attn_output")

    return attn_output
