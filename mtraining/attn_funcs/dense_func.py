#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# This file modifies the official modeling_llama.py file at runtime to
# 1. register the flash attention function to nnscaler and update related code
# 2. replace the un-fused RMSNorm with apex's fused version
import torch
from torch import Tensor
from transformers.utils import is_flash_attn_2_available
from typing import List, Optional, Tuple, Union, Any, Dict
if is_flash_attn_2_available():
    from flash_attn.bert_padding import pad_input
    from flash_attn import flash_attn_func, flash_attn_varlen_func

from nnscaler.ir import IRTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.runtime.device import DeviceGroup
from nnscaler.graph.parser.register import register_op

from minference.dist_ops.zigzag_attention import zigzag_ring_flash_attn_func
from minference.dist_ops.striped_attention import stripe_flash_attn_func

from .utils import nnscaler_upad_input


def fa_attn_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        causal=True,
    ):
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = nnscaler_upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

# ---------------------------------------------------------------------------
def zigzag_ring_attention_forward(
    module: torch.nn.Module,
    query: Tensor, # [B, H, N, D]
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[Tensor, None]: 
    return wrap_zigzag_attn_func(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        layer_idx=module.layer_idx,
        softmax_scale=scaling,
        dropout_p=dropout,
        causal=True,
    ), None


def wrap_zigzag_attn_func(
        q: Tensor, k: Tensor, v: Tensor, 
        layer_idx: int,
        softmax_scale: Tensor=None,
        dropout_p: float=0.0, 
        causal: bool=True, 
        window_size: Tuple[int]=(-1, -1),
        alibi_slopes: Tensor=None, deterministic: bool=False,
        return_attn_probs: bool=False,
        process_group: Tuple[int]=None
) -> Tensor:
    if process_group is None or len(process_group) == 1:
        # there is an additional checker for the `softmax_scale`, which is equivalent
        # to the behavior of the original flash_attn_func.
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        output = flash_attn_func(q, k, v, 0.0, softmax_scale, causal)
        return output

    assert causal == True, "zigzag_ring is meaningless for causal=False"
    assert len(q.shape) == 4, "q must have shape [bs, ql, qh, dim]"
    assert len(k.shape) == 4, "k must have shape [bs, kl, kh, dim]"
    assert len(v.shape) == 4, "v must have shape [bs, vl, vh, dim]"
    qbsz, qlen, qheads, qdim = q.shape
    kbsz, klen, kheads, kdim = k.shape
    vbsz, vlen, vheads, vdim = v.shape
    assert qbsz == kbsz == vbsz, "batch size must be the same"
    assert qlen == klen == vlen, "sequence length must be the same"
    assert kheads == vheads, "number of k and v heads must be the same"
    assert qheads % kheads == 0, "number of q heads must be a multiple of k heads"
    assert qdim == kdim == vdim, "dimension must be the same"

    local_process_group = DeviceGroup().get_group(process_group)
    output = zigzag_ring_flash_attn_func(
        q, k, v,
        layer_idx, 
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        local_process_group,
    ).contiguous()
    return output

# ---------------------------------------------------------------------------
def stripe_ring_attention_forward(
    module: torch.nn.Module,
    query: Tensor, # [B, H, N, D]
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[Tensor, None]: 
    return wrap_striped_attn_func(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        layer_idx=module.layer_idx,
        softmax_scale=scaling,
        dropout_p=dropout,
        causal=True,
    ), None


def wrap_striped_attn_func(
        q: Tensor, k: Tensor, v: Tensor, layer_idx: int,
        granularity: int=1,
        softmax_scale: Tensor=None,
        dropout_p: float=0.0, causal: bool=True, window_size: Tuple[int]=(-1, -1),
        alibi_slopes: Tensor=None, deterministic: bool=False,
        return_attn_probs: bool=False,
        process_group: Tuple[int]=None
    ) -> Tensor:
    if process_group is None or len(process_group) == 1:
        # there is an additional checker for the `softmax_scale`, which is equivalent
        # to the behavior of the original flash_attn_func.
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        output = flash_attn_func(q, k, v, 0.0, softmax_scale, causal)
        return output

    assert len(q.shape) == 4, "q must have shape [bs, ql, qh, dim]"
    assert len(k.shape) == 4, "k must have shape [bs, kl, kh, dim]"
    assert len(v.shape) == 4, "v must have shape [bs, vl, vh, dim]"
    qbsz, qlen, qheads, qdim = q.shape
    kbsz, klen, kheads, kdim = k.shape
    vbsz, vlen, vheads, vdim = v.shape
    assert qbsz == kbsz == vbsz, "batch size must be the same" 
    assert qlen == klen == vlen, "sequence length must be the same"
    assert kheads == vheads, "number of k and v heads must be the same"
    assert qheads % kheads == 0, "number of q heads must be a multiple of k heads"
    assert qdim == kdim == vdim, "dimension must be the same"

    local_process_group = DeviceGroup().get_group(process_group)
    output = stripe_flash_attn_func(
        q, k, v,
        layer_idx,
        dropout_p,
        softmax_scale,
        granularity,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        local_process_group,
    ).contiguous()
    return output



# ---------------------------------------------------------------------------
def flash_attention_anno(query_states, key_states, value_states, attention_mask, *args, **kwargs) -> str:
    if query_states.shape[2] != key_states.shape[2]:
        assert query_states.shape[2] % key_states.shape[2] == 0
        group_size = query_states.shape[2] // key_states.shape[2]
        assert query_states.shape[2] == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    if isinstance(attention_mask, IRTensor):
        return f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^, b l^ -> b l^ {q_anno} vd^'
    else:
        return f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^ -> b l^ {q_anno} vd^'


def emit_ring(node: IRFwOperation, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
    """Special rule to generate zigzag_attn node"""

    signature = node.signature

    offset = (runtime_devid // plan_ndevs) * plan_ndevs
    scale_unit_dev_ids = [local_rank + offset for local_rank in range(plan_ndevs)]

    kw_pairs = list()
    for key, val in kwargs.items():
        code = f'{key}={val}'
        kw_pairs.append(code)

    sub_input = node.inputs()[0]
    full_input = sub_input.parent
    partition_dims = [i for i, (s, f) in enumerate(zip(sub_input.shape, full_input.shape)) if s != f]
    assert len(partition_dims) <= 1, f"support no more than one partition dim, but got {partition_dims}"
    if not partition_dims:
        kw_pairs.append("process_group=None")
    else:
        # if the 'process_group' is None, we will use the local attention (flash_attn_func)
        if partition_dims[0] == 0: # partition on batch dim
            # partition the bsz dim, use local flash_attn_func
            kw_pairs.append("process_group=None")
        elif partition_dims[0] == 1: # partition on sequence dim
            # the synchronization should occur across scaleunits
            kw_pairs.append(f"process_group={scale_unit_dev_ids}")
        elif partition_dims[0] == 2:
            # partition on num_head dim
            kw_pairs.append("process_group=None")
        else:
            raise ValueError(f'unsupported partition dim: {partition_dims[0]}')
                
    args = ", ".join(list(args) + kw_pairs)
    return f"{signature}({args})"

def ring_attn_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    if query_states.shape[2] != key_states.shape[2]:
        assert query_states.shape[2] % key_states.shape[2] == 0
        group_size = query_states.shape[2] // key_states.shape[2]
        assert query_states.shape[2] == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    return f'b l {q_anno} hd^, b l {kv_anno} hd^, b l {kv_anno} vd^ -> b l {q_anno} vd^'


register_op(flash_attention_anno)(fa_attn_forward)
register_op(ring_attn_anno, emit_fn=emit_ring)(wrap_zigzag_attn_func)
register_op(ring_attn_anno, emit_fn=emit_ring)(wrap_striped_attn_func)
