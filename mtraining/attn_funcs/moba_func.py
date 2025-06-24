import os
import yaml
import torch
import torch.distributed as dist

from torch import Tensor
from typing import Tuple, Optional, List, Dict, Any

from nnscaler.ir.operator import IRFwOperation
from nnscaler.runtime.device import DeviceGroup
from nnscaler.graph.parser.register import register_op

from minference.ops.moba import moba_attn_func
from minference.ops.op_utils.moba_utils import MoBAConfig
from minference.dist_ops.moba_zigzag import moba_zigzag_func

def load_moba_config(moba_config_dict: Dict[str, Any]):
    moba_config = MoBAConfig(**moba_config_dict) 
    return moba_config

def moba_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor, # [B, H, N, D]
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]: 
    seq_len = query.shape[2]
    moba_topk, moba_chunk_size = module.moba_topk, module.moba_chunk_size
    implementation = module.implementation
    
    if implementation == "default":
        return wrapped_moba_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            seq_len,
            moba_topk, moba_chunk_size,
        ), None
    elif implementation == "zigzag":
        layer_idx = module.layer_idx
        return wrapped_moba_zigzag_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            seq_len,
            moba_topk, moba_chunk_size,
            layer_idx,
            attention_mask, dropout, scaling, sliding_window, softcap,
        ), None
    else:
        raise ValueError(f"Unsupported MoBA implementation: {implementation}. "
                         f"Supported implementations are 'default' and 'zigzag'.")


# ------------------------------------------
def wrapped_moba_func(
    q: Tensor, k: Tensor, v: Tensor, 
    seq_len: int,
    moba_topk: int, moba_chunk_size: int,
):
    return moba_attn_func(
        q, k, v,
        seq_len,
        moba_chunk_size, moba_topk,
    )

def wrapped_moba_zigzag_func(
    query: Tensor, # [B, N, H, D]
    key: Tensor, 
    value: Tensor, 
    seq_len: int,
    moba_topk: int, moba_chunk_size: int,
    layer_idx: int,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    process_group: Tuple[int]=None,
):
    if process_group is None or len(process_group) == 1:
        # there is an additional checker for the `softmax_scale`, which is equivalent
        # to the behavior of the original flash_attn_func.
        from flash_attn import flash_attn_func
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** (-0.5)
        output = flash_attn_func(query, key, value, 0.0, softmax_scale, True)
        return output

    batch_size, block_seq_len, q_heads, head_dim = query.shape
    assert batch_size == 1, "Current implementation only supports batch size = 1"

    local_process_group = DeviceGroup().get_group(process_group)
    output = moba_zigzag_func(
        query, key, value,
        layer_idx, 
        seq_len,
        moba_chunk_size,
        moba_topk,
        dropout, softmax_scale,
        True, # causal,
        (-1, -1), # window_size,
        None, # alibi_slopes,
        False, # deterministic,
        False, # return_softmax,
        local_process_group, # group
    ).contiguous()
    return output.view(batch_size, block_seq_len, q_heads, head_dim)

# --------------------------------------------------
def moba_attn_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    if query_states.shape[2] != key_states.shape[2]:
        assert query_states.shape[2] % key_states.shape[2] == 0
        group_size = query_states.shape[2] // key_states.shape[2]
        assert query_states.shape[2] == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    return f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^ -> b l^ {q_anno} vd^'

def moba_zigzag_attn_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    num_q_heads, num_kv_heads = query_states.shape[2], key_states.shape[2]
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        group_size = num_q_heads // num_kv_heads
        assert num_q_heads == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    attn_anno = f'b l {q_anno} hd^, b l {kv_anno} hd^, b l {kv_anno} vd^ -> b l {q_anno} vd^'
    return attn_anno


def emit_moba_zigzag(node: IRFwOperation, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
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

if __name__ != "__main__":
    register_op(moba_attn_anno)(wrapped_moba_func)
    register_op(moba_zigzag_attn_anno, emit_fn=emit_moba_zigzag)(wrapped_moba_zigzag_func)
