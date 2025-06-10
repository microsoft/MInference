import os
import yaml
import copy
import torch
from torch import Tensor
import torch.distributed as dist

from functools import partial
from flash_attn import flash_attn_func
from typing import Tuple, Optional, Dict, Any, List

from nnscaler.runtime.device import DeviceGroup
from nnscaler.graph.parser.register import register_op
from nnscaler.ir import IRTensor
from nnscaler.ir.operator import IRFwOperation

from minference.ops.xattention_fa import xattn_flash_attn_func
from minference.dist_ops.xattn_zigzag import xattn_zigzag_func

def xattn_attention_forward(
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
    granularity, xattn_params = module.granularity, module.xattn_params
    implementation = module.implementation
    layer_idx = module.layer_idx

    if implementation == "default":
        head_indices = torch.arange(module.config.num_attention_heads, device=query.device)
        return wrapped_xattn_func_(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            head_indices,
            granularity, xattn_params,
            dropout, scaling, sliding_window
        )
    elif implementation == "zigzag":
        return wrapped_xattn_zigzag_func_(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            layer_idx,
            granularity, xattn_params,
            dropout=dropout, scaling=scaling, sliding_window=sliding_window,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation for xattn_attention_forward: {implementation}")

# ------------------------------------------
# Non-CP version
def wrapped_xattn_func_(
    q: Tensor, k: Tensor, v: Tensor, # [B, N, H, D]
    head_indices: torch.Tensor,
    granularity: int,
    xattn_params: Dict[str, Any],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
):
    return wrapped_xattn_func(
        q, k, v,
        head_indices,
        granularity, xattn_params,
        dropout, scaling, sliding_window
    ), None

def wrapped_xattn_func(
    q: Tensor, k: Tensor, v: Tensor, 
    head_indices: torch.Tensor,
    granularity: int,
    xattn_params: Dict[str, Any],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
):
    sliding_window = -1 if sliding_window is None else sliding_window
    return xattn_flash_attn_func(
        q, k, v,
        head_indices.detach().cpu().numpy().tolist(),
        xattn_params,
        granularity, 
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
        window_size=(sliding_window, sliding_window),
        alibi_slopes=None,
        deterministic=False,
    )




# ------------------------------------------
# Zigzag Version
def wrapped_xattn_zigzag_func_(
    q: Tensor, k: Tensor, v: Tensor, # [B, N, H, D]
    layer_idx: int,
    granularity: int,
    xattn_params: Dict[str, Any],
    causal: bool=True, 
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
):
    return wrapped_xattn_zigzag_func(
        q, k, v,
        layer_idx,
        granularity, xattn_params,
        causal=causal,
        dropout=dropout, scaling=scaling, sliding_window=sliding_window,
    ), None


def wrapped_xattn_zigzag_func(
    q: Tensor, k: Tensor, v: Tensor, # [B, N, H, D]
    layer_idx: int,
    granularity: int,
    xattn_params: Dict[str, Any],
    causal: bool=True, 
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    process_group: Tuple[int]=None,
):
    if process_group is None or len(process_group) == 1:
        # there is an additional checker for the `scaling`, which is equivalent
        # to the behavior of the original flash_attn_func.
        if scaling is None:
            scaling = q.shape[-1] ** (-0.5)
        output = flash_attn_func(q, k, v, 0.0, scaling, causal)
        return output

    group = DeviceGroup().get_group(process_group)

    xattn_params = copy.copy(xattn_params)
    xattn_params.pop("chunk_size", None)
    return xattn_zigzag_func(
        q, k, v,
        layer_idx,
        xattn_params,
        granularity,
        dropout_p=dropout, 
        softmax_scale=scaling, 
        causal=causal,
        group=group,
    ).contiguous()

def xattn_attn_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    if query_states.shape[2] != key_states.shape[2]:
        assert query_states.shape[2] % key_states.shape[2] == 0
        group_size = query_states.shape[2] // key_states.shape[2]
        assert query_states.shape[2] == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    return f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^, {q_anno} -> b l^ {q_anno} vd^'

def emit_xattn_zigzag(node: IRFwOperation, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
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

def xattn_zigzag_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    if query_states.shape[2] != key_states.shape[2]:
        assert query_states.shape[2] % key_states.shape[2] == 0
        group_size = query_states.shape[2] // key_states.shape[2]

        assert query_states.shape[2] == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    return f'b l {q_anno} hd^, b l {kv_anno} hd^, b l {kv_anno} vd^ -> b l {q_anno} vd^'

if __name__ != "__main__":
    register_op(xattn_attn_anno)(wrapped_xattn_func)
    register_op(xattn_zigzag_anno, emit_fn=emit_xattn_zigzag)(wrapped_xattn_zigzag_func)
