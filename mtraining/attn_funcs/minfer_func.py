#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# This file modifies the official modeling_llama.py file at runtime to
# 1. register the flash attention function to nnscaler and update related code
# 2. replace the un-fused RMSNorm with apex's fused version
import json
import torch
import logging
logger = logging.getLogger(__name__)

from typing import List, Optional, Tuple, Dict, Callable
from transformers.utils import logging, is_flash_attn_2_available
if is_flash_attn_2_available(): from flash_attn import flash_attn_func

from nnscaler.runtime.device import DeviceGroup
from nnscaler.graph.parser.register import register_op
from nnscaler.ir import IRTensor
from nnscaler.ir.operator import IRFwOperation

from minference.ops.minference_attn import minference_flash_attn_func
from minference.ops.minference_attn_triton import minference_flash_attn_triton_func
from minference.dist_ops import (
    minfer_stripe_func, minfer_stripe_triton_func,
    minfer_zigzag_func, minfer_dr_stripe_func, minfer_dr_stripe_triton_func,
)


# =======================================================
def minfer_op(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,

    bsz: int,
    q_len: int,
    head_dim: int,
    layer_idx: int,

    pattern_dict: Dict[int, Tuple[str, int, int, int]],
    attn_dropout: float=0.,
    granularity: int = 128,
    group: Optional[torch.distributed.ProcessGroup] = None,
):
    v_sizes = [pattern_dict[head_indices[idx].item()][1] for idx in range(query_states.size(1))]
    s_sizes = [pattern_dict[head_indices[idx].item()][2] for idx in range(query_states.size(1))]
    if torch.version.hip is None:
        attn_output = minference_flash_attn_func(
            query_states.transpose(1, 2).contiguous(),
            key_states.transpose(1, 2).contiguous(),
            value_states.transpose(1, 2).contiguous(),
            v_sizes, s_sizes,
            attn_dropout,
            softmax_scale=None,
            granularity=granularity,
            causal=True,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=False,
            group=group,
        )
    else:
        attn_output = minference_flash_attn_triton_func(
            query_states.transpose(1, 2).contiguous(),
            key_states.transpose(1, 2).contiguous(),
            value_states.transpose(1, 2).contiguous(),
            v_sizes, s_sizes,
            attn_dropout,
            softmax_scale=None,
            granularity=granularity,
            causal=True,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=False,
            group=group,
        )
    return attn_output.contiguous()


def minfer_stripe_op(
    query_states: torch.Tensor, # [batch_size, num_heads, num_tokens, head_dim]
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,
    bsz: int,
    q_len: int,
    head_dim: int,
    layer_idx: int,
    
  
    pattern_dict: Dict[int, Tuple[str, int, int, int]],
    attn_dropout: float=0.,
    granularity: int = 128,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
):
    if (process_group is None or len(process_group) == 1):
        softmax_scale = query_states.shape[-1] ** (-0.5)

        output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attn_dropout, softmax_scale, causal=True)
        return output
    group = DeviceGroup().get_group(process_group)

    v_sizes = [pattern_dict[head_indices[idx].item()][1] for idx in range(query_states.size(1))]
    s_sizes = [pattern_dict[head_indices[idx].item()][2] for idx in range(query_states.size(1))]
    if torch.version.hip is None:
        attn_output = minfer_stripe_func(
            query_states.transpose(1, 2).contiguous(),
            key_states.transpose(1, 2).contiguous(),
            value_states.transpose(1, 2).contiguous(),
            v_sizes, s_sizes,
            layer_idx,
            attn_dropout,
            softmax_scale=None,
            granularity=granularity,
            causal=True,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=False,
            group=group,
        ) # expect:  b {q_anno} l^ vd^'
    else:
        attn_output = minfer_stripe_triton_func(
            query_states.transpose(1, 2).contiguous(),
            key_states.transpose(1, 2).contiguous(),
            value_states.transpose(1, 2).contiguous(),
            v_sizes, s_sizes,
            layer_idx,
            attn_dropout,
            softmax_scale=None,
            granularity=granularity,
            causal=True,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=False,
            group=group,
        )
    return attn_output.contiguous()


def minfer_zigzag_op(
    query_states: torch.Tensor, # [batch_size, num_heads, num_tokens, head_dim]
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,

    bsz: int,
    q_len: int,
    head_dim: int,
    layer_idx: int,
    
    pattern_dict: Dict[int, Tuple[str, int, int, int]],
    attn_dropout: float=0.,
    granularity: int = 128,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
):
    if process_group is None or len(process_group) == 1:
        # there is an additional checker for the `softmax_scale`, which is equivalent
        # to the behavior of the original flash_attn_func.
        softmax_scale = query_states.shape[-1] ** (-0.5)
        output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attn_dropout, softmax_scale, causal=True)
        return output
    group = DeviceGroup().get_group(process_group)

    v_sizes = [pattern_dict[head_indices[idx].item()][1] for idx in range(query_states.size(1))]
    s_sizes = [pattern_dict[head_indices[idx].item()][2] for idx in range(query_states.size(1))]
    if torch.version.hip is None:
        attn_output = minfer_zigzag_func(
            query_states.transpose(1, 2).contiguous(),
            key_states.transpose(1, 2).contiguous(),
            value_states.transpose(1, 2).contiguous(),
            v_sizes, s_sizes,
            layer_idx,
            attn_dropout,
            softmax_scale=None,
            granularity=granularity,
            causal=True,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=False,
            group=group,
        ) # expect:  b {q_anno} l^ vd^'
    else:
        raise NotImplementedError("Triton-only version is not implemented for MInfer w. zigzag")
    return attn_output.contiguous()

def minfer_dr_stripe_op(
    query_states: torch.Tensor, # [batch_size, num_heads, num_tokens, head_dim]
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,
    bsz: int,
    q_len: int,
    head_dim: int,
    layer_idx: int,
    
    pattern_dict: Dict[int, Tuple[str, int, int, int]],
    attn_dropout: float=0.,
    granularity: int = 128,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
):
    if (process_group is None or len(process_group) == 1):
        # there is an additional checker for the `softmax_scale`, which is equivalent
        # to the behavior of the original flash_attn_func.
        softmax_scale = query_states.shape[-1] ** (-0.5)

        output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attn_dropout, softmax_scale, causal=True)
        return output

    group = DeviceGroup().get_group(process_group)
    v_sizes = [pattern_dict[head_indices[idx].item()][1] for idx in range(query_states.size(1))]
    s_sizes = [pattern_dict[head_indices[idx].item()][2] for idx in range(query_states.size(1))]

    if torch.version.hip is None:
        attn_output = minfer_dr_stripe_func(
            query_states.transpose(1, 2).contiguous(),
            key_states.transpose(1, 2).contiguous(),
            value_states.transpose(1, 2).contiguous(),
            v_sizes, s_sizes,
            layer_idx,
            attn_dropout,
            softmax_scale=None,
            granularity=granularity,
            causal=True,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=False,
            group=group,
        ) # expect:  b {q_anno} l^ vd^'
    else:
        attn_output = minfer_dr_stripe_triton_func(
            query_states.transpose(1, 2).contiguous(),
            key_states.transpose(1, 2).contiguous(),
            value_states.transpose(1, 2).contiguous(),
            v_sizes, s_sizes,
            layer_idx,
            attn_dropout,
            softmax_scale=None,
            granularity=granularity,
            causal=True,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=False,
            group=group,
        )
    
    return attn_output.contiguous()



MINFER_IMPLEMENTATIONS: Dict[str, Callable] = {
    "default": minfer_op,
    "stripe": minfer_stripe_op,
    "zigzag": minfer_zigzag_op,
    "dr_stripe": minfer_dr_stripe_op,
}

def emit_minfer_ring(node: IRFwOperation, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
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
        elif partition_dims[0] == 1:
            # partition on num_head dim
            kw_pairs.append("process_group=None")
        elif partition_dims[0] == 2: # partition on sequence dim
            # the synchronization should occur across scaleunits
            kw_pairs.append(f"process_group={scale_unit_dev_ids}")
        else:
            raise ValueError(f'unsupported partition dim: {partition_dims[0]}')

    args = ", ".join(list(args) + kw_pairs)
    return f"{signature}({args})"

def minfer_attn_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    if query_states.shape[1] != key_states.shape[1]:
        assert query_states.shape[1] % key_states.shape[1] == 0
        group_size = query_states.shape[1] // key_states.shape[1]
        assert query_states.shape[1] == value_states.shape[1] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    return f'b {q_anno} l^ hd^, b {kv_anno} s^ hd^, b {kv_anno} s^ vd^, {q_anno} -> b l^ {q_anno} vd^'

def minfer_attn_ring_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    if query_states.shape[1] != key_states.shape[1]:
        assert query_states.shape[1] % key_states.shape[1] == 0
        group_size = query_states.shape[1] // key_states.shape[1]
        assert query_states.shape[1] == value_states.shape[1] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    return f'b {q_anno} l hd^, b {kv_anno} l hd^, b {kv_anno} l vd^, {q_anno} -> b l {q_anno} vd^'

if __name__ != "__main__":
    register_op(minfer_attn_anno)(minfer_op)
    register_op(minfer_attn_ring_anno, emit_fn=emit_minfer_ring)(minfer_stripe_op)
    register_op(minfer_attn_ring_anno, emit_fn=emit_minfer_ring)(minfer_zigzag_op)
    register_op(minfer_attn_ring_anno, emit_fn=emit_minfer_ring)(minfer_dr_stripe_op)

class MInferAttnFunc:
    def __init__(self):
        self.initialized = False
    
    def init_minfer_params(
        self,
        config_path: str,
        minfer_implementation: str, # "fa", "stripe", "zigzag"
        granularity: int = 128,
    ):
        assert minfer_implementation in MINFER_IMPLEMENTATIONS, f"minfer_implementation should be one of {MINFER_IMPLEMENTATIONS}, but got {self.minfer_implementation}"
        self.minfer_implementation: str = minfer_implementation

        self.config_path = config_path
        self.all_pattern_dict = json.load(open(self.config_path))
        self.granularity = granularity

        self.initialized = True
    
    def get_pattern_dict(self, layer_idx):
        return {int(ii): jj for ii, jj in self.all_pattern_dict[layer_idx].items()}
    
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        head_indices: torch.Tensor,
        attn_module_config: Dict[str, int],
        attn_dropout: float=0.0,
    ):
        bsz, q_len = query_states.shape[0], query_states.shape[2]
        head_dim, layer_idx = attn_module_config["head_dim"], attn_module_config["layer_idx"]
        
        pattern_dict = self.get_pattern_dict(layer_idx)
        minfer_args = (
            query_states, key_states, value_states, 
            head_indices,
            bsz, q_len, head_dim, layer_idx,
            pattern_dict, attn_dropout, self.granularity,
        )

        if self.minfer_implementation == "default":
            return minfer_op(*minfer_args)
        elif self.minfer_implementation == "stripe":
            return minfer_stripe_op(*minfer_args)
        elif self.minfer_implementation == "zigzag":
            return minfer_zigzag_op(*minfer_args)
        elif self.minfer_implementation == "dr_stripe":
            return minfer_dr_stripe_op(*minfer_args)
        else:
            raise ValueError(f"Unsupported minfer_implementation: {self.minfer_implementation}")

def minfer_attention_forward(
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
    attn_module_config = {
        "num_heads": module.config.num_attention_heads,
        "head_dim": module.head_dim,
        "layer_idx": module.layer_idx,
    }
    head_indices = torch.arange(attn_module_config["num_heads"], device=query.device, dtype=torch.int32)

    return module.minfer_attn_func.forward(
        query, key, value, head_indices,
        attn_module_config,
        dropout,
    ), None
