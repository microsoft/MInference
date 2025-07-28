from ..ops.streaming_kernel import tri_shape_kernel

try:
    from flash_attn import flash_attn_func
except ImportError:
    from ..ops.flash_attn_triton import _flash_attn_triton_decoding as flash_attn_func

from ..modules.minference_forward import minference_prefill_forward
from ..modules.flexprefill import flexprefill_forward
import copy

import torch


def tri_mix_forward(query_states, key_states, value_states, prefill_kwargs):
    starting_layer = prefill_kwargs["attn_forward_config"].get("starting_layer", 0)
    layer_idx = prefill_kwargs["layer_idx"]

    bsz, head_num, q_len, head_dim = query_states.shape
    if layer_idx < starting_layer:
        # flash attention
        result = flash_attn_func(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
                0.0, softmax_scale=None, causal=q_len != 1,
        ).transpose(1, 2)
    else:
        result = tri_shape_kernel(query_states, key_states, value_states, prefill_kwargs)

    return result


def tri_mix_minference_forward(q, k, v, prefill_kwargs):
    layer_idx = prefill_kwargs["layer_idx"]
    starting_layer = prefill_kwargs["attn_forward_config"].get("starting_layer", 0)

    if layer_idx < starting_layer:
        # minference
        minference_prefill_kwargs = copy.deepcopy(prefill_kwargs)
        minference_prefill_kwargs["attn_forward_config"]["starting_layer"] = 0
        result =  minference_prefill_forward(q, k, v, minference_prefill_kwargs)
    else:
        result = tri_shape_kernel(q, k, v, prefill_kwargs)
    return result
