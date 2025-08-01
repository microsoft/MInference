from typing import Optional, Tuple
import os
import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Model,
    repeat_kv,
    apply_rotary_pos_emb,
)
import types
from .tuple_kv_cache import enable_tuple_kv_cache_for_qwen
from ..ulysses import UlyssesAttention, UlyssesAttentionDecode
import math
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache

class BinaryMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bar):
        return torch.where(bar.to(x.device)==1., x, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Straight-through estimator


def scaled_attn(
    full_query_states: torch.Tensor,
    full_key_states: torch.Tensor,
    full_value_states: torch.Tensor,
    scaling_factors: torch.Tensor,
    length_context: int, 
    num_key_value_groups: int,
    sink_size: int = 128,
    recent_size: int = 1024,
):

    dtype = full_query_states.dtype
    q_len = full_query_states.shape[1]
    hdim = full_query_states.shape[-1]
    prefill_query_states = full_query_states[:, :length_context, :, :]
    prefill_key_states = full_key_states[:, :length_context, :, :]
    prefill_value_states = full_value_states[:, :length_context, :, :]

    with torch.no_grad():
        
        prefill_attn_output = _flash_attention_forward(
            prefill_query_states,
            prefill_key_states,
            prefill_value_states,
            None,
            q_len,
            dropout=0.0,
            is_causal=True,
        )

    decode_query_states = full_query_states[:, length_context:, :, :]

    full_key_states_masked = full_key_states.clone()
    full_key_states = full_key_states.transpose(1, 2).to(torch.float32)
    
    full_key_states_masked = (full_key_states_masked * scaling_factors[:, :1, :, :]).transpose(1, 2).to(torch.float32)
    
    decode_query_states = decode_query_states.transpose(1, 2).to(torch.float32)
    full_value_states = full_value_states.transpose(1, 2).to(torch.float32)

    full_key_states = repeat_kv(full_key_states, num_key_value_groups)
    full_key_states_masked = repeat_kv(full_key_states_masked, num_key_value_groups)
    full_value_states = repeat_kv(full_value_states, num_key_value_groups)
    
    attn_weights = torch.matmul(decode_query_states, full_key_states.transpose(2, 3)) / math.sqrt(hdim)
    attn_weights_masked = torch.matmul(decode_query_states, full_key_states_masked.transpose(2, 3)) / math.sqrt(hdim)

    query_len = q_len - length_context

    causal_mask = torch.full((query_len, query_len), fill_value=torch.finfo(attn_weights.dtype).min).to(attn_weights.dtype).to(attn_weights.dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, ...].to(attn_weights.device)

    attn_weights[:, :, -query_len:, -query_len:] += causal_mask
    attn_weights_masked[:, :, -query_len:, -query_len:] += causal_mask

    full_mask = torch.zeros(query_len, q_len).to(attn_weights.device)
    full_mask[:, -recent_size:] = 1
    full_mask[:, -query_len-recent_size:-recent_size] = torch.triu(torch.full((query_len, query_len), fill_value=1), diagonal=1)
    full_mask[:, :sink_size] = 1
    full_mask = full_mask[None, None, ...]

    attn_weights = attn_weights * full_mask + attn_weights_masked * (1 - full_mask)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(decode_query_states.dtype)
    attn_output = torch.matmul(attn_weights, full_value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    streaming_attn_output = torch.cat((prefill_attn_output, attn_output), dim=1)
    
    return streaming_attn_output.to(dtype)


def scaled_attn_forward_qwen(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    length_context: Optional[int] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    scaling_factors = (
        self.scaling_factors.clamp(0, 1)
        .view(1, 1, self.num_key_value_heads, self.head_dim)
    )
    if self.mask_round is not None:
        scaling_factors = BinaryMask.apply(scaling_factors, torch.Tensor(self.mask_round).reshape(scaling_factors.shape))

    full_query_states = self.q_proj(hidden_states)
    full_key_states = self.k_proj(hidden_states)
    full_value_states = self.v_proj(hidden_states)
    full_query_states = full_query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    )
    full_key_states = full_key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )
    full_value_states = full_value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )

    cos, sin = position_embeddings

    full_query_states, full_key_states = apply_rotary_pos_emb(
        full_query_states,
        full_key_states,
        cos,
        sin,
        unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    )
    
    decode_attn_output = self.scaled_attn_func(
        full_query_states,
        full_key_states,
        full_value_states,
        scaling_factors,
        length_context=length_context,
        num_key_value_groups=self.num_key_value_groups,
        sink_size=self.sink_size,
        recent_size=self.recent_size,
    )

    decode_attn_output = decode_attn_output.reshape(bsz, q_len, -1).contiguous()
    decode_attn_output = self.o_proj(decode_attn_output)

    return decode_attn_output, None

def full_attn_forward_qwen(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    length_context: Optional[int] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    with torch.no_grad():
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        full_query_states = self.q_proj(hidden_states)
        full_key_states = self.k_proj(hidden_states)
        full_value_states = self.v_proj(hidden_states)
        full_query_states = full_query_states.view(
            hidden_shape
        )
        full_key_states = full_key_states.view(
            hidden_shape
        )
        full_value_states = full_value_states.view(
            hidden_shape
        )

        cos, sin = position_embeddings

        full_query_states, full_key_states = apply_rotary_pos_emb(
            full_query_states,
            full_key_states,
            cos,
            sin,
            unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
        )
        
        full_attn_output = self.full_attn_func(
            full_query_states,
            full_key_states,
            full_value_states,
            attention_mask,
            q_len,
            dropout=0.,
            is_causal=self.is_causal,
        )
        full_attn_output = full_attn_output.reshape(bsz, q_len, -1).contiguous()
        full_attn_output = self.o_proj(full_attn_output)

    return full_attn_output, None

   

def enable_qwen_training(
    model: Qwen2ForCausalLM,
    sink_size,
    recent_size,
    scaling_factors=None,
    initial_value=1.0,
    enable_ulysses_attention=False,
):
    enable_tuple_kv_cache_for_qwen(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_parameter(
            "scaling_factors",
            nn.Parameter(
                (scaling_factors[idx].to(device, dtype))
                if scaling_factors is not None
                else (torch.ones(
                    model.config.num_key_value_heads * module.head_dim,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                * initial_value)
            ),
        )
        module.scaling_factors.requires_grad = True
        module.num_heads = model.config.num_attention_heads
        module.num_key_value_heads = model.config.num_key_value_heads
        module.sink_size = sink_size
        module.recent_size = recent_size

        if not enable_ulysses_attention:
            module.scaled_attn_func = scaled_attn
            module.full_attn_func = _flash_attention_forward
        else:
            module.scaled_attn_func = UlyssesAttentionDecode(
                attn_func=scaled_attn,
            )
            module.full_attn_func = UlyssesAttention(
                attn_func=_flash_attention_forward,
            )


def set_scaling_factors_qwen(model):
    scaling_factors = []
    if isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "scaling_factors"):
                continue
            scaling_factors.append(module.scaling_factors)
    elif isinstance(model, Qwen2Model):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "scaling_factors"):
                continue
            scaling_factors.append(module.scaling_factors)
    else:
        raise ValueError("Model type not supported")

    return scaling_factors


def set_scaling_factors_qwen(model):
    scaling_factors = []
    if isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "scaling_factors"):
                continue
            scaling_factors.append(module.scaling_factors)
    elif isinstance(model, Qwen2Model):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "scaling_factors"):
                continue
            scaling_factors.append(module.scaling_factors)
    else:
        raise ValueError("Model type not supported")

    return scaling_factors


def map_scaling_factors_qwen(model, func):
    if isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "scaling_factors"):
                continue
            func(module.scaling_factors)
    elif isinstance(model, Qwen2Model):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "scaling_factors"):
                continue
            func(module.scaling_factors)
    else:
        raise ValueError("Model type not supported")
