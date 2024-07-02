# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json

import torch
import transformers
from transformers.cache_utils import *
from transformers.models.llama.modeling_llama import *

from .modules.inf_llm import InfLLMGenerator, inf_llm_forward
from .modules.minference_forward import (
    gather_last_q_vertical_slash_topk_v4,
    gather_last_q_vertical_slash_topk_vllm,
    init_minference_parameters,
    minference_forward,
    minference_kv_cache_cpu_forward,
    minference_vllm_forward,
    minference_with_snapkv_forward,
    search_pattern,
    sum_all_diagonal_matrix,
)
from .ops.streaming_kernel import stream_llm_forward
from .utils import patch_glm_4_1m

KV_CACHE_CPU_DEVICE = "cpu"


class RotaryEmbeddingESM(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self,
        dim: int,
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
        is_glm4: bool = False,
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None
        self.is_glm4 = is_glm4
        self.dim = dim

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x, length, right, cos, sin):
        dtype = x.dtype
        if self.is_glm4:
            cos = cos[right - length : right, ...]
            return apply_rotary_pos_emb_glm4(x, cos)
        if cos.dim() == 2:
            cos = cos[right - length : right, :]
            sin = sin[right - length : right, :]
        elif cos.dim() == 3:
            cos = cos[:, right - length : right, :]
            sin = sin[:, right - length : right, :]
        elif cos.dim() == 4:
            cos = cos[:, :, right - length : right, :]
            sin = sin[:, :, right - length : right, :]

        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    def _update_cos_sin_tables(self, x, seq_dim):
        seq_len = x.size(seq_dim)
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq).float()
            if self.is_glm4:
                cache = torch.stack(
                    [torch.cos(freqs), torch.sin(freqs)], dim=-1
                ).bfloat16()
                self._cos_cached, self._sin_cached = cache, None
            else:
                emb = torch.cat((freqs, freqs), dim=-1)
                if x.dim() == 2:
                    self._cos_cached = emb.cos()
                    self._sin_cached = emb.sin()
                elif x.dim() == 3:
                    self._cos_cached = emb.cos()[None, :, :]
                    self._sin_cached = emb.sin()[None, :, :]
                elif x.dim() == 4:
                    self._cos_cached = emb.cos()[None, None, :, :]
                    self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached

    def _update_cos_sin_tables_len(self, seq_len, device, dim=None):
        if seq_len > self._seq_len_cached:
            if dim is None:
                assert self._cos_cached is not None
                dim = self._cos_cached.dim()

            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            if self.is_glm4:
                cache = torch.stack(
                    [torch.cos(freqs), torch.sin(freqs)], dim=-1
                ).bfloat16()
                self._cos_cached, self._sin_cached = cache, None
            else:
                emb = torch.cat((freqs, freqs), dim=-1)
                if dim == 2:
                    self._cos_cached = emb.cos()
                    self._sin_cached = emb.sin()
                elif dim == 3:
                    self._cos_cached = emb.cos()[None, :, :]
                    self._sin_cached = emb.sin()[None, :, :]
                elif dim == 4:
                    self._cos_cached = emb.cos()[None, None, :, :]
                    self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb_one_angle(self, x: torch.Tensor, index):
        dtype = x.dtype
        cos, sin = self._update_cos_sin_tables_len(max(index, x.shape[-2]), x.device)
        if self.is_glm4:
            return apply_rotary_pos_emb_glm4(x, cos)
        if cos.dim() == 2:
            cos = cos[index - 1 : index, :]
            sin = sin[index - 1 : index, :]
        elif cos.dim() == 3:
            cos = cos[:, index - 1 : index, :]
            sin = sin[:, index - 1 : index, :]
        elif cos.dim() == 4:
            cos = cos[:, :, index - 1 : index, :]
            sin = sin[:, :, index - 1 : index, :]

        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_dim=-2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dim=seq_dim
        )
        return (
            self.apply_rotary_pos_emb(
                q, q.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached
            ),
            self.apply_rotary_pos_emb(
                k, k.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached
            ),
        )


@torch.jit.script
def apply_rotary_pos_emb_glm4(
    x: torch.Tensor, rope_cache: torch.Tensor
) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    # import ipdb;ipdb.set_trace()
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


ATTN_FORWRAD = {
    "streaming": stream_llm_forward,
    "minference": minference_forward,
    "inf_llm": inf_llm_forward,
}


def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions

        # for GLM-4
        if "q_proj" not in self.__dict__["_modules"]:
            query_pos = self.num_heads * self.head_dim
            key_value_pos = query_pos // self.num_key_value_groups
            self.q_proj = torch.nn.Linear(
                hidden_states.size(-1),
                query_pos,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self.k_proj = torch.nn.Linear(
                hidden_states.size(-1),
                key_value_pos,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self.v_proj = torch.nn.Linear(
                hidden_states.size(-1),
                key_value_pos,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            self.q_proj.weight.copy_(self.qkv_proj.weight[:query_pos, :])
            self.k_proj.weight.copy_(
                self.qkv_proj.weight[query_pos : query_pos + key_value_pos, :]
            )
            self.v_proj.weight.copy_(
                self.qkv_proj.weight[query_pos + key_value_pos :, :]
            )

            self.q_proj.bias.copy_(self.qkv_proj.bias[:query_pos])
            self.k_proj.bias.copy_(
                self.qkv_proj.bias[query_pos : query_pos + key_value_pos]
            )
            self.v_proj.bias.copy_(self.qkv_proj.bias[query_pos + key_value_pos :])

            del self.qkv_proj

        ret = forward(
            self,
            hidden_states,
            hidden_states,
            position_ids,
            use_cache,
            past_key_value,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None

        return o, None, pkv

    return hf_forward


def hf_437_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, transformers.cache_utils.Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    **kwargs,
):
    # With static cache, the `past_key_values` is None
    # TODO joao: standardize interface for the different Cache classes and remove of this if
    has_static_cache = False
    if past_key_values is None:
        past_key_values = getattr(
            getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None
        )
        has_static_cache = past_key_values is not None

    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, transformers.cache_utils.Cache):
            past_length = (
                cache_position[0]
                if cache_position is not None
                else past_key_values.get_seq_length()
            )
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = (
                past_length
                if max_cache_length is None
                else torch.min(max_cache_length, past_length)
            )
        # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            cache_length = past_length = cache_position[0]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        # TODO: use `next_tokens` directly instead.
        model_inputs = {"input_ids": input_ids.contiguous()}

    input_length = (
        position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    )
    if cache_position is None:
        cache_position = torch.arange(
            past_length, past_length + input_length, device=input_ids.device
        )
    else:
        cache_position = cache_position[-input_length:]

    if has_static_cache:
        past_key_values = None

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def prepare_inputs_for_generation_snapkv(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    if past_key_values is None:  # [SnapKV]
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            # max_cache_length = None
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


def forward_llama_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    residual = hidden_states.clone()
    batch, seq_len, embed_dim = hidden_states.shape

    for start_idx in range(0, seq_len, 32000):
        end_idx = min(seq_len, start_idx + 32000)
        hidden_states[:, start_idx:end_idx, :] = self.input_layernorm(
            hidden_states[:, start_idx:end_idx, :]
        )

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    for start_idx in range(0, seq_len, 32000):
        end_idx = min(seq_len, start_idx + 32000)
        part_hidden_states = hidden_states[:, start_idx:end_idx, :].clone()
        part_hidden_states = self.post_attention_layernorm(part_hidden_states)
        part_hidden_states = self.mlp(part_hidden_states)
        hidden_states[:, start_idx:end_idx, :] += part_hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def forward_llama_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    batch, seq_len, embed_dim = hidden_states.shape
    for start_idx in range(0, seq_len, 32000):
        end_idx = min(seq_len, start_idx + 32000)
        hidden_states[:, start_idx:end_idx, :] = self.norm(
            hidden_states[:, start_idx:end_idx, :]
        )

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def forward_llama_for_causal_lm(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    # assert labels is not None
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    torch.cuda.empty_cache()

    hidden_states = outputs[0]
    if labels is not None:
        loss_fct = CrossEntropyLoss(reduction="sum")
        valid_seq_len = input_ids.shape[-1] - 1
        valid_seq_len_slide_win = torch.sum(labels[:, 1:] >= 0).item()
        # print("valid_seq_len_slide_win", valid_seq_len)
        loss = 0.0

        for start_idx in range(0, valid_seq_len, 32000):
            end_idx = min(start_idx + 32000, valid_seq_len)
            shift_logits = self.lm_head(
                hidden_states[..., start_idx:end_idx, :]
            ).float()
            shift_labels = labels[..., start_idx + 1 : end_idx + 1].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss += loss_fct(shift_logits, shift_labels)

        loss /= valid_seq_len_slide_win
        logits = None
    else:
        if self.config.to_dict().get("is_ppl", False):
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[:, -1:]).float()
        loss = None

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
    )


def minference_patch(model, config):
    from transformers import LlamaForCausalLM

    if config.kv_cache_cpu:
        global KV_CACHE_CPU_DEVICE
        KV_CACHE_CPU_DEVICE = config.kv_cache_cpu_device
        model.config.kv_cache_cpu_device = config.kv_cache_cpu_device
        return minference_patch_kv_cache_cpu(model)
    if config.use_snapkv:
        return minference_patch_with_snapkv(model)

    model = patch_glm_4_1m(model)

    Attention = model.model.layers[0].self_attn.__class__
    Model = model.model.__class__
    DecoderLayer = model.model.layers[0].__class__

    forward = minference_forward()

    def update_module(m):
        if isinstance(m, Attention):
            m.init_minference_parameters = init_minference_parameters.__get__(
                m, Attention
            )
            m.gather_last_q_vertical_slash_topk_v4 = (
                gather_last_q_vertical_slash_topk_v4.__get__(m, Attention)
            )
            m.forward = forward.__get__(m, Attention)
        if isinstance(m, DecoderLayer):
            m.forward = forward_llama_decoder_layer.__get__(m, DecoderLayer)

    model.apply(update_module)
    model.prepare_inputs_for_generation = hf_437_prepare_inputs_for_generation.__get__(
        model, model.__class__
    )
    model.model._use_sdpa = False

    model.model._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask_inference.__get__(
            model.model, model.model.__class__
        )
    )
    model.model.forward = forward_llama_model.__get__(
        model.model, model.model.__class__
    )
    model.forward = forward_llama_for_causal_lm.__get__(model, model.__class__)

    print("Patched model for minference..")
    return model


def minference_patch_kv_cache_cpu(model):
    from transformers import LlamaForCausalLM

    transformers.cache_utils.DynamicCache.update = cpu_cache_update
    transformers.cache_utils.DynamicCache.get = cpu_cache_get

    model = patch_glm_4_1m(model)

    Attention = model.model.layers[0].self_attn.__class__
    Model = model.model.__class__
    DecoderLayer = model.model.layers[0].__class__

    forward = minference_kv_cache_cpu_forward()

    def update_module(m):
        if isinstance(m, Attention):
            m.init_minference_parameters = init_minference_parameters.__get__(
                m, Attention
            )
            m.gather_last_q_vertical_slash_topk_v4 = (
                gather_last_q_vertical_slash_topk_v4.__get__(m, Attention)
            )
            m.forward = forward.__get__(m, Attention)
        if isinstance(m, DecoderLayer):
            m.forward = forward_llama_decoder_layer.__get__(m, DecoderLayer)

    model.apply(update_module)
    model.prepare_inputs_for_generation = hf_437_prepare_inputs_for_generation.__get__(
        model, model.__class__
    )
    model.model._use_sdpa = False

    model.model._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask_inference.__get__(
            model.model, model.model.__class__
        )
    )
    model.model.forward = forward_llama_model.__get__(
        model.model, model.model.__class__
    )
    model.forward = forward_llama_for_causal_lm.__get__(model, model.__class__)

    print("Patched model for MInference load KV Cache to CPU.")
    return model


def minference_patch_with_snapkv(model):
    from transformers import LlamaForCausalLM

    model = patch_glm_4_1m(model)

    Attention = model.model.layers[0].self_attn.__class__
    Model = model.model.__class__
    DecoderLayer = model.model.layers[0].__class__

    forward = minference_with_snapkv_forward()

    def update_module(m):
        if isinstance(m, Attention):
            m.init_minference_parameters = init_minference_parameters.__get__(
                m, Attention
            )
            m.gather_last_q_vertical_slash_topk_v4 = (
                gather_last_q_vertical_slash_topk_v4.__get__(m, Attention)
            )
            m.forward = forward.__get__(m, Attention)
        if isinstance(m, DecoderLayer):
            m.forward = forward_llama_decoder_layer.__get__(m, DecoderLayer)

    model.apply(update_module)
    model.prepare_inputs_for_generation = prepare_inputs_for_generation_snapkv.__get__(
        model, model.__class__
    )
    model.model._use_sdpa = False

    model.model._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask_inference.__get__(
            model.model, model.model.__class__
        )
    )
    model.model.forward = forward_llama_model.__get__(
        model.model, model.model.__class__
    )
    model.forward = forward_llama_for_causal_lm.__get__(model, model.__class__)

    print("Patched model for minference with SanpKV..")
    return model


def llama_model_forward_vllm(
    self,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.get_input_embeddings(input_ids)
    residual = None
    for i in range(len(self.layers)):
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions,
            hidden_states,
            kv_caches[i],
            attn_metadata,
            residual,
            layer_idx=i,
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def llama_layer_forward_vllm(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    residual: Optional[torch.Tensor],
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        layer_idx=layer_idx,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


def llama_attn_forward_vllm(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    layer_idx: int,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata, self.kv_scale, layer_idx)
    output, _ = self.o_proj(attn_output)
    return output


def vllm_attn_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    attn_metadata,
    kv_scale: float = 1.0,
    layer_idx: int = 0,
) -> torch.Tensor:
    return self.impl.forward(
        query, key, value, kv_cache, attn_metadata, kv_scale, layer_idx
    )


def minference_patch_vllm(
    llm,
    config_file,
):
    from vllm.attention import Attention
    from vllm.model_executor.models.llama import (
        LlamaAttention,
        LlamaDecoderLayer,
        LlamaForCausalLM,
        LlamaModel,
    )

    config = json.load(open(config_file))
    attn_forward = minference_vllm_forward(config)

    def update_module(m):
        if isinstance(m, Attention):
            m.forward = vllm_attn_forward.__get__(m, Attention)

            m = m.impl
            m_cls = m.__class__
            m.gather_last_q_vertical_slash_topk_vllm = (
                gather_last_q_vertical_slash_topk_vllm.__get__(m, m_cls)
            )
            m.forward = attn_forward.__get__(m, m_cls)
        if isinstance(m, LlamaDecoderLayer):
            m.forward = llama_layer_forward_vllm.__get__(m, LlamaDecoderLayer)
        if isinstance(m, LlamaModel):
            m.forward = llama_model_forward_vllm.__get__(m, LlamaModel)
        if isinstance(m, LlamaAttention):
            m.forward = llama_attn_forward_vllm.__get__(m, LlamaAttention)

    llm.llm_engine.model_executor.driver_worker.model_runner.model.apply(update_module)

    print("Patched model for minference with vLLM..")
    return llm


def patch_hf(
    model,
    attn_type: str = "inf_llm",
    attn_kwargs: dict = {},
    base=None,
    distance_scale=None,
    **kwargs,
):
    attn_kwargs.update(kwargs)
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM
    from transformers.models.llama.modeling_llama import (
        BaseModelOutputWithPast,
        LlamaAttention,
        LlamaModel,
    )
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralModel,
    )
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        *args,
        **kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        if use_cache:
            pkv = tuple()

        else:
            pkv = None

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias,
                past_key_value=(
                    past_key_values[i] if past_key_values is not None else None
                ),
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                _cache = layer_outputs[2 if output_attentions else 1]
                pkv = pkv + (_cache,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # hidden_states = self.norm(hidden_states)
        for start_idx in range(0, hidden_states.size(1), 32000):
            end_idx = min(hidden_states.size(1), start_idx + 32000)
            hidden_states[:, start_idx:end_idx, :] = self.norm(
                hidden_states[:, start_idx:end_idx, :]
            )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, pkv, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    forward = huggingface_forward(ATTN_FORWRAD[attn_type](**attn_kwargs))
    model = patch_glm_4_1m(model)

    is_glm4 = False
    if isinstance(model, LlamaForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, MistralForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, Qwen2ForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif model.__class__.__name__ == "MiniCPMForCausalLM":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif model.__class__.__name__ == "Phi3ForCausalLM":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif model.__class__.__name__ == "ChatGLMForConditionalGeneration":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
        base = model.model.layers[0].self_attn.rotary_emb.rope_ratio * 10000
        is_glm4 = True
    else:
        raise ValueError("Only supports llama, mistral and qwen2 models.")

    hf_rope = model.model.layers[0].self_attn.rotary_emb
    base = base if base is not None else hf_rope.base
    distance_scale = distance_scale if distance_scale is not None else 1.0
    rope = RotaryEmbeddingESM(hf_rope.dim, base, distance_scale, is_glm4=is_glm4)
    model.model.position_bias = rope
    model.model.hf_position_bias = hf_rope
    DecoderLayer = model.model.layers[0].__class__

    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)
        if isinstance(m, DecoderLayer):
            m.forward = forward_llama_decoder_layer.__get__(m, DecoderLayer)

    model.apply(set_forward)

    model._old_prepare_inputs_for_generation = model.prepare_inputs_for_generation
    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(
        model, model.__class__
    )
    model.model._old_forward = model.model.forward
    model.model.forward = model_forward.__get__(model.model, Model)
    model.forward = forward_llama_for_causal_lm.__get__(model, model.__class__)

    if attn_type == "inf_llm":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model.config._name_or_path, trust_remote_code=True
        )
        model = InfLLMGenerator(model, tokenizer)

    print("Patched model ...")
    return model


def fp8_cache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`torch.Tensor`):
            The new key states to cache.
        value_states (`torch.Tensor`):
            The new value states to cache.
        layer_idx (`int`):
            The index of the layer to cache the states for.
        cache_kwargs (`Dict[str, Any]`, `optional`):
            Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

    Return:
        A tuple containing the updated key and value states.
    """
    # Update the number of seen tokens
    if layer_idx == 0:
        self.seen_tokens += key_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states.to(torch.float8_e5m2))
        self.value_cache.append(value_states.to(torch.float8_e5m2))
    else:
        self.key_cache[layer_idx] = torch.cat(
            [self.key_cache[layer_idx], key_states.to(torch.float8_e5m2)], dim=-2
        )
        self.value_cache[layer_idx] = torch.cat(
            [self.value_cache[layer_idx], value_states.to(torch.float8_e5m2)], dim=-2
        )

    return self.key_cache[layer_idx].to(key_states.dtype), self.value_cache[
        layer_idx
    ].to(key_states.dtype)


def cpu_cache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if layer_idx == 0:
        if "_seen_tokens" in self.__dict__:
            self._seen_tokens += key_states.shape[-2]
        else:
            self.seen_tokens += key_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states.to(KV_CACHE_CPU_DEVICE))
        self.value_cache.append(value_states.to(KV_CACHE_CPU_DEVICE))
    else:
        self.key_cache[layer_idx] = torch.cat(
            [self.key_cache[layer_idx], key_states.to(KV_CACHE_CPU_DEVICE)], dim=-2
        )
        self.value_cache[layer_idx] = torch.cat(
            [self.value_cache[layer_idx], value_states.to(KV_CACHE_CPU_DEVICE)], dim=-2
        )


def cpu_cache_get(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if layer_idx == 0:
        if "_seen_tokens" in self.__dict__:
            self._seen_tokens += key_states.shape[-2]
        else:
            self.seen_tokens += key_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        return key_states, value_states
    elif KV_CACHE_CPU_DEVICE == "cpu":
        key_states = torch.cat(
            [self.key_cache[layer_idx][:, head_idx : head_idx + 1].cuda(), key_states],
            dim=-2,
        )
        value_states = torch.cat(
            [
                self.value_cache[layer_idx][:, head_idx : head_idx + 1].cuda(),
                value_states,
            ],
            dim=-2,
        )
        return key_states, value_states
    key_states = torch.cat(
        [self.key_cache[layer_idx][:, head_idx : head_idx + 1], key_states],
        dim=-2,
    )
    value_states = torch.cat(
        [
            self.value_cache[layer_idx][:, head_idx : head_idx + 1],
            value_states,
        ],
        dim=-2,
    )
    return key_states, value_states
