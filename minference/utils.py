# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import functools
import inspect
import types
from typing import Optional

import torch
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.glm.modeling_glm import GlmMLP, GlmRotaryEmbedding
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    logger,
)


def update_kwargs(
    self,
    outputs,
    model_kwargs,
    is_encoder_decoder: bool = False,
    num_new_tokens: int = 1,
):
    # update past_key_values keeping its naming used in model code
    cache_name, cache = self._extract_past_from_model_output(outputs)
    model_kwargs[cache_name] = cache
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat(
            [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
        )

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [
                    decoder_attention_mask,
                    decoder_attention_mask.new_ones(
                        (decoder_attention_mask.shape[0], 1)
                    ),
                ],
                dim=-1,
            )

    if model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = (
            model_kwargs["cache_position"][-1:] + num_new_tokens
        )
    else:
        past_positions = model_kwargs.pop("cache_position")
        new_positions = torch.arange(
            past_positions[-1] + 1,
            past_positions[-1] + num_new_tokens + 1,
            dtype=past_positions.dtype,
        ).to(past_positions.device)
        model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
    return model_kwargs


def causal_model_forward(original_forward):
    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        if kwargs.get("num_logits_to_keep", None) == 1:
            kwargs["return_last_logit"] = True
            kwargs.pop("num_logits_to_keep")
        if kwargs.get("position_ids", None) is None:
            kv_cache = kwargs.get("past_key_values")
            input_ids = kwargs.get("input_ids")
            past_seen_tokens = kv_cache.get_seq_length() if kv_cache is not None else 0
            pos_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
            pos_ids = pos_ids.unsqueeze(0)
            kwargs["position_ids"] = pos_ids
            kv_cache.pos_ids = pos_ids
        return original_forward(*args, **kwargs)

    return new_forward


def prepare_input(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    model_inputs = {}
    if self._supports_cache_class:
        model_inputs["cache_position"] = cache_position
    elif cache_position is None:
        past_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        )

    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        if (
            inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]
        ):  # Exception 1 or Exception 3
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif (
            input_ids.shape[1] != cache_position.shape[0]
        ):  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]

    input_ids_key = (
        "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    )
    if not self.config.is_encoder_decoder:
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs[input_ids_key] = input_ids.clone(
                memory_format=torch.contiguous_format
            )
            model_inputs["inputs_embeds"] = None
    else:
        model_inputs[input_ids_key] = input_ids.clone(
            memory_format=torch.contiguous_format
        )

    if (
        attention_mask is not None
        and kwargs.get("position_ids") is None
        and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        kwargs[
            "position_ids"
        ] = position_ids  # placed in kwargs for further processing (see below)

    for model_input_name in ["position_ids", "token_type_ids"]:
        model_input = kwargs.get(model_input_name)
        if model_input is not None:
            if past_key_values:
                model_input = model_input[:, -input_ids.shape[1] :]
                model_input = model_input.clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs[input_ids_key].shape
            device = model_inputs[input_ids_key].device

        base_model = getattr(self, self.base_model_prefix, None)
        if base_model is None:
            causal_mask_creation_function = getattr(
                self, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        else:
            causal_mask_creation_function = getattr(
                base_model,
                "_prepare_4d_causal_attention_mask_with_cache_position",
                None,
            )
        if causal_mask_creation_function is None:
            logger.warning_once(
                f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                "writing code, see Llama for an example implementation. If you're a user, please report this "
                "issue on GitHub."
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask

    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value

    model_inputs.pop("labels", None)
    model_inputs.pop("cache_position", None)
    model_inputs["return_last_logit"] = True

    return model_inputs


@torch.jit.script
def apply_rotary_pos_emb_glm_legacy(
    x: torch.Tensor, rope_cache: torch.Tensor
) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
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


def glm_forward(
    prefill_forward, decoding_forward, attn_forward_config, class_name: str
):
    def attn_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        mixed_x_layer = self.query_key_value(hidden_states)
        q_len = mixed_x_layer.size(1)
        bsz = mixed_x_layer.size(0)
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            num_kv_groups = (
                self.num_attention_heads_per_partition
                // self.num_multi_query_groups_per_partition
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = torch.split(
                mixed_x_layer,
                [
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )

        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [
            k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]
        ]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb_glm_legacy(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb_glm_legacy(key_layer, rotary_pos_emb)

        if kv_cache is not None:
            cache_kwargs = {
                "attn_forward_config": attn_forward_config,
                "attention_mask": attention_mask,
                "num_key_value_groups": num_kv_groups,
                "query_states": query_layer,
                "update_global_past_kv": getattr(self, "update_global_past_kv", True),
            }
            (
                key_layer,
                value_layer,
            ) = kv_cache.update(  # DynamicCache/KvcompressCache
                key_layer,
                value_layer,
                self.layer_number - 1,
                cache_kwargs,
            )

        if q_len == kv_cache.get_seq_length(self.layer_number - 1):  # prefilling
            if prefill_forward is not None:  # eg, a-shape/tri-shape/minference
                prefill_kwargs = {
                    "attention_mask": attention_mask,
                    "layer_idx": self.layer_number - 1,
                    "attn_forward_config": attn_forward_config,
                }
                attn_output = prefill_forward(  # [bsz, num_heads, q_len, head_dim]
                    query_layer,
                    key_layer,
                    value_layer,
                    prefill_kwargs,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()

            else:  # if not specified, use flash attention
                attn_output = (
                    _flash_attention_forward(  # [bsz, q_len, num_heads, head_dim]
                        query_layer.transpose(1, 2),
                        key_layer.transpose(1, 2),
                        value_layer.transpose(1, 2),
                        attention_mask,
                        q_len,
                        sliding_window=getattr(self, "sliding_window", None),
                        is_causal=True,
                    )
                )

        else:  # decoding
            # assert q_len == 1
            if decoding_forward is not None:  # eg, retr_attn
                decoding_kwargs = {
                    "layer_idx": self.layer_number - 1,
                    "attn_forward_config": attn_forward_config,
                    "position_ids": kv_cache.pos_ids,
                    "num_key_value_groups": num_kv_groups,
                }
                attn_output = decoding_forward(  # [bsz, num_heads, q_len, head_dim]
                    query_layer,
                    key_layer,
                    value_layer,
                    decoding_kwargs,
                )
                attn_output = attn_output.transpose(
                    1, 2
                )  # [bsz, q_len, num_heads, head_dim]
            else:
                attn_output = _flash_attention_forward(
                    query_layer.transpose(1, 2),
                    key_layer.transpose(1, 2),
                    value_layer.transpose(1, 2),
                    attention_mask,
                    q_len,
                    sliding_window=getattr(self, "sliding_window", None),
                    is_causal=True,
                )

        assert attn_output.size(1) == q_len
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        output = self.dense(attn_output)

        return output, kv_cache

    def transformer_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):
        if kv_caches is None:
            # if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches,
                    use_cache=use_cache,
                )
            hidden_states, kv_caches = layer_ret

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, kv_caches, all_hidden_states, all_self_attentions

    forwards = {
        "attn_forward": attn_forward,
        "transformer_forward": transformer_forward,
    }
    return forwards[class_name]


def convert_glm_4_1m(model):
    # Support THUDM/glm-4-9b-chat-1m
    from transformers.models.llama.modeling_llama import LlamaAttention

    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    except ImportError:
        pass

    def patch_forward(model):
        if model.__class__.__name__ == "ChatGLMForConditionalGeneration":
            model.forward = types.MethodType(LlamaForCausalLM.forward, model)
            model.prepare_inputs_for_generation = types.MethodType(
                LlamaForCausalLM.prepare_inputs_for_generation, model
            )
            model._update_model_kwargs_for_generation = types.MethodType(
                LlamaForCausalLM._update_model_kwargs_for_generation, model
            )

        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                patch_forward(module)
            if module.__class__.__name__ == "ChatGLMModel":
                model._modules[name].forward = types.MethodType(
                    LlamaModel.forward, model._modules[name]
                )
                rope_emb = GlmRotaryEmbedding(
                    dim=model._modules[name].rotary_pos_emb.dim,
                    max_position_embeddings=model._modules[name].config.seq_length,
                    base=model._modules[name].rotary_pos_emb.rope_ratio,
                )
                rope_emb.to(model._modules[name].device)
                model._modules[name].rotary_emb = rope_emb
                model._modules[name]._update_causal_mask = types.MethodType(
                    LlamaModel._update_causal_mask, model._modules[name]
                )
            if module.__class__.__name__ == "ChatGLMPreTrainedModel":
                model._modules[name].forward = types.MethodType(
                    LlamaPreTrainedModel.forward, model._modules[name]
                )
            if module.__class__.__name__ == "GLMBlock":
                model._modules[name].forward = types.MethodType(
                    LlamaDecoderLayer.forward, model._modules[name]
                )
            if module.__class__.__name__ == "SelfAttention":
                model._modules[name].forward = types.MethodType(
                    LlamaAttention.forward, model._modules[name]
                )
            if module.__class__.__name__ == "FlashAttention2":
                model._modules[name].forward = types.MethodType(
                    LlamaFlashAttention2.forward, model._modules[name]
                )
            if module.__class__.__name__ == "MLP":
                model._modules[name].forward = types.MethodType(
                    GlmMLP.forward, model._modules[name]
                )

    if model.__class__.__name__ == "ChatGLMForConditionalGeneration":
        model.model = model.transformer
        model.model.embed_tokens = model.model.embedding.word_embeddings
        model.lm_head = model.model.output_layer
        model.model.norm = model.model.encoder.final_layernorm
        model.model.layers = model.model.encoder.layers

        del model.model.embedding
        del model.model.output_layer
        del model.model.encoder.final_layernorm
        del model.transformer
        del model.model.encoder

        for layer_idx in range(0, len(model.model.layers)):
            model.model.layers[layer_idx].self_attn = model.model.layers[
                layer_idx
            ].self_attention
            model.model.layers[layer_idx].self_attn.qkv_proj = model.model.layers[
                layer_idx
            ].self_attn.query_key_value
            model.model.layers[layer_idx].self_attn.o_proj = model.model.layers[
                layer_idx
            ].self_attn.dense
            del model.model.layers[layer_idx].self_attn.dense
            del model.model.layers[layer_idx].self_attention
            del model.model.layers[layer_idx].self_attn.query_key_value
            model.model.layers[layer_idx].self_attn.config = model.config
            model.model.layers[layer_idx].self_attn.layer_idx = layer_idx
            config = model.config
            model.model.layers[
                layer_idx
            ].self_attn.attention_dropout = config.attention_dropout
            model.model.layers[layer_idx].self_attn.hidden_size = config.hidden_size
            model.model.layers[
                layer_idx
            ].self_attn.num_heads = config.num_attention_heads
            model.model.layers[layer_idx].self_attn.head_dim = (
                config.hidden_size // config.num_attention_heads
            )
            model.model.layers[
                layer_idx
            ].self_attn.num_key_value_heads = config.multi_query_group_num
            model.model.layers[layer_idx].self_attn.num_key_value_groups = (
                config.num_attention_heads // config.multi_query_group_num
            )
            model.model.layers[
                layer_idx
            ].self_attn.max_position_embeddings = config.seq_length
            model.model.layers[layer_idx].self_attn.rope_theta = config.rope_ratio
            model.model.layers[layer_idx].self_attn.is_causal = True
            model.model.layers[layer_idx].mlp.gate_up_proj = model.model.layers[
                layer_idx
            ].mlp.dense_h_to_4h
            model.model.layers[layer_idx].mlp.down_proj = model.model.layers[
                layer_idx
            ].mlp.dense_4h_to_h
            model.model.layers[layer_idx].mlp.activation_fn = ACT2FN["silu"]

            del model.model.layers[layer_idx].mlp.dense_h_to_4h
            del model.model.layers[layer_idx].mlp.dense_4h_to_h
        model.model.gradient_checkpointing = False
        model.config.pretraining_tp = 1
        patch_forward(model)
        torch.cuda.empty_cache()

    return model
