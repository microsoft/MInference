# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import types

import torch
from transformers.models.llama.modeling_llama import *


class GlmRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb_glm4(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    # Keep half for later concatenation
    q, q_pass = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    k, k_pass = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]

    # Apply rotary embeddings on the first half
    q_embed = (q * cos.to(q.device)) + (rotate_half(q) * sin.to(q.device))
    k_embed = (k * cos.to(q.device)) + (rotate_half(k) * sin.to(q.device))

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def patch_glm_4_1m(model):
    # Support THUDM/glm-4-9b-chat-1m

    @torch.no_grad()
    def rope_forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

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
                # model._modules[name].rotary_emb.forward = types.MethodType(rope_forward, model._modules[name].rotary_emb)
                model._modules[name].rotary_emb = GlmRotaryEmbedding(
                    model._modules[name].rotary_emb.dim, model.config.seq_length
                )
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

    if model.__class__.__name__ == "ChatGLMForConditionalGeneration":
        model.model = model.transformer
        del model.transformer
        model.model.embed_tokens = model.model.embedding.word_embeddings
        del model.model.embedding
        model.lm_head = model.model.output_layer
        del model.model.output_layer
        model.model.norm = model.model.encoder.final_layernorm
        del model.model.encoder.final_layernorm
        model.model.layers = model.model.encoder.layers
        model.model.rotary_emb = model.model.rotary_pos_emb
        del model.model.encoder
        for layer_idx in range(0, len(model.model.layers)):
            model.model.layers[layer_idx].self_attn = model.model.layers[
                layer_idx
            ].self_attention
            del model.model.layers[layer_idx].self_attention
            model.model.layers[layer_idx].self_attn.qkv_proj = model.model.layers[
                layer_idx
            ].self_attn.query_key_value
            del model.model.layers[layer_idx].self_attn.query_key_value
            model.model.layers[layer_idx].self_attn.o_proj = model.model.layers[
                layer_idx
            ].self_attn.dense
            del model.model.layers[layer_idx].self_attn.dense
            model.model.layers[
                layer_idx
            ].self_attn.rotary_emb = model.model.rotary_pos_emb
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
        model.model.gradient_checkpointing = False
        model.config.pretraining_tp = 1
        patch_forward(model)
        torch.cuda.empty_cache()

    return model
