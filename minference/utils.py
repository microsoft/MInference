# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch


def patch_glm_4_1m(model):
    # Support THUDM/glm-4-9b-chat-1m
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
        torch.cuda.empty_cache()
    return model
