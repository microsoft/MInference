# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import *

from ..utils import apply_rotary_pos_emb_glm4
from .pyramid import PyramidKVCluster
from .quest import *
from .snap_kv import SnapKVCluster, StreamingLLMKVCluster


def prepare_inputs_for_generation_kvcompression(
    method: str, config, original_prepare_inputs_for_generation
):
    def new_prepare_inputs_for_generation(self, *args, **kwargs):
        outputs = original_prepare_inputs_for_generation(*args, **kwargs)
        use_cache = kwargs.get("use_cache", True)
        if use_cache and not isinstance(
            outputs["past_key_values"], method_to_cache_obj[method]
        ):
            cache_obj: Cache = method_to_cache_obj[method]
            config.num_layers = self.config.num_hidden_layers
            outputs["past_key_values"] = cache_obj(config)
        if self._supports_num_logits_to_keep():
            outputs["num_logits_to_keep"] = 1
        return outputs

    return new_prepare_inputs_for_generation


def snapkv_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    if "q_proj" in self.__dict__["_modules"]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    else:
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        key_value_pos = query_pos // self.num_key_value_groups
        query_states, key_states, value_states = torch.split(
            qkv, [query_pos, key_value_pos, key_value_pos], -1
        )

    # [bsz, q_len, num_heads, head_dim]
    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    if query_states.shape[-1] != cos.shape[-1]:  # glm-4 rope
        query_states, key_states = apply_rotary_pos_emb_glm4(
            query_states, key_states, cos, sin
        )
    else:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(  # kvcompress
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.layer_idx,
            self.num_key_value_groups,
            cache_kwargs,
        )

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


class SnapKVCache(Cache):
    def __init__(self, config):
        super().__init__()
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.window_size = config.window_size if hasattr(config, "window_size") else 32
        self.max_capacity_prompt = (
            config.max_capacity_prompt
            if hasattr(config, "max_capacity_prompt")
            else 4096
        )
        self.kernel_size = config.kernel_size if hasattr(config, "kernel_size") else 5
        self.pooling = config.pooling if hasattr(config, "pooling") else "avgpool"

        self.kv_clusters = []
        self.kv_cluster_class = SnapKVCluster

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        # if prefill, then compress
        # if decode, then update

        # [bsz, num_heads, q_len, head_dim]

        query_states = cache_kwargs["query_states"]
        attention_mask = cache_kwargs["attention_mask"]
        num_key_value_groups = cache_kwargs["num_key_value_groups"]

        if key_states.size(1) != query_states.size(1): # GQA
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        prefilling = False
        if len(self.kv_clusters) == layer_idx:
            # prefilling
            prefilling = True
            kv_cluster = self.kv_cluster_class(
                self.window_size,
                self.max_capacity_prompt,
                self.kernel_size,
                self.pooling,
            )
            self.kv_clusters.append(kv_cluster)

            key_compress, value_compress = self.kv_clusters[layer_idx].update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
            )

        if len(self.key_cache) == layer_idx:
            self.key_cache.append(key_compress)
            self.value_cache.append(value_compress)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        torch.cuda.empty_cache()
        if prefilling:
            return key_states, value_states
        else:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def to_legacy_cache(self):
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values):
        cache = cls()
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx]
            cache.update(key_states, value_states, layer_idx)
        return cache


class PyramidKVCache(SnapKVCache):
    def __init__(self, config):
        super().__init__(config)

        self.num_layers = config.num_layers
        self.kv_cluster_class = PyramidKVCluster

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        # if prefill, then compress
        # if decode, then update

        query_states = cache_kwargs["query_states"]
        attention_mask = cache_kwargs["attention_mask"]
        num_key_value_groups = cache_kwargs["num_key_value_groups"]

        if key_states.size(1) != query_states.size(1): # GQA
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        prefilling = False
        if len(self.kv_clusters) == layer_idx:
            # prefilling
            prefilling = True
            kv_cluster = self.kv_cluster_class(
                self.num_layers,
                self.window_size,
                self.max_capacity_prompt,
                self.kernel_size,
                self.pooling,
                layer_idx=layer_idx,
            )
            self.kv_clusters.append(kv_cluster)

            key_compress, value_compress = self.kv_clusters[layer_idx].update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
            )

        if len(self.key_cache) == layer_idx:
            self.key_cache.append(key_compress)
            self.value_cache.append(value_compress)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        if prefilling:
            return key_states, value_states
        else:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]


class StreamingLLMKVCache(SnapKVCache):
    def __init__(self, config):
        if not hasattr(config, "window_size"):
            config.max_capacity_prompt = (
                config.max_capacity_prompt
                if hasattr(config, "max_capacity_prompt")
                else 4096
            )
            config.window_size = config.max_capacity_prompt - 128
        super().__init__(config)
        self.kv_cluster_class = StreamingLLMKVCluster


method_to_cache_obj = {
    "": DynamicCache,
    "dense": DynamicCache,
    "snapkv": SnapKVCache,
    "pyramidkv": PyramidKVCache,
    "streaming": StreamingLLMKVCache,
    "quest": DynamicCache,
}
