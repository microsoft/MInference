# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import functools
from typing import List

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.glm.modeling_glm import (
    apply_rotary_pos_emb as apply_rotary_pos_emb_glm4,
)

from .kivi import KiviCache
from .pyramidkv import PyramidKVCluster
from .quest import *
from .retr_attn import RetrAttnCache
from .snapkv import SnapKVCluster, StreamingLLMKVCluster


def prepare_inputs_for_generation_kvcompression(
    method: str, config, original_prepare_inputs_for_generation
):
    @functools.wraps(original_prepare_inputs_for_generation)
    def new_prepare_inputs_for_generation(self, *args, **kwargs):
        outputs = original_prepare_inputs_for_generation(*args, **kwargs)
        use_cache = kwargs.get("use_cache", True)
        if use_cache and not isinstance(
            outputs.get("past_key_values", None), method_to_cache_obj[method]
        ):
            cache_obj: Cache = method_to_cache_obj[method]
            config.num_layers = self.config.num_hidden_layers
            outputs["past_key_values"] = cache_obj(config)
        if (
            hasattr(self, "_supports_num_logits_to_keep")
            and self._supports_num_logits_to_keep()
        ):
            outputs["num_logits_to_keep"] = 1
        if (
            hasattr(self, "_supports_logits_to_keep")
            and self._supports_logits_to_keep()
        ):
            outputs["logits_to_keep"] = 1
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
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "query_states": query_states,
            "attention_mask": attention_mask,
            "num_key_value_groups": self.num_key_value_groups,
        }
        key_states, value_states = past_key_value.update(  # kvcompress
            key_states,
            value_states,
            self.layer_idx,
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


class BaseKVCache(Cache):
    def __init__(self):
        super().__init__()
        # Used in `generate` to keep tally of how many tokens the cache has seen
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.kv_clusters = {}
        self.kv_cluster_granularity = "layer"

        self.temp_key_cache = []
        self.temp_value_cache = []

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        raise NotImplementedError(
            "Make sure to implement `get_kv_cluster_class_config` in a subclass."
        )

    def get_kv_cluster_class(self, layer_idx: int, head_idx=None):
        cluster_name, cluster_class, cluster_config = self.get_kv_cluster_class_config(
            layer_idx, head_idx
        )
        if cluster_name not in self.kv_clusters:
            self.kv_clusters[cluster_name] = cluster_class(**cluster_config)
        return self.kv_clusters[cluster_name]

    def compresssed_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        layer_idx: int,
    ):
        if self.kv_cluster_granularity == "layer":
            kv_cluster = self.get_kv_cluster_class(layer_idx)

            key_compress, value_compress = kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
            )
            self.key_cache.append(key_compress)
            self.value_cache.append(value_compress)
        else:
            assert (
                False
            ), f"kv_cluster_granularity {self.kv_cluster_granularity} not supported"

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        # if prefill, then compress; if decode, then update
        # [bsz, num_heads, q_len, head_dim]

        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        query_states = cache_kwargs["query_states"]
        attention_mask = cache_kwargs["attention_mask"]
        num_key_value_groups = cache_kwargs["num_key_value_groups"]

        if key_states.size(1) != query_states.size(1):  # GQA
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        q_len = query_states.shape[-2]
        initializing_kv_cluster = False
        if (
            len(self.key_cache) == layer_idx
        ):  # initialize kv_cluster, ie, the first query/context
            initializing_kv_cluster = True
            self.compresssed_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
                layer_idx,
            )
        else:  # the follow up queries/contexts
            if update_global_past_kv:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            else:  # add KVs to temp_kv_cache
                if len(self.temp_key_cache) == layer_idx:
                    self.temp_key_cache.append(key_states)
                    self.temp_value_cache.append(value_states)
                else:
                    self.temp_key_cache[layer_idx] = torch.cat(
                        [self.temp_key_cache[layer_idx], key_states], dim=-2
                    )
                    self.temp_value_cache[layer_idx] = torch.cat(
                        [self.temp_value_cache[layer_idx], value_states], dim=-2
                    )

        torch.cuda.empty_cache()
        if not initializing_kv_cluster:  # return the compressed KV cache
            if self.temp_key_cache:  # concat global past_kv and temp_kv_cache
                key_states = torch.cat(
                    [self.key_cache[layer_idx], self.temp_key_cache[layer_idx]], dim=-2
                )
                value_states = torch.cat(
                    [self.value_cache[layer_idx], self.temp_value_cache[layer_idx]],
                    dim=-2,
                )
            else:
                key_states = self.key_cache[layer_idx]
                value_states = self.value_cache[layer_idx]
        key_states = repeat_kv(key_states, query_states.size(1) // key_states.size(1))
        value_states = repeat_kv(
            value_states, query_states.size(1) // value_states.size(1)
        )
        return key_states, value_states

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

    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[-1].shape[
                -2
            ]  # seq_len of temp_kv_cache
        self.temp_key_cache = []
        self.temp_value_cache = []


class SnapKVCache(BaseKVCache):
    def __init__(self, config):
        super().__init__()
        self.window_size = config.attn_kwargs.get("window_size", 32)
        self.max_capacity_prompt = config.attn_kwargs.get("max_capacity_prompt", 4096)
        self.kernel_size = config.attn_kwargs.get("kernel_size", 5)
        self.pooling = config.attn_kwargs.get("pooling", "avgpool")

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "window_size": self.window_size,
            "max_capacity_prompt": self.max_capacity_prompt,
            "kernel_size": self.kernel_size,
            "pooling": self.pooling,
        }
        cluster_name = ",".join(["snapv"] + [str(i) for i in cluster_config.values()])
        return cluster_name, SnapKVCluster, cluster_config


class PyramidKVCache(SnapKVCache):
    def __init__(self, config):
        super().__init__(config)
        self.num_layers = config.num_layers

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "num_hidden_layers": self.num_layers,
            "window_size": self.window_size,
            "max_capacity_prompt": self.max_capacity_prompt,
            "kernel_size": self.kernel_size,
            "pooling": self.pooling,
            "layer_idx": layer_idx,
        }
        cluster_name = ",".join(
            ["pyramidv"] + [str(i) for i in cluster_config.values()]
        )
        return cluster_name, PyramidKVCluster, cluster_config


class StreamingLLMKVCache(SnapKVCache):
    def __init__(self, config):
        n_local = config.attn_kwargs.get("n_local", 3968)
        n_init = config.attn_kwargs.get("n_init", 128)
        config.attn_kwargs["window_size"] = n_local
        config.attn_kwargs["max_capacity_prompt"] = n_local + n_init
        super().__init__(config)

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "window_size": self.window_size,
            "max_capacity_prompt": self.max_capacity_prompt,
        }
        cluster_name = ",".join(
            ["streamingllm"] + [str(i) for i in cluster_config.values()]
        )
        return cluster_name, StreamingLLMKVCluster, cluster_config


class DynamicCacheWithRepeat(DynamicCache):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_key_cache = []
        self.temp_value_cache = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if update_global_past_kv:  # add KVs to global past_kv
            assert len(self.temp_key_cache) == 0 and len(self.temp_value_cache) == 0, (
                "when you updating global past_kv, make sure the temp_kv_cache is empty. "
                "User past_key_values.clear_temp_kv_cache() to empty the temp_kv_cache"
            )

            # prefilling
            if len(self.key_cache) == layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:  # decoding
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
        else:  # add KVs to temp_kv_cache, this is used when you have a common context but different query, the KVs of the query will be added to temp_kv_cache, and will be cleaned in the next query
            if len(self.temp_key_cache) == layer_idx:
                self.temp_key_cache.append(key_states)
                self.temp_value_cache.append(value_states)
            else:  # decoding
                self.temp_key_cache[layer_idx] = torch.cat(
                    [self.temp_key_cache[layer_idx], key_states], dim=-2
                )
                self.temp_value_cache[layer_idx] = torch.cat(
                    [self.temp_value_cache[layer_idx], value_states], dim=-2
                )

        if self.temp_key_cache:  # concat global past_kv and temp_kv_cache
            key_states, value_states = torch.cat(
                [self.key_cache[layer_idx], self.temp_key_cache[layer_idx]], dim=-2
            ), torch.cat(
                [self.value_cache[layer_idx], self.temp_value_cache[layer_idx]], dim=-2
            )
        else:
            key_states, value_states = (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
            )

        # repeat kv if needed
        query_states = cache_kwargs.get("query_states", None)
        if query_states is not None:
            key_states = repeat_kv(
                key_states, query_states.size(1) // key_states.size(1)
            )
            value_states = repeat_kv(
                value_states, query_states.size(1) // value_states.size(1)
            )
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[-1].shape[
                -2
            ]  # seq_len of temp_kv_cache
        self.temp_key_cache = []
        self.temp_value_cache = []


method_to_cache_obj = {
    "": DynamicCacheWithRepeat,
    "dense": DynamicCacheWithRepeat,
    "snapkv": SnapKVCache,
    "pyramidkv": PyramidKVCache,
    "streamingllm": StreamingLLMKVCache,
    "quest": DynamicCacheWithRepeat,
    "retr_attn": RetrAttnCache,
    "kivi": KiviCache,
}
