from .snap_kv import *
from .quest import *

class SnapKVCache(Cache):
    def __init__(self, config):
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.window_size = config.window_size if hasattr(config, "window_size") else 32
        self.max_capacity_prompt = config.max_capacity_prompt if hasattr(config, "max_capacity_prompt") else 4096
        self.kernel_size = config.kernel_size if hasattr(config, "kernel_size") else 5
        self.pooling = config.pooling if hasattr(config, "pooling") else "avgpool"

        self.kv_clusters = []

    def update(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer_idx,
        num_key_value_groups,
        cache_kwargs
    ):
        # if prefill, then compress
        # if decode, then update
        
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        prefilling = False
        if len(self.kv_clusters) == layer_idx:
            # prefilling
            prefilling = True
            kv_cluster = SnapKVCluster(
                self.window_size, self.max_capacity_prompt,
                self.kernel_size, self.pooling
            )
            self.kv_clusters.append(kv_cluster)

            key_compress, value_compress = self.kv_clusters[layer_idx].update_kv(
                key_states, query_states, value_states, attention_mask, num_key_value_groups
            )

        if len(self.key_cache) == layer_idx:
            self.key_cache.append(key_compress)
            self.value_cache.append(value_compress)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

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

method_to_cache_obj = {
    'snapkv': SnapKVCache,
}