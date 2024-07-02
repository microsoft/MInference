# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os

from .configs.model2path import MODEL2PATH


class MInferenceConfig:
    MINFERENCE_ATTENTION_TYPES = [
        "minference",
        "vllm",
    ]
    STASTIC_ATTENTION_TYPES = [
        "minference_with_dense",
        "static",
        "dilated1",
        "dilated2",
        "streaming",
        "inf_llm",
        "hf",
    ]

    def __init__(
        self,
        attn_type: str = "minference",
        model_name: str = None,
        config_path: str = None,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        kv_cache_cpu_device: str = "cpu",
        use_snapkv: bool = False,
        is_search: bool = False,
        attn_kwargs: dict = {},
        **kwargs,
    ):
        super(MInferenceConfig, self).__init__()
        assert (
            attn_type in self.MINFERENCE_ATTENTION_TYPES + self.STASTIC_ATTENTION_TYPES
        ), f"The attention_type {attn_type} you specified is not supported."
        self.attn_type = attn_type
        self.config_path = self.update_config_path(config_path, model_name)
        self.model_name = model_name
        self.is_search = is_search
        self.starting_layer = starting_layer
        self.kv_cache_cpu = kv_cache_cpu
        self.kv_cache_cpu_device = kv_cache_cpu_device
        self.use_snapkv = use_snapkv
        self.attn_kwargs = attn_kwargs

    def update_config_path(self, config_path: str, model_name: str):
        if self.attn_type in self.STASTIC_ATTENTION_TYPES:
            return ""
        if config_path is not None:
            return config_path
        assert (
            model_name in MODEL2PATH
        ), f"The model {model_name} you specified is not supported. You are welcome to add it and open a PR :)"
        return MODEL2PATH[model_name]
