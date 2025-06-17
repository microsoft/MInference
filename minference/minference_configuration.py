# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from .configs.model2path import MODEL2PATH


class MInferenceConfig:
    MINFERENCE_ATTENTION_TYPES = [
        "minference",
        "vllm_minference",
    ]
    OTHER_ATTENTION_TYPES = [
        # original implement
        "hf",
        "vllm",
        # our custom implement
        "dense",
        "static",  # minference w/ static
        "dilated1",
        "dilated2",
        "a_shape",
        "tri_shape",
        "vllm_a_shape",
        "vllm_tri_shape",
        "inf_llm",
        "flexprefill",
        "vllm_flexprefill",
        "xattention",
    ]
    KV_TYPES = [
        "dense",
        "streamingllm",
        "snapkv",
        "pyramidkv",
        "quest",
        "retr_attn",
        "kivi",
    ]

    def __init__(
        self,
        attn_type: str = "minference",
        model_name: str = None,
        config_path: str = None,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        kv_cache_cpu_device: str = "cpu",
        kv_type: str = "dense",
        is_search: bool = False,
        attn_kwargs: dict = {},
        **kwargs,
    ):
        super(MInferenceConfig, self).__init__()
        attn_type, kv_type = self.update_config_type(attn_type, kv_type)
        assert (
            attn_type in self.MINFERENCE_ATTENTION_TYPES + self.OTHER_ATTENTION_TYPES
        ), f"The attn_type {attn_type} you specified is not supported."
        assert (
            kv_type in self.KV_TYPES
        ), f"The kv_type {kv_type} you specified is not supported."
        print(
            f"<---- MInference Config Detail ----> attn_type {attn_type}, kv_type {kv_type}"
        )
        self.attn_type = attn_type
        self.config_path = self.update_config_path(config_path, model_name)
        self.model_name = model_name
        self.is_search = is_search
        self.starting_layer = starting_layer
        self.kv_cache_cpu = kv_cache_cpu
        self.kv_cache_cpu_device = kv_cache_cpu_device
        self.kv_type = kv_type
        self.attn_kwargs = {
            "is_search": is_search,
            "starting_layer": starting_layer,
            "config_path": config_path,
            **attn_kwargs,
        }

    def update_config_path(self, config_path: str = None, model_name: str = None):
        if self.attn_type in self.OTHER_ATTENTION_TYPES:
            return ""
        if config_path is not None:
            return config_path
        assert (
            model_name in MODEL2PATH
        ), f"The model {model_name} you specified is not supported. You are welcome to add it and open a PR :)"
        return MODEL2PATH[model_name]

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def update_config_type(self, attn_type: str, kv_type: str):
        if kv_type == "":
            kv_type = "dense"
        if attn_type == "minference_with_dense":
            attn_type = "dense"
        return attn_type, kv_type

    @classmethod
    def get_available_attn_types(cls):
        return cls.MINFERENCE_ATTENTION_TYPES + cls.OTHER_ATTENTION_TYPES

    @classmethod
    def get_available_kv_types(cls):
        return cls.KV_TYPES
