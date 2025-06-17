# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import os

from .minference_configuration import MInferenceConfig
from .patch import minference_patch, minference_patch_vllm, new_patch, patch_hf

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MInference:
    def __init__(
        self,
        attn_type: str = "minference",
        model_name: str = None,
        config_path: str = None,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        kv_type: str = "dense",
        is_search: bool = False,
        attn_kwargs: dict = {},
        **kwargs,
    ):
        super(MInference, self).__init__()
        self.config = MInferenceConfig(
            attn_type=attn_type,
            model_name=model_name,
            config_path=config_path,
            starting_layer=starting_layer,
            kv_cache_cpu=kv_cache_cpu,
            kv_type=kv_type,
            is_search=is_search,
            attn_kwargs=attn_kwargs,
            **kwargs,
        )

    def __call__(self, model):
        return self.patch_model(model)

    def patch_model(self, model):
        if self.config.kv_type == "retr_attn":
            self.config.attn_kwargs.setdefault(
                "max_seq_length", model.config.max_position_embeddings
            )
            self.config.attn_kwargs.setdefault("max_new_tokens", 1024)
            self.config.attn_kwargs.setdefault(
                "num_layers", model.config.num_hidden_layers
            )
            self.config.attn_kwargs.setdefault("top_k", 4096)
            self.config.attn_kwargs.setdefault("from_layer", 0)

        if self.config.kv_type == "kivi":
            self.config.attn_kwargs.setdefault("bits", 2)
            self.config.attn_kwargs.setdefault("group_size", 32)
            self.config.attn_kwargs.setdefault("residual_length", 32)

        if self.config.kv_type in ["snapkv", "pyramidkv"]:
            self.config.attn_kwargs.setdefault("window_size", 32)
            self.config.attn_kwargs.setdefault("max_capacity_prompt", 4096)
            self.config.attn_kwargs.setdefault("kernel_size", 5)
            self.config.attn_kwargs.setdefault("pooling", "avgpool")

        if self.config.kv_type == "quest":
            self.config.attn_kwargs.setdefault("chunk_size", 16)
            self.config.attn_kwargs.setdefault("token_budget", 1024)

        if self.config.kv_type == "streamingllm":
            self.config.attn_kwargs.setdefault("n_local", 3968)
            self.config.attn_kwargs.setdefault("n_init", 128)

        if self.config.attn_type == "flexprefill":
            self.config.attn_kwargs.setdefault("gamma", 0.9)
            self.config.attn_kwargs.setdefault("tau", 0.1)
            self.config.attn_kwargs.setdefault("min_budget", None)
            self.config.attn_kwargs.setdefault("max_budget", None)
            self.config.attn_kwargs.setdefault("block_size", 128)

        if "vllm" not in self.config.attn_type:
            model.config.starting_layer = self.config.starting_layer
            model.config.config_path = self.config.config_path

        if self.config.attn_type == "minference":
            if not self.config.is_search:
                with open(self.config.config_path, "r") as f:
                    self.config.attn_kwargs.setdefault("best_pattern", json.load(f))
            model = new_patch(model, self.config)

        elif self.config.attn_type == "a_shape":
            self.config.attn_kwargs.setdefault("n_local", 3968)
            self.config.attn_kwargs.setdefault("n_init", 128)
            model = new_patch(model, self.config)

        elif self.config.attn_type == "tri_shape":
            self.config.attn_kwargs.setdefault("n_local", 3968)
            self.config.attn_kwargs.setdefault("n_init", 128)
            self.config.attn_kwargs.setdefault("n_last", 100)
            model = new_patch(model, self.config)

        elif self.config.attn_type in ["flexprefill", "dense", "xattention"]:
            model = new_patch(model, self.config)

        elif self.config.attn_type == "dilated1":
            model.config.dilated1 = True
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "static":
            model.config.static_pattern = True
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "dilated2":
            model.config.dilated2 = True
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "streaming2":
            model = patch_hf(
                model,
                attn_type="a_shape",
                attn_kwargs={"n_local": 3968, "n_init": 128, **self.config.attn_kwargs},
            )
        elif self.config.attn_type in ["hf", "vllm"]:
            pass
        elif self.config.attn_type == "inf_llm":
            model = patch_hf(
                model,
                attn_type="inf_llm",
                attn_kwargs={
                    "block_size": 128,
                    "n_init": 128,
                    "n_local": 4096,
                    "topk": 16,
                    "repr_topk": 4,
                    "max_cached_block": 32,
                    "exc_block_size": 512,
                    "base": 1000000,
                    "distance_scale": 1.0,
                    "dense_decoding": True,
                    **self.config.attn_kwargs,
                },
            )
        elif self.config.attn_type == "vllm_minference":
            model = minference_patch_vllm(
                model, self.config.config_path, self.config.attn_kwargs
            )
        elif self.config.attn_type == "vllm_flexprefill":
            patch_config = {
                "flexprefill": True,
                "flexprefill_kwargs": {},
                **self.config.attn_kwargs,
            }
            model = minference_patch_vllm(model, self.config.config_path, patch_config)
        elif self.config.attn_type == "vllm_a_shape":
            patch_config = {
                "a_shape": True,
                "streaming_kwargs": {
                    "n_local": 3968,
                    "n_init": 128,
                },
                **self.config.attn_kwargs,
            }
            model = minference_patch_vllm(model, self.config.config_path, patch_config)
        elif self.config.attn_type == "vllm_tri_shape":
            patch_config = {
                "tri_shape": True,
                "streaming_kwargs": {
                    "n_local": 3968,
                    "n_init": 128,
                },
                **self.config.attn_kwargs,
            }
            model = minference_patch_vllm(model, self.config.config_path, patch_config)
        else:
            raise ValueError(
                f"The attention type {self.config.attn_type} you specified is not supported."
            )
        return model
