# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os

from .minference_configuration import MInferenceConfig
from .patch import minference_patch, minference_patch_vllm, patch_hf

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MInference:
    def __init__(
        self,
        attn_type: str = "minference",
        model_name: str = None,
        config_path: str = None,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        use_snapkv: bool = False,
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
            use_snapkv=use_snapkv,
            is_search=is_search,
            attn_kwargs=attn_kwargs,
            **kwargs,
        )

    def __call__(self, model):
        return self.patch_model(model)

    def patch_model(self, model):
        if self.config.attn_type != "vllm":
            model.config.starting_layer = self.config.starting_layer
            model.config.config_path = self.config.config_path

        if self.config.attn_type == "minference":
            model.config.is_search = self.config.is_search
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "minference_with_dense":
            model.config.dense = True
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "dilated1":
            model.config.dilated1 = True
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "static":
            model.config.static_pattern = True
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "dilated2":
            model.config.dilated2 = True
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "streaming":
            model.config.streaming = True
            model.config.streaming_kwargs = {
                "n_local": 3968,
                "n_init": 128,
                **self.config.attn_kwargs,
            }
            model = minference_patch(model, self.config)

        elif self.config.attn_type == "streaming2":
            model = patch_hf(
                model,
                attn_type="streaming",
                attn_kwargs={"n_local": 3968, "n_init": 128, **self.config.attn_kwargs},
            )
        elif self.config.attn_type == "hf":
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
        elif self.config.attn_type == "vllm":
            model = minference_patch_vllm(model, self.config.config_path)
        else:
            raise ValueError(
                f"The attention type {self.config.attn_type} you specified is not supported."
            )
        return model
