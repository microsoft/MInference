import os

from .minference_configuration import MInferenceConfig
from .patch import patch_hf, minference_patch, minference_patch_with_snapkv, minference_path_wo_cache


class MInference:
    def __init__(
        self,
        attn_type="minference",
        model_name=None,
        config_path=None,
        is_search=False,
        starting_layer=-1,
        **kwargs,
    ):
        super(MInference, self).__init__()
        self.config = MInferenceConfig(
            attn_type=attn_type,
            model_name=model_name,
            config_path=config_path,
            is_search=is_search,
            starting_layer=starting_layer,
            **kwargs,
        )

    def patch_model(self, model):
        model.config.starting_layer = self.config.starting_layer
        model.config.config_path = self.config.config_path

        if self.config.attn_type == "minference":
            model.config.is_search = self.config.is_search
            model = minference_patch(model)

        elif self.config.attn_type == "minference_wo_cache":
            model = minference_path_wo_cache(model)

        elif self.config.attn_type == "dilated1":
            model.config.dilated1 = True
            model = minference_patch(model)
        
        elif self.config.attn_type == "static":
            model.config.static_pattern = True
            model = minference_patch(model)
        
        elif self.config.attn_type == "dilated2":
            model.config.dilated2 = True
            model = minference_patch(model)

        elif self.config.attn_type == 'minference_with_snapkv':
            model = minference_patch_with_snapkv(model)
        elif self.config.attn_type == "streaming":
            model = patch_hf(model, attn_type="streaming", attn_kwargs={'n_local': 3968, 'n_init': 128})
        elif self.config.attn_type == "inf_llm":
            model = patch_hf(
                model, attn_type="inf_llm",
                attn_kwargs={
                    'block_size': 128,
                    'n_init': 128,
                    'n_local': 4096,
                    'topk': 16,
                    'repr_topk': 4,
                    'max_cached_block': 32,
                    'exc_block_size': 512,
                    'base': 1000000,
                    'distance_scale': 1.0,
                    'dense_decoding': True,
                }
            )
        else:
            raise ValueError(
                f"The attention type {self.config.attn_type} you specified is not supported."
            )
        return model

