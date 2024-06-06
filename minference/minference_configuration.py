import os

from .configs.model2path import MODEL2PATH


class MInferenceConfig:
    ATTENTION_TYPES = [
        "minference",
        "minference_wo_cache",
        "dilated1",
        "static",
        "dilated2",
        "minference_with_snapkv",
        "streaming",
        "inf_llm",
        "vllm",
    ]

    def __init__(
        self,
        attn_type="minference",
        model_name=None,
        config_path=None,
        starting_layer=-1,
        is_search=False,
        **kwargs,
    ):
        super(MInferenceConfig, self).__init__()
        assert (
            attn_type in self.ATTENTION_TYPES
        ), f"The attention_type {attn_type} you specified is not supported."
        self.attn_type = attn_type
        self.config_path = self.update_config_path(config_path, model_name)
        self.model_name = model_name
        self.is_search = is_search
        self.starting_layer = starting_layer

    def update_config_path(self, config_path: str, model_name: str):
        if config_path is not None:
            return config_path
        assert (
            model_name in MODEL2PATH
        ), f"The model {model_name} you specified is not supported. You are welcome to add it and open a PR :)"
        return MODEL2PATH[model_name]
