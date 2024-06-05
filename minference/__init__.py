# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# flake8: noqa
from .version import VERSION as __version__
from .patch import patch_hf, minference_patch, minference_patch_with_snapkv, minference_path_wo_cache
from .models_patch import MInference
from .minference_configuration import MInferenceConfig

from .kernels.streaming_kernel import streaming_forward
from .kernels.pit_sparse_flash_attention_v2 import pit_sparse_flash_attention_forward
from .kernels.block_sparse_flash_attention import block_sparse_flash_attention_forward

__all__ = [
    "MInference", "minference_patch", "minference_path_wo_cache", "minference_patch_with_snapkv", "patch_hf",
    "pit_sparse_flash_attention_forward", "block_sparse_flash_attention_forward", "streaming_forward",
    "MInferenceConfig",
]
