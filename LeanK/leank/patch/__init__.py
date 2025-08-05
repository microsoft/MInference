# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os

import numpy as np
import torch

from .llama import (
    enable_llama_training,
    full_attn_forward_llama,
    get_scaling_factors_llama,
    map_scaling_factors_llama,
    scaled_attn_forward_llama,
    set_scaling_factors_llama,
)
from .qwen import (
    enable_qwen_training,
    full_attn_forward_qwen,
    map_scaling_factors_qwen,
    scaled_attn_forward_qwen,
    set_scaling_factors_qwen,
)


def enable_training(
    model,
    sink_size,
    recent_size,
    initial_value=1.0,
    enable_ulysses_attention=False,
    scaling_factors=None,
):
    if "llama" in model.config.model_type:
        enable_llama_training(
            model,
            sink_size,
            recent_size,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            scaling_factors=scaling_factors,
        )
    elif "qwen" in model.config.model_type:
        enable_qwen_training(
            model,
            sink_size,
            recent_size,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            scaling_factors=scaling_factors,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def get_scaling_factors(model):
    if "llama" in model.config.model_type:
        return get_scaling_factors_llama(model)
    elif "qwen" in model.config.model_type:
        return set_scaling_factors_qwen(model)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def set_scaling_factors(model, scaling_factors):
    if "llama" in model.config.model_type:
        model = set_scaling_factors_llama(model, scaling_factors)
    elif "qwen" in model.config.model_type:
        model = set_scaling_factors_qwen(model, scaling_factors)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")
    return model


def map_scaling_factors(model, func):
    if "llama" in model.config.model_type:
        return map_scaling_factors_llama(model, func)
    elif "qwen" in model.config.model_type:
        return map_scaling_factors_qwen(model, func)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def load_scaling_factors(load_dir, filename="scaling_factors.tsv"):
    scaling_factors = np.loadtxt(
        os.path.join(load_dir, filename),
        dtype=float,
        delimiter="\t",
    )
    scaling_factors = np.clip(scaling_factors, 0, 1)
    scaling_factors = torch.tensor(scaling_factors, dtype=torch.float32)
    return scaling_factors
