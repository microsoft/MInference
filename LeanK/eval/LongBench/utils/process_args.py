# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    k_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV_cache quantization bits."},
    )
    v_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV_cache quantization bits."},
    )
    k_quant_dim: Optional[str] = field(
        default="token",
        metadata={"help": "KV_cache quantization bits."},
    )
    v_quant_dim: Optional[str] = field(
        default="token",
        metadata={"help": "KV_cache quantization bits."},
    )
    group_size: Optional[int] = field(
        default=32,
        metadata={"help": "KV_cache quantization group size."},
    )
    residual_length: Optional[int] = field(
        default=128,
        metadata={"help": "KV_cache residual length."},
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    load_quant: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a quantized model"},
    )
    w_bit: Optional[int] = field(
        default=4,
        metadata={"help": "The model weight bit width."},
    )
    lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA"},
    )
    lora_mode: Optional[str] = field(
        default="q",
        metadata={"help": "LoRA mode"},
    )
    lora_r: Optional[int] = field(
        default=1,
        metadata={"help": "LoRA r"},
    )
    lora_alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "LoRA dropout"},
    )
    enable_leank: Optional[str] = field(
        default=True,
        metadata={"help": "enable LeanK decoding"},
    )
    enable_think: Optional[str] = field(
        default=False,
        metadata={"help": "enable ThinK decoding"},
    )


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default="c4",
        metadata={"help": "The dataset used for fine-tuning the model."},
    )
    eval_tasks: Optional[str] = field(
        default="wikitext",
        metadata={"help": "The dataset used for evaluation."},
    )
    tasks: Optional[str] = field(
        default="wikitext",
        metadata={"help": "The dataset used for evaluation."},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The batch size."},
    )
    num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "The number of fewshot examples."},
    )
    output_path: Optional[str] = field(
        default="./outputs",
        metadata={"help": "The output path."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )


def process_args():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    return model_args, data_args
