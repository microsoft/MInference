# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


def parse_args():
    parser = argparse.ArgumentParser(description="kv_reduction")

    parser.add_argument(
        "--model_name", type=str, default="Meta-Llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--config_name", type=str, default=None)

    parser.add_argument("--dataset_format", type=str, default="multiple_passkey")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--context_length_min", type=int, default=1024)
    parser.add_argument("--context_length_max", type=int, default=4096)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=20)
    parser.add_argument("--depth_ratio_num_intervals", type=int, default=10)
    parser.add_argument("--num_passkeys", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sink_size", type=int, default=64)
    parser.add_argument("--recent_size", type=int, default=256)
    parser.add_argument("--deploy_sink_size", type=int, default=None)
    parser.add_argument("--deploy_recent_size", type=int, default=None)
    parser.add_argument("--reg_weight", type=float, default=0.05)
    parser.add_argument("--initial_value", type=float, default=1.0)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--enable_pp", action="store_true")
    parser.add_argument("--enable_tp", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--min_needle_depth_ratio", type=float, default=0)
    parser.add_argument("--max_needle_depth_ratio", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rope_theta", type=float, default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--align", type=int, default=32)
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--stage1_rst_path", type=str, default=None)

    parser.add_argument(
        "--supervision",
        type=str,
        default="distill",
        choices=["classify", "distill"],
    )

    # Eval params
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--task", type=str, default="default")
    parser.add_argument("--attn_load_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--passkey_length", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--generation_length", type=int, default=256)
    parser.add_argument("--stride_length", type=int, default=256)
    parser.add_argument("--prefilling_chunk_size", type=int, default=4096)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    args = parser.parse_args()

    args.device = parse_device(args.device)
    return args


def parse_device(device: str):
    if "," in device:
        return [int(d) for d in device.split(",")]
    elif device in ["auto", "cpu"]:
        return device
    return f"cuda:{device}"


def get_model(model_name):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    if hasattr(model.config, "sliding_window") and model.config.sliding_window is None:
        model.config.sliding_window = model.config.max_position_embeddings

    return model


def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=False, trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer


def convert_to_list(scaling_factors):
    num_pruned_layers = len(scaling_factors)
    for idx in range(num_pruned_layers):
        scaling_factors[idx] = (
            scaling_factors[idx].detach().flatten().float().cpu().tolist()
        )
    return scaling_factors


def visualize_patterns(scaling_factors):
    img = np.array(scaling_factors)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="coolwarm", interpolation="nearest", aspect="auto")
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.colorbar(fraction=0.046, pad=0.04)
    # scale the color to 0-1
    plt.clim(0, 1)
    plt.tight_layout()
    plt.title("Ratio of Full Attention Computations")
    return fig


def seed_everything(seed):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_scaling_factors(scaling_factors, output_filename):
    np.savetxt(
        output_filename,
        np.array(scaling_factors),
        delimiter="\t",
    )


def sparsify_scaling_factors(scaling_factors, ratio, round_to):
    l = len(scaling_factors)
    h, d = scaling_factors[0].shape

    scaling_factors = torch.stack(scaling_factors).flatten().float().cpu().numpy()

    threshold = np.quantile(scaling_factors, ratio)
    mask = (scaling_factors >= threshold).astype(float)

    head_k = (mask.reshape(l, h, d).sum(-1) + round_to // 2) // round_to * round_to

    ind = np.argsort(scaling_factors.reshape(l, h, d))[:, :, ::-1]
    reverse_ind = np.argsort(ind)
    mask_round = reverse_ind < head_k[:, :, None]
    return mask_round
