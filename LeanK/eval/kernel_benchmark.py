# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from minference.ops.leank_flash_decoding import leank_flashattn
import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
from minference.modules.leank import reorder_channel_mask
from tilelang_flash_decoding import flashattn
import numpy as np

llama_pattern = torch.load("../../minference/configs/leank/llama3.1-8b-instruct.pth")

supported_dims = [0, 32, 64, 96, 128]
layer_boundaries = []
layer_counts = []
for i in range(llama_pattern.shape[0]):
    layer_mask = llama_pattern[i]
    channel_mask = layer_mask.sum(dim=-1)
    layer_full_attn_channels = reorder_channel_mask(
        channel_mask, 
        layer_mask, 
        supported_dims,
    )
    dim_cnt = (layer_mask.sum(dim=-1) == supported_dims[0]).sum().item()
    boundaries = [dim_cnt]
    counts = [supported_dims[0]]
    for d in supported_dims[1:]:
        nheads = (layer_mask.sum(dim=-1) == d).sum().item()
        if nheads > 0:
            dim_cnt += nheads
            boundaries.append(dim_cnt)
            counts.append(d)
    layer_boundaries.append(boundaries)
    layer_counts.append(counts)
    
def get_heuristic_config() -> dict:
    # Get CUDA device properties
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    print(f"CUDA device capability: {sm_version}")
    if sm_version == 89:
        return {
            "block_N": 128,
            "block_H": 64,
            "num_split": 8,
            "num_stages": 0,
            "threads": 128
        }
    else:
        return {
            "block_N": 64,
            "block_H": 64,
            "num_split": 2,
            "num_stages": 1,
            "threads": 128
        }
        
def main(batch: int = 1,
         heads: int = 32,
         groups: int = 8,
         kv_seqlen: int = 8192,
         dim: int = 128,
         tune: bool = False):
    batch, heads, groups, kv_seqlen, dim = batch, heads, groups, kv_seqlen, dim
    qk_flops = 2 * batch * heads * kv_seqlen * dim
    kv_fulllen = 1024
    kv_seqlen -= 1024
    pv_flops = 2 * batch * heads * kv_seqlen * dim
    total_flops = qk_flops + pv_flops
    heads_per_group = heads // groups
    layer_idx = 0
    
    config = get_heuristic_config()
    
    if (not tune):
        kernel = flashattn(batch, heads, groups, kv_seqlen, kv_seqlen, dim, **config)
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        base_latency = profiler.do_bench(warmup=500)
        print("Dense Attention (Tile-lang): {:.2f} ms".format(base_latency))
    
    times = []
            
    for boundaries, counts in zip(layer_boundaries, layer_counts):
        l, r = boundaries[0], -1
        number_groups = []
        kernel_kwargs = []
        for boundary, count in zip(boundaries[1:], counts[1:]):
            r = boundary
            number_groups.append(r - l)
            l = r
        
        n_groups = len(number_groups)
        if n_groups < 3:
            number_groups += [0] * (3 - n_groups)
        
        kernel_kwargs = [batch, heads]
        kernel_kwargs += [i * heads_per_group for i in number_groups]
        kernel_kwargs += [groups]
        kernel_kwargs += number_groups
        kernel_kwargs += [kv_seqlen, kv_seqlen, kv_fulllen, kv_fulllen, dim]
        kernel_kwargs += counts[1:]
        if n_groups < 3:
            kernel_kwargs += [0] * (3 - n_groups)
        kernel_kwargs += [n_groups]
        kernel_kwargs += ["bfloat16"]
        
        layer_idx += 1
        if (not tune):
            program = leank_flashattn(*kernel_kwargs)(**config)
            kernel = tilelang.compile(program, out_idx=[5 * (n_groups + 1) + 1 + (n_groups > 0)])
            profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
            latency = profiler.do_bench(warmup=500)
            print(f"Layer {layer_idx}", "LeanK Attention: {:.2f} ms".format(latency))
            times.append(latency)
        else:
            best_result = leank_flashattn(**kernel_kwargs, tune=tune)
            best_latency = best_result.latency
            best_config = best_result.config
            print(f"Best latency: {best_latency}")
            print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
            print(f"Best config: {best_config}")

    print("---------------------")
    print("Dense Attention (Tile-lang): {:.2f} ms".format(base_latency))
    print("LeanK Decoding (Average): {:.2f} ms".format(np.mean(times)))

def get_heuristic_config() -> dict:
    # Get CUDA device properties
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    print(f"CUDA device capability: {sm_version}")
    if sm_version == 89:
        return {
            "block_N": 128,
            "block_H": 64,
            "num_split": 8,
            "num_stages": 0,
            "threads": 128
        }
    else:
        return {
            "block_N": 64,
            "block_H": 64,
            "num_split": 8,
            "num_stages": 1,
            "threads": 128
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--kv_seqlen', type=int, default=32768, help='kv sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    main(args.batch, args.heads, args.groups, args.kv_seqlen, args.dim, args.tune)