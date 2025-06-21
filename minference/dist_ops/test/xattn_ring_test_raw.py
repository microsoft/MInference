# tests/test_minference_sparse_attention.py
"""
Distributed correctness tests for Minference sparse-attention kernels.

Run with:
    pytest -q -s tests/test_minference_sparse_attention.py
or manually choose GPUs, e.g.
    CUDA_VISIBLE_DEVICES=0,1 pytest -q -s …

The test spawns one process per GPU with torch.multiprocessing, so it does
**not** require `pytest-xdist`.  It will be skipped automatically if you have
fewer than two visible CUDA devices.
"""
from __future__ import annotations

import os
import random
from types import SimpleNamespace
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minference.ops.utils import set_seed, check_correctness_by_row
from minference.dist_ops.xattn_zigzag import xattn_zigzag_func
from minference.ops.xattention_fa import xattn_flash_attn_func

# ------------- constants ------------------------------------------------------
_ATOL = 1e-1
_RTOL = 1e-1
_WORLD_SIZE = 4 

# ------------- helpers --------------------------------------------------------
def _init_process_group(rank: int, world_size: int, port: str) -> None:
    """Initialise NCCL backend for the current worker."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank % min(world_size, torch.cuda.device_count())),
            "LOCAL_WORLD_SIZE": str(min(world_size, torch.cuda.device_count())),
        }
    )
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



def _run_worker(
    rank: int,
    world_size: int,
    port: str,
    cfg: SimpleNamespace,
) -> None:
    """Worker function executed in every spawned GPU process."""
    _init_process_group(rank, world_size, port)

    # Short-hand variables
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    set_seed(2025 + rank)

    # ----------------- generate identical tensors on every rank --------------
    if rank == 0:
        rand_or_one = (
            torch.randn if not cfg.ones else lambda s, **k: torch.ones(*s, **k)
        )
        q = rand_or_one(
            (cfg.batch_size, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim),
            dtype=dtype,
            device=device,
        )
        k = rand_or_one(
            (cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim),
            dtype=dtype,
            device=device,
        )
        v = rand_or_one(
            (cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim),
            dtype=dtype,
            device=device,
        )
        dout = rand_or_one(
            (cfg.batch_size, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim),
            dtype=dtype,
            device=device,
        )
    else:
        # placeholders that will be overwritten by broadcast
        shape_q = (cfg.batch_size, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim)
        shape_kv = (cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)
        q = torch.empty(shape_q, device=device, dtype=dtype)
        k = torch.empty(shape_kv, device=device, dtype=dtype)
        v = torch.empty(shape_kv, device=device, dtype=dtype)
        dout = torch.empty(shape_q, device=device, dtype=dtype)

    # Make every rank see the same data
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)

    # ----------------- slice local context -----------------------------------
    local_ctx = cfg.seq_len // world_size
    sl = slice(rank * local_ctx, (rank + 1) * local_ctx)

    q_local = q[:, sl].clone().detach().requires_grad_()
    k_local = k[:, sl].clone().detach().requires_grad_()
    v_local = v[:, sl].clone().detach().requires_grad_()
    dout_local = dout[:, sl].clone()

    # ----------------- forward / backward on the candidate kernel ------------
    out_local = xattn_zigzag_func(
        q_local, k_local, v_local,
        layer_idx=0,
        xattn_params=cfg.xattn_params,
        granularity=128,
    )
    torch.autograd.backward(out_local, dout_local)

    # ----------------- gather outputs & grads for reference comparison -------
    out_gather = [torch.empty_like(out_local) for _ in range(world_size)]
    dist.all_gather(out_gather, out_local)
    final_out = torch.cat(out_gather, dim=1)

    grads = []
    for g in (q_local.grad, k_local.grad, v_local.grad):
        tmp = [torch.empty_like(g) for _ in range(world_size)]
        dist.all_gather(tmp, g)
        grads.append(torch.cat(tmp, dim=1))

    # ---------------------------------------
    if rank == 0:
        q_ref = q.detach().clone().requires_grad_()
        k_ref = k.detach().clone().requires_grad_()
        v_ref = v.detach().clone().requires_grad_()

        single_machine_params = cfg.xattn_params.copy()
        single_machine_params["chunk_size"] = cfg.seq_len // _WORLD_SIZE
        out_ref = xattn_flash_attn_func(
            q_ref, k_ref, v_ref,
            head_indices=list(range(cfg.num_qo_heads)),
            xattn_params=single_machine_params,
            granularity=128,
        )
        torch.autograd.backward(out_ref, dout)
        ref_grads = (q_ref.grad, k_ref.grad, v_ref.grad)

        # ----------------- assertions ----------------------------------------
        if check_correctness_by_row(
            cfg.seq_len, final_out, out_ref, "forward output", ATOL=_ATOL, RTOL=_RTOL
        ):
            check_correctness_by_row(
                cfg.seq_len, grads[0], ref_grads[0], "Q-grad", ATOL=_ATOL, RTOL=_RTOL
            )
            check_correctness_by_row(
                cfg.seq_len, grads[1], ref_grads[1], "K-grad",
                ATOL=_ATOL, RTOL=_RTOL
            )
            check_correctness_by_row(
                cfg.seq_len, grads[2], ref_grads[2], "V-grad",  
                ATOL=_ATOL, RTOL=_RTOL
            )

    dist.destroy_process_group()


# ------------- pytest entry-point --------------------------------------------
def test_xattention_kernels(
    seq_len: int = 4096,
    batch_sz: int = 1,
    head_dim: int = 64,
    ones: bool = True,
    num_qo_heads: int = 2,
    num_kv_heads: int = 2,
    
    stride: int = 16,
    threshold: float = 0.9,
):
    """
    Compare every sparse kernel against the dense Flash-Attention reference on
    both forward pass and input-gradient w.r.t Q/K/V.
    """
    port = str(random.randint(12000, 20000))
    xattn_params = {
        "stride": stride,
        "norm": 1,
        "softmax": True,
        "threshold": threshold,
        "select_mode": "inverse",
        "use_triton": True,
        "causal": True,
        "kdb": 1,
        "keep_sink": False,
        "keep_recent": False
    }
    cfg = SimpleNamespace(
        batch_size=batch_sz,
        seq_len=seq_len,
        head_dim=head_dim,
        ones=ones,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        xattn_params=xattn_params,
    )
  
    print(f"=" * 80)
    print(f"Testing XAttention (w. Zigzag) with configuration:\n{cfg}")
    print(f"=" * 80)
    mp.spawn(
        _run_worker,
        args=(_WORLD_SIZE, port, cfg),
        nprocs=_WORLD_SIZE,
        join=True,
    )

if __name__ == "__main__":
    # Run the test with default parameters
    test_xattention_kernels(
        seq_len=512 * 1024,
        batch_sz=1,
        head_dim=64,
        ones=False,
        num_qo_heads=4,
        num_kv_heads=1,
        stride=16,
        threshold=0.95,
    )