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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minference.ops.utils import set_seed
from minference.dist_ops.minfer_zigzag import minfer_zigzag_func
from minference.dist_ops.minfer_striped import minfer_stripe_func
from minference.dist_ops.minfer_dr_striped import minfer_dr_stripe_func
from minference.ops.pit_sparse_flash_attention_v3 import minference_flash_attn_func

# ------------- constants ------------------------------------------------------
_ATOL = 1e-2
_RTOL = 1e-2
_WORLD_SIZE = 2 

_ATTENTION_IMPLS: dict[str, Callable] = {
    "minfer_zigzag": minfer_zigzag_func,
    "minfer_stripe": minfer_stripe_func,
    "minfer_dr_stripe": minfer_dr_stripe_func,
}

# ------------- helpers --------------------------------------------------------
def _init_process_group(rank: int, world_size: int, port: str) -> None:
    """Initialise NCCL backend for the current worker."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _run_worker(
    rank: int,
    world_size: int,
    port: str,
    cfg: SimpleNamespace,
    attn_op_name: str,
) -> None:
    """Worker function executed in every spawned GPU process."""
    _init_process_group(rank, world_size, port)

    # Short-hand variables
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    set_seed(2025 + rank)

    attn_op: Callable = _ATTENTION_IMPLS[attn_op_name]

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
    out_local = attn_op(
        q_local,
        k_local,
        v_local,
        cfg.v_size,
        cfg.s_size,
        layer_idx=0,
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

    # ----------------- reference: dense Flash-Attention ----------------------
    if rank == 0:
        q_ref = q.detach().clone().requires_grad_()
        k_ref = k.detach().clone().requires_grad_()
        v_ref = v.detach().clone().requires_grad_()

        out_ref = minference_flash_attn_func(
            q_ref,
            k_ref,
            v_ref,
            cfg.v_size,
            cfg.s_size,
            causal=True,
        )
        torch.autograd.backward(out_ref, dout)
        ref_grads = (q_ref.grad, k_ref.grad, v_ref.grad)

        # ----------------- assertions ----------------------------------------
        torch.testing.assert_close(
            final_out, out_ref, atol=_ATOL, rtol=_RTOL, msg="forward mismatch"
        )
        for got, ref, name in zip(
            grads,
            ref_grads,
            ("Q-grad", "K-grad", "V-grad"),
        ):
            torch.testing.assert_close(got, ref, atol=_ATOL, rtol=_RTOL, msg=name)

    dist.destroy_process_group()


# ------------- pytest entry-point --------------------------------------------
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD_SIZE, reason="Not enough GPUs")
@pytest.mark.parametrize("seq_len",   [512, 4096, 8192])
@pytest.mark.parametrize("batch_sz",  [1])
@pytest.mark.parametrize("head_dim",  [32, 64])
@pytest.mark.parametrize("sparsity", [0.9, 0.95, 1.])
@pytest.mark.parametrize("ones", [False, True])
@pytest.mark.parametrize("num_qo_heads", [2, 4, 4])
@pytest.mark.parametrize("num_kv_heads", [2, 1, 4])
@pytest.mark.parametrize("attn_op_name",
    ["minfer_zigzag", "minfer_stripe", "minfer_dr_stripe"]
)
def test_sparse_attention_kernels(
    seq_len: int,
    batch_sz: int,
    head_dim: int,
    sparsity: float,
    ones: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    attn_op_name: str,
):
    """
    Compare every sparse kernel against the dense Flash-Attention reference on
    both forward pass and input-gradient w.r.t Q/K/V.
    """
    port = str(random.randint(12000, 20000))

    cfg = SimpleNamespace(
        batch_size=batch_sz,
        seq_len=seq_len,
        head_dim=head_dim,
        sparsity=sparsity,
        ones=ones,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
    )
    # derived sizes used by both candidate and reference kernels
    cfg.v_size = [int((1 - cfg.sparsity) * 0.1 * cfg.seq_len)] * cfg.num_qo_heads
    cfg.s_size = [int((1 - cfg.sparsity) * 0.2 * cfg.seq_len)] * cfg.num_qo_heads

    mp.spawn(
        _run_worker,
        args=(_WORLD_SIZE, port, cfg, attn_op_name),
        nprocs=_WORLD_SIZE,
        join=True,
    )
