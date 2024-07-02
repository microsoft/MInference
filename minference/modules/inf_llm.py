# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Refer to the code in https://github.com/thunlp/InfLLM/blob/main/inf_llm/attention/context_manager.py

from copy import deepcopy
from typing import Optional, Tuple

import torch
from flash_attn import flash_attn_func
from transformers.modeling_outputs import CausalLMOutput

from ..ops.streaming_kernel import TritonMultiStageDotProductionAttention


class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        self.num_units = num_units
        self.unit_size = unit_size
        self.dtype = dtype
        self.data = torch.empty((num_units, unit_size), device="cuda", dtype=dtype)
        self.idle_set = set(list(range(num_units)))

    def alloc(self):
        assert len(self.idle_set) > 0

        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        assert idx not in self.idle_set
        self.idle_set.add(idx)


class MemoryUnit:
    def __init__(
        self,
        kv: Tuple[torch.Tensor, torch.Tensor],
        cache: CudaCache,
        load_to_cache: bool = False,
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda:
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        if pin_memory:
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)

        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
            gpu_data[1].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> bool:
        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None

            return False, target_event

        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)
        if target is not None:
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)

        else:
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event

    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    def offload(self):
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None


class VectorTensor:
    def __init__(self, hidden_size, element_dtype):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, hidden_size), dtype=element_dtype, device="cuda"
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size

    def append_cache(self):
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:], device="cuda", dtype=self.data.dtype
        )
        new_data[: self.cache_size, ...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size
        assert tensor.is_contiguous()

        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length : self.length + append_l, ...].copy_(tensor)

        self.length += append_l

    def get_data(self):
        return self.data[: self.length, ...]

    def get_topk(self, tensor: torch.Tensor, topk):  # inner product
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size
        logits = torch.matmul(self.data[: self.length], tensor[:, None]).squeeze(dim=-1)
        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits.topk(topk, dim=0).indices.cpu().tolist()

    def __len__(self):
        return self.length


class Faiss:
    def __init__(self, hidden_size, element_dtype):
        import faiss

        # We use the CPU index here because the GPU index requires a long initialization time
        self.index = faiss.IndexFlatIP(hidden_size)
        self.hidden_size = hidden_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dim() == 2 and tensor.size(1) == self.hidden_size
        self.index.add(tensor.cpu().float().numpy().astype("float32"))

    def get_data(self):
        raise ValueError

    def get_topk(self, tensor: torch.Tensor, topk):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size
        xq = tensor[None, :].cpu().float().numpy().astype("float32")
        topk_index = self.index.search(xq, topk)[1][0].tolist()
        return topk_index

    def __len__(self):
        return self.index.ntotal


GLOBAL_STREAM = None


class ContextManager:
    def __init__(
        self,
        position_embedding,
        n_init,
        n_local,
        block_size,
        max_cached_block,
        topk,
        exc_block_size,
        score_decay: Optional[float] = None,
        repr_topk: int = 1,
        cache_strategy="lru",
        chunk_topk_calc: Optional[int] = None,
        async_global_stream: bool = False,
        pin_memory: bool = False,
        faiss: bool = False,
        perhead: bool = False,
        dense_decoding: bool = False,
    ):
        self.length = 0
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        self.score_decay = score_decay
        assert exc_block_size <= n_local  # no global token in input
        self.topk = topk
        self.Attn = TritonMultiStageDotProductionAttention
        self.initialized = False
        self.repr_topk = repr_topk
        self.cache_strategy = cache_strategy
        self.load_count = 0
        self.chunk_topk_calc = chunk_topk_calc
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        self.faiss = faiss
        self.perhead = perhead

        self.dense_decoding = dense_decoding

        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        assert cache_strategy in ["lru", "lru-s"]

        if cache_strategy == "lru-s":
            self.calc_block_score = True
        else:
            self.calc_block_score = False

    def remove_lru_blocks(
        self, u, num_remove: Optional[int] = None, ignore_blocks=None
    ):
        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block

        if num_remove <= 0:
            return

        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += 1

            if removed >= num_remove:
                return

    def get_block_k(self, k, score):
        assert isinstance(score, torch.Tensor)
        assert k.dim() >= 2
        k = self.from_group_kv(k)
        assert k.shape[:-1] == score.shape
        assert k.shape[-2] == self.block_size
        score_topk = score.topk(self.repr_topk, dim=-1).indices
        assert score_topk.shape == (self.num_units, self.unit_size, self.repr_topk)
        ret = torch.gather(
            k,
            -2,
            score_topk[:, :, :, None].expand(
                self.num_units, self.unit_size, self.repr_topk, self.dim_head
            ),
        )
        return ret

    def from_group_kv(self, tensor):
        assert tensor.dim() == 4
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_units, self.unit_size_kv, 1, length, dim_head))
        tensor = tensor.expand(
            (self.num_units, self.unit_size_kv, num_group, length, dim_head)
        ).reshape((self.num_units, self.num_heads, length, dim_head))
        return tensor

    def init(self, local_q, local_k, local_v, global_q, global_k, global_v):
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert _t.size(1) == num_heads or _t.size(1) == num_heads_kv
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.num_units = batch_size
        self.unit_size = num_heads
        self.unit_size_kv = num_heads_kv

        self.global_blocks = [[] for _ in range(self.num_units)]  # [[memory_unit]]
        self.cached_blocks = [
            {} for _ in range(self.num_units)
        ]  # [[block_id: block_score]
        self.num_global_block = 0

        if self.faiss:
            self.block_k = [
                Faiss(dim_head * self.unit_size, global_k.dtype)
                for _ in range(self.num_units)
            ]
        else:
            self.block_k = [
                VectorTensor(dim_head * self.unit_size, global_k.dtype)
                for _ in range(self.num_units)
            ]

        self.local_k = torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=local_k.dtype,
            device=local_k.device,
        )
        self.local_v = torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=local_v.dtype,
            device=local_v.device,
        )

        if self.dense_decoding:
            self.dense_k = torch.empty(
                (self.num_units, self.unit_size_kv, 0, dim_head),
                dtype=local_k.dtype,
                device=local_k.device,
            )
            self.dense_v = torch.empty(
                (self.num_units, self.unit_size_kv, 0, dim_head),
                dtype=local_v.dtype,
                device=local_v.device,
            )

        self.global_remainder = (
            torch.empty(
                (self.num_units, self.unit_size_kv, 0, dim_head),
                dtype=global_k.dtype,
                device=global_k.device,
            ),
            torch.empty(
                (self.num_units, self.unit_size_kv, 0, dim_head),
                dtype=global_v.dtype,
                device=global_v.device,
            ),
        )

        self.global_remainder_local_score = torch.empty(
            (self.num_units, self.unit_size, 0),
            dtype=global_k.dtype,
            device=global_k.device,
        )

        self.init_k = torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=global_k.dtype,
            device=global_k.device,
        )
        self.init_v = torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=global_k.dtype,
            device=global_k.device,
        )
        self.init_exc = False
        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        buffer_len = (
            self.topk * self.block_size
            + self.exc_block_size
            + self.block_size
            + self.n_init
        )
        self.global_buffer = torch.zeros(
            (2, self.num_units, self.unit_size_kv, buffer_len, dim_head),
            dtype=global_k.dtype,
            device=global_k.device,
        )
        self.global_buffer_block_id_list = [
            [-1] * self.topk for _ in range(self.num_units)
        ]
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.num_units,
            self.unit_size_kv * self.block_size * dim_head * 2,
            local_k.dtype,
        )

        self.initialized = True

    def calc_block_topk(self, global_h_q):
        if not self._use_chunk_topk:
            if self.num_global_block <= self.topk:
                return [
                    list(range(len(self.global_blocks[0])))
                    for _ in range(self.num_units)
                ]

            global_h_q = global_h_q.mean(dim=2, keepdim=False)
            assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
            global_h_q = global_h_q.reshape(
                self.num_units, self.dim_head * self.unit_size
            )
            ret = []
            for u in range(self.num_units):
                ret.append(self.block_k[u].get_topk(global_h_q[u], self.topk))

        else:
            return self._cached_topk[self._topk_cur]

        return ret

    def get_global_hidden_and_mask(self, len_q, block_topk):
        assert len(block_topk) == self.num_units
        global_block_map = [[] for _ in range(self.num_units)]
        global_remainder_len = max(
            self._global_remainder_ed
            - self._global_remainder_st
            + len_q
            - self.n_local,
            0,
        )
        init_len = self.init_k.size(-2)
        sliding_window = None

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        block_num = len(block_topk[0])
        for u in range(self.num_units):
            assert len(block_topk[u]) == block_num

            block_topk[u].sort()
            global_block_map[u] = deepcopy(self.global_buffer_block_id_list[u])
            for b_idx in block_topk[u]:
                if b_idx in global_block_map[u]:
                    continue

                st = -1
                ed = -1
                for j in range(self.topk):
                    if (
                        global_block_map[u][j] == -1
                        or global_block_map[u][j] not in block_topk[u]
                    ):
                        st = j * self.block_size
                        ed = st + self.block_size
                        global_block_map[u][j] = b_idx
                        break

                assert b_idx in self.cached_blocks[u]
                self.global_blocks[u][b_idx].load(
                    (global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :])
                )

        init_st = block_num * self.block_size
        init_ed = init_st + init_len
        if (
            self.global_buffer_init_st != init_st
            or self.global_buffer_init_ed != init_ed
        ):
            global_h_k[:, :, init_st:init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st:init_ed, :].copy_(self.init_v, non_blocking=True)

        ed = init_ed

        rmd_st = init_ed
        rmd_ed = rmd_st + global_remainder_len
        ed = rmd_ed
        global_h_k[:, :, rmd_st:rmd_ed, :].copy_(
            self.global_remainder[0][
                :,
                :,
                self._global_remainder_st : self._global_remainder_st
                + global_remainder_len,
                :,
            ],
            non_blocking=True,
        )
        global_h_v[:, :, rmd_st:rmd_ed, :].copy_(
            self.global_remainder[1][
                :,
                :,
                self._global_remainder_st : self._global_remainder_st
                + global_remainder_len,
                :,
            ],
            non_blocking=True,
        )

        sliding_window = (self.global_remainder[0].size(-2) + rmd_st, self.n_local)

        self.global_buffer_block_id_list = deepcopy(global_block_map)
        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        for u in range(self.num_units):
            assert max(global_block_map[u][block_num:] + [-1]) == -1
            assert min(global_block_map[u][:block_num] + [0]) > -1
            global_block_map[u] = list(global_block_map[u][:block_num])

        global_h_k = global_h_k[:, :, :ed, :]
        global_h_v = global_h_v[:, :, :ed, :]
        return global_h_k, global_h_v, sliding_window, global_block_map, block_num

    def update_block_score(
        self, global_score: torch.FloatTensor, global_block_map, global_block_num
    ):
        if global_score is not None:
            global_score = global_score[:, :, : global_block_num * self.block_size]
            assert global_score.shape == (
                self.num_units,
                self.unit_size,
                global_block_num * self.block_size,
            )
            global_score = global_score.view(
                self.num_units, self.unit_size, global_block_num, self.block_size
            )
            global_score = global_score.sum(dim=-1).sum(dim=1)
            assert global_score.shape == (self.num_units, global_block_num)
            global_score = global_score.to(
                device="cpu", non_blocking=False
            )  # (num_units, global_block_num)
            for u in range(self.num_units):
                for k, v in self.cached_blocks[u].items():
                    self.cached_blocks[u][k] = v * self.score_decay
                score = global_score[u].tolist()
                assert len(score) >= len(global_block_map[u])
                for s, i in zip(score, global_block_map[u]):
                    self.cached_blocks[u][i] += s

    def _append(self, local_q, local_k, local_v, global_q):
        # get local_h_q, local_h_k, local_h_v
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        # calc local result first to overlap host-device communication
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, get_score=True, sliding_window=self.n_local
        )

        # calc topk global repr k and load cache
        with torch.cuda.stream(GLOBAL_STREAM):
            block_topk = self.calc_block_topk(global_q)

            for u in range(self.num_units):
                num_remove = len(self.cached_blocks[u]) - self.max_cached_block
                for bidx in block_topk[u]:
                    if bidx not in self.cached_blocks[u]:
                        num_remove += 1

                # update cache
                self.remove_lru_blocks(u, num_remove, block_topk[u])

            if self.cache_strategy == "lru":
                self.load_count += 1
                for u in range(self.num_units):
                    for bidx in block_topk[u]:
                        self.cached_blocks[u][bidx] = self.load_count

            elif self.cache_strategy == "lru-s":
                for u in range(self.num_units):
                    for bidx in block_topk[u]:
                        self.cached_blocks[u][bidx] = 0
            else:
                raise ValueError

            # get global_h_k, global_h_v, global_mask
            #    Beacuse exc_block_size <= n_local, no global_k, global_v used in global part
            global_h_q = global_q
            (
                global_h_k,
                global_h_v,
                global_sliding_window,
                global_block_map,
                global_block_num,
            ) = self.get_global_hidden_and_mask(local_h_q.size(-2), block_topk)

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # calc global result
        attn.append(
            global_h_q,
            global_h_k,
            global_h_v,
            end=True,
            get_score=self.calc_block_score,
            sliding_window=global_sliding_window,
            complement_sliding_window=True,
        )

        o, score_list = attn.get_result()
        loc_score = score_list[0]
        glb_score = score_list[1]

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # update global score
        with torch.cuda.stream(GLOBAL_STREAM):
            self.update_block_score(glb_score, global_block_map, global_block_num)

        return o.view((self.batch_size, self.num_heads, -1, self.dim_head)), loc_score

    def get_batched_topk(self, global_q):
        length = global_q.shape[2]
        exc_num = (length + self.exc_block_size - 1) // self.exc_block_size
        exc_block_num = length // self.exc_block_size
        ret = []
        if self.num_global_block <= self.topk:
            for _ in range(exc_num):
                ret.append(
                    [
                        list(range(len(self.global_blocks[0])))
                        for _ in range(self.num_units)
                    ]
                )
            return ret

        global_h_q = global_q
        assert global_h_q.dim() == 4
        assert global_h_q.shape[:2] == (self.num_units, self.unit_size)
        assert global_h_q.shape[3] == self.dim_head

        block_k = torch.cat(
            [self.block_k[u].get_data()[None, :, :] for u in range(self.num_units)],
            dim=0,
        )
        assert block_k.shape == (
            self.num_units,
            self.num_global_block,
            self.dim_head * self.unit_size,
        )
        block_k = (
            block_k.reshape(
                self.num_units, self.num_global_block, self.unit_size, self.dim_head
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        if exc_block_num > 0:
            tmp_global_h_q = (
                global_h_q[:, :, : exc_block_num * self.exc_block_size, :]
                .reshape(
                    self.num_units,
                    self.unit_size,
                    exc_block_num,
                    self.exc_block_size,
                    self.dim_head,
                )
                .mean(dim=-2)
            )
            assert tmp_global_h_q.shape == (
                self.num_units,
                self.unit_size,
                exc_block_num,
                self.dim_head,
            )
            block_score = torch.matmul(tmp_global_h_q, block_k.transpose(-1, -2)).mean(
                dim=1
            )  # (num_units, exc_block_num, num_global_block)
            assert block_score.shape == (
                self.num_units,
                exc_block_num,
                self.num_global_block,
            )

            indices = block_score.topk(self.topk, dim=-1).indices.cpu()
            for b in range(exc_block_num):
                tmp = []
                for u in range(self.num_units):
                    tmp.append(indices[u, b].tolist())
                    assert len(tmp[-1]) == self.topk

                ret.append(tmp)

        if exc_block_num != exc_num:
            tmp_global_h_q = (
                global_h_q[:, :, exc_block_num * self.exc_block_size :, :]
                .reshape(
                    self.num_units,
                    self.unit_size,
                    length - exc_block_num * self.exc_block_size,
                    self.dim_head,
                )
                .mean(dim=-2, keepdim=True)
            )
            assert tmp_global_h_q.shape == (
                self.num_units,
                self.unit_size,
                1,
                self.dim_head,
            )
            block_score = torch.matmul(tmp_global_h_q, block_k.transpose(-1, -2))
            assert block_score.shape == (
                self.num_units,
                self.unit_size,
                1,
                self.num_global_block,
            )
            block_score = block_score.squeeze(dim=2).mean(dim=1)
            assert block_score.shape == (self.num_units, self.num_global_block)
            indices = block_score.topk(self.topk, dim=-1).indices.cpu()
            tmp = []
            for u in range(self.num_units):
                tmp.append(indices[u].tolist())
                assert len(tmp[-1]) == self.topk

            ret.append(tmp)

        return ret

    def append_global(self, exc_length, kv_length, local_score):
        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st

        global_remainder_len = global_remainder_ed - global_remainder_st

        assert local_score.shape[:3] == (self.num_units, self.unit_size, kv_length)
        local_score = local_score[:, :, -exc_length - self.n_local :]
        self.global_remainder_local_score[
            :, :, global_remainder_ed - local_score.size(-1) : global_remainder_ed
        ].add_(local_score)

        if not self.init_exc and global_remainder_len > self.n_local:
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = min(
                self.n_init - self.init_k.size(-2), global_remainder_len - self.n_local
            )
            self.init_k = torch.cat(
                (
                    self.init_k,
                    global_k[
                        :,
                        :,
                        global_remainder_st : global_remainder_st + append_init_len,
                        :,
                    ],
                ),
                dim=-2,
            )
            self.init_v = torch.cat(
                (
                    self.init_v,
                    global_v[
                        :,
                        :,
                        global_remainder_st : global_remainder_st + append_init_len,
                        :,
                    ],
                ),
                dim=-2,
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True

        while global_remainder_len - self.block_size >= self.n_local:
            global_remainder_len -= self.block_size
            for u in range(self.num_units):
                self.global_blocks[u].append(
                    (
                        MemoryUnit(
                            (
                                self.global_remainder[0][
                                    u,
                                    :,
                                    global_remainder_st : global_remainder_st
                                    + self.block_size,
                                    :,
                                ],
                                self.global_remainder[1][
                                    u,
                                    :,
                                    global_remainder_st : global_remainder_st
                                    + self.block_size,
                                    :,
                                ],
                            ),
                            self.cuda_cache,
                            False,
                            self.pin_memory,
                        )
                    )
                )

            global_block_k = self.get_block_k(
                self.global_remainder[0][
                    :, :, global_remainder_st : global_remainder_st + self.block_size, :
                ],
                self.global_remainder_local_score[
                    :, :, global_remainder_st : global_remainder_st + self.block_size
                ],
            )
            assert global_block_k.shape == (
                self.num_units,
                self.unit_size,
                self.repr_topk,
                self.dim_head,
            )
            global_block_k = global_block_k.mean(dim=-2, keepdim=False)
            global_block_k = global_block_k.reshape(
                self.num_units, self.unit_size * self.dim_head
            )
            global_block_k = global_block_k[:, None, :]

            self.num_global_block += 1
            for u in range(self.num_units):
                self.block_k[u].append(global_block_k[u])
            global_remainder_st += self.block_size

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

    def append(
        self,
        local_q,
        local_k,
        local_v,
        global_q,
        global_k,
        global_v,
    ):
        batch_size = local_q.size(0)
        input_length = local_q.size(-2)

        if self.perhead:
            num_heads = local_q.size(1)
            num_heads_kv = local_v.size(1)

            def repeat_kv(t):
                t = t.view(batch_size, num_heads_kv, 1, input_length, -1)
                t = t.expand(
                    batch_size,
                    num_heads_kv,
                    num_heads // num_heads_kv,
                    input_length,
                    -1,
                )
                t = t.reshape(batch_size * num_heads, 1, input_length, -1)
                return t

            local_q = local_q.view(batch_size * num_heads, 1, input_length, -1)
            local_k = repeat_kv(local_k)
            local_v = repeat_kv(local_v)
            global_q = global_q.view(batch_size * num_heads, 1, input_length, -1)
            global_k = repeat_kv(global_k)
            global_v = repeat_kv(global_v)

        if not self.initialized:
            self.init(local_q, local_k, local_v, global_q, global_k, global_v)

        input_length = local_q.size(-2)

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # append local and global tensor
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        if self.dense_decoding:
            self.dense_k = torch.cat((self.dense_k, local_k), dim=-2)
            self.dense_v = torch.cat((self.dense_v, local_v), dim=-2)

        # append global remainder
        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )

            self.global_remainder_local_score = torch.cat(
                (
                    self.global_remainder_local_score,
                    torch.zeros(
                        (self.num_units, self.unit_size, global_k.size(-2)),
                        dtype=global_k.dtype,
                        device=global_k.device,
                    ),
                ),
                dim=-1,
            )

        with torch.cuda.stream(GLOBAL_STREAM):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        use_chunk_topk = self.chunk_topk_calc is not None and input_length > 1
        self._use_chunk_topk = use_chunk_topk
        if use_chunk_topk:
            exc_block_num = input_length // self.exc_block_size
            exc_block_per_topk_chunk = self.chunk_topk_calc // self.exc_block_size
            calc_cur_list = [
                i * self.exc_block_size
                for i in range(0, exc_block_num + 1, exc_block_per_topk_chunk)
            ]
            if calc_cur_list[-1] < input_length:
                calc_cur_list.append(input_length)
            self._topk_cur = 0
            self._topk_calc_cur = -1

        o_list = []

        for st in range(0, input_length, self.exc_block_size):
            ed = min(st + self.exc_block_size, input_length)
            if use_chunk_topk and calc_cur_list[self._topk_calc_cur + 1] < ed:
                # calculate topk and sync with host here
                assert ed <= calc_cur_list[self._topk_calc_cur + 2]
                self._topk_calc_cur += 1
                with torch.cuda.stream(GLOBAL_STREAM):
                    self._cached_topk = self.get_batched_topk(
                        global_q[
                            :,
                            :,
                            calc_cur_list[self._topk_calc_cur] : calc_cur_list[
                                self._topk_calc_cur + 1
                            ],
                            :,
                        ]
                    )
                self._topk_cur = 0

            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length
            chunk_o, local_score = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st:kv_ed, :],
                self.local_v[:, :, kv_st:kv_ed, :],
                global_q[:, :, st:ed, :],
            )
            o_list.append(chunk_o)

            # append global
            with torch.cuda.stream(GLOBAL_STREAM):
                self.append_global(ed - st, kv_ed - kv_st, local_score)

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

            if use_chunk_topk:
                self._topk_cur += 1

        self.length += input_length

        # update local and global tensor
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local :, :]
            self.local_v = self.local_v[:, :, -self.n_local :, :]

        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st :, :],
                self.global_remainder[1][:, :, self._global_remainder_st :, :],
            )
            self.global_remainder_local_score = self.global_remainder_local_score[
                :, :, self._global_remainder_st :
            ]

        ret = torch.cat(o_list, dim=-2)

        if self.perhead:
            ret = ret.view(batch_size, num_heads, input_length, -1)

        return ret

    def size(self, *args, **kwargs):
        return self.length


def inf_llm_forward(
    n_local,
    n_init,
    topk,
    block_size,
    max_cached_block,
    exc_block_size,
    repr_topk: int = 1,
    cache_strategy="lru",
    score_decay=None,
    chunk_topk_calc=None,
    async_global_stream=True,
    pin_memory=False,
    faiss=False,
    perhead=False,
    dense_decoding=False,
    *args,
    **kwargs
):
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        position_bias: Optional[torch.Tensor],
        use_cache: bool,
        past_key_value,
        project_q,
        project_k,
        project_v,
        attention_out,
        dim_head,
        num_heads,
        num_heads_kv,
    ):
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        # assert use_cache

        h_q = project_q(query)  # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)  # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)  # (batch, len_k, num_heads * dim_head)

        h_q = (
            h_q.view(batch_size, len_q, num_heads, dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (batch, num_heads, len_q, dim_head)
        h_k = (
            h_k.view(batch_size, len_k, num_heads_kv, dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (batch, num_heads_kv, len_k, dim_head)
        h_v = (
            h_v.view(batch_size, len_k, num_heads_kv, dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (batch, num_heads_kv, len_k, dim_head)

        if len_q == 1 and dense_decoding:
            past_k = past_key_value.dense_k
            past_v = past_key_value.dense_v

            h_k = torch.cat((past_k, h_k), dim=-2)
            h_v = torch.cat((past_v, h_v), dim=-2)

            past_key_value.dense_k = h_k
            past_key_value.dense_v = h_v

            h_q, h_k = position_bias(h_q, h_k)

            # (batch_size, seqlen, nheads, headdim)
            h_q = h_q.transpose(1, 2)
            h_k = h_k.transpose(1, 2)
            h_v = h_v.transpose(1, 2)

            # (batch_size, seqlen, nheads, headdim)
            o = flash_attn_func(h_q, h_k, h_v, causal=True)

            o = o.reshape(batch_size, len_q, dim_head * num_heads)
            o = attention_out(o)

            if use_cache:
                return o, past_key_value
            else:
                return o

        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias,
                n_init,
                n_local,
                block_size,
                max_cached_block,
                topk,
                exc_block_size,
                score_decay,
                repr_topk,
                cache_strategy,
                chunk_topk_calc,
                async_global_stream,
                pin_memory,
                faiss,
                perhead,
                dense_decoding=dense_decoding,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        o = past_key_value.append(
            local_q,
            local_k,
            local_v,
            global_q,
            global_k,
            global_v,
        )

        o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
        o = o.reshape(batch_size, len_q, dim_head * num_heads)
        o = attention_out(o)

        if use_cache:
            return o, past_key_value
        else:
            return o

    return forward


class GreedySearch:
    def __init__(self, model, tokenizer):
        model.eval()
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer
        self.past_kv = None

    def clear(self):
        self.past_kv = None

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = (
                torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()
            )

        return model_inputs

    def generate(self, text=None, input_ids=None, **kwargs):
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs["input_ids"]

        with torch.inference_mode():
            result = self._decode(input_ids, **kwargs)

        self.clear()
        return result

    def _decode(
        self,
        input_ids,
        max_length=100,
        extra_end_token_ids=[],
        chunk_size: int = 4096,
        output=False,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        attention_mask = torch.ones_like(input_ids)
        assert input_ids.size(0) == 1
        length = input_ids.size(1)
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        past_key_values = self.past_kv
        if output:
            output_text = ""

        for i in range(max_length + 1):
            if i == 0:
                if chunk_size is None:
                    chunk_size = input_ids.size(1)
                for st in range(0, input_ids.size(1) - 1, chunk_size):
                    ed = min(input_ids.size(1) - 1, st + chunk_size)
                    out = self.model(
                        input_ids=input_ids[:, st:ed],
                        attention_mask=attention_mask[:, :ed],
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                    )
                    logits, past_key_values = out.logits, out.past_key_values

                out = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                )
                logits, past_key_values = out.logits, out.past_key_values
            else:
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits, past_key_values = out.logits, out.past_key_values

            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
            if word.item() in end_token_ids or i == max_length:
                break

            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.size(0), 1),
                        dtype=torch.int,
                        device=attention_mask.device,
                    ),
                ),
                dim=-1,
            )
            if output:
                tmp = self.tokenizer.decode(input_ids.squeeze(0)[length:])
                if len(tmp) > len(output_text):
                    import sys

                    sys.stdout.write(tmp[len(output_text) :])
                    sys.stdout.flush()
                    output_text = tmp

        self.past_kv = past_key_values

        if output:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # return [self.tokenizer.decode(input_ids.squeeze(0)[length:])]
        return input_ids


class InfLLMGenerator(GreedySearch):
    def generate(
        self,
        input_ids=None,
        generation_config=None,
        pad_token_id=None,
        max_new_tokens=None,
    ):
        if max_new_tokens is not None:
            max_new_tokens = max_new_tokens
        else:
            max_new_tokens = generation_config.max_new_tokens
        return super().generate(
            text=None,
            input_ids=input_ids,
            max_length=max_new_tokens,
            chunk_size=8192,
            extra_end_token_ids=[pad_token_id] if pad_token_id is not None else [],
        )

    @torch.no_grad()
    def __call__(self, input_ids=None, *args, **kwargs):
        # chunked forward
        chunk_size = 8192
        all_logits = torch.empty(0, dtype=torch.bfloat16).to(input_ids.device)
        for st in range(0, input_ids.size(1), chunk_size):
            torch.cuda.empty_cache()
            ed = min(input_ids.size(1), st + chunk_size)
            out = self.model(
                input_ids=input_ids[:, st:ed],
            )
            logits = out.logits.to(torch.bfloat16)
            all_logits = torch.cat((all_logits, logits), dim=1)

        return CausalLMOutput(logits=all_logits)
