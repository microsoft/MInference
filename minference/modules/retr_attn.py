# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import os
import threading
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.utils.import_utils import _is_package_available

try:
    from flash_attn import flash_attn_func
except ImportError:
    from ..ops.flash_attn_triton import _flash_attn_triton_decoding as flash_attn_func

if _is_package_available("papyfaiss"):
    import papyfaiss


class VectorDB_KV_Cache:
    def __init__(self, max_length, temp_cache_size):
        self.key_cache = []
        self.value_cache = []
        self.update_thread = None
        self.gpu_key_cache = []
        self.gpu_value_cache = []
        self.temp_gpu_key_cache = []
        self.temp_gpu_value_cache = []
        self.start_size = 128
        self.recent_size = 512
        self.temp_cache = temp_cache_size  # set equal to max generation length
        self.cache_size = 640
        self.seen1 = None
        self.seen2 = None
        self.temp_seen = 0
        self.max_size = max_length

    def async_update(
        self, key_states, value_states, layer_idx, do_strllm_until_layer, core
    ):
        def update_task(key_states, value_states, layer_idx):
            self.key_vec_cache_update(key_states, layer_idx, core)
            self.val_vec_cache_update(value_states, layer_idx, do_strllm_until_layer)

        assert self.start_size + self.recent_size == self.cache_size

        start_size = self.start_size
        recent_size = self.recent_size
        cache_size = self.cache_size

        bs, head_num, qlen, dim = key_states.shape

        if layer_idx <= do_strllm_until_layer:
            assert 0 < cache_size <= qlen  # cache_size is not 0 when using pattern

            key_tensor = torch.zeros(
                (bs, head_num, cache_size, dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            value_tensor = torch.zeros(
                (bs, head_num, cache_size, dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )

            key_tensor[:, :, :, :] = torch.cat(
                (
                    key_states[:, :, :start_size, :],
                    key_states[:, :, qlen - recent_size : qlen, :],
                ),
                dim=2,
            )
            value_tensor[:, :, :, :] = torch.cat(
                (
                    value_states[:, :, :start_size, :],
                    value_states[:, :, qlen - recent_size : qlen, :],
                ),
                dim=2,
            )

            self.gpu_key_cache.append(key_tensor)
            self.gpu_value_cache.append(value_tensor)

            key_states = key_states[:, :, start_size : qlen - recent_size, :]
            value_states = value_states[:, :, start_size : qlen - recent_size, :]

        else:
            self.gpu_key_cache.append(None)
            self.gpu_value_cache.append(None)

        self.temp_gpu_key_cache.append(
            torch.zeros(
                (bs, head_num, self.temp_cache, dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
        )
        self.temp_gpu_value_cache.append(
            torch.zeros(
                (bs, head_num, self.temp_cache, dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
        )

        if 0 < layer_idx < do_strllm_until_layer:
            self.sync()

        key_states = key_states.to(torch.float32).detach().cpu().numpy()
        value_states = value_states.detach().cpu()

        self.update_thread = threading.Thread(
            target=update_task, args=(key_states, value_states, layer_idx)
        )
        self.update_thread.start()

    def sync(self):
        self.update_thread.join()

    # sync update, with load data
    def sync_key_update(
        self, key_states, layer_idx, do_strllm_until_layer, core, insert_db
    ):
        if insert_db:
            start_size = self.start_size
            recent_size = self.recent_size
            if layer_idx <= do_strllm_until_layer and recent_size > 0:
                bs, head_num, _, dim = key_states.shape
                key_tensor = torch.zeros(
                    (bs, head_num, 1, dim),
                    dtype=key_states.dtype,
                    device=key_states.device,
                )
                key_tensor = self.gpu_key_cache[layer_idx][
                    :, :, start_size : start_size + 1, :
                ]
                tempk = torch.cat(
                    [
                        self.gpu_key_cache[layer_idx][:, :, :start_size, :],
                        self.gpu_key_cache[layer_idx][:, :, start_size + 1 :, :],
                    ],
                    dim=-2,
                )
                self.gpu_key_cache[layer_idx] = torch.cat([tempk, key_states], dim=-2)
            else:
                key_tensor = key_states

            if key_tensor.dtype == torch.bfloat16:
                key_states = key_tensor.float().detach().cpu().numpy()
            else:
                key_states = key_tensor.detach().cpu().numpy()
            key_cache = self.key_vec_cache_update(key_states, layer_idx, core)
            return key_cache, self.gpu_key_cache[layer_idx]
        else:
            if layer_idx == 0:
                self.temp_seen += 1
            self.temp_gpu_key_cache[layer_idx][
                :, :, self.temp_seen - 1 : self.temp_seen, :
            ] = key_states
            return self.key_cache[layer_idx], torch.cat(
                (
                    self.gpu_key_cache[layer_idx],
                    self.temp_gpu_key_cache[layer_idx][:, :, : self.temp_seen, :],
                ),
                dim=-2,
            )

    def sync_value_update(
        self, value_states, layer_idx, do_strllm_until_layer, insert_db
    ):
        if insert_db:
            start_size = self.start_size
            recent_size = self.recent_size
            if layer_idx <= do_strllm_until_layer and recent_size > 0:
                bs, head_num, _, dim = value_states.shape
                value_tensor = torch.zeros(
                    (bs, head_num, 1, dim),
                    dtype=value_states.dtype,
                    device=value_states.device,
                )
                value_tensor = self.gpu_value_cache[layer_idx][
                    :, :, start_size : start_size + 1, :
                ]
                tempv = torch.cat(
                    [
                        self.gpu_value_cache[layer_idx][:, :, :start_size, :],
                        self.gpu_value_cache[layer_idx][:, :, start_size + 1 :, :],
                    ],
                    dim=-2,
                )
                self.gpu_value_cache[layer_idx] = torch.cat(
                    [tempv, value_states], dim=-2
                )
            else:
                value_tensor = value_states

            value_states = value_tensor.detach().cpu()
            value_cache = self.val_vec_cache_update(
                value_states, layer_idx, do_strllm_until_layer
            )
            return value_cache, self.gpu_value_cache[layer_idx]
        else:
            self.temp_gpu_value_cache[layer_idx][
                :, :, self.temp_seen - 1 : self.temp_seen, :
            ] = value_states
            return self.value_cache[layer_idx], torch.cat(
                (
                    self.gpu_value_cache[layer_idx],
                    self.temp_gpu_value_cache[layer_idx][:, :, : self.temp_seen, :],
                ),
                dim=-2,
            )

    # papyfaiss update vectordb cache
    def key_vec_cache_update(self, key_states, layer_idx, core):
        def cpp_add(task_id, key, key_cache):
            key_cache.add(key[task_id], task_id, core)

        if len(self.key_cache) <= layer_idx:
            # Flat index
            try:
                index = papyfaiss.FlatIndex(
                    head_num=key_states.shape[1], dim=key_states.shape[3]
                )
            except:
                assert (
                    False
                ), "Please install papyfaiss. Refer to https://github.com/microsoft/RetrievalAttention"
            # index = papyfaiss.FlatIndexSQ(head_num=key_states.shape[1], dim=key_states.shape[3])
            # IVF index
            # index = papyfaiss.IVFIndexSQ(head_num=key_states.shape[1], dim=key_states.shape[3], n_centroids=512, quant="SQ8", use_gpu=True)
            # index.set_nprobe(150, 150)
            self.key_cache.append(index)

            # pool = ThreadPool(core)
            # tasks = range(key_states.shape[1])
            # pool.map(lambda task_id: cpp_add(task_id, key_states[0], self.key_cache[layer_idx]), tasks)
            # pool.close()
            # pool.join()
            self.key_cache[layer_idx].paraadd(key_states[0], core)
        else:
            self.key_cache[layer_idx].paraadd(key_states[0], core)

        return self.key_cache[layer_idx]

    def val_vec_cache_update(
        self,
        value_states,  # (batch_size, nheads, seqlen, head_dim)
        layer_idx: int,
        do_strllm_until_layer,
    ):
        if len(self.value_cache) <= layer_idx:  # prompt phase
            self.value_cache.append(value_states)
            return self.value_cache[layer_idx]
        else:  # decode phase
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value_states[:, :, :, :]), dim=-2
            )
            return self.value_cache[layer_idx]


class RetrAttnCache(Cache):
    def __init__(self, config):
        super().__init__()
        self.max_length = config.attn_kwargs["max_seq_length"]
        self.temp_cache_size = config.attn_kwargs["max_new_tokens"]
        self._seen_tokens = 0

        self.core = os.cpu_count() // 2
        self.do_strllm_until_layer = config.attn_kwargs["num_layers"] - 1
        self.num_layers = config.attn_kwargs["num_layers"]
        self.vector_db_cache = VectorDB_KV_Cache(self.max_length, self.temp_cache_size)

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        query_states = cache_kwargs.get("query_states", None)
        q_len = query_states.size(-2)
        insert_db = cache_kwargs.get("update_global_past_kv", True)

        if layer_idx == 0:
            self._seen_tokens += key_states.size(-2)

        # if query_states.size(-2) != 1:  # prefill
        # if len(self.vector_db_cache.value_cache) == layer_idx: # initializing vector_db_cache
        if q_len == self._seen_tokens:
            self.vector_db_cache.async_update(
                key_states.contiguous(),
                value_states,
                layer_idx,
                self.do_strllm_until_layer,
                self.core,
            )
            if layer_idx == self.num_layers - 1:
                self.vector_db_cache.sync()

            key_states = repeat_kv(
                key_states, query_states.size(1) // key_states.size(1)
            )
            value_states = repeat_kv(
                value_states, query_states.size(1) // value_states.size(1)
            )
            # assert len(self.vector_db_cache.value_cache) == layer_idx + 1
            return key_states, value_states
        else:
            if q_len == 1:  # the decoding
                key_cache, gpu_key_cache = self.vector_db_cache.sync_key_update(
                    key_states,
                    layer_idx,
                    self.do_strllm_until_layer,
                    self.core,
                    insert_db,
                )
                value_cache, gpu_value_cache = self.vector_db_cache.sync_value_update(
                    value_states, layer_idx, self.do_strllm_until_layer, insert_db
                )
            else:  # the follow-up queries
                for i in range(q_len):  # insert query token one by one
                    key_cache, gpu_key_cache = self.vector_db_cache.sync_key_update(
                        key_states[:, :, i : i + 1, :],
                        layer_idx,
                        self.do_strllm_until_layer,
                        self.core,
                        insert_db,
                    )
                    (
                        value_cache,
                        gpu_value_cache,
                    ) = self.vector_db_cache.sync_value_update(
                        value_states[:, :, i : i + 1, :],
                        layer_idx,
                        self.do_strllm_until_layer,
                        insert_db,
                    )
            return (key_cache, gpu_key_cache), (value_cache, gpu_value_cache)

    def get_seq_length(self, layer_idx=0):
        if len(self.vector_db_cache.gpu_key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def clear_temp_kv_cache(self):
        self._seen_tokens -= self.vector_db_cache.temp_seen
        self.vector_db_cache.temp_seen = 0


def retr_attn(
    q,
    key_cache,
    value_cache,
    decoding_kwargs,
):
    key_cache, gpu_key_cache = key_cache
    value_cache, gpu_value_cache = value_cache

    kv_heads = gpu_key_cache.size(1)
    num_key_value_groups = decoding_kwargs.get(
        "num_key_value_groups", q.size(1) // kv_heads
    )
    layer_idx = decoding_kwargs.get("layer_idx", None)
    top_k = decoding_kwargs["attn_forward_config"].get("top_k", 2000)
    from_layer = decoding_kwargs["attn_forward_config"].get("from_layer", 0)
    core = os.cpu_count() // 2

    device = q.device
    q_dtype = q.dtype

    qq = q.clone()
    if q.dtype == torch.bfloat16:
        q = q.float().detach().cpu().numpy()
    else:
        q = q.detach().cpu().numpy()

    # retrieval
    top_k = top_k if layer_idx >= from_layer else 124_000
    if key_cache.get_index_type() == "Flat":
        distances, indices = key_cache.search(q[0], num_key_value_groups, top_k, core)
    elif "IVF" in key_cache.get_index_type():
        distances, indices = key_cache.search(
            q[0], layer_idx, num_key_value_groups, top_k, core
        )

    retrieval_attn_weights = distances.reshape(
        (q.shape[0], q.shape[1], q.shape[2], top_k)
    ) / math.sqrt(q.shape[3])
    retrieval_indices = indices.reshape((q.shape[1], q.shape[2], top_k))

    retrieval_attn_weights = torch.from_numpy(retrieval_attn_weights)
    retrieval_lse = torch.log(torch.sum(torch.exp(retrieval_attn_weights), dim=-1)).to(
        device
    )
    retrieval_attn_weights = nn.functional.softmax(
        retrieval_attn_weights, dim=-1, dtype=torch.float32
    )
    retrieval_attn_weights = nn.functional.dropout(
        retrieval_attn_weights, p=0, training=False
    )

    retrieval_out = torch.full(
        (q.shape[0], q.shape[1], q.shape[2], q.shape[3]), 0, dtype=torch.float32
    )
    for head_idx in range(q.shape[1]):
        for q_idx in range(q.shape[2]):
            query2key = head_idx // num_key_value_groups
            retrieval_attn_values = value_cache[0][query2key][
                retrieval_indices[head_idx][q_idx]
            ]  # (bs, nhead, true_k, head_dim)
            retrieval_out[:, head_idx:, q_idx, :] = torch.matmul(
                retrieval_attn_weights[:, head_idx, q_idx, :],
                retrieval_attn_values.to(torch.float32),
            )

    retrieval_out = retrieval_out.to(q_dtype).to(device)
    if gpu_key_cache is not None:
        flash_out, flash_lse, _ = flash_attn_func(
            qq.transpose(1, 2),
            gpu_key_cache.transpose(1, 2),
            gpu_value_cache.transpose(1, 2),
            dropout_p=0,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=True,
        )

        flash_lse = flash_lse.transpose(-2, -1).unsqueeze(dim=-1)
        retrieval_lse = retrieval_lse.transpose(-2, -1).unsqueeze(dim=-1)
        new_lse = retrieval_lse + torch.log(1 + torch.exp(flash_lse - retrieval_lse))

        final_out = (
            torch.exp(retrieval_lse - new_lse) * retrieval_out.transpose(1, 2)
            + torch.exp(flash_lse - new_lse) * flash_out
        )
        retrieval_out = final_out.to(q_dtype)

    retrieval_out = retrieval_out.transpose(1, 2)
    return retrieval_out


def llama_retr_flash_attention_forward(
    self,
    top_k: int,
    from_layer: int,
    total_layers: int,
    insert_db,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    dropout_rate = self.attention_dropout if self.training else 0.0

    do_strllm_until_layer = total_layers - 1
    core = os.cpu_count() // 2

    # prefill phase, using flash attn
    if key_states.shape[2] != 1:
        # (1) async update
        # past_key_value.sync_key_update(key_states, self.layer_idx)
        # past_key_value.sync_value_update(value_states, self.layer_idx)
        past_key_value.async_update(
            key_states.contiguous(),
            value_states,
            self.layer_idx,
            do_strllm_until_layer,
            core,
        )

        # (2) computation
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            causal=True,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        # (3) sync
        if self.layer_idx == total_layers - 1:
            past_key_value.sync()

    # decoding phase
    else:
        key_cache, gpu_key_cache = past_key_value.sync_key_update(
            key_states, self.layer_idx, do_strllm_until_layer, core, insert_db
        )
        value_cache, gpu_value_cache = past_key_value.sync_value_update(
            value_states, self.layer_idx, do_strllm_until_layer, insert_db
        )

        attn_output = retr_attn(
            query_states,
            key_cache,
            value_cache,
            gpu_key_cache,
            gpu_value_cache,
            self.num_key_value_groups,
            self.layer_idx,
            top_k,
            from_layer,
            core,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        if attn_output.dtype != torch.bfloat16:
            attn_output = attn_output.to(torch.bfloat16)
        attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
