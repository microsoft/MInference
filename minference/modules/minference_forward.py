# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import inspect
import json
import os
from importlib import import_module

from transformers.models.llama.modeling_llama import *
from transformers.utils.import_utils import _is_package_available

if _is_package_available("vllm"):
    from vllm.attention.backends.flash_attn import *

from ..ops.block_sparse_flash_attention import block_sparse_attention
from ..ops.pit_sparse_flash_attention_v2 import vertical_slash_sparse_attention
from ..ops.streaming_kernel import streaming_forward, streaming_forward2
from .snap_kv import *

last_q = 64
arange = torch.arange(last_q, device="cuda")
LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]
ROPE_TYPE = None
SEARCH_MASK = None

def set_rope_type(self):
    global ROPE_TYPE
    if ROPE_TYPE is not None:
        return
    if "seq_len" in inspect.signature(self.rotary_emb.forward).parameters:
        ROPE_TYPE = "seq_len"
    elif "max_seq_len" in inspect.signature(self.rotary_emb.forward).parameters:
        ROPE_TYPE = "max_seq_len"
    else:
        ROPE_TYPE = "position_ids"

def get_cos_sin(self, value_states, kv_seq_len, position_ids):
    if ROPE_TYPE == "seq_len":
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    elif ROPE_TYPE == "max_seq_len":
        cos = self.rotary_emb(kv_seq_len)
        if position_ids is not None:
            cos = cos[position_ids]
        else:
            cos = cos[None, :kv_seq_len]
        sin = None
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
    return cos, sin


def init_minference_parameters(self):
    config = self.config.to_dict()
    self.starting_layer = config.get("starting_layer", 0)
    self.is_search = config.get("is_search", False)

    self.ne_inf = None
    self.config_path = config.get("config_path", "")
    if (
        self.config_path is not None and
        os.path.exists(self.config_path) and
        self.layer_idx < len(json.load(open(self.config_path)))
    ):
        self.best_pattern = {int(ii): jj for ii, jj in json.load(open(self.config_path))[self.layer_idx].items()}
    else:
        self.best_pattern = {}
    self.vertical, self.slash = None, None

    # import apply_rotary_pos_emb
    if "apply_rotary_pos_emb" not in self.__dict__:
        global apply_rotary_pos_emb
        model_path = self.rotary_emb.__class__.__module__
        apply_rotary_pos_emb = getattr(import_module(model_path), "apply_rotary_pos_emb")
        self.apply_rotary_pos_emb = True

def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), -1) # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)) # Change the strides
    sum_diags = torch.sum(mat_strided, 2) # Sums the resulting matrix's columns
    return sum_diags[:,:,1:]

def gather(t, dim, i):
    """A broadcasting version of torch.gather."""
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))

def gather_qkv(q, k, v, attention_mask):
    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1)) + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    return attn_output

def search_pattern(q, k, head):
    q_len = q.shape[2]
    head_dim = q.shape[-1]

    def vertical_and_slash(vertical_size, slash_size):
        last_q = 64
        q_len = q.shape[2]
        qk_idxs = [ii + q_len for ii in list(range(-last_q, 0, 1))]
        qk = torch.matmul(q[:,:,qk_idxs,:], k.transpose(2, 3))/ math.sqrt(head_dim) + attention_mask[:,:,qk_idxs]
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical_topk = torch.topk(-vertical, q_len - vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-30:] = torch.inf
        slash_topk = slash
        slash = torch.topk(slash, slash_size, -1).indices - (q_len - 1)
        slash = torch.stack([torch.sparse.spdiags(torch.ones(slash_size, q_len), slash.cpu()[0][_], (q_len, q_len)).to_dense() for _ in range(1)]).to(q.device)

        est_attn = torch.ones_like(attn_weights)
        dim = 3
        est_attn = est_attn.scatter(3, vertical_topk.expand(*est_attn.shape[:dim], vertical_topk.shape[dim], *est_attn.shape[dim + 1 :]), 0)
        est_attn = est_attn + slash

        est_attn = (est_attn > 0).float()
        est_attn = torch.tril(est_attn)
        attn_weights_x = attn_weights * est_attn
        res3 = attn_weights_x[:,:,2500:].sum(-1).mean(-1).squeeze().float().detach().cpu().numpy()
        return res3

    def stream_llm(vertical_size, slash_size):
        q_len = q.shape[2]

        mask = torch.triu(torch.tril(torch.ones(q_len, q_len), 0), -slash_size).to(q)
        mask[:,:vertical_size] = 1
        mask = mask.unsqueeze(0).unsqueeze(1)

        est_attn = torch.tril(mask)
        attn_weights_x = attn_weights * est_attn
        res3 = attn_weights_x[:,:,2500:].sum(-1).mean(-1).squeeze().float().detach().cpu().numpy()
        return res3

    def block_sparse(topk_ratio, slash_size=None):
        block_num = (q_len -1) // 32 + 1
        block_q = torch.zeros(1,1,block_num * 32,head_dim).to(q)
        block_q[:,:,:q_len] = q
        block_q = block_q.reshape(1,1,block_num,32,-1).mean(-2)
        block_k = torch.zeros(1,1,block_num * 32,head_dim).to(k)
        block_k[:,:,:q_len] = k
        block_k = block_k.reshape(1,1,block_num,32,-1).mean(-2)

        qk = torch.matmul(block_q, block_k.transpose(2, 3)) + attention_mask[:,:,:block_num,:block_num]
        est_attn = torch.ones_like(qk)
        block_topk = torch.topk(-qk, block_num - block_num//topk_ratio, -1).indices

        dim = 3
        est_attn = est_attn.scatter(3, block_topk.expand(*est_attn.shape[:dim], block_topk.shape[dim], *est_attn.shape[dim + 1 :]), 0)
        est_attn = est_attn.unsqueeze(3).unsqueeze(-1).repeat(1,1,1,32,1,32).reshape(1,1,block_num * 32, block_num * 32)[...,:q_len,:q_len]
        est_attn = torch.tril(est_attn)

        attn_weights_x = attn_weights * est_attn
        res2 = attn_weights_x[:,:,2500:].sum(-1).mean(-1).squeeze().float().detach().cpu().numpy()
        return res2

    global SEARCH_MASK
    if SEARCH_MASK is None:
        attention_mask = torch.full((q_len, q_len), torch.finfo(q.dtype).min, device="cuda")
        mask_cond = torch.arange(attention_mask.size(-1), device="cuda")
        attention_mask.masked_fill_(mask_cond < (mask_cond + 1).view(attention_mask.size(-1), 1), 0)
        attention_mask = attention_mask[None, None, :]
        SEARCH_MASK = attention_mask
    else:
        attention_mask = SEARCH_MASK
    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim) + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    best_s, best_v, best_score, best_ty = 0, 0, 0, ""
    all_info = []
    for ty, fc in [("stream_llm", stream_llm), ("vertical_and_slash", vertical_and_slash), ("block_sparse", block_sparse)]:
        if ty == "stream_llm":
            vs_list = [(100, 800)]
        elif ty == "vertical_and_slash":
            vs_list = [(30, 800), (100, 750), (500, 700), (3500, 100)]
        else:
            vs_list = [(8, 1)]
        for v_size, s_size in vs_list:
            score = fc(v_size, s_size)
            score = score.item()
            all_info.append([ty, v_size, s_size, score])
            if score > best_score:
                best_score = score
                best_s, best_v = s_size, v_size
                best_ty = ty
    if best_ty == "stream_llm":
        best_ty = "vertical_and_slash"
    if best_ty == "block_sparse":
        best_ty, best_v, best_s = "vertical_and_slash", 1000, 6096
    print(head, best_ty, best_v, best_s, best_score)
    return (best_ty, best_v, best_s, best_score)

def search_pattern_v2(q, k, v, head):
    q_len = q.shape[2]
    head_dim = q.shape[-1]
    def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        last_q = 64
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK, qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-30:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
    def dense(q, k, v, vertical_size=None, slash_size=None):
        return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2), 0.0, softmax_scale=None, causal=q_len != 1).view(bsz, 1, q_len, head_dim)
    def block_sparse_kernel(q, k, v, vertical_size=None, slash_size=None):
        topk = 100
        return block_sparse_attention(q, k, v, topk)

    best_s, best_v, best_score, best_ty = 0, 0, float("inf"), ""
    bsz = q.shape[0]
    all_info = []
    ref = dense(q, k, v)
    for ty, fc in [("stream_llm", streaming_forward), ("vertical_and_slash", vertical_and_slash_kernel), ("block_sparse", block_sparse_kernel)]:
        if ty == "stream_llm":
            vs_list = [(100, 800)]
        elif ty == "vertical_and_slash":
            vs_list = [(30, 800), (100, 800), (100, 750), (500, 700), (3500, 100), (1000, 4096)]
        else:
            vs_list = [(10, 1)]
        for v_size, s_size in vs_list:
            score = fc(q, k, v, v_size, s_size)
            # delta = (ref - score).abs().sum()
            delta = ((ref - score).abs() > 5e-3).sum()
            score = delta.item()
            all_info.append([ty, v_size, s_size, score])
            if score < best_score:
                best_score = score
                best_s, best_v = s_size, v_size
                best_ty = ty
    print(head, best_ty, best_v, best_s, best_score)
    return all_info

def shift_matrix(mat):
    b, h, _, n = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), -1) # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((1, 1, n, n + 2 * n), (1, n * (2 * n + n), 2 * n + n - 1, 1)) # Change the strides
    return mat_strided[...,2 * n-1:-1]

def repeat(self, q, k, v, attention_mask):
    q_len = q.shape[2]
    if q_len == 1:
        return gather_qkv(q, k, v, attention_mask)
    qk = torch.matmul(q[:,:,-1:,:], k.transpose(2, 3)) / math.sqrt(self.head_dim)
    qk = qk.repeat(1,1,q_len, 1)
    qk = shift_matrix(qk) + attention_mask
    attn_weights = nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    return attn_output

def gather_last_q_vertical_slash_topk_v4(self, q, k, v, head_id):
    kv_seq_len = k.size(2)

    def vertical_and_slash(attn_weights, vertical_size, slash_size):
        last_q = 64
        q_len = q.shape[2]
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        qk_idxs = [ii + q_len for ii in list(range(-last_q, 0, 1))]
        qk = torch.matmul(q[:,:,qk_idxs,:], k.transpose(2, 3))/ math.sqrt(self.head_dim) + attention_mask[:,:,qk_idxs]
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = -self.ne_inf
        vertical_topk = torch.topk(-vertical, q_len - vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-30:] = -self.ne_inf
        slash_topk = slash
        slash = torch.topk(slash, slash_size, -1).indices - (q_len - 1)
        slash = torch.stack([torch.sparse.spdiags(torch.ones(slash_size, q_len), slash.cpu()[0][_], (q_len, q_len)).to_dense() for _ in range(1)]).to(q.device)

        est_attn = torch.ones_like(attn_weights)
        dim = 3
        est_attn = est_attn.scatter(3, vertical_topk.expand(*est_attn.shape[:dim], vertical_topk.shape[dim], *est_attn.shape[dim + 1 :]), 0)
        est_attn = est_attn + slash

        est_attn = (est_attn > 0).float()
        est_attn = torch.tril(est_attn)
        est_attn = (est_attn == 0).int() * self.ne_inf
        attn_weights = attn_weights + est_attn
        if self.kv_cache_compressed_v4:
            self.vertical = torch.topk(vertical, vertical_size * 4, -1).indices
            self.slash = (torch.topk(slash_topk, slash_size * 4, -1).indices - (q_len - 1)).unsqueeze(2)
        return attn_weights

    def stream_llm(attn_weights, vertical_size, slash_size):
        q_len = q.shape[2]
        vertical_size, slash_size = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        mask = torch.triu(torch.tril(torch.ones(q_len, q_len), 0), -slash_size).to(q)
        mask[:,:vertical_size] = 1
        mask = mask.unsqueeze(0).unsqueeze(1)

        est_attn = torch.tril(mask)
        est_attn = (est_attn == 0).int() * self.ne_inf
        attn_weights = attn_weights + est_attn
        if self.kv_cache_compressed_v4:
            self.vertical = torch.Tensor(list(range(vertical_size * 4))).long().to(q.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.slash = torch.Tensor(list(range(-slash_size * 4, 1))).long().to(q.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return attn_weights

    def block_sparse(attn_weights, topk_ratio, slash_size=None, block_size=8):
        block_num = (q_len -1) // block_size + 1
        block_q = torch.zeros(1,1,block_num * block_size,head_dim).to(q)
        block_q[:,:,:q_len] = q
        block_q = block_q.reshape(1,1,block_num,block_size,-1).mean(-2)
        block_k = torch.zeros(1,1,block_num * block_size,head_dim).to(k)
        block_k[:,:,:q_len] = k
        block_k = block_k.reshape(1,1,block_num,block_size,-1).mean(-2)

        qk = torch.matmul(block_q, block_k.transpose(2, 3)) + attention_mask[:,:,:block_num,:block_num]
        est_attn = torch.ones_like(qk)
        block_topk = torch.topk(-qk, block_num - block_num//topk_ratio, -1).indices

        dim = 3
        est_attn = est_attn.scatter(3, block_topk.expand(*est_attn.shape[:dim], block_topk.shape[dim], *est_attn.shape[dim + 1 :]), 0)
        est_attn = est_attn.unsqueeze(3).unsqueeze(-1).repeat(1,1,1,block_size,1,block_size).reshape(1,1,block_num * block_size, block_num * block_size)[...,:q_len,:q_len]
        est_attn = torch.tril(est_attn)
        est_attn = (est_attn == 0).int()
        attn_weights = attn_weights + est_attn
        return attn_weights

    def dialted(q,k,v, type):
        q_len = q.shape[2]
        n_init = min(1024, q_len)
        vertical_topk = torch.arange(0, n_init, device=q.device)[None, None, None, :]

        slash = torch.arange(0, q_len, device=q.device)
        if type == 'dilated1':
            # 8k local with 1 interval
            slash = slash[-8192::2][None, None, :]
        elif type == 'dilated2':
            # 2k dense local + 4k local with 1 interval
            slash = torch.cat([slash[-2048:], slash[-6144:-2048:2]], 0)[None, None, :]

        slash = (q_len - 1) - slash
        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        last_q = min(64, q_len)
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-100:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    def vertical_and_slash_kernel_extend(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size + 100, 30)), min(q_len, max(slash_size, 50))
        last_q = min(64, q_len)
        last_start = 100
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q-last_start:-last_start,:], k)
        qk[:, :, :, -last_start:] = -torch.inf
        qk[:, :, :, -last_q-last_start:-last_start] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q-last_start:-last_start], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical[...,-100:] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-100:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    def vertical_and_slash_kernel_static(q, k, v, vertical_size, slash_size):
        if "vs" in self.__dict__:
            vertical_topk, slash = self.vs
        else:
            vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
            last_q = 64
            qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k)
            qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK, qk[:, :, :, -last_q:], -torch.inf)
            qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
            vertical = qk.sum(-2, keepdim=True)
            vertical[...,:30] = torch.inf
            vertical_topk = torch.topk(vertical, vertical_size, -1).indices

            slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
            slash[...,-30:] = torch.inf
            slash_topk = slash
            slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices
            self.vs = vertical_topk, slash

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
    def dense(q, k, v, vertical_size=None, slash_size=None):
        return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2), 0.0, softmax_scale=None, causal=q_len != 1).view(bsz, 1, q_len, self.head_dim)
    def block_sparse_kernel(q, k, v, vertical_size=None, slash_size=None):
        topk = 100
        return block_sparse_attention(q, k, v, topk)

    q_len = q.shape[2]
    bsz = q.shape[0]

    if q_len == 1:
        return dense(q, k, v)

    if self.config.to_dict().get("dilated1", False):
        return dialted(q, k, v, 'dilated1')
    if self.config.to_dict().get("dilated2", False):
        return dialted(q, k, v, 'dilated2')
    if self.config.to_dict().get("dense", False):
        return dense(q, k, v)
    if self.config.to_dict().get("streaming", False):
        return streaming_forward(q, k, v, self.config.streaming_kwargs["n_init"], self.config.streaming_kwargs["n_local"])

    ty, vertical_size, slash_size, _ = self.best_pattern.get(head_id, ("vertical_and_slash", 1000, 6096, 1))

    if self.config.to_dict().get("static_pattern", False):
        return vertical_and_slash_kernel_static(q, k, v, vertical_size, slash_size)
    if self.config.to_dict().get("vs_only", False):
        return vertical_and_slash_kernel(q, k, v, vertical_size, slash_size)

    fc = {
        "stream_llm": streaming_forward,
        "vertical_and_slash": vertical_and_slash_kernel,
        "block_sparse": block_sparse_kernel,
    }[ty]
    return fc(q, k, v, vertical_size, slash_size)

def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin)

def minference_forward():
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        **kwargs,
    ):
        self.init_minference_parameters()
        self.ne_inf = torch.finfo(hidden_states.dtype).min

        bsz, q_len, _ = hidden_states.size()

        if "q_proj" in self.__dict__["_modules"]:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
            query_pos = self.num_heads * self.head_dim
            key_value_pos = query_pos // self.num_key_value_groups
            query_states, key_states, value_states = torch.split(qkv, [query_pos, key_value_pos, key_value_pos], -1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        set_rope_type(self)
        cos, sin = get_cos_sin(self, value_states, kv_seq_len, position_ids)
        if ROPE_TYPE == "max_seq_len":
            query_states = apply_rotary_pos_emb(query_states, cos)
            key_states = apply_rotary_pos_emb(key_states, cos)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if self.is_search:
            if os.path.exists(self.config_path):
                config_list = json.load(open(self.config_path))
                if self.layer_idx < len(config_list):
                    assert False
            else:
                config_list = []
            config = {}
            print("Layer", self.layer_idx)
        if q_len != 1:
            output = torch.empty_like(query_states)
            for head in range(query_states.size(1)):
                q = query_states[:, head, :, :].unsqueeze(1)
                k = key_states[:, head, :, :].unsqueeze(1)
                v = value_states[:, head, :, :].unsqueeze(1)
                if self.is_search:
                    config[head] = search_pattern(q, k, head)
                if self.layer_idx >= self.starting_layer and not self.is_search:
                    attn_output = self.gather_last_q_vertical_slash_topk_v4(q, k, v, head)
                elif is_flash_attn_2_available():
                    attn_output = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2), 0.0, softmax_scale=None, causal=q_len != 1).view(bsz, 1, q_len, self.head_dim)
                else:
                    attn_output = gather_qkv(q, k, v, attention_mask)
                output[:, head:head + 1] = attn_output
            if self.is_search:
                config_list.append(config)
                with open(self.config_path, 'w') as json_file:
                    json.dump(config_list, json_file)
        else:
            output =  flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1,2), 0.0, softmax_scale=None, causal=q_len != 1).view(bsz, query_states.size(1), q_len, self.head_dim)
        attn_output = output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward

def minference_kv_cache_cpu_forward():
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        **kwargs,
    ):
        self.init_minference_parameters()
        self.ne_inf = torch.finfo(hidden_states.dtype).min

        bsz, q_len, hidden_dim = hidden_states.size()
        kv_seq_len = q_len
        if use_cache and past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        set_rope_type(self)
        cos, sin = get_cos_sin(self, hidden_states, kv_seq_len, position_ids)
        cache_kwargs = {"sin": sin, "cos": cos}
        kv_cache_cpu_device = self.config.to_dict().get("kv_cache_cpu_device", "cpu")

        attn_out = torch.empty_like(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        act_num_heads = self.num_heads // self.num_key_value_groups
        if use_cache:
            k = torch.zeros(bsz, act_num_heads, q_len, self.head_dim).to(hidden_states.dtype).to(kv_cache_cpu_device)
            v = torch.zeros(bsz, act_num_heads, q_len, self.head_dim).to(hidden_states.dtype).to(kv_cache_cpu_device)
        part_k, part_v = None, None
        for head in range(self.num_heads):
            if "q_proj" in self.__dict__["_modules"]:
                part_q = F.linear(hidden_states, self.q_proj.weight.view(self.num_heads, self.head_dim, hidden_dim)[head]).unsqueeze(2)
                if self.q_proj.bias is not None:
                    part_q += self.q_proj.bias.view(self.num_heads, self.head_dim)[head]
            else:
                query_pos = self.num_heads * self.head_dim
                part_q = F.linear(hidden_states, self.qkv_proj.weight[:query_pos].view(self.num_heads, self.head_dim, hidden_dim)[head]).unsqueeze(2)
                if self.qkv_proj.bias is not None:
                    part_q += self.qkv_proj.bias[:query_pos].view(self.num_heads, self.head_dim)[head]

            if ROPE_TYPE == "max_seq_len":
                part_q = apply_rotary_pos_emb(part_q.transpose(1, 2), cos)
            else:
                part_q = apply_rotary_pos_emb_single(part_q.transpose(1, 2), cos, sin, position_ids)

            if head % self.num_key_value_groups == 0:
                if "q_proj" in self.__dict__["_modules"]:
                    part_k = F.linear(hidden_states, self.k_proj.weight.view(act_num_heads, self.head_dim, hidden_dim)[head // self.num_key_value_groups]).unsqueeze(2)
                    part_v = F.linear(hidden_states, self.v_proj.weight.view(act_num_heads, self.head_dim, hidden_dim)[head // self.num_key_value_groups]).unsqueeze(2).transpose(1, 2)
                    if self.k_proj.bias is not None:
                        part_k += self.k_proj.bias.view(act_num_heads, self.head_dim)[head // self.num_key_value_groups]
                    if self.v_proj.bias is not None:
                        part_v += self.v_proj.bias.view(act_num_heads, self.head_dim)[head // self.num_key_value_groups]
                else:
                    query_pos = self.num_heads * self.head_dim
                    part_k = F.linear(hidden_states, self.qkv_proj.weight[query_pos:].view(2, act_num_heads, self.head_dim, hidden_dim)[0][head // self.num_key_value_groups]).unsqueeze(2)
                    part_v = F.linear(hidden_states, self.qkv_proj.weight[query_pos:].view(2, act_num_heads, self.head_dim, hidden_dim)[1][head // self.num_key_value_groups]).unsqueeze(2).transpose(1, 2)
                    if self.qkv_proj.bias is not None:
                        part_k += self.qkv_proj.bias[query_pos:].view(2, act_num_heads, self.head_dim)[0][head // self.num_key_value_groups]
                        part_v += self.qkv_proj.bias[query_pos:].view(2, act_num_heads, self.head_dim)[1][head // self.num_key_value_groups]

                if ROPE_TYPE == "max_seq_len":
                    part_k = apply_rotary_pos_emb(part_k.transpose(1, 2), cos)
                else:
                    part_k = apply_rotary_pos_emb_single(part_k.transpose(1, 2), cos, sin, position_ids)
                if use_cache and past_key_value is not None:
                    k[:,head // self.num_key_value_groups] = part_k.to(kv_cache_cpu_device)
                    v[:,head // self.num_key_value_groups] = part_v.to(kv_cache_cpu_device)
                    part_k, part_v = past_key_value.get(part_k, part_v, self.layer_idx, head // self.num_key_value_groups, cache_kwargs)

            if self.layer_idx >= self.starting_layer:
                part_o = self.gather_last_q_vertical_slash_topk_v4(part_q, part_k, part_v, head)
            else:
                part_o = flash_attn_func(part_q, part_k, part_v.transpose(1, 2), 0.0, softmax_scale=None, causal=True).view(bsz, part_q.shape[1], self.head_dim)
            attn_out[:, :, head, :] = part_o

        if use_cache and past_key_value is not None:
            past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        torch.matmul(attn_out.view(bsz, q_len, hidden_dim), self.o_proj.weight.T, out=hidden_states)
        torch.cuda.empty_cache()
        return (hidden_states, None, past_key_value)

    return forward

def minference_with_snapkv_forward():
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        **kwargs,
    ):
        self.init_minference_parameters()
        self.ne_inf = torch.finfo(hidden_states.dtype).min

        init_snapkv(self)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )

            if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
                if self.kv_seq_len != 0:
                    kv_seq_len += self.kv_seq_len
                else:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        set_rope_type(self)
        cos, sin = get_cos_sin(self, value_states, kv_seq_len, position_ids)
        if ROPE_TYPE == "max_seq_len":
            query_states = apply_rotary_pos_emb(query_states, cos)
            key_states = apply_rotary_pos_emb(key_states, cos)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            if key_states.shape[-2] == kv_seq_len: # [SnapKV] add kv_cluster
                self.kv_seq_len = kv_seq_len # [SnapKV] register kv_seq_len
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                self.kv_seq_len += q_len
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.layer_idx >= self.starting_layer:
            assert query_states.size(1) == key_states.size(1) == value_states.size(1)
            output = torch.empty_like(query_states)
            for head in range(query_states.size(1)):
                q = query_states[:, head, :, :].unsqueeze(1)
                k = key_states[:, head, :, :].unsqueeze(1)
                v = value_states[:, head, :, :].unsqueeze(1)
                output[:, head:head + 1] = self.gather_last_q_vertical_slash_topk_v4(q, k, v, head)

            attn_output = output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

        else:
            output = torch.empty_like(query_states)
            for head in range(query_states.size(1)):
                q = query_states[:, head, :, :].unsqueeze(1)
                k = key_states[:, head, :, :].unsqueeze(1)
                v = value_states[:, head, :, :].unsqueeze(1)
                if is_flash_attn_2_available():
                    attn_output = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2), 0.0, softmax_scale=None, causal=q_len != 1).view(bsz, 1, q.shape[2], self.head_dim)
                else:
                    attn_output = gather_qkv(q, k, v, attention_mask)
                output[:, head:head + 1] = attn_output
            attn_output = output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

    return forward

def gather_last_q_vertical_slash_topk_vllm(self, q, k, v, head_id):
    kv_seq_len = k.size(2)
    head_dim = q.size(-1)

    def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        last_q = min(64, q_len)
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k)

        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:], qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-100:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    def block_sparse_kernel(q, k, v, vertical_size=None, slash_size=None):
        topk = 100
        return block_sparse_attention(q, k, v, topk)

    def dense(q, k, v, vertical_size=None, slash_size=None):
        return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2), 0.0, softmax_scale=None, causal=q_len != 1).view(bsz, 1, q_len, head_dim)

    q_len = q.shape[2]
    bsz = q.shape[0]

    ty, vertical_size, slash_size, _ = self.best_pattern[head_id]

    if q_len == 1:
        return dense(q, k, v)

    fc = {
        "stream_llm": streaming_forward,
        "vertical_and_slash": vertical_and_slash_kernel,
        "block_sparse": block_sparse_kernel,
    }[ty]
    return fc(q, k, v, vertical_size, slash_size)

def minference_vllm_forward(
    pattern_config
):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata[FlashAttentionMetadata],
        kv_scale: float,
        layer_idx: int,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        self.best_pattern = {int(ii): jj for ii, jj in pattern_config[layer_idx].items()}
        def repeat_kv(hidden_states, n_rep):
            sqlen, num_head, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :].expand(sqlen, num_head, n_rep, head_dim)
            return hidden_states.reshape(sqlen, num_head * n_rep, head_dim)

        def minference_prefill_func(
            q, k, v,
        ):
            # (seq_len, num_heads, head_size)
            if q.size(-2) != k.size(-2):
                k = repeat_kv(k, q.size(-2) // k.size(-2))
                v = repeat_kv(v, q.size(-2) // v.size(-2))

            output = torch.empty_like(q)
            for head in range(q.size(-2)):
                q_head = q[:, head, :].unsqueeze(1)
                k_head = k[:, head, :].unsqueeze(1)
                v_head = v[:, head, :].unsqueeze(1)

                # (1, seq_len, num_heads, head_size)
                q_head = q_head[None, ...]
                k_head = k_head[None, ...]
                v_head = v_head[None, ...]

                q_head = q_head.transpose(1, 2)
                k_head = k_head.transpose(1, 2)
                v_head = v_head.transpose(1, 2)

                out = self.gather_last_q_vertical_slash_topk_vllm(q_head, k_head, v_head, head)

                out = out.transpose(1, 2).squeeze(0).contiguous()
                output[:, head:head+1, :] = out
            return output

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype,
                                                kv_scale)

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                # (seq_len, num_heads, head_size)
                # out = flash_attn_varlen_func(
                #     q=query,
                #     k=key,
                #     v=value,
                #     cu_seqlens_q=prefill_meta.seq_start_loc,
                #     cu_seqlens_k=prefill_meta.seq_start_loc,
                #     max_seqlen_q=prefill_meta.max_prompt_len,
                #     max_seqlen_k=prefill_meta.max_prompt_len,
                #     softmax_scale=self.scale,
                #     causal=True,
                #     window_size=self.sliding_window,
                #     alibi_slopes=self.alibi_slopes,
                # )
                out = minference_prefill_func(query, key, value)
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                output[:num_prefill_tokens] = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.subquery_start_loc,
                    prefill_meta.prompt_lens_tensor,
                    prefill_meta.context_lens,
                    prefill_meta.max_subquery_len,
                    self.alibi_slopes,
                )
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.context_lens,
                decode_meta.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

    return forward
