from functools import partial
from .minference_forward import *
import math
import numpy as np
from ..ops.grid_attention import multimodal_grid_attention
from ..ops.convert_idx import mix_modality_vertical_slash_indexes_triton
from ..ops.pit_sparse_flash_attention_v2 import _triton_mixed_sparse_attention, flops_counter
from ..ops.fa_cross import streaming_cross_attention
from ..cuda import convert_vertical_slash_indexes
from ..ops.streaming_kernel import a_shape_kernel
from transformers.utils.import_utils import _is_package_available

from collections import defaultdict

def compute_overlap_matrix(sequences):
    def pairwise_overlap(a, b):
        if len(a) <= len(b):
            shorter, longer = a, b
        else:
            shorter, longer = b, a
        mask = torch.isin(shorter, longer)
        count = torch.sum(mask).item()
        return (count / len(shorter)) * 100  # Return tensor-friendly value

    n = len(sequences)
    matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # Avoid redundant pairs: compute only when j >= i
            overlap = pairwise_overlap(sequences[i], sequences[j])
            matrix[i][j] = overlap
            if i != j:  # Mirror the value to the symmetric position
                matrix[j][i] = -1
    return matrix

def create_vs_mask(n, indices, range_start, range_end, target_mask):
    h = range_end - range_start
    w = n
    diag_values = torch.tensor([idx - range_start for idx in indices], dtype=torch.long)
    
    rows = torch.arange(h).unsqueeze(1)  # Shape (h, 1)
    cols = torch.arange(w).unsqueeze(0)  # Shape (1, w)
    diff = rows - cols  # Shape (h, w)
    
    # Expand dimensions for broadcasting and check equality
    mask = (diff.unsqueeze(-1) == diag_values.view(1, 1, -1)).any(dim=-1).int()
    
    target_mask[range_start:range_end] = mask
    return target_mask

def dense_attention(q, k):
    q_len = q.size(-2)
    causal_mask = torch.triu(torch.ones(q_len, q_len), diagonal=1).to(q.device) * torch.finfo(q.dtype).min
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    return attn_weights

def dense_forward(
    q, k, v, # [bsz, num_heads, seqlen, head_dim]
):
    attn_out = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2), v.transpose(1, 2),
        causal=True,
    )
    attn_out = attn_out.transpose(1, 2).to(q.dtype)
    return attn_out
    
def a_shape(attn_scores, n_init, n_local):
    q_len = attn_scores.size(-2)
    mask = torch.tril(torch.triu(torch.ones(q_len, q_len), diagonal=-(n_local-1)))
    mask[:,:n_init] = 1
    mask = mask.to(attn_scores.device)
    est_attn = torch.where(mask.bool(), attn_scores, 0)
    attn_recall = est_attn.sum(-1).mean(-1).squeeze().float().item()
    
    return attn_recall

def grid_attn(attn_scores, stride, h=True, v=True, s=True, n_local=None, section_size=None):
    def get_grid_indices_efficient(q_len, section_size, stride, shift=0):
        base_indices = torch.arange(0, section_size, stride)
        n_full_sections = q_len // (section_size)
        section_offsets = torch.arange(n_full_sections + 1) * (section_size)
        indices = (base_indices.view(-1, 1) + section_offsets.view(1, -1) + shift).flatten()
        return indices[indices < q_len]

    q_len = attn_scores.size(-2)
    indices = torch.arange(q_len)
    x, y = torch.meshgrid(indices, indices, indexing='ij')
    candidate_shifts = [0, 1] + (np.array([-1]) + stride).tolist()
    
    best_recall = -1
    best_shift = 0
    for shift in candidate_shifts:
        if s:
            if n_local:
                mask = (((((x - y) + shift) % stride == 0) | (x - y) <= n_local) & (x >= y)).float()
            else:
                mask = ((((x - y) + shift) % stride == 0) & (x >= y)).float()
        else:
            if n_local:
                mask = (((x - y) <= n_local) & (x >= y)).float()
            else:
                mask = torch.zeros_like(attn_scores)

        if h:
            if section_size:
                mask[get_grid_indices_efficient(q_len, section_size, stride, shift),:] = 1
            else:
                mask[shift::stride,:] = 1
        if v:
            if section_size:
                mask[:,get_grid_indices_efficient(q_len, section_size, stride, shift)] = 1
            else:
                mask[:,shift::stride] = 1
        mask = mask.to(attn_scores.device)
        est_attn = torch.where(mask.bool(), attn_scores, 0)
        attn_recall = est_attn.sum(-1).mean(-1).squeeze().float().item()

        if attn_recall > best_recall:
            best_recall = attn_recall
            best_shift = shift
    
    return best_recall, best_shift

def vs_attn(q, k, attn_scores, v_size, s_size, last_q=64):
    q_len = attn_scores.size(-2)
    qk = torch.einsum(f'mk, nk -> mn', q[-last_q:,:], k) / math.sqrt(q.size(-1))
    causal_mask = torch.triu(torch.ones(last_q, q_len), diagonal=((q_len-last_q)+1)).to(q.device) * torch.finfo(q.dtype).min
    qk = qk + causal_mask
    qk = nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)
    vertical = qk.sum(-2, keepdim=True)
    vertical[...,:30] = torch.inf
    vertical_topk = torch.topk(vertical, v_size, -1).indices

    slash = sum_all_diagonal_matrix(qk[None, None, ...])[...,:-last_q + 1]
    slash[...,-30:] = torch.inf
    slash = torch.topk(slash, s_size, -1).indices - (q_len - 1)
    slash = torch.stack([torch.sparse.spdiags(torch.ones(s_size, q_len), slash.cpu()[0][_], (q_len, q_len)).to_dense() for _ in range(1)]).to(q.device)

    slash[..., :, vertical_topk] = 1
    est_attn = torch.where(slash[0, ...].bool(), attn_scores, 0)
    attn_recall = est_attn.sum(-1).mean(-1).squeeze().float().item()
    
    return attn_recall

def dynamic_grid(q, k, attn_scores, v_size, h_size, local_size = 1024):
    q_len = attn_scores.size(-2)
    last_q = 64
    qk = torch.einsum(f'mk, nk -> mn', q[-last_q:,:], k) / math.sqrt(q.size(-1))
    causal_mask = torch.triu(torch.ones(last_q, q_len), diagonal=((q_len-last_q)+1)).to(q.device) * torch.finfo(q.dtype).min
    qk = qk + causal_mask
    qk = nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)
    vertical = qk.sum(-2, keepdim=True)
    vertical[...,:30] = torch.inf
    vertical_topk = torch.topk(vertical, v_size, -1).indices

    qk = torch.matmul(q[local_size:, :], k[(local_size-last_q):local_size, :].T)/ math.sqrt(q.size(-1))
    qk = nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)

    row_std = qk.std(1, keepdim=True)
    horizental = row_std.top(row_std, h_size, largest=False)

    raise NotImplementedError # TODO: implement dynamic grid

def sum_pool_conv(x, k, is_2d=False):
    N, M = x.shape
    if not is_2d:
        pad = (k - (M % k)) % k  # Padding to reach next multiple of k
        x_padded = torch.nn.functional.pad(x, (0, pad))
        # Add channel dimension: (N, M) -> (N, 1, M)
        x = x_padded.unsqueeze(1)
    else:
        pad_m = (k - (M % k)) % k  # Padding to reach next multiple of k
        pad_n = (k - (N % k)) % k  # Padding to reach next multiple of k
        x_padded = torch.nn.functional.pad(x, (0, pad_m, 0, pad_n))
        # Add channel dimension: (N, M) -> (N, 1, M)
        x = x_padded[None, None, ...]
    # Create a Conv1d layer with kernel_size=k, stride=k, and fixed weights=1
    if not is_2d:
        conv = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            stride=k,
            bias=False
        )
    else:
        conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(k, k),
            stride=(k, k),
            bias=False
        )
    # Freeze the weights to 1 (no learning)
    conv.weight.data = torch.ones_like(conv.weight).to(x.device)
    conv.weight.requires_grad = False
    # Apply convolution and squeeze channel dimension
    result = conv(x).squeeze(1)
    return result

def create_typed_spans(boundaries, types):
    # Convert inputs to numpy arrays if they aren't already
    boundaries = np.asarray(boundaries)
    
    # Create masks for t and v types
    t_mask = np.array([t == 't' for t in types])
    v_mask = np.array([t == 'v' for t in types])
    
    # Get the start and end points for each type
    t_starts = boundaries[:-1][t_mask]
    t_ends = boundaries[1:][t_mask]
    v_starts = boundaries[:-1][v_mask]
    v_ends = boundaries[1:][v_mask]
    
    # Create arrays for each type using np.arange
    t_arrays = [np.arange(start, end) for start, end in zip(t_starts, t_ends)]
    v_arrays = [np.arange(start, end) for start, end in zip(v_starts, v_ends)]
    
    # Concatenate all arrays
    # result = np.concatenate(t_arrays + v_arrays)
    result = np.concatenate(v_arrays+t_arrays)
    
    return torch.tensor(result)

def row_wise_analysis(attn_scores, n_rows=None, use_conv=False, tau=0.95):
    if use_conv:
        attn_scores = sum_pool_conv(attn_scores, 64, is_2d=True).flatten(1).sort(descending=True).values.cumsum(1)
        n_rows = attn_scores.size(0)
        tau_per_row = (attn_scores < tau * n_rows).sum(1)
    else:
        tau_per_row = (attn_scores[-n_rows:].sort(descending=True).values.cumsum(1) < tau * n_rows).sum(1)
        std = tau_per_row.to(torch.float).std()
        mean = tau_per_row.float().mean()
        quantile = torch.quantile(tau_per_row.float(), 0.95)
        print(f"[{mean:.1f}", f"{std:.1f}", f"{quantile:.1f}]")
    return tau_per_row

def pattern_search_attn_delta(
    q, k, v,
    layer_idx,
    head_idx,
    input_boundaries,
    frame_stride,
    stride,
    model_name,
):
    assert len(q.shape) == 2
    assert len(k.shape) == 2

    if input_boundaries is None:
        input_boundaries = (0, len(q)-1, len(q))
    if stride is None:
        stride = frame_stride

    frame_stride = int(frame_stride)
    stride = int(stride)
    
    search_space = [
        # ("dynamic_grid", 100, 100),
        # ("grid_attn", frame_stride, True, False, False, 1024),
        ("grid_attn", frame_stride, False, True, False, 1024),
        # ("grid_attn", frame_stride, False, False, True, 1024),
        ("grid_attn", frame_stride, True, True, False, 1024),
        # ("grid_attn", frame_stride, False, True, True, 1024),
        # ("grid_attn", frame_stride, True, True, True, 1024),
        # ("a_shape", 128, 1024),
        # ("a_shape", 128, 2048),
        ("a_shape", 128, 4096),
        # ("grid_attn", stride, True, False, False, 1024),
        ("grid_attn", stride, False, True, False, 1024),
        # ("grid_attn", stride, False, False, True, 1024),
        ("grid_attn", stride, True, True, False, 1024),
        # ("grid_attn", stride, False, True, True, 1024),
        # ("grid_attn", stride, True, True, True, 1024),
        # ("dynamic_grid", 100, 100),
        # ("dynamic_grid", 100, 500),
        # ("dynamic_grid", 500, 500),
        # ("dynamic_grid", 500, 1000),
        # ("dynamic_grid", 1000, 1000),
        # ("vs_attn", 30, 800),
        ("vs_attn", 1000, 1024, 257),
        ("vs_attn", 1000, 2048, 257),
        # ("vs_attn", 2000, 2048, 257),
        # ("vs_attn", 1000, 3096, 257),
        # ("vs_attn", 2000, 3096, 257),
        # ("vs_attn", 1000, 4096, 257),
        # ("vs_attn", 2000, 4096, 257),
        # ("vs_attn", 1000, 1024, 64),
        # ("vs_attn", 1000, 2048, 64),
        # ("vs_attn", 2000, 2048, 64),
        # ("vs_attn", 1000, 3096, 64),
        # ("vs_attn", 2000, 3096, 64),
        # ("vs_attn", 1000, 4096, 64),
        # ("vs_attn", 2000, 4096, 64),
        # ("vs_attn", 3500, 200),
        # ("vs_attn", 1000, 2500),
    ]

    attn_scores = dense_attention(q, k)
    vision_start = input_boundaries[0]
    vision_end = input_boundaries[1]
    attn_scores = attn_scores[
        ...,
        vision_start:vision_end,
        # :vision_end
        vision_start:vision_end
    ]
    q = q[..., vision_start:vision_end, :]
    k = k[..., vision_start:vision_end, :]

    name2func = {
        "grid_attn": partial(grid_attn, attn_scores),
        "a_shape": partial(a_shape, attn_scores),
        "vs_attn": partial(vs_attn, q, k, attn_scores),
        "dynamic_grid": partial(dynamic_grid, q, k, attn_scores),
    }

    best_attn_recall = 0
    best_pattern = None
    minimal_delta = 1e-2
    recalles = {}
    for attn_type, *args in search_space:
        if attn_type == "grid_attn" and model_name == "longvila":
            args.append(frame_stride)
        attn_recall = name2func[attn_type](*args)
        if attn_type == "grid_attn":
            attn_recall, shift = attn_recall
            args += (shift,)
        if attn_recall - best_attn_recall > minimal_delta: # if delta is larger than minimal_delta, update best_pattern
            best_attn_recall = attn_recall
            best_pattern = (attn_type, *args)
        recalles['_'.join(map(str, (attn_type, *args)))] = attn_recall
    return best_pattern, recalles

def pattern_search(
    q, k, v,
    config,
):
    layer_idx = config["layer_idx"]
    input_boundaries = config["input_boundaries"]
    frame_stride = config["frame_start_indices"][1] - config["frame_start_indices"][0]
    stride = config["stride"]
    model_name = config["model_name"]

    q_len = q.size(-2)
    if q_len > 25000: # use delta on attn output
        # pattern_search_func
        raise NotImplementedError
    else: # use delta on attn score
        pattern_search_func = pattern_search_attn_delta
    
    all_best_patterns = {}
    all_recalles = {}
    for head in range(q.size(1)):
        # if layer_idx != 27 or head != 13:
        #     continue
        q_head = q[0, head, :, :]
        k_head = k[0, head, :, :]
        # v_head = v[0, head, :, :]
        best_pattern, recalles = pattern_search_func(q_head, k_head, None, layer_idx, head, input_boundaries, frame_stride, stride, model_name)
        all_best_patterns[head] = best_pattern
        all_recalles[head] = recalles

    patterns_file = f"mminference_best_patterns_{model_name}.json"
    recalls_file = f"mminference_best_recalls_{model_name}.json"
    try:
        with open(patterns_file, "r+") as f:
            best_patterns = json.load(f)
            best_patterns[f"layer-{layer_idx}"] = all_best_patterns
            f.seek(0)
            json.dump(best_patterns, f, indent=1)
            f.truncate()
        with open(recalls_file, "r+") as f:
            recalls = json.load(f)
            recalls[f"layer-{layer_idx}"] = all_recalles
            f.seek(0)
            json.dump(recalls, f, indent=1)
            f.truncate()
    except FileNotFoundError:
        with open(patterns_file, "w") as f:
            json.dump({f"layer-{layer_idx}": all_best_patterns}, f, indent=1)
        with open(recalls_file, "w") as f:
            json.dump({f"layer-{layer_idx}": all_recalles}, f, indent=1)

    return all_best_patterns

def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size, last_q=182):
    q_len, head_dim = q.size(2), q.size(3)
    vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
    last_q = min(last_q, q_len)
    qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k) / math.sqrt(head_dim)
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

def tri_shape_kernel(q, k, v, n_init=30, n_local=1024, n_last=100):
    q_len = q.size(2)

    q1, q2 = q[:,:,:-n_last], q[:,:,-n_last:]
    y1 = streaming_forward(q1, k[:,:,:-n_last], v[:,:,:-n_last], n_init, n_local)

    if _is_package_available("flash_attn"):
        y2 = flash_attn_func(
            q2.transpose(1, 2),
            k.transpose(1, 2), v.transpose(1, 2),
            causal=True,
        )
        y2 = y2.transpose(1, 2).to(q.dtype)
    else:
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q2, k) / math.sqrt(q.shape[-1])
        arange = torch.arange(n_last, device="cuda")
        mask = arange[None, None, :, None] >= arange[None, None, None, :]
        qk[:, :, :, -n_last:] = torch.where(mask, qk[:, :, :, -n_last:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)
        y2 = torch.einsum(f'bhmn, bhnk -> bhmk', qk, v)
    return torch.cat([y1, y2], dim=2)

def mminference_kernel(
    q, k, v,
    head_idx,
    config,
):
    layer_idx = config["layer_idx"]
    input_boundaries = config["input_boundaries"] # [v_start, v_end, seq_len]
    frame_stride = config["frame_start_indices"][1] - config["frame_start_indices"][0]
    stride = config["stride"]
    model_name = config["model_name"]
    grid_attn_func = None

    n_ctx, n_heads, seq_len, head_dim = q.shape
    best_pattern = config["attn_forward_config"]["best_pattern"][str(layer_idx)][str(head_idx)]

    attn_type = best_pattern[0]
    attn_vars = best_pattern[1]

    if attn_type == 'grid_attn':
        if model_name != 'longvila':
            vis_stride, use_hline, use_vline, _, local_window, _, grid_shift = attn_vars
            sink_tokens = (input_boundaries[0] - 1) + grid_shift # include the initial shift tokens to sink, to let the grid start at the right position
            sink_tokens = sink_tokens.item()
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            # rows = vis_start:vis_end:257 and 0:256:8
            if vis_stride == 257:
                vis_start = input_boundaries[0] + grid_shift
                attn_vars = (sink_tokens, local_window, vis_start, input_boundaries[1], frame_stride)
                grid_attn_func = multimodal_grid_attention
            else:
                rows = torch.arange(input_boundaries[0], input_boundaries[1], frame_stride).to(q.device) # [num_rows]
                cols = torch.arange(0, frame_stride, vis_stride).to(q.device) # [num_cols]
                attn_vars = (sink_tokens, local_window, rows, cols)
                grid_attn_func = streaming_cross_attention
                # vis_start = input_boundaries[0] + grid_shift
                # vis_end = q.size(2) - ((q.size(2) - vis_start) % vis_stride)
                # attn_vars = (sink_tokens, local_window, vis_start, vis_end, vis_stride)
                # grid_attn_func = multimodal_grid_attention
        else:
            vis_stride, use_hline, use_vline, _, local_window, grid_shift = attn_vars
            sink_tokens = (input_boundaries[0] - 1) + grid_shift # include the initial shift tokens to sink, to let the grid start at the right position
            vis_start = input_boundaries[0] + grid_shift
            attn_vars = (sink_tokens, local_window, vis_start, input_boundaries[1], vis_stride)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            grid_attn_func = multimodal_grid_attention
            # attn_vars = ()
        
    attn_ops = {
        "a_shape": partial(tri_shape_kernel, n_last=(q.size(2) - input_boundaries[1])),
        # "a_shape": streaming_forward,
        "vs_attn": partial(vertical_and_slash_kernel),
        "grid_attn": grid_attn_func,
        "dense": dense_forward,
        # "grid_attn": partial(streaming_forward, n_init=20, n_local=1024), # backup kernel
    }

    return attn_ops[attn_type](q, k, v, *attn_vars)

def topk_vs(q, k, last_q_range, vertical_size, slash_size, local_window=100, init_window=30):
    q_len, head_dim = q.size(2), q.size(3)
    last_q_size = last_q_range[1] - last_q_range[0]
    last_q_start, last_q_end = last_q_range

    vertical_size, slash_size  = min(last_q_end, max(vertical_size, 30)), min(last_q_end, max(slash_size, 50))
    qk = torch.einsum(
        f'bhmk, bhnk -> bhmn',
        q[:,:,last_q_start:last_q_end,:],
        k[:,:,:last_q_end,:]
    ) / math.sqrt(head_dim) # [bsz, n_heads, last_q_start:last_q_end, :last_q_end]

    qk[:, :, :, last_q_start:] = torch.where(
        LAST_Q_MASK[...,-last_q_size:,-last_q_size:].to(q.device),
        qk[:, :, :, last_q_start:],
        -torch.inf
    )
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    vertical = qk.sum(-2, keepdim=True)
    vertical[...,:init_window] = torch.inf
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices

    slash = sum_all_diagonal_matrix(qk)[...,:-last_q_size + 1]
    slash[...,-local_window:] = torch.inf
    slash = (last_q_end - 1) - torch.topk(slash, slash_size, -1).indices

    return vertical_topk, slash

def pad_modality(
    q, k, v,
    modality_spans,
    modality_type_idx,
    block_size=64,
):
    bsz, n_heads, seq_len, head_dim = q.shape
    modality_span_size = modality_spans[:, 1] - modality_spans[:, 0]
    blocks_per_modality = ((modality_span_size + block_size - 1) // block_size)
    modality_type_idx = torch.repeat_interleave(modality_type_idx, blocks_per_modality) # [bsz, n_heads, N_ROWS]

    num_zeros = blocks_per_modality * block_size - modality_span_size
    pad_zeros = [
        torch.zeros(
            (bsz, n_heads, pad_size, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        for pad_size in num_zeros
    ]

    q = q.split(modality_span_size.tolist(), dim=2)
    k = k.split(modality_span_size.tolist(), dim=2)
    v = v.split(modality_span_size.tolist(), dim=2)

    q = torch.cat([_ for pair in zip(q, pad_zeros) for _ in pair], dim=2)
    k = torch.cat([_ for pair in zip(k, pad_zeros) for _ in pair], dim=2)
    v = torch.cat([_ for pair in zip(v, pad_zeros) for _ in pair], dim=2)

    return q, k, v, [_ for _ in zip(modality_span_size, num_zeros)], modality_type_idx

def depad_modality(
    attn_output,
    depad_idx,
):
    attn_output = attn_output.split(depad_idx, dim=2)
    attn_output = torch.cat(attn_output[::2], dim=2)

    return attn_output

def vs_sparse_attention_mix_modality(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    modality_mask: torch.Tensor, # [N_ROWS]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = block_size_M - (context_size & (block_size_M - 1))
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N,
    )
    # flops_counter(block_count, column_count, context_size)
    block_count[0,0,modality_mask==0] = 0
    column_count[0,0,modality_mask==0] = 0
    out = _triton_mixed_sparse_attention(
        query, key, value, seqlens,
        block_count, block_offset, column_count, column_index,
        sm_scale, block_size_M, block_size_N,
    )
    return out[..., :context_size, :head_dim]

def mix_modality_vs_legacy(
    q, k, v,
    config
):
    frame_stride = config["frame_start_indices"][1] - config["frame_start_indices"][0]
    modality_types = config["modality_types"]
    input_boundaries = config["input_boundaries"]
    layer_idx = config["layer_idx"]

    assert len(input_boundaries) == (len(modality_types))
    input_boundaries = np.concatenate([input_boundaries, [q.size(2)]], axis=0)
    
    type2idx = {type: i for i, type in enumerate(sorted(list(set(modality_types))))}
    idx2type = {i: type for type, i in type2idx.items()}
    last_q_for_each_modality = {}
    modality_spans = torch.stack([
        torch.tensor(input_boundaries[:-1]), torch.tensor(input_boundaries[1:])
    ], dim=-1).to(q.device)
    for modality_type, modality_span in zip(modality_types[::-1], reversed(modality_spans)):
        if type2idx[modality_type] in last_q_for_each_modality:
            continue
        last_q_size = min(64, modality_span[1] - modality_span[0])
        last_q_for_each_modality[type2idx[modality_type]] = (
            modality_span[1] - last_q_size,
            modality_span[1]
        )

    bsz, n_heads, seq_len, head_dim = q.shape
    attn_output = torch.empty_like(q)
    modality_type_idx = torch.tensor(
        [type2idx[modality_type] for modality_type in modality_types],
        dtype=torch.int32,
        device=q.device,
    )
    row_wise_modality_type = chunk_overlap(
        modality_spans, modality_type_idx,
        64, 0, q.size(2)
    )
    for head_idx in range(n_heads):
        q_head = q[:, head_idx:head_idx+1, :, :] # [bsz, 1, seq_len, head_dim]
        k_head = k[:, head_idx:head_idx+1, :, :]
        v_head = v[:, head_idx:head_idx+1, :, :]

        best_pattern = config["attn_forward_config"]["best_pattern"][str(layer_idx)][str(head_idx)]
        do_shuffle, attn_type, attn_vars = best_pattern
        if do_shuffle == "no_shuffle":
            out = vertical_and_slash_kernel(
                q_head,
                k_head,
                v_head,
                vertical_size = attn_vars[0],
                slash_size = attn_vars[1],
            )
            attn_output[:, head_idx:head_idx+1, :, :] = out
            continue
        for modality_type_idx in range(len(type2idx)): # num_modality
            v_idx_m, s_idx_m = topk_vs(
                q_head,
                k_head,
                last_q_for_each_modality[type2idx[modality_types[modality_type_idx]]],
                vertical_size = attn_vars[0],
                slash_size = attn_vars[1]
            )

            out = vs_sparse_attention_mix_modality(
                q_head, k_head, v_head,
                v_idx_m, s_idx_m,
                row_wise_modality_type[modality_type_idx],
            )
            for modality_type, modality_span in zip(modality_types, modality_spans):
                if modality_type_idx == type2idx[modality_type]:
                    span_start, span_end = modality_span
                    attn_output[:, head_idx:head_idx+1, span_start:span_end, :] = out[..., span_start:span_end, :head_dim]
    return attn_output

def chunk_overlap(spans, labels, chunk_size, start, end):
    total_length = end - start
    num_rows = (total_length + chunk_size - 1) // chunk_size
    
    chunk_starts = start + torch.arange(0, num_rows, device=spans.device) * chunk_size
    chunk_ends = chunk_starts + chunk_size
    
    spans_start = spans[:, 0].unsqueeze(1)  # Shape (N, 1)
    spans_end = spans[:, 1].unsqueeze(1)    # Shape (N, 1)
    
    overlap = (spans_start < chunk_ends.unsqueeze(0)) & (spans_end > chunk_starts.unsqueeze(0))
    
    unique_labels, inverse_indices = torch.unique(labels, sorted=True, return_inverse=True)
    num_labels = unique_labels.size(0)
    onehot = torch.nn.functional.one_hot(inverse_indices, num_classes=num_labels).float()
    
    overlap_per_label = torch.matmul(onehot.T, overlap.float())  # Shape (num_labels, num_chunks)
    result = (overlap_per_label > 0).int()
    return result

def pad_and_stack_vs_idx(
    indices, # list of [bsz, n_heads, n_local]
    pad_value=-1
):
    max_dim = max(t.size(1) for t in indices)
    padded_tensors = []
    for t in indices:
        if t.size(1) < max_dim:
            # Pad the second dimension (dim=1) to max_dim
            pad = (0, max_dim - t.size(1))
            # For 2D: (pad_left, pad_right) on dim=1, no padding on dim=0
            padded = F.pad(t, pad, value=pad_value)
        else:
            padded = t
        padded_tensors.append(padded)
    return torch.stack(padded_tensors, dim=0)

def fill_in_attn_output(
    attn_output,
    out,
    current_modality_type,
    modality_span,
    modality_type_idx
):
    attn_output[:, modality_type_idx == current_modality_type, :, :] = out
    return attn_output

def fill_in_attn_output(
    target, incoming,
    current_modality_type,
    modality_span,
    modality_type_idx,
    head_idx,
):
    mask = torch.zeros(target.size(2), dtype=torch.bool, device=target.device)
    
    for span, label in zip(modality_span, modality_type_idx):
        if label == current_modality_type:
            start, end = span
            mask[start:end] = True  # end is exclusive
    
    target[:, head_idx:head_idx+1, mask] = incoming[:, :, mask]
    return target

def mix_modality_vs(
    q, k, v,
    config
):
    modality_types = config["modality_types"]
    input_boundaries = config["input_boundaries"]

    assert len(input_boundaries) == (len(modality_types))
    input_boundaries = np.concatenate([input_boundaries, [q.size(2)]], axis=0)
    seqlens = torch.tensor([q.size(2)], dtype=torch.int32, device=q.device)
    
    type2idx = {type: i for i, type in enumerate(sorted(list(set(modality_types))))}
    last_q_for_each_modality = {}
    modality_spans = torch.stack([
        torch.tensor(input_boundaries[:-1]), torch.tensor(input_boundaries[1:])
    ], dim=-1).to(q.device)
    for modality_type, modality_span in zip(modality_types[::-1], reversed(modality_spans)):
        if type2idx[modality_type] in last_q_for_each_modality:
            continue
        last_q_size = min(64, modality_span[1] - modality_span[0])
        last_q_for_each_modality[type2idx[modality_type]] = (
            modality_span[1] - last_q_size,
            modality_span[1]
        )

    bsz, n_heads, seq_len, head_dim = q.shape
    attn_output = torch.empty_like(q)
    modality_type_idx = torch.tensor(
        [type2idx[modality_type] for modality_type in modality_types],
        dtype=torch.int32,
        device=q.device,
    )
    row_wise_modality_type = chunk_overlap(
        modality_spans, modality_type_idx,
        64, 0, q.size(2)
    )
    for i in range(n_heads):
        q_head = q[:, i, :, :] # [bsz, seq_len, head_dim]
        k_head = k[:, i, :, :]
        v_head = v[:, i, :, :]

        v_idx = []
        s_idx = []
        for mi in range(len(type2idx)):
            v_idx_m, s_idx_m = topk_vs(
                q_head,
                k_head,
                last_q_for_each_modality[type2idx[modality_types[mi]]],
                vertical_size = 3000,
                slash_size = 6096,
            )
            v_idx.append(v_idx_m)
            s_idx.append(s_idx_m)
        v_idx = pad_and_stack_vs_idx(v_idx)
        s_idx = pad_and_stack_vs_idx(s_idx)

        (
            block_count, block_offset, column_count, column_index # [bsz, n_modality, n_heads, n_row, n_slashes]
        ) = mix_modality_vertical_slash_indexes_triton(
            seqlens,
            v_idx, # [n_modality, bsz, n_heads, n_local]
            s_idx, # [n_modality, bsz, n_heads, n_local]
            row_wise_modality_type,
        )

        for m_idx in range(len(type2idx)):
            out = _triton_mixed_sparse_attention(
                q_head, k_head, v_head,
                seqlens,
                block_count[:, m_idx], block_offset[:, m_idx], column_count[:, m_idx], column_index[:, m_idx], # [bsz, n_heads, n_row, n_slashes]
                sm_scale=head_dim ** -0.5, block_size_M=64, block_size_N=64,    
            ) # [bsz, 1, seq_len, head_dim]
            attn_output = fill_in_attn_output(
                attn_output,
                out[..., :seq_len, :head_dim],
                m_idx,
                modality_spans,
                modality_type_idx,
                i,
            )
    return attn_output

def shuffle_q_vs_one_modality_one_last_q(
    q, k, attn_scores,
    config,
    vertical_size=1000,
    slash_size=2048,
    print_overlap=False,
    calculate_recall=False,
    head_id=None,
):
    modality_types = config["modality_types"]
    input_boundaries = config["input_boundaries"]
    layer_idx = config["layer_idx"]

    if len(q.shape) == 2:
        q = q[None, None, ...]
        k = k[None, None, ...]

    assert len(input_boundaries) == (len(modality_types))
    input_boundaries = np.concatenate([input_boundaries, [q.size(2)]], axis=0)

    bsz, n_heads, seq_len, head_dim = q.shape
    seqlens = torch.tensor([q.size(2)], dtype=torch.int32, device=q.device)
    
    type2idx = {type: i for i, type in enumerate(sorted(list(set(modality_types))))}
    idx2type = {i: type for type, i in type2idx.items()}
    # if layer_idx == 0:
    #     print(type2idx)
    last_q_for_each_chunks = []
    modality_spans = torch.stack([
        torch.tensor(input_boundaries[:-1]), torch.tensor(input_boundaries[1:])
    ], dim=-1).to(q.device)
    for modality_type, modality_span in zip(modality_types[::-1], reversed(modality_spans)):
        last_q_size = min(64, modality_span[1] - modality_span[0])
        last_q_for_each_chunks.append((
            (modality_span[1] - last_q_size, modality_span[1]),
            type2idx[modality_type],
            modality_span,
        ))
    
    all_vs_idxs = defaultdict(list)
    for head_idx in range(n_heads):
        q_head = q[:, head_idx:head_idx+1, :, :]
        k_head = k[:, head_idx:head_idx+1, :, :]

        all_vs_idxs[head_idx] = defaultdict(list)
        for chunk_idx in range(len(last_q_for_each_chunks)):
            v_idxs, s_idxs = topk_vs(
                q_head, k_head,
                last_q_for_each_chunks[chunk_idx][0],
                vertical_size=vertical_size,
                slash_size=slash_size,
                local_window=100,
                init_window=30,
            )
            all_vs_idxs[head_idx][last_q_for_each_chunks[chunk_idx][1]].append((
                v_idxs, s_idxs, last_q_for_each_chunks[chunk_idx][2],
                last_q_for_each_chunks[chunk_idx][1],
            ))
    
    if print_overlap:
        # calculate the overlap between chunks
        data = []
        for head_idx in range(n_heads):
            for modality_type in all_vs_idxs[head_idx]:
                v_idxs = [tuple[0][0,0,0] for tuple in all_vs_idxs[head_idx][modality_type]][:2]
                overlap = compute_overlap_matrix(v_idxs)
                data.append((head_idx, modality_type, overlap[0,-1].item()))
        print(data)
    
    if calculate_recall:
        recalls = {}
        for head_idx in range(n_heads):
            mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=q.device)
            # mask[:100, :100] = True
            all_spans = all_vs_idxs[head_idx][0] + all_vs_idxs[head_idx][1]
            for the_span in all_spans:
                span_start, span_end = the_span[2][0], the_span[2][1]
                v_idxs, s_idxs = the_span[0], the_span[1]
                modality_type = idx2type[the_span[3]]
                # s_idxs [bsz, n_heads, n_slashes]
                mask[span_start:span_end] = torch.stack(
                    [
                        torch.sparse.spdiags(
                            torch.ones(s_idxs.size(-1), seq_len),
                            (0 - s_idxs).cpu()[0][0],
                            (seq_len, seq_len)
                        ).to_dense()
                        for _ in range(1)
                    ]
                )[0][span_start:span_end]
                mask[span_start:span_end, v_idxs[0,0,0]] = True

                span_recall = torch.where(
                    mask[span_start:span_end],
                    attn_scores[span_start:span_end],
                    0,
                ).sum(-1).mean(-1).squeeze().float().item()

                with open(f"modality-wise-recall.txt", "a+") as f:
                    f.write(f"Layer {layer_idx} Head {head_id} Modality {modality_type} Span {span_start}-{span_end} Recall {span_recall}\n")

            recalls[head_idx] = torch.where(mask, attn_scores, 0).sum(-1).mean(-1).squeeze().float().item()
    return recalls[0]


def shuffle_q_vs(
    q, k, attn_scores,
    config,
    vertical_size=1000,
    slash_size=2048,
    print_overlap=False,
    calculate_recall=False,
    head_id=None,
):
    modality_types = config["modality_types"]
    input_boundaries = config["input_boundaries"]
    layer_idx = config["layer_idx"]

    if len(q.shape) == 2:
        q = q[None, None, ...]
        k = k[None, None, ...]

    assert len(input_boundaries) == (len(modality_types))
    input_boundaries = np.concatenate([input_boundaries, [q.size(2)]], axis=0)

    bsz, n_heads, seq_len, head_dim = q.shape
    seqlens = torch.tensor([q.size(2)], dtype=torch.int32, device=q.device)
    
    type2idx = {type: i for i, type in enumerate(sorted(list(set(modality_types))))}
    idx2type = {i: type for type, i in type2idx.items()}
    # if layer_idx == 0:
    #     print(type2idx)
    last_q_for_each_modality = {}
    idx2span = defaultdict(list)
    modality_spans = torch.stack([
        torch.tensor(input_boundaries[:-1]), torch.tensor(input_boundaries[1:])
    ], dim=-1).to(q.device)
    for modality_type, modality_span in zip(modality_types[::-1], reversed(modality_spans)):
        idx2span[type2idx[modality_type]].append(modality_span)
        if type2idx[modality_type] in last_q_for_each_modality:
            continue
        last_q_size = min(64, modality_span[1] - modality_span[0])
        last_q_for_each_modality[type2idx[modality_type]] = (
            (modality_span[1] - last_q_size, modality_span[1]),
            modality_span,
        )
    
    all_vs_idxs = {}
    for head_idx in range(n_heads):
        q_head = q[:, head_idx:head_idx+1, :, :]
        k_head = k[:, head_idx:head_idx+1, :, :]

        all_vs_idxs[head_idx] = {}
        for modality_type in last_q_for_each_modality:
            v_idxs, s_idxs = topk_vs(
                q_head, k_head,
                last_q_for_each_modality[modality_type][0],
                vertical_size=vertical_size,
                slash_size=slash_size,
                local_window=100,
                init_window=30,
            )
            all_vs_idxs[head_idx][modality_type] = (v_idxs, s_idxs)
    
    if calculate_recall:
        recalls = {}
        for head_idx in range(n_heads):
            mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=q.device)
            # mask[:100, :100] = True
            for modality_type in all_vs_idxs[head_idx]:
                v_idxs, s_idxs = all_vs_idxs[head_idx][modality_type]
                s_mask_for_modality = torch.stack(
                    [
                        torch.sparse.spdiags(
                            torch.ones(s_idxs.size(-1), seq_len),
                            (0 - s_idxs).cpu()[0][0],
                            (seq_len, seq_len)
                        ).to_dense()
                        for _ in range(1)
                    ]
                )[0]
                for span in idx2span[modality_type]:
                    span_start, span_end = span[0], span[1]
                    mask[span_start:span_end] = s_mask_for_modality[span_start:span_end]
                    mask[span_start:span_end, v_idxs[0,0,0]] = True

                    span_recall = torch.where(
                        mask[span_start:span_end],
                        attn_scores[span_start:span_end],
                        0,
                    ).sum(-1).mean(-1).squeeze().float().item()

                    with open(f"modality-wise-recall-one-lastq.txt", "a+") as f:
                        f.write(f"Layer {layer_idx} Head {head_id} Modality {idx2type[modality_type]} Span {span_start}-{span_end} Recall {span_recall}\n")

            recalls[head_idx] = torch.where(mask, attn_scores, 0).sum(-1).mean(-1).squeeze().float().item()
    return recalls[0]

def mix_modality_pattern_search_attn_delta(
    q, k, v,
    layer_idx,
    head_idx,
    input_boundaries,
    frame_stride,
    stride,
    model_name,
    config,
):
    assert len(q.shape) == 2
    assert len(k.shape) == 2

    # search params:
    #     do_modality_shuffle: choose from ['no_shuffle', 'shuffle_q']
    #       - no_shuffle: choose pattern from ['a_shape', 'vs_attn'] # grid_attn is not available for mix modality
    #       - shuffle_q: choose pattern from ['vs_attn']
    #       - 2d_shuffle: two pattern can be used.
    #         - V2V & T2V: choose pattern from ['vs_attn', 'grid_attn'] # if grid_attn then only columns should be used
    #         - T2T: choose pattern from ['vs_attn'] # should not consider a_shape here. If it exhibits a_shape, shuffle_q should work better.
    #         - V2T: choose pattern from ['block_attn']

    search_space = [
        # no_shuffle + a_shape
        # ('no_shuffle', 'a_shape', 128, 2048),
        # ('no_shuffle', 'a_shape', 128, 4096),

        # no_shuffle + vs_attn
        # ('no_shuffle', 'vs_attn', 1000, 2048),
        ('no_shuffle', 'vs_attn', 1000, 2048),
        # ('no_shuffle', 'vs_attn', 3000, 2048),

        # shuffle_q + vs_attn
        # ('shuffle_q', 'vs_attn', 1000, 2048),
        ('shuffle_q', 'vs_attn', 1000, 2048),
        # ('shuffle_q', 'vs_attn', 3000, 2048),

        # 2d_shuffle + vs_attn
        # (
        #     '2d_shuffle',
        #     [
        #         ('grid_attn', frame_stride, True, True, False, 1024),
        #         ('vs_attn', 1000, 2048),
        #     ]
        # ),
        # (
        #     '2d_shuffle',
        #     [
        #         ('vs_attn', 1000, 2048),
        #         ('vs_attn', 1000, 2048),
        #     ]
        # ),
    ]

    attn_scores = dense_attention(q, k)

    name2func = {
        "no_shuffle": {
            "grid_attn": partial(grid_attn, attn_scores),
            "a_shape": partial(a_shape, attn_scores),
            "vs_attn": partial(vs_attn, q, k, attn_scores),
        },
        "shuffle_q": {
            "vs_attn": partial(shuffle_q_vs, q, k, attn_scores, config, print_overlap=False, calculate_recall=True, head_id=head_idx),
        },
        "2d_shuffle": {
            "vs_attn": partial(vs_attn, q, k, attn_scores),
        },
    }

    best_attn_recall = 0
    best_pattern = None
    minimal_delta = 1e-2
    recalles = {}
    for do_shuffle, attn_type, *args in search_space:
        if attn_type == "grid_attn" and model_name == "longvila":
            args.append(frame_stride)
        attn_recall = name2func[do_shuffle][attn_type](*args)
        if attn_type == "grid_attn":
            attn_recall, shift = attn_recall
            args += (shift,)
        if attn_recall - best_attn_recall > minimal_delta: # if delta is larger than minimal_delta, update best_pattern
            best_attn_recall = attn_recall
            best_pattern = (attn_type, *args)
        recalles['_'.join(map(str, (do_shuffle, attn_type, *args)))] = attn_recall
    return best_pattern, recalles


def mix_modality_pattern_search(
    q, k, v,
    config,
):
    layer_idx = config["layer_idx"]
    input_boundaries = config["input_boundaries"]
    frame_stride = config["frame_start_indices"][1] - config["frame_start_indices"][0]
    stride = config["stride"]
    model_name = config["model_name"]

    q_len = q.size(-2)
    if q_len > 25000: # use delta on attn output
        # pattern_search_func
        raise NotImplementedError
    else: # use delta on attn score
        pattern_search_func = mix_modality_pattern_search_attn_delta
    
    all_best_patterns = {}
    all_recalles = {}
    for head in range(q.size(1)):
        # if layer_idx != 27 or head != 13:
        #     continue
        q_head = q[0, head, :, :]
        k_head = k[0, head, :, :]
        # v_head = v[0, head, :, :]
        best_pattern, recalles = pattern_search_func(q_head, k_head, None, layer_idx, head, input_boundaries, frame_stride, stride, model_name, config)
        all_best_patterns[head] = best_pattern
        all_recalles[head] = recalles

    patterns_file = f"mminference_mix_modality_best_patterns_{model_name}.json"
    recalls_file = f"mminference_mix_modality_best_recalls_{model_name}.json"
    try:
        with open(patterns_file, "r+") as f:
            best_patterns = json.load(f)
            best_patterns[f"layer-{layer_idx}"] = all_best_patterns
            f.seek(0)
            json.dump(best_patterns, f, indent=1)
            f.truncate()
        with open(recalls_file, "r+") as f:
            recalls = json.load(f)
            recalls[f"layer-{layer_idx}"] = all_recalles
            f.seek(0)
            json.dump(recalls, f, indent=1)
            f.truncate()
    except FileNotFoundError:
        with open(patterns_file, "w") as f:
            json.dump({f"layer-{layer_idx}": all_best_patterns}, f, indent=1)
        with open(recalls_file, "w") as f:
            json.dump({f"layer-{layer_idx}": all_recalles}, f, indent=1)

    return all_best_patterns


def mminference_prefill_forward(
    query_states,
    key_states,
    value_states,
    prefill_kwargs,
):
    is_search = prefill_kwargs["attn_forward_config"].get("is_search", False)
    input_boundaries = prefill_kwargs["input_boundaries"]
    mix_modality = len(input_boundaries) > 3 # mix modality input
    if mix_modality:
        if is_search:
            # attn_output = mix_modality_vs(query_states, key_states, value_states, prefill_kwargs)
            # torch.save(attn_output, f"debug/attn_output_{prefill_kwargs['layer_idx']}_mix_modality_vs.pt")
            # shuffle_q_vs(query_states, key_states, value_states, prefill_kwargs)
            mix_modality_pattern_search(query_states, key_states, value_states, prefill_kwargs)
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                0.0,
                softmax_scale=None,
                causal=True,
            ).transpose(1, 2).contiguous()
            # torch.save(flash_attn_output, f"debug/flash_attn_output_{prefill_kwargs['layer_idx']}_mix_modality_vs.pt")
        else:
            attn_output = mix_modality_vs_legacy(query_states, key_states, value_states, prefill_kwargs)
            # attn_output = mix_modality_vs(query_states, key_states, value_states, prefill_kwargs)
        
        return attn_output

    if is_search:
        pattern_search(
            query_states, key_states, value_states,
            prefill_kwargs,
        )
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            0.0,
            softmax_scale=None,
            causal=True,
        ).transpose(1, 2).contiguous()
    else:
        n_ctx, n_heads, seq_len, head_dim = query_states.shape
        attn_output = torch.empty_like(query_states)
        for head in range(n_heads):
            q = query_states[:, head:head+1, :, :]
            k = key_states[:, head:head+1, :, :]
            v = value_states[:, head:head+1, :, :]
            attn_output[0, head:head+1, :, :] = mminference_kernel(
                q, k, v, head, prefill_kwargs
            )
    return attn_output