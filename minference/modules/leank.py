from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from typing import Union, List, Optional
import torch
import tilelang
import tilelang.language as T
try:
    from minference.ops.leank_flash_decoding import leank_flashattn
except:
    pass

def reorder_linear_weights(
    linear_module: torch.nn.Linear,
    channel_mask: torch.Tensor,
    repeat_num,
    reorder_channel,
    boundaries: List[int],
):
    assert reorder_channel in ["in", "out"]
    channel_mask = torch.repeat_interleave(
        channel_mask, repeats=repeat_num
    ).to(linear_module.weight.device)
    reordered_weight = torch.zeros_like(linear_module.weight)
    if linear_module.bias is not None:
        reordered_bias = torch.zeros_like(linear_module.bias)
    
    l, r = 0, 0
    for boundary in boundaries:
        mask = (channel_mask == boundary)
        r += int(mask.sum())
        if reorder_channel == "in":
            reordered_weight[:, l:r] =  linear_module.weight.data[:, mask.bool()]
        else:
            reordered_weight[l:r] =  linear_module.weight.data[mask.bool()]
        if linear_module.bias is not None:
            reordered_bias[l:r] =  linear_module.bias.data[mask.bool()]
        l = r

    linear_module.weight.data = reordered_weight
    if linear_module.bias is not None:
        linear_module.bias.data = reordered_bias
    return linear_module


def reorder_channel_mask(
    channel_mask: torch.Tensor,
    layer_full_attn_channels: torch.Tensor,
    boundaries: List[int],
):
    new_rst = torch.zeros_like(layer_full_attn_channels)
    l, r = 0, 0
    for boundary in boundaries:
        mask = (channel_mask == boundary)
        r += int(mask.sum())
        new_rst[l:r] = layer_full_attn_channels[mask]
        l = r
    return new_rst


def mask_channels_key(key_states, mask, remained_count):
    bs, nh, seq_len, dim = key_states.shape
    return key_states.transpose(1, 2) \
                    .reshape(bs, seq_len, dim * nh)[..., mask.flatten().bool()] \
                    .reshape(bs, seq_len, nh, remained_count).transpose(1, 2)
      
                    
def mask_channels_query(query_states, mask, remained_count):
    bs, nh, dim = query_states.shape
    ng = mask.shape[0]
    heads_per_group = nh // ng
    return query_states.reshape(bs, ng, heads_per_group, -1).transpose(1, 2) \
        .reshape(bs, heads_per_group, -1)[:, :,  mask.flatten().bool()] \
        .reshape(bs, heads_per_group, ng, remained_count).transpose(1, 2).reshape(bs, -1, remained_count)
   
    
def get_round_seqlen_and_split_hueristic(seqlen):
    supported_lengths = [4096, 8192] + [2**i + 8192 for i in range(13, 18)]
    num_split_hueristic = [2, 2, 4, 4, 8, 8, 16]
    last_length, last_n_split = supported_lengths[-1], num_split_hueristic[-1]
    for l, ns in zip(supported_lengths[::-1], num_split_hueristic[::-1]):
        if seqlen > l:
            return last_length, last_n_split
        last_length = l
        last_n_split = ns
    return last_length, last_n_split


class LeanKCache(DynamicCache):
    def __init__(self, config):
        super().__init__()
        self.sink_size = config.attn_kwargs.get("sink_size", 128)
        self.recent_size = config.attn_kwargs.get("recent_size", 768)
        self.full_size = self.sink_size + self.recent_size
        self.accumu_size = config.attn_kwargs.get("accumu_size", 128)
        # key caches
        self.key_cache_mid = []
        self.key_cache_full = []
        # value caches
        self.value_cache_full = []
        self.value_cache_mid = []
        self.mid_seq_len = -1
        import gc; gc.collect() # fix tilelang kernel compile related issues
    
    def update(self, key_states, value_states, layer_idx, mask, boundaries, counts, cache_kwargs):
        # Update the cache
        if key_states is not None:
            if len(self.key_cache_full) <= layer_idx:
                # adding cache for the first time
                self.key_cache_full.append(torch.cat((key_states[:, :, :self.sink_size], key_states[:, :, -self.recent_size:]), dim=-2))
                self.value_cache_full.append(torch.cat((value_states[:, :, :self.sink_size], value_states[:, :, -self.recent_size:]), dim=-2))
                
                self.key_cache_mid.append({})
                self.value_cache_mid.append({})
                
                l, r = boundaries[0], -1
                for boundary, count in zip(boundaries[1:], counts[1:]):
                    r = boundary
                    # Tilelang 0.1.5 requires contiguous inputs
                    self.key_cache_mid[layer_idx][count] = mask_channels_key(key_states[:, l:r, self.sink_size: - self.recent_size], mask[l:r], count).contiguous()
                    self.value_cache_mid[layer_idx][count] = value_states[:, l:r, self.sink_size: -self.recent_size].contiguous()
                    l = r
                
                self.mid_seq_len = value_states.shape[-2] - self.full_size

            else:
                self.key_cache_full[layer_idx] = torch.cat([self.key_cache_full[layer_idx], key_states], dim=-2)
                self.value_cache_full[layer_idx] = torch.cat([self.value_cache_full[layer_idx], value_states], dim=-2)

                if self.key_cache_full[layer_idx].shape[-2] >= self.full_size + self.accumu_size:
                    
                    self.value_cache_full[layer_idx] = torch.cat((self.value_cache_full[layer_idx][:, :, : self.sink_size], 
                                                                  self.value_cache_full[layer_idx][:, :, -self.recent_size :]), dim=-2)
                    
                    l, r = boundaries[0], -1
                    for boundary, count in zip(boundaries[1:], counts[1:]):
                        r = boundary
                        self.key_cache_mid[layer_idx][count] = torch.cat((self.key_cache_mid[layer_idx][count], 
                                                                    mask_channels_key(self.key_cache_full[layer_idx][:, l:r, self.sink_size: - self.recent_size], mask[l:r], count)), dim=2).contiguous()
                        self.value_cache_mid[layer_idx][count] = torch.cat((self.value_cache_mid[layer_idx][count],
                                                                            self.value_cache_full[layer_idx][:,l:r, self.sink_size : -self.recent_size]), dim=-2).contiguous()
                        l = r
                    
                    self.mid_seq_len += self.accumu_size
                    
                    self.key_cache_full[layer_idx] = torch.cat((self.key_cache_full[layer_idx][:, :, :self.sink_size], self.key_cache_full[layer_idx][:, :, -self.recent_size:]), dim=-2)
        
        torch.cuda.empty_cache()
        return self.key_cache_full[layer_idx], self.key_cache_mid[layer_idx], self.value_cache_mid[layer_idx], self.value_cache_full[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        is_empty_layer = (
            len(self.key_cache_full) == 0  # no cache in any layer
            or len(self.key_cache_full) <= layer_idx  # the layer has no cache
        )
        if is_empty_layer:
            return 0
        return self.value_cache_full[layer_idx].shape[-2] + self.mid_seq_len

def leank_forward(
    query_states: torch.Tensor,
    key_states_full: torch.Tensor,
    key_states_mid: list[torch.Tensor],
    value_states_full: torch.Tensor,
    value_states_mid: torch.Tensor,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> torch.Tensor: 
    bsz, ng, full_kvlen, dim = key_states_full.shape
    dtype = key_states_full.dtype
    device = key_states_full.device
    
    nh = query_states.shape[1]
    heads_per_group = nh // ng
    
    if len(key_states_mid) == 0:
        kv_seqlen = 0
    else:
        kv_seqlen = next(iter(key_states_mid.items()))[1].shape[-2]
        maskmid = torch.ones((bsz, kv_seqlen), dtype=torch.uint8, device=device)
        
    need_compile = False
    rouned_seq_len, num_split = get_round_seqlen_and_split_hueristic(kv_seqlen) 
    last_length = kwargs["last_length"]
    if last_length != rouned_seq_len:
        last_length = rouned_seq_len
        if kwargs["layer_idx"]== 0:
            print(f"--- Compiling LeanK decoding kernel, might take some time ---")
        need_compile = True
    
    if kwargs["attention_mask"] is not None:
        mask = kwargs["attention_mask"].to(torch.uint8)
    else:
        mask = torch.ones((bsz, full_kvlen), dtype=torch.uint8, device=device)
    glse = torch.full((bsz, nh, num_split + 1), -torch.inf, dtype=dtype, device=device)
    Output_partial = torch.zeros(bsz, nh, num_split + 1, dim, dtype=dtype, device=device)
    
    args = []
    number_groups = []
    
    args.append(query_states.squeeze(-2))
    args.append(key_states_full)
    args.append(value_states_full)
    
    args.append(mask)
    args.append(glse)
    args.append(Output_partial)
    
    boundaries = kwargs["boundaries"]
    counts = kwargs["counts"]
    l, r = boundaries[0], -1
    for boundary, count in zip(boundaries[1:], counts[1:]):
        r = boundary
        args.append(mask_channels_query(query_states[:, l * heads_per_group: r * heads_per_group, 0], kwargs["full_attn_channels"][l:r], count))
        args.append(key_states_mid[count])
        args.append(value_states_mid[count])
        args.append(glse[:, l * heads_per_group: r * heads_per_group].contiguous())
        args.append(Output_partial[:, l * heads_per_group: r * heads_per_group].contiguous())
        number_groups.append(r - l)
        l = r
        
    n_groups = len(number_groups)
    if n_groups > 0:
        args.append(maskmid)
    if n_groups < 4:
        number_groups += [0] * (4 - n_groups)
    
    kernel_kwargs = [T.symbolic("batch"), nh]
    kernel_kwargs += [i * heads_per_group for i in number_groups]
    kernel_kwargs += [ng]
    kernel_kwargs += number_groups
    kernel_kwargs += [last_length, T.symbolic("cur_len"), kwargs["full_size"], T.symbolic("cur_full_len"), dim]
    kernel_kwargs += counts[1:]
    if n_groups < 4:
        kernel_kwargs += [0] * (4 - n_groups)
    kernel_kwargs += [n_groups]
    kernel_kwargs += ["float16" if dtype == torch.float16 else "bfloat16"]
    
    if need_compile or kwargs["kernel"] is None:
        program = leank_flashattn(*kernel_kwargs)(num_split=num_split, **kwargs["kernel_config"])
        kernel = tilelang.compile(program, out_idx=[5 * (n_groups + 1) + 1 + (n_groups > 0)])
    else:
        kernel = kwargs["kernel"]
        
    attn_output = kernel(*args)
    return attn_output.unsqueeze(2), last_length, kernel


def patch_leank(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    config, 
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    align = config.attn_kwargs.get("round_to", 32)
    supported_dims = [x for x in range(0, model.config.hidden_size // model.config.num_attention_heads + 1, align)]
    leank_pattern = torch.load(config.attn_kwargs.get("leank_path"))

    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        module.sink_size = config.attn_kwargs.get("sink_size", 128)
        module.recent_size = config.attn_kwargs.get("recent_size", 768)
        module.accumu_size = config.attn_kwargs.get("recent_size", 128)
        layer_full_attn_channels = leank_pattern[idx].to(device).to(dtype).reshape(-1, module.head_dim)
        channel_mask = layer_full_attn_channels.sum(dim=-1)
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            channel_mask,
            module.num_key_value_groups * module.head_dim,
            "out",
            supported_dims,
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            channel_mask,
            module.head_dim,
            "out",
            supported_dims,
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            channel_mask,
            module.head_dim,
            "out",
            supported_dims,
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            channel_mask,
            module.num_key_value_groups * module.head_dim,
            "in",
            supported_dims,
        )
        layer_full_attn_channels = reorder_channel_mask(
            channel_mask, 
            layer_full_attn_channels, 
            supported_dims,
        )
        module.register_buffer(
            "full_attn_channels",
            layer_full_attn_channels,
        )
        
        dim_cnt = (layer_full_attn_channels.sum(dim=-1) == supported_dims[0]).sum().item()
        boundaries = [dim_cnt]
        counts = [supported_dims[0]]
        for d in supported_dims[1:]:
            nheads = (layer_full_attn_channels.sum(dim=-1) == d).sum().item()
            if nheads > 0:
                dim_cnt += nheads
                boundaries.append(dim_cnt)
                counts.append(d)
        setattr(module, "boundaries", boundaries)
        setattr(module, "counts", counts)
        
        module.kernel_config = {"block_N": 64, "block_H": 64, "num_stages": 2, "threads": 128}
        module.last_length = -1
        module.kernel = None