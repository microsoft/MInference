from .phi3 import (
    Phi3Config, Phi3ForCausalLM, Phi3Attention, 
    apply_rotary_pos_emb, repeat_kv,
    PHI_ATTN_FUNCS
)

from .qwen2 import (
    Qwen2Config, Qwen2ForCausalLM, Qwen2Attention, 
    apply_rotary_pos_emb, repeat_kv,
    QWEN_ATTN_FUNCS
)


MODEL_TO_ATTN_FUNC = {
    "microsoft/Phi-3-mini-4k-instruct": PHI_ATTN_FUNCS,
    "Qwen/Qwen2.5-3B": QWEN_ATTN_FUNCS
}


MODEL_ID_TO_MODEL_CLS = {
    "microsoft/Phi-3-mini-4k-instruct": Phi3ForCausalLM,
    "Qwen/Qwen2.5-3B": Qwen2ForCausalLM
}

MODEL_ID_TO_PREFIX = {
    "microsoft/Phi-3-mini-4k-instruct": "Phi3",
    "Qwen/Qwen2.5-3B": "Qwen2",
}

