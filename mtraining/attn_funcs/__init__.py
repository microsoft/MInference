from typing import Dict, Callable

from .dense_func import fa_attn_forward, stripe_ring_attention_forward, zigzag_ring_attention_forward
from .minfer_func import minfer_attention_forward
from .moba_func import moba_attention_forward
from .xattn_func import xattn_attention_forward


class AttnType:
    BASELINE: str = "baseline"
    ZIGZAG_RING: str = "zigzag_ring"
    STRIPE_RING: str = "stripe_ring"

    MINFER: str = "minfer"
    MOBA: str = "moba"
    XATTN: str = "xattn"

ATTN_TO_FUNC = {
    AttnType.BASELINE: fa_attn_forward,
    AttnType.ZIGZAG_RING: zigzag_ring_attention_forward,
    AttnType.STRIPE_RING: stripe_ring_attention_forward,

    AttnType.MINFER: minfer_attention_forward,
    AttnType.MOBA: moba_attention_forward,
    AttnType.XATTN: xattn_attention_forward,
}

def overwrite_attn_implementation(
    attn_dict: Dict[str, Callable],
    attn_type: AttnType,
):
    attn_func: Callable = ATTN_TO_FUNC[attn_type]
    print(f"Overwriting attention implementation to {attn_type} ({attn_func.__name__})")

    for attn_name in attn_dict:
        attn_dict[attn_name] = attn_func