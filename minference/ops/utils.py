import os
import torch

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def use_triton():
    return torch.version.hip is not None or os.getenv("FORCE_TRITON", "0") == "1"