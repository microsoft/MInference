import os
from dataclasses import dataclass

@dataclass
class ExprData:
    global_batch_size: int
    micro_batch_size: int
    reuse_type: str

EXPR_DATA = ExprData(64, 1, "match")

def update_expr_data(args):
    global EXPR_DATA
    EXPR_DATA = ExprData(args.global_batch_size, args.micro_batch_size, args.reuse_type)
    # print(f"Updated global batch size to {EXPR_DATA.global_batch_size}, micro batch size to {EXPR_DATA.micro_batch_size}, reuse type to {EXPR_DATA.reuse_type}")
