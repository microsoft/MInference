# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]


def l1_loss(x):
    numel = x.numel()
    l1 = x.abs().sum()
    return l1 / numel
