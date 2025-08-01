def l1_loss(x):
    numel = x.numel()
    l1 = x.abs().sum()
    return l1 / numel
