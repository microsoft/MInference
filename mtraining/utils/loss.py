#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.utils.checkpoint as ckpt

from nnscaler.graph.parser.register import register_op


def linear_cross_entropy(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """
    Compute the cross entropy loss of a linear layer.

    Args:

        x: [token_num, hidden_size], the last hidden state of the model
        w: [dict_size, hidden_size], the weight matrix of the last linear layer
        y: [token_num], the target token index
        padding_idx: int, the index of padding token

    Returns:
    
        losses: [token_num], the cross entropy loss of each token
    """
    logits = torch.nn.functional.linear(x, w)
    normalized_logits = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    losses = torch.nn.functional.nll_loss(normalized_logits, y, reduction='none', ignore_index=padding_idx)
    return losses


def chunk_linear_cross_entropy(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, padding_idx: int, chunk_size: int) -> torch.Tensor:
    """
    In order to reduce the memory usage when the sequence length and dictionary size are large, we can split the input
    tensor into chunks and compute the cross entropy loss of each chunk separately.
    You can register this function with annotation 'b l d^, n^ d^, b l -> b l'.

    Args:
    
        x: [bsz, seq_len, hidden_size], the last hidden state of the model
        w: [dict_size, hidden_size], the weight matrix of the last linear layer
        y: [bsz, seq_len], the target token index
        padding_idx: int, the index of padding token
        chunk_size: int, the size of each chunk

    Returns:
        
        losses: [bsz, seq_len], the cross entropy loss of each token
    """
    bsz, seq_len, hidden_size = x.size()
    token_num = bsz * seq_len
    x = x.view(token_num, hidden_size)
    y = y.view(token_num)

    if token_num % chunk_size != 0:
        raise ValueError(f"token_num {token_num} is not divisible by chunk_size {chunk_size}")

    chunk_num = token_num // chunk_size
    xs = x.view(chunk_num, chunk_size, hidden_size)
    ys = y.view(chunk_num, chunk_size)
    losses = []
    for i in range(chunk_num):
        loss = ckpt.checkpoint(linear_cross_entropy, xs[i], w, ys[i], padding_idx, use_reentrant=False)
        losses.append(loss)
    losses = torch.stack(losses).view(bsz, seq_len)
    return losses


register_op('b l d^, n^ d^, b l -> b l')(chunk_linear_cross_entropy)
