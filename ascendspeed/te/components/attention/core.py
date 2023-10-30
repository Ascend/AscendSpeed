import math
from typing import Optional, Union

import torch
from .attention_mask import AttentionMask


def _matmul_with_mask(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if mask is None:
        return a @ b

    att = a @ b
    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        att[~mask] = float("-inf")
    else:
        # mask is presumed additive
        # repeat if batch sizes don't match
        if (
            mask.ndim == 3
            and mask.shape[0] != att.shape[0]
            and (att.shape[0] % mask.shape[0]) == 0
        ):
            repeat_factor = att.shape[0] // mask.shape[0]
            mask = mask.repeat([repeat_factor, 1, 1])
        att += mask
    return att


def _softmax(a: torch.Tensor, causal: bool = False) -> torch.Tensor:
    return torch.softmax(a, dim=a.ndim - 1)


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def _apply_dropout(att, dropout):
    if dropout is None:
        return att
    
    att = dropout(att)
    return att


def scaled_query_key_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    att_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # assume we have (N, S, hs) instead of (B, nh, S, hs), with N = B x nh
    # this is needed due to limitations in sparse_bmm for now
    last_dimension = k.size(-1)
    if last_dimension <= 0:
        raise Exception("the last dimension of key should be greater than zero")
    
    # Self-attend: (N, S, hs) x (N, hs, S) -> (N, S, S)
    q = q / math.sqrt(last_dimension)

    # Matmul with mask
    if att_mask is not None and isinstance(att_mask, AttentionMask):
        mask = att_mask.values
    else:
        mask = att_mask

    att = _matmul_with_mask(q, k.transpose(-2, -1), mask)

    # Softmax to get the attention probabilities
    is_causal = isinstance(att_mask, AttentionMask) and att_mask.is_causal
    att = _softmax(att, causal=is_causal)
    return att


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask: Optional[torch.Tensor] = None,
    dropout: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    # general common self-attention interface
    att = scaled_query_key_softmax(q, k, att_mask=att_mask)

    #  Optional dropout, could be part of the masking in the future
    att = _apply_dropout(att, dropout)

    # Get to the predicted values, for all heads
    # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
    y = bmm(att, v)
    return y


def group_attention(
    q:torch.Tensor,
    k:torch.Tensor,
    v:torch.Tensor,
    groups: int,
    scale: float
) -> torch.Tensor:
    # group attention method to decrease attention flops
    qs = torch.chunk(q, groups, dim=1)
    ks = torch.chunk(k, groups, dim=1)
    vs = torch.chunk(v, groups, dim=1)
    outputs = []

    epsilon = 1e-8
    for i in range(groups):
        j = (i + 1) % groups
        kst = ks[j].transpose(1, 2)
        si = torch.bmm(qs[i], kst) / (scale + epsilon)
        pi = torch.softmax(si, dim=-1)
        oi = torch.bmm(pi, vs[j])
        outputs.append(oi)

    outputs = torch.cat(outputs, dim=1)
    return outputs
