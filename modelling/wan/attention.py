# https://github.com/Wan-Video/Wan2.2/blob/388807310646ed5f318a99f8e8d9ad28c5b65373/wan/modules/attention.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
from torch import Tensor

try:
    import flash_attn

except ModuleNotFoundError:
    flash_attn = None


# TODO: see if FA is needed
def flash_attention(
    q: Tensor,  # [B, Lq, Nq, C1]
    k: Tensor,  # [B, Lk, Nk, C1]
    v: Tensor,  # [B, Lk, Nk, C2]
    q_lens: Tensor | None = None,  # [B]
    k_lens: Tensor | None = None,  # [B]
    dropout_p: float = 0.0,
) -> Tensor:
    B, Lq = q.shape[:2]
    Lk = k.shape[1]

    # preprocess query
    if q_lens is None:
        q = q.flatten(0, 1)
        q_lens = torch.tensor([Lq] * B, dtype=torch.int32, device=q.device)
    else:
        q = torch.cat([u[:v] for u, v in zip(q, q_lens)])

    # preprocess key, value
    if k_lens is None:
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
        k_lens = torch.tensor([Lk] * B, dtype=torch.int32, device=k.device)
    else:
        k = torch.cat([u[:v] for u, v in zip(k, k_lens)])
        v = torch.cat([u[:v] for u, v in zip(v, k_lens)])

    out = flash_attn.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True),
        cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True),
        max_seqlen_q=Lq,
        max_seqlen_k=Lk,
        dropout_p=dropout_p,
    )

    return out.unflatten(0, (B, Lq))
