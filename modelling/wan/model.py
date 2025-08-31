# https://github.com/Wan-Video/Wan2.2/blob/388807310646ed5f318a99f8e8d9ad28c5b65373/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..utils import load_hf_state_dict


def sinusoidal_embedding_1d(dim: int, position: Tensor, theta: float = 1e4) -> Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    freqs = torch.outer(
        position,
        theta ** (-torch.arange(half, dtype=position.dtype, device=position.device).div(half)),
    )
    x = torch.cat([freqs.cos(), freqs.sin()], dim=1)
    return x.float()


def rope_params(seqlen: int, dim: int, theta: float = 1e4, device: torch.types.Device = None) -> Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(seqlen, dtype=torch.float64, device=device),
        1.0 / theta ** torch.arange(0, dim, 2, dtype=torch.float64, device=device).div(dim),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: Tensor, rope_embeds: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    # x: [B, L, num_heads, head_dim]
    # rope_embeds: [1024, head_dim / 2] (complex number)
    rope_F, rope_H, rope_W = rope_embeds
    F = rope_F.shape[0]
    H = rope_H.shape[0]
    W = rope_W.shape[0]

    rope_F = rope_F.view(F, 1, 1, -1).expand(F, H, W, -1)
    rope_H = rope_H.view(1, H, 1, -1).expand(F, H, W, -1)
    rope_W = rope_W.view(1, 1, W, -1).expand(F, H, W, -1)
    rope = torch.cat([rope_F, rope_H, rope_W], dim=-1).reshape(F * H * W, 1, -1)

    x_f64 = torch.view_as_complex(x.to(torch.float64).unflatten(-1, (-1, 2)))
    return torch.view_as_real(x_f64 * rope).flatten(3).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: Tensor) -> Tensor:
        x_f32 = x.float()
        out = x_f32 * torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out.to(x.dtype)

    def forward(self, x: Tensor):
        return self._norm(x) * self.weight


class LayerNorm(nn.LayerNorm):
    """nn.LayerNorm doesn't work if x is FP32 and weight/bias is BF16"""

    def forward(self, x: Tensor):
        if self.elementwise_affine:
            weight = self.weight.float()
            bias = self.bias.float()
        else:
            weight = None
            bias = None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)


class Linear(nn.Linear):
    """Mimic autocast logic (kinda)"""

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias)


class FP32Linear(nn.Linear):
    """Mimic torch.autocast(dtype=torch.float32) behavior"""

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), self.weight.float(), bias)


class WanSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128, eps=1e-6) -> None:
        assert dim % head_dim == 0
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim

        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.o = Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: Tensor, rope_embeds: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        # x: [B, L, D]
        shape = (x.shape[0], -1, self.num_heads, self.head_dim)
        q = apply_rope(self.norm_q(self.q(x)).view(shape), rope_embeds)
        k = apply_rope(self.norm_k(self.k(x)).view(shape), rope_embeds)
        v = self.v(x).view(shape)
        return self.o(dispatch_attn(q, k, v).flatten(2))


class WanCrossAttention(WanSelfAttention):
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        # x: [B, L1, C]
        # context: [B, L2, C]
        shape = (x.shape[0], -1, self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).view(shape)
        k = self.norm_k(self.k(context)).view(shape)
        v = self.v(context).view(shape)
        return self.o(dispatch_attn(q, k, v).flatten(2))


def modulate(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """NOTE: this always returns FP32"""
    return x.float() * (1.0 + scale.float()) + bias.float()


class WanAttentionBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, head_dim: int = 128, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, head_dim, eps)

        self.norm3 = LayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = WanCrossAttention(dim, head_dim, eps)

        self.norm2 = LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x: Tensor, e: Tensor, rope_embeds: tuple[Tensor, Tensor, Tensor], context: Tensor) -> Tensor:
        # x: [B, L, C]
        # e: [B, L, 6, C]
        assert e.dtype == torch.float32
        b0, s0, s1, b2, s2, s3 = (self.modulation.unsqueeze(0) + e).unbind(2)

        # x will become FP32 since e is FP32
        x = x + self.self_attn(modulate(self.norm1(x), s0, b0), rope_embeds) * s1
        x = x + self.cross_attn(self.norm3(x), context)
        x = x + self.ffn(modulate(self.norm2(x), s2, b2)) * s3
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6) -> None:
        super().__init__()
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps, elementwise_affine=False)
        self.head = FP32Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        # x: [B, L, C]
        # e: [B, L, C]
        assert e.dtype == torch.float32
        bias, scale = (self.modulation + e.unsqueeze(2)).unbind(2)
        return self.head(modulate(self.norm(x.float()), scale, bias))


@dataclass
class WanConfig:
    dim: int
    ffn_dim: int
    num_layers: int
    in_dim: int = 16
    out_dim: int = 16
    patch_size: tuple[int, int, int] = (1, 2, 2)  # nF, nH, nW
    text_len: int = 512
    freq_dim: int = 256
    text_dim: int = 4096
    head_dim: int = 128
    eps: float = 1e-6


class WanModel(nn.Module):
    def __init__(self, cfg: WanConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = nn.Conv3d(cfg.in_dim, cfg.dim, cfg.patch_size, cfg.patch_size)
        self.text_embedding = nn.Sequential(
            Linear(cfg.text_dim, cfg.dim),
            nn.GELU(approximate="tanh"),
            Linear(cfg.dim, cfg.dim),
        )
        self.time_embedding = nn.Sequential(
            FP32Linear(cfg.freq_dim, cfg.dim),
            nn.SiLU(),
            FP32Linear(cfg.dim, cfg.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            FP32Linear(cfg.dim, cfg.dim * 6),
        )
        self.blocks = nn.ModuleList(
            [WanAttentionBlock(cfg.dim, cfg.ffn_dim, cfg.head_dim, eps=cfg.eps) for _ in range(cfg.num_layers)]
        )
        self.head = Head(cfg.dim, cfg.out_dim, cfg.patch_size, cfg.eps)

    def get_rope(self, F: int, H: int, W: int, device: torch.types.Device = None) -> tuple[Tensor, Tensor, Tensor]:
        d = self.cfg.head_dim
        return (
            rope_params(F, d - 4 * (d // 6), device=device),
            rope_params(H, 2 * (d // 6), device=device),
            rope_params(W, 2 * (d // 6), device=device),
        )

    def forward(self, x: Tensor, t: Tensor, context: Tensor, y: Tensor | None = None) -> Tensor:
        # x: [B, C, F, H, W]
        # t: [B] or [B, F*H*W]
        # context: [B, L, C]
        # y: [B, C, F, H, W]
        if y is not None:
            x = torch.cat([x, y], dim=0)

        x = self.patch_embedding(x.to(self.patch_embedding.weight.dtype))
        F_, H, W = x.shape[2:]
        seqlen = F_ * H * W
        x = x.flatten(2).transpose(1, 2)  # (B, F*H*W, C)

        # time embeddings -> modulation
        if t.ndim == 1:
            t = t.view(-1, 1).expand(t.shape[0], seqlen)
        time_embeds = sinusoidal_embedding_1d(self.cfg.freq_dim, t.flatten()).unflatten(0, (t.shape[0], seqlen))
        e = self.time_embedding(time_embeds)
        e0 = self.time_projection(e).unflatten(2, (6, -1))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        if context.shape[1] < 512:
            context = F.pad(context, (0, 0, 0, 512 - context.shape[1]))
        elif context.shape[1] > 512:
            context = context[:, :512]
        context = self.text_embedding(context)

        rope_embeds = self.get_rope(F_, H, W, device=x.device)
        for block in self.blocks:
            x = block(x, e0, rope_embeds, context)

        x = self.head(x, e)

        nF, nH, nW = self.cfg.patch_size
        out_dim = self.cfg.out_dim
        x = x.reshape(x.shape[0], F_, H, W, nF, nH, nW, out_dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(x.shape[0], out_dim, F_ * nF, H * nH, W * nW)
        return x


def load_wan(name: str):
    repo_id, filename, cfg = {
        "wan2.2-t2v-a14b-high": (
            "Wan-AI/Wan2.2-T2V-A14B",
            "high_noise_model/diffusion_pytorch_model.safetensors.index.json",
            WanConfig(dim=5120, ffn_dim=13_824, num_layers=40),
        ),
        "wan2.2-t2v-a14b-low": (
            "Wan-AI/Wan2.2-T2V-A14B",
            "low_noise_model/diffusion_pytorch_model.safetensors.index.json",
            WanConfig(dim=5120, ffn_dim=13_824, num_layers=40),
        ),
        "wan2.2-ti2v-5b": (
            "Wan-AI/Wan2.2-TI2V-5B",
            "diffusion_pytorch_model.safetensors.index.json",
            WanConfig(dim=3072, ffn_dim=14_336, num_layers=30, in_dim=48, out_dim=48),
        ),
    }[name]

    with torch.device("meta"):
        model = WanModel(cfg)

    state_dict = load_hf_state_dict(repo_id, filename)
    model.load_state_dict(state_dict, assign=True)
    return model
