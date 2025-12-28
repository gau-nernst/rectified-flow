# https://github.com/Wan-Video/Wan2.2/blob/38880731/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..rope import RopeND, apply_rope
from ..utils import FP32Linear, Linear, load_hf_state_dict


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


class WanSelfAttention(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.o = Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        # x: [B, L, D]
        q = self.norm_q(self.q(x)).unflatten(-1, (-1, self.head_dim))
        k = self.norm_k(self.k(x)).unflatten(-1, (-1, self.head_dim))
        v = self.v(x).unflatten(-1, (-1, self.head_dim))

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        out = dispatch_attn(q, k, v).flatten(2)
        return self.o(out)


class WanCrossAttention(WanSelfAttention):
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        # x: [B, L1, C]
        # context: [B, L2, C]
        q = self.norm_q(self.q(x)).unflatten(-1, (-1, self.head_dim))
        k = self.norm_k(self.k(c)).unflatten(-1, (-1, self.head_dim))
        v = self.v(c).unflatten(-1, (-1, self.head_dim))

        out = dispatch_attn(q, k, v).flatten(2)
        return self.o(out)


def modulate(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """NOTE: this always returns FP32"""
    return x.float() * (1.0 + scale.float()) + bias.float()


class WanAttentionBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, eps)

        self.norm3 = LayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = WanCrossAttention(dim, eps)

        self.norm2 = LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x: Tensor, e: Tensor, rope: Tensor, c: Tensor) -> Tensor:
        # x: [B, L, C]
        # e: [B, L, 6, C]
        assert e.dtype == torch.float32
        b0, s0, s1, b2, s2, s3 = (self.modulation.unsqueeze(0) + e).unbind(2)

        # x will become FP32 since e is FP32
        x = x + self.self_attn(modulate(self.norm1(x), s0, b0), rope) * s1
        x = x + self.cross_attn(self.norm3(x), c)
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
    rope_dims: tuple[int, int, int] = (44, 42, 42)
    patch_size: tuple[int, int, int] = (1, 2, 2)  # nF, nH, nW
    text_len: int = 512
    freq_dim: int = 256
    text_dim: int = 4096
    eps: float = 1e-6


class WanModel(nn.Module):
    def __init__(self, cfg: WanConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = nn.Conv3d(cfg.in_dim, cfg.dim, cfg.patch_size, cfg.patch_size)

        # use fp64 for RoPE. this has an impact on visual quality
        # TODO: check if maxlens are enough
        self.pos_embed = RopeND(cfg.rope_dims, (128, 512, 512), theta=1e4, dtype=torch.float64)

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
            [WanAttentionBlock(cfg.dim, cfg.ffn_dim, eps=cfg.eps) for _ in range(cfg.num_layers)]
        )
        self.head = Head(cfg.dim, cfg.out_dim, cfg.patch_size, cfg.eps)

    def forward(self, vid: Tensor, t: Tensor, txt: Tensor, y: Tensor | None = None) -> Tensor:
        # vid: [B, C, F, H, W]
        # t: [B] or [B, F*H*W]
        # txt: [B, 512, C]
        # y: [B, C, F, H, W] - not used?
        if y is not None:
            vid = torch.cat([vid, y], dim=0)

        vid = self.patch_embedding(vid.to(self.patch_embedding.weight.dtype))
        B, C, F_, H, W = vid.shape  # shape after patchify
        seqlen = F_ * H * W
        vid = vid.flatten(2).transpose(1, 2)  # (B, F*H*W, C)

        # time embeddings -> modulation
        # TODO: unify sinusoidal_embedding_1d with FLUX
        if t.ndim == 1:
            t = t.view(-1, 1).expand(t.shape[0], seqlen)
        time_embeds = sinusoidal_embedding_1d(self.cfg.freq_dim, t.flatten()).unflatten(0, (t.shape[0], seqlen))
        e = self.time_embedding(time_embeds)
        e0 = self.time_projection(e).unflatten(2, (6, -1))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        assert txt.shape[1] == 512
        txt = self.text_embedding(txt)

        rope = self.pos_embed.create((0, 0, 0), (F_, H, W))
        for block in self.blocks:
            vid = block(vid, e0, rope, txt)

        vid = self.head(vid, e)

        pF, pH, pW = self.cfg.patch_size
        out_dim = self.cfg.out_dim
        vid = vid.reshape(B, F_, H, W, pF, pH, pW, out_dim)
        vid = vid.permute(0, 7, 1, 4, 2, 5, 3, 6)
        vid = vid.reshape(B, out_dim, F_ * pF, H * pH, W * pW)
        return vid


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
