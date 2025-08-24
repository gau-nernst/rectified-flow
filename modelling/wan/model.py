# https://github.com/Wan-Video/Wan2.2/blob/388807310646ed5f318a99f8e8d9ad28c5b65373/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import load_hf_state_dict
from .attention import flash_attention


def sinusoidal_embedding_1d(dim: int, position: Tensor, theta: float = 10_000.0) -> Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    freqs = torch.outer(
        position,
        theta ** (-torch.arange(half, dtype=position.dtype, device=position.device).div(half)),
    )
    x = torch.cat([freqs.cos(), freqs.sin()], dim=1)
    return x.float()


def rope_params(max_seq_len: int, dim: int, theta: float = 10_000.0, device: torch.types.Device = None) -> Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float, device=device),
        1.0 / theta ** torch.arange(0, dim, 2, dtype=torch.float64, device=device).div(dim),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(x: Tensor, grid_sizes: Tensor, rope_embeds: Tensor) -> Tensor:
    N = x.shape[2]
    C = x.shape[3] // 2
    rope_embeds = rope_embeds.split([C - 2 * (C // 3), C // 3, C // 3], dim=1)

    # loop over samples
    # NOTE: grid_sizes doesn't need to be a tensor?
    output = []
    for i, (F, H, W) in enumerate(grid_sizes.tolist()):
        seq_len = F * H * W

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, N, -1, 2))
        rope_i = torch.cat(
            [
                rope_embeds[0][:F].view(F, 1, 1, -1).expand(F, H, W, -1),
                rope_embeds[1][:H].view(1, H, 1, -1).expand(F, H, W, -1),
                rope_embeds[2][:W].view(1, 1, W, -1).expand(F, H, W, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * rope_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


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


class WanSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128, eps=1e-6) -> None:
        assert dim % head_dim == 0
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: Tensor,  # [B, L, num_heads, C / num_heads]
        seq_lens: Tensor,  # [B]
        grid_sizes: Tensor,  # [B, 3]
        rope_embeds: Tensor,  # [1024, C / num_heads / 2]
    ) -> Tensor:
        q = self.norm_q(self.q(x)).unflatten(-1, (-1, self.head_dim))
        k = self.norm_k(self.k(x)).unflatten(-1, (-1, self.head_dim))
        v = self.v(x).unflatten(-1, (-1, self.head_dim))

        q = rope_apply(q, grid_sizes, rope_embeds)
        k = rope_apply(k, grid_sizes, rope_embeds)
        x = flash_attention(q, k, v, k_lens=seq_lens)

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        # x: [B, L1, C]
        # context: [B, L2, C]
        q = self.norm_q(self.q(x)).unflatten(-1, (-1, self.head_dim))
        k = self.norm_k(self.k(context)).unflatten(-1, (-1, self.head_dim))
        v = self.v(context).unflatten(-1, (-1, self.head_dim))

        x = flash_attention(q, k, v)

        x = x.flatten(2)
        x = self.o(x)
        return x


def modulate(x: Tensor, e: Tensor) -> Tensor:
    assert e.dtype == torch.float32
    return (x * (1 + e[:, :, 0]) + e[:, :, 1]).to(x.dtype)


class WanAttentionBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, head_dim: int = 128, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, head_dim, eps)

        self.norm3 = nn.LayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = WanCrossAttention(dim, head_dim, eps)

        self.norm2 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: Tensor,  # [B, L, C]
        e: Tensor,  # [B, L1, 6, C]
        seq_lens: Tensor,  # [B]
        grid_sizes: Tensor,  # [B, 3]
        rope_embeds: Tensor,  # [1024, C/num_heads/2]
        context: Tensor,
    ) -> Tensor:
        e = self.modulation.unsqueeze(0) + e

        y = self.self_attn(modulate(self.norm1(x), e[:, :, :2]), seq_lens, grid_sizes, rope_embeds)
        x = (x + y * e[:, :, 2]).to(x.dtype)

        x = x + self.cross_attn(self.norm3(x), context)

        y = self.ffn(modulate(self.norm2(x), e[:, :, 3:5]))
        x = (x + y * e[:, :, 5]).to(x.dtype)

        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size, eps: float = 1e-6) -> None:
        super().__init__()
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        assert e.dtype == torch.float32
        e = self.modulation.unsqueeze(0) + e.unsqueeze(2)
        x = self.head(modulate(self.norm(x).to(x.dtype), e))
        return x


class FP32Linear(nn.Linear):
    """Mimic torch.autocast(dtype=torch.float32) behavior"""

    def forward(self, input: Tensor):
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(input.float(), self.weight.float(), bias)


@dataclass
class WanConfig:
    dim: int
    ffn_dim: int
    num_layers: int
    in_dim: int = 16
    out_dim: int = 16
    patch_size: tuple[int, int, int] = (1, 2, 2)  # t, h, w
    text_len: int = 512
    freq_dim: int = 256
    text_dim: int = 4096
    head_dim: int = 128
    eps: float = 1e-6


class WanModel(nn.Module):
    def __init__(self, cfg: WanConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # embeddings
        self.patch_embedding = nn.Conv3d(cfg.in_dim, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.dim, cfg.dim),
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

        # blocks
        self.blocks = nn.ModuleList(
            [WanAttentionBlock(cfg.dim, cfg.ffn_dim, cfg.head_dim, eps=cfg.eps) for _ in range(cfg.num_layers)]
        )

        # head
        self.head = Head(cfg.dim, cfg.out_dim, cfg.patch_size, cfg.eps)

    def get_rope(self, device: torch.types.Device = None):
        d = self.cfg.head_dim
        return torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6), device=device),
                rope_params(1024, 2 * (d // 6), device=device),
                rope_params(1024, 2 * (d // 6), device=device),
            ],
            dim=1,
        )

    def forward(
        self,
        x: list[Tensor],  # each is [Cin, F, H, W]
        t: Tensor,  # [B]
        context: list[Tensor],  # each is [L, C]
        seq_len: int,
        y: list[Tensor] | None = None,  # same shape as x
    ) -> list[Tensor]:
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        # TODO: device for grid_sizes and seq_lens
        x = [self.patch_embedding(u.to(self.patch_embedding.weight.dtype)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in x])

        x = [u.flatten(1) for u in x]  # each is (dim, F*H*W)
        seq_lens = torch.tensor([u.shape[1] for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len  # NOTE: this will graph-break?

        # pad and stack. transpose to (B, seq_len, dim)
        x = torch.stack([F.pad(u, (0, seq_len - u.shape[1])).T for u in x], dim=0)

        # time embeddings
        if t.ndim == 1:  # this is a bit strange
            t = t.expand(t.shape[0], seq_len)
        time_embeds = sinusoidal_embedding_1d(self.cfg.freq_dim, t.flatten()).unflatten(0, (t.shape[0], seq_len))
        e = self.time_embedding(time_embeds)
        e0 = self.time_projection(e).unflatten(2, (6, -1))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context = torch.stack([F.pad(u, (0, 0, 0, self.cfg.text_len - u.shape[0])) for u in context], dim=0)
        context = self.text_embedding(context.to(self.text_embedding[0].weight.dtype))

        rope_embeds = self.get_rope(x.device)
        for block in self.blocks:
            x = block(
                x=x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                rope_embeds=rope_embeds,
                context=context,
            )

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x: list[Tensor], grid_sizes: Tensor):
        # x: each is [L, Cout, prod(patch_size)]
        # grid_sizes: shape [B, 3] - F/H/W patches
        c = self.cfg.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.cfg.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.cfg.patch_size)])
            out.append(u)
        return out


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
