# https://github.com/Wan-Video/Wan2.2/blob/388807310646ed5f318a99f8e8d9ad28c5b65373/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..utils import load_hf_state_dict
from .attention import flash_attention


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x: Tensor, grid_sizes: Tensor, freqs: Tensor) -> Tensor:
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


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

    def forward(self, x: Tensor, seq_lens: Tensor, grid_sizes: Tensor, freqs: Tensor) -> Tensor:
        # x: [B, L, num_heads, C / num_heads]
        # seq_lens: [B]
        # grid_sizes: [B, 3]
        # freqs: [1024, C / num_heads / 2]
        B, S, N, D = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(B, S, N, D)
        k = self.norm_k(self.k(x)).view(B, S, N, D)
        v = self.v(x).view(B, S, N, D)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
        )

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x: Tensor, context: Tensor, context_lens: Tensor) -> Tensor:
        # x: [B, L1, C]
        # context: [B, L2, C]
        # context_lens: [B]
        B, N, D = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(B, -1, N, D)
        k = self.norm_k(self.k(context)).view(B, -1, N, D)
        v = self.v(context).view(B, -1, N, D)

        x = flash_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        x = self.o(x)
        return x


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
        freqs: Tensor,  # [1024, C/num_heads/2]
        context: Tensor,
        context_lens: Tensor,
    ) -> Tensor:
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = self.modulation.unsqueeze(0) + e
        assert e.dtype == torch.float32

        # self-attention
        # NOTE: once we remove autocast, we don't need .to(x.dtype) anymore
        y = self.self_attn(self.norm1(x).to(x.dtype) * (1 + e[:, :, 1]) + e[:, :, 0], seq_lens, grid_sizes, freqs)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[:, :, 2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x).to(x.dtype), context, context_lens)
            y = self.ffn(self.norm2(x).to(x.dtype) * (1 + e[:, :, 4]) + e[:, :, 3])
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[:, :, 5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
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
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x).to(x.dtype) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


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
            nn.Linear(cfg.freq_dim, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim * 6),
        )

        # blocks
        self.blocks = nn.ModuleList(
            [WanAttentionBlock(cfg.dim, cfg.ffn_dim, cfg.head_dim, eps=cfg.eps) for _ in range(cfg.num_layers)]
        )

        # head
        self.head = Head(cfg.dim, cfg.out_dim, cfg.patch_size, cfg.eps)
        self.init_freqs()

    def init_freqs(self):
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        d = self.cfg.head_dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
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
        # params
        self.freqs = self.freqs.to(self.patch_embedding.weight.device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.shape[1] for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len  # NOTE: this will graph-break?
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.shape[1], u.shape[2])], dim=1) for u in x])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(sinusoidal_embedding_1d(self.cfg.freq_dim, t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.cfg.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.cfg.text_len - u.shape[0], u.shape[1])]) for u in context])
        )

        for block in self.blocks:
            x = block(
                x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
            )

        # head
        x = self.head(x, e)

        # unpatchify
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
    model.init_freqs()

    state_dict = load_hf_state_dict(repo_id, filename)
    model.load_state_dict(state_dict, assign=True)
    return model
