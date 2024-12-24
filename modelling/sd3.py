# https://github.com/Stability-AI/sd3.5/blob/fbf8f483f992d8d6ad4eaaeb23b1dc5f523c3b3a/mmditx.py

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .flux import LastLayer, timestep_embedding
from .utils import load_hf_state_dict


def attention(q: Tensor, k: Tensor, v: Tensor, num_heads: int) -> Tensor:
    B, L, _ = q.shape
    q, k, v = map(lambda t: t.view(B, L, num_heads, -1).transpose(1, 2), (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(B, L, -1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, x: Tensor):
        return self.proj(x).flatten(2).transpose(1, 2)  # NCHW -> NLC


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.in_dim = in_dim

    def forward(self, t: Tensor, dtype: torch.dtype) -> Tensor:
        t_freq = timestep_embedding(t, self.in_dim).to(dtype)
        return self.mlp(t_freq)


class VectorEmbedder(nn.Sequential):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # NOTE: official code does not cast x to FP32 before normalization. seems like a mistake.
        return self._norm(x.float()).to(x.dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, pre_only: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.ln_q = RMSNorm(head_dim)
        self.ln_k = RMSNorm(head_dim)
        self.proj = nn.Identity() if pre_only else nn.Linear(dim, dim)  # no output projection if pre_only=True

    def pre_attention(self, x: Tensor):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.ln_q(q.view(B, L, self.num_heads, -1)).view(B, L, -1)
        k = self.ln_k(k.view(B, L, self.num_heads, -1)).view(B, L, -1)
        return q, k, v

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        x = self.proj(x)
        return x


class MLP(nn.Sequential):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)


class DismantledBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        pre_only: bool = False,
        x_self_attn: bool = False,
    ) -> None:
        super().__init__()
        self.x_self_attn = x_self_attn
        self.pre_only = pre_only  # no output projection and MLP if pre_only=True

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads, pre_only=pre_only)

        if x_self_attn:
            assert not pre_only
            self.attn2 = SelfAttention(hidden_size, num_heads)
            self.n_mods = 9
        else:
            self.n_mods = 6 if not pre_only else 2

        if not pre_only:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio))

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, self.n_mods * hidden_size))

    def pre_attention(self, x: Tensor, c: Tensor):
        modulation = self.adaLN_modulation(c).chunk(self.n_mods, dim=1)
        x_norm = self.norm1(x)
        qkv = self.attn.pre_attention(modulate(x_norm, modulation[0], modulation[1]))

        if not self.x_self_attn:
            return qkv, modulation[2:]
        else:
            qkv2 = self.attn2.pre_attention(modulate(x_norm, modulation[6], modulation[7]))
            return qkv, (qkv2, *modulation[2:6], modulation[8])

    def post_attention(self, x: Tensor, attn: Tensor, *args: Tensor):
        if self.pre_only:
            return x

        if not self.x_self_attn:
            gate_msa, shift_mlp, scale_mlp, gate_mlp = args
            x = x + gate_msa.unsqueeze(1) * self.attn.proj(attn)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x

        else:
            # extra self-attn for x only. only used in SD3.5-medium
            (q2, k2, v2), gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2 = args
            attn2 = attention(q2, k2, v2, self.attn2.num_heads)
            x = x + gate_msa.unsqueeze(1) * self.attn.proj(attn)
            x = x + gate_msa2.unsqueeze(1) * self.attn2.proj(attn2)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x


class JointBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, context_pre_only: bool, x_self_attn: bool) -> None:
        super().__init__()
        self.context_block = DismantledBlock(hidden_size, num_heads, pre_only=context_pre_only)
        self.x_block = DismantledBlock(hidden_size, num_heads, x_self_attn=x_self_attn)

    def forward(self, context: Tensor, x: Tensor, c: Tensor):
        (context_q, context_k, context_v), context_intermediates = self.context_block.pre_attention(context, c)
        (x_q, x_k, x_v), x_intermediates = self.x_block.pre_attention(x, c)

        # joint self-attention
        q = torch.cat([context_q, x_q], dim=1)
        k = torch.cat([context_k, x_k], dim=1)
        v = torch.cat([context_v, x_v], dim=1)
        attn = attention(q, k, v, self.x_block.attn.num_heads)
        context_attn = attn[:, : context.shape[1]]
        x_attn = attn[:, context.shape[1] :]

        # output projection and MLP
        context = self.context_block.post_attention(context, context_attn, *context_intermediates)
        x = self.x_block.post_attention(x, x_attn, *x_intermediates)
        return context, x


class SD3(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        depth: int = 24,
        adm_in_channels: int = 2048,  # concat of OpenAI CLIP/L and OpenCLIP/bigG
        context_dim: int = 4096,  # T5
        pos_embed_size: int = 192,
        num_x_self_attn_layers: int = 0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed_size = pos_embed_size

        # scale width linearly with depth
        # apply magic --> this defines a head_size of 64
        hidden_size = 64 * depth
        num_heads = depth

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size)
        self.context_embedder = nn.Linear(context_dim, hidden_size)
        self.register_buffer("pos_embed", torch.zeros(1, pos_embed_size * pos_embed_size, hidden_size))

        self.joint_blocks = nn.ModuleList()
        for i in range(depth):
            block = JointBlock(
                hidden_size,
                num_heads,
                context_pre_only=i == depth - 1,
                x_self_attn=i < num_x_self_attn_layers,
            )
            self.joint_blocks.append(block)
        self.final_layer = LastLayer(hidden_size, patch_size, in_channels)

    def crop_pos_embed(self, h: int, w: int):
        h = h // self.patch_size
        w = w // self.patch_size
        assert h <= self.pos_embed_size, (h, self.pos_embed_size)
        assert w <= self.pos_embed_size, (w, self.pos_embed_size)
        top = (self.pos_embed_size - h) // 2
        left = (self.pos_embed_size - w) // 2
        pos_embed = self.pos_embed.view(1, self.pos_embed_size, self.pos_embed_size, -1)
        pos_embed = pos_embed[:, top : top + h, left : left + w, :]
        pos_embed = pos_embed.flatten(1, 2)
        return pos_embed

    def forward(self, x: Tensor, t: Tensor, context: Tensor, y: Tensor, skip_layers: list = []) -> Tensor:
        # NOTE: t should be [0,1] i.e. w/o x1000 scaling
        N, _, H, W = x.shape
        x = self.x_embedder(x) + self.crop_pos_embed(H, W)
        c = self.t_embedder(t, dtype=x.dtype) + self.y_embedder(y)  # (N, D)
        context = self.context_embedder(context)

        for i, block in enumerate(self.joint_blocks):
            if i in skip_layers:
                continue
            context, x = block(context, x, c)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        p = self.patch_size
        x = x.view(N, H // p, W // p, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(N, -1, H, W)
        return x


def load_sd3_5(size: str = "medium"):
    if size == "large":
        depth = 38
        pos_embed_size = 192
        num_x_self_attn_layers = 0
    elif size == "medium":
        depth = 24
        pos_embed_size = 384
        num_x_self_attn_layers = 13
    else:
        raise ValueError(size)

    with torch.device("meta"):
        model = SD3(depth=depth, pos_embed_size=pos_embed_size, num_x_self_attn_layers=num_x_self_attn_layers)

    state_dict = load_hf_state_dict(f"stabilityai/stable-diffusion-3.5-{size}", f"sd3.5_{size}.safetensors")
    prefix = "model.diffusion_model."
    state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}

    # FP16
    model.load_state_dict(state_dict, assign=True)
    return model
