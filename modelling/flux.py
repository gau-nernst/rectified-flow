# https://github.com/black-forest-labs/flux/blob/7e14a05ed7280f7a34ece612f7324fcc2ec9efbb/src/flux/math.py
# https://github.com/black-forest-labs/flux/blob/7e14a05ed7280f7a34ece612f7324fcc2ec9efbb/src/flux/modules/layers.py
# https://github.com/black-forest-labs/flux/blob/7e14a05ed7280f7a34ece612f7324fcc2ec9efbb/src/flux/model.py

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import load_hf_state_dict
from .attn import dispatch_attention


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, impl: str = "pt") -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = dispatch_attention(q, k, v, impl=impl)
    x = x.transpose(-2, -3).flatten(-2)  # (B, num_head, L, head_dim) -> (B, L, num_head * head_dim)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = pos.unsqueeze(-1) * omega
    out = torch.stack([out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1)
    out = out.unflatten(-1, (2, 2))
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t.float()
    half = dim // 2
    freqs = torch.exp(-(math.log(max_period) / half) * torch.arange(0, half, dtype=torch.float32, device=t.device))
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLPEmbedder(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return x.to(dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q), self.key_norm(k)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    # def forward(self, x: Tensor, pe: Tensor) -> Tensor:
    #     qkv = self.qkv(x)
    #     q, k, v = qkv.unflatten(2, (3 * self.num_heads, -1)).transpose(1, 2).chunk(3, dim=1)
    #     q, k = self.norm(q, k)
    #     x = attention(q, k, v, pe=pe)
    #     x = self.proj(x)
    #     return x


class ModulationOut(NamedTuple):
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool) -> None:
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.silu = nn.SiLU()
        self.lin = nn.Linear(dim, self.multiplier * dim)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(self.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, attn_impl: str = "pt"
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.attn_impl = attn_impl

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.unflatten(2, (3 * self.num_heads, -1)).permute(0, 2, 1, 3).chunk(3, dim=1)
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.unflatten(2, (3 * self.num_heads, -1)).permute(0, 2, 1, 3).chunk(3, dim=1)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, impl=self.attn_impl)
        txt_attn = attn[:, : txt.shape[1]].contiguous()
        img_attn = attn[:, txt.shape[1] :].contiguous()

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, attn_impl: str = "pt") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn_impl = attn_impl
        head_dim = hidden_size // num_heads

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)  # qkv and mlp_in
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)  # proj and mlp_out

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.unflatten(2, (3 * self.num_heads, -1)).permute(0, 2, 1, 3).chunk(3, dim=1)
        q, k = self.norm(q, k)
        attn = attention(q, k, v, pe=pe, impl=self.attn_impl)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


@dataclass
class FluxConfig:
    in_channels: int = 64
    out_channels: int = 64
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: tuple[int] = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    guidance_embed: bool = True  # False for schnell


class Flux(nn.Module):
    def __init__(self, config: FluxConfig | None = None) -> None:
        super().__init__()
        config = config or FluxConfig()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}")
        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(f"Got {config.axes_dim} but expected positional dim {pe_dim}")

        self.pe_embedder = EmbedND(pe_dim, config.theta, config.axes_dim)
        self.img_in = nn.Linear(config.in_channels, config.hidden_size)
        self.time_in = MLPEmbedder(256, config.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, config.hidden_size)
        self.guidance_in = MLPEmbedder(256, config.hidden_size) if config.guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(config.context_in_dim, config.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(config.hidden_size, config.num_heads, config.mlp_ratio, config.qkv_bias)
                for _ in range(config.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth_single_blocks)
            ]
        )
        self.final_layer = LastLayer(config.hidden_size, 1, config.out_channels)

    @staticmethod
    def create_img_ids(bsize: int, height: int, width: int):
        img_ids = torch.zeros(height, width, 3)
        img_ids[..., 1] = torch.arange(height)[:, None]
        img_ids[..., 2] = torch.arange(width)[None, :]
        return img_ids.view(1, height * width, 3).expand(bsize, -1, -1)

    def forward(self, img: Tensor, timesteps: Tensor, txt: Tensor, y: Tensor, guidance: Tensor | None = None) -> Tensor:
        # we integrate patchify and unpatchify into model's forward pass
        B, _, H, W = img.shape
        img = img.view(B, -1, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, H * W // 4, -1)

        img = self.img_in(img)
        txt = self.txt_in(txt)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype)) + self.vector_in(y)

        if guidance is not None:  # allow no guidance_embed
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        img_ids = self.create_img_ids(B, H // 2, W // 2).cuda()
        txt_ids = torch.zeros(*txt.shape[:2], 3, device=txt.device)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec, pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        img = img.view(B, H // 2, W // 2, -1, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, -1, H, W)
        return img


def _load_flux(repo_id: str, filename: str, prefix: str | None = None):
    state_dict = load_hf_state_dict(repo_id, filename, prefix=prefix)

    # support depth-pruned FLUX, such as Flex.1-alpha
    config = FluxConfig(depth=0, depth_single_blocks=0)
    for key in state_dict.keys():
        if key.startswith("double_blocks."):
            config.depth = max(config.depth, int(key.split(".")[1]) + 1)
        elif key.startswith("single_blocks."):
            config.depth_single_blocks = max(config.depth_single_blocks, int(key.split(".")[1]) + 1)

    with torch.device("meta"):
        model = Flux(config)
    model.load_state_dict(state_dict, assign=True)
    return model


def load_flux(name: str = "flux-dev"):
    repo_id, filename, prefix = {
        "flux-dev": ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", None),
        "flux-schnell": ("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors", None),
        "flex1-alpha": ("ostris/Flex.1-alpha", "Flex.1-alpha.safetensors", "model.diffusion_model."),
        "flex2-preview": ("ostris/Flex.2-preview", "Flex.2-preview.safetensors", None),
    }[name]

    # BF16
    return _load_flux(repo_id, filename, prefix=prefix)


# https://github.com/black-forest-labs/flux/blob/805da8571a0b49b6d4043950bd266a65328c243b/src/flux/modules/image_embedders.py#L66
def load_flux_redux(dtype: torch.dtype = torch.bfloat16):
    import timm

    siglip = timm.create_model(
        "vit_so400m_patch14_siglip_378.webli",
        pretrained=True,
        dynamic_img_size=True,
    )  # 428M params
    siglip.forward = siglip.forward_features  # with 378x378 input, output is (B, 729, 1152)

    in_dim = 1152  # siglip
    out_dim = 4096  # t5
    with torch.device("meta"):
        redux = nn.Sequential()  # 64.5M params
        redux.redux_up = nn.Linear(in_dim, out_dim * 3)
        redux.silu = nn.SiLU()
        redux.redux_down = nn.Linear(out_dim * 3, out_dim)

    state_dict = load_hf_state_dict("black-forest-labs/FLUX.1-Redux-dev", "flux1-redux-dev.safetensors")
    redux.load_state_dict(state_dict, assign=True)
    return nn.Sequential(siglip, redux).to(dtype=dtype)
