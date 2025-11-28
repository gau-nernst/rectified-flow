# https://github.com/black-forest-labs/flux/blob/7e14a05e/src/flux/math.py
# https://github.com/black-forest-labs/flux/blob/7e14a05e/src/flux/modules/layers.py
# https://github.com/black-forest-labs/flux/blob/7e14a05e/src/flux/model.py

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attn import dispatch_attn
from .utils import load_hf_state_dict


def rope(pos: Tensor, dim: int, theta: float = 1e4) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = pos.unsqueeze(-1) * omega
    out = torch.stack([out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1)
    out = out.unflatten(-1, (2, 2))  # [..., dim/2, 2, 2]
    return out.float()


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # x: (B, L, nH, D)
    # freqs_cis: (L, D/2, 2, 2)
    x_ = x.float().unflatten(-1, (-1, 1, 2))  # [B, L, nH, D/2, 1, 2]
    freqs_cis = freqs_cis.unsqueeze(-4)  # [L, 1, D/2, 2, 2]
    out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    return out.reshape(x.shape).type_as(x)


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


def fix_qk_norm_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    for k in list(state_dict.keys()):
        if k.endswith(".norm.query_norm.scale"):
            state_dict[k.replace(".norm.query_norm.scale", ".q_norm.weight")] = state_dict.pop(k)
        elif k.endswith(".norm.key_norm.scale"):
            state_dict[k.replace(".norm.key_norm.scale", ".k_norm.weight")] = state_dict.pop(k)


class SelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.RMSNorm(128, eps=1e-6)
        self.k_norm = nn.RMSNorm(128, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.register_load_state_dict_pre_hook(fix_qk_norm_hook)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    x = F.layer_norm(x, x.shape[-1:], eps=1e-6)
    return (1.0 + scale) * x + shift


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool) -> None:
        super().__init__()
        self.multiplier = 6 if double else 3
        self.silu = nn.SiLU()
        self.lin = nn.Linear(dim, self.multiplier * dim)

    def forward(self, vec: Tensor) -> tuple[Tensor, ...]:
        return self.lin(self.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)


class DoubleStreamBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, attn_impl: str = "pt") -> None:
        super().__init__()
        self.head_dim = 128
        mlp_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_impl = attn_impl

        self.img_mod = Modulation(dim, double=True)
        self.img_attn = SelfAttention(dim)
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim),
        )

        self.txt_mod = Modulation(dim, double=True)
        self.txt_attn = SelfAttention(dim)
        self.txt_mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = self.img_mod(vec)
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = self.txt_mod(vec)

        # prepare image for attention
        img_qkv = self.img_attn.qkv(modulate(img, img_shift1, img_scale1))
        img_q, img_k, img_v = img_qkv.unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)

        # prepare txt for attention
        txt_qkv = self.txt_attn.qkv(modulate(txt, txt_shift1, txt_scale1))
        txt_q, txt_k, txt_v = txt_qkv.unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)

        # run actual attention
        q = apply_rope(torch.cat((self.txt_attn.q_norm(txt_q), self.img_attn.q_norm(img_q)), dim=1), pe)
        k = apply_rope(torch.cat((self.txt_attn.k_norm(txt_k), self.img_attn.k_norm(img_k)), dim=1), pe)
        v = torch.cat((txt_v, img_v), dim=1)

        attn = dispatch_attn(q, k, v, impl=self.attn_impl).flatten(2)
        txt_attn, img_attn = attn.split([txt.shape[1], img.shape[1]], dim=1)

        # calculate the img blocks
        img = img + img_gate1 * self.img_attn.proj(img_attn)
        img = img + img_gate2 * self.img_mlp(modulate(img, img_shift2, img_scale2))

        # calculate the txt blocks
        txt = txt + txt_gate1 * self.txt_attn.proj(txt_attn)
        txt = txt + txt_gate2 * self.txt_mlp(modulate(txt, txt_shift2, txt_scale2))
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, attn_impl: str = "pt") -> None:
        super().__init__()
        self.dim = dim
        self.attn_impl = attn_impl
        self.head_dim = 128
        self.mlp_dim = int(dim * mlp_ratio)

        self.linear1 = nn.Linear(dim, dim * 3 + self.mlp_dim)  # qkv and mlp_in
        self.linear2 = nn.Linear(dim + self.mlp_dim, dim)  # proj and mlp_out

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.register_load_state_dict_pre_hook(fix_qk_norm_hook)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(dim, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        shift, scale, gate = self.modulation(vec)
        x_mod = modulate(x, shift, scale)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.dim, self.mlp_dim], dim=-1)

        q, k, v = qkv.unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        q = apply_rope(self.q_norm(q), pe)
        k = apply_rope(self.k_norm(k), pe)
        attn = dispatch_attn(q, k, v, impl=self.attn_impl).flatten(2)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + gate * output


class LastLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec)[:, None, :].chunk(2, dim=-1)
        return self.linear(modulate(x, shift, scale))


@dataclass
class FluxConfig:
    in_channels: int = 64
    out_channels: int = 64
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    depth: int = 19
    depth_single_blocks: int = 38
    rope_dims: tuple[int, int, int] = (16, 56, 56)
    theta: float = 1e4
    guidance_embed: bool = True  # False for schnell


class Flux(nn.Module):
    def __init__(self, cfg: FluxConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or FluxConfig()
        self.cfg = cfg
        self.img_in = nn.Linear(cfg.in_channels, cfg.hidden_size)
        self.time_in = MLPEmbedder(256, cfg.hidden_size)
        self.vector_in = MLPEmbedder(cfg.vec_in_dim, cfg.hidden_size)
        self.guidance_in = MLPEmbedder(256, cfg.hidden_size) if cfg.guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(cfg.context_in_dim, cfg.hidden_size)

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(cfg.hidden_size, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(cfg.hidden_size, cfg.mlp_ratio) for _ in range(cfg.depth_single_blocks)]
        )
        self.final_layer = LastLayer(cfg.hidden_size, 1, cfg.out_channels)

    def build_rope(self, H: int, W: int, device: torch.types.Device = None):
        rope_d0, rope_dh, rope_dw = self.cfg.rope_dims
        theta = self.cfg.theta

        rope_0 = rope(torch.arange(1, device=device), rope_d0, theta)  # [1, 8, 2, 2]
        rope_h = rope(torch.arange(H, device=device), rope_dh, theta)  # [H, 28, 2, 2]
        rope_w = rope(torch.arange(W, device=device), rope_dw, theta)  # [W, 28, 2, 2]

        rope_0 = rope_0.unflatten(0, (1, 1)).expand(H, W, -1, -1, -1)
        rope_h = rope_h.unflatten(0, (H, 1)).expand(H, W, -1, -1, -1)
        rope_w = rope_w.unflatten(0, (1, W)).expand(H, W, -1, -1, -1)

        return torch.cat([rope_0, rope_h, rope_w], dim=2).flatten(0, 1)  # [H*W, 64, 2, 2]

    def forward(self, img: Tensor, timesteps: Tensor, txt: Tensor, y: Tensor, guidance: Tensor | None = None) -> Tensor:
        # we integrate patchify and unpatchify into model's forward pass
        B, _, H, W = img.shape
        txt_len = txt.shape[1]
        img = img.to(self.img_in.weight.dtype)
        img = img.view(B, -1, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, H * W // 4, -1)

        img = self.img_in(img)
        txt = self.txt_in(txt)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype)) + self.vector_in(y)

        if guidance is not None:  # allow no guidance_embed
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        rope_img = self.build_rope(H // 2, W // 2, img.device)
        rope_txt = self.build_rope(1, 1, img.device).expand(txt_len, -1, -1, -1)
        pe = torch.cat([rope_txt, rope_img], dim=0)

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, pe)

        img = torch.cat([txt, img], dim=1)
        for block in self.single_blocks:
            img = block(img, vec, pe)
        img = img[:, txt_len:]

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
        "flux-krea-dev": ("black-forest-labs/FLUX.1-Krea-dev", "flux1-krea-dev.safetensors", None),
        "flex1-alpha": ("ostris/Flex.1-alpha", "Flex.1-alpha.safetensors", "model.diffusion_model."),
        "flex2-preview": ("ostris/Flex.2-preview", "Flex.2-preview.safetensors", None),
    }[name]

    # BF16
    return _load_flux(repo_id, filename, prefix=prefix)


# https://github.com/black-forest-labs/flux/blob/805da857/src/flux/modules/image_embedders.py#L66
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
