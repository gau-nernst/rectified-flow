# https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/model.py

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import LastLayer, MLPEmbedder, Modulation, SelfAttention, modulate, timestep_embedding
from ..rope import RopeND, apply_rope
from ..utils import create_name_map_hook, load_hf_state_dict


class SwiGLU(nn.Module):
    def forward(self, x: Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


# compared to Flux.1
# - per-layer modulation projection is replaced with a single model-wide projection.
# - no bias.
# - gelu is replaced with swiglu
class DoubleStreamBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, attn_impl: str = "pt", eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        mlp_dim = int(dim * mlp_ratio)
        self.attn_impl = attn_impl

        self.img_attn = SelfAttention(dim, bias=False, eps=eps)
        self.txt_attn = SelfAttention(dim, bias=False, eps=eps)

        self.img_mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim * 2, bias=False), SwiGLU(), nn.Linear(mlp_dim, dim, bias=False)
        )
        self.txt_mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim * 2, bias=False), SwiGLU(), nn.Linear(mlp_dim, dim, bias=False)
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        mod_img: tuple[Tensor, ...],
        mod_txt: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = mod_img
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = mod_txt

        img_q, img_k, img_v = self.img_attn.forward_qkv(modulate(img, img_shift1, img_scale1))
        txt_q, txt_k, txt_v = self.txt_attn.forward_qkv(modulate(txt, txt_shift1, txt_scale1))

        q = apply_rope(torch.cat((txt_q, img_q), dim=1), pe)
        k = apply_rope(torch.cat((txt_k, img_k), dim=1), pe)
        v = torch.cat((txt_v, img_v), dim=1)
        attn = dispatch_attn(q, k, v, impl=self.attn_impl).flatten(2)
        txt_attn, img_attn = attn.split([txt.shape[1], img.shape[1]], dim=1)

        img = img + img_gate1 * self.img_attn.proj(img_attn)
        img = img + img_gate2 * self.img_mlp(modulate(img, img_shift2, img_scale2))

        txt = txt + txt_gate1 * self.txt_attn.proj(txt_attn)
        txt = txt + txt_gate2 * self.txt_mlp(modulate(txt, txt_shift2, txt_scale2))

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, attn_impl: str = "pt", eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        self.dim = dim
        self.mlp_dim = int(dim * mlp_ratio)
        self.attn_impl = attn_impl
        self.eps = eps

        self.linear1 = nn.Linear(dim, dim * 3 + self.mlp_dim * 2, bias=False)  # qkv and mlp_in
        self.linear2 = nn.Linear(dim + self.mlp_dim, dim, bias=False)  # proj and mlp_out

        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.mlp_act = SwiGLU()

        remap_pairs = [
            ("norm.query_norm.scale", "q_norm.weight"),
            ("norm.key_norm.scale", "k_norm.weight"),
        ]
        self.register_load_state_dict_pre_hook(create_name_map_hook(remap_pairs))

    def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, ...]) -> Tensor:
        shift, scale, gate = mod
        x_mod = modulate(x, shift, scale, eps=self.eps)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.dim, self.mlp_dim * 2], dim=-1)

        q, k, v = qkv.unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        q = apply_rope(self.q_norm(q), pe)
        k = apply_rope(self.k_norm(k), pe)
        attn = dispatch_attn(q, k, v, impl=self.attn_impl).flatten(2)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + gate * output


# default is Klein-4B
class Flux2Config(NamedTuple):
    img_dim: int = 128
    txt_dim: int = 7680
    dim: int = 3072
    mlp_ratio: float = 3.0
    num_double_blocks: int = 5
    num_single_blocks: int = 20
    patch_size: int = 2
    cfg_distill: bool = False


class Flux2(nn.Module):
    def __init__(self, cfg: Flux2Config = Flux2Config()) -> None:
        super().__init__()
        self.cfg = cfg

        # input projections
        self.img_in = nn.Linear(cfg.img_dim, cfg.dim, bias=False)
        self.txt_in = nn.Linear(cfg.txt_dim, cfg.dim, bias=False)
        self.time_in = MLPEmbedder(256, cfg.dim, bias=False)
        if cfg.cfg_distill:
            self.guidance_in = MLPEmbedder(256, cfg.dim, bias=False)

        # 4D rope
        self.pos_embed = RopeND(dims=(32, 32, 32, 32), max_lens=(512, 512, 512, 512), theta=2e3)

        self.double_stream_modulation_img = Modulation(cfg.dim, double=True, bias=False)
        self.double_stream_modulation_txt = Modulation(cfg.dim, double=True, bias=False)
        self.single_stream_modulation = Modulation(cfg.dim, double=False, bias=False)

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(cfg.dim, cfg.mlp_ratio) for _ in range(cfg.num_double_blocks)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(cfg.dim, cfg.mlp_ratio) for _ in range(cfg.num_single_blocks)]
        )
        self.final_layer = LastLayer(cfg.dim, 1, cfg.img_dim, bias=False)

    def make_rope(self, H: int, W: int, L: int):
        # main image:         https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/sampling.py#L93
        # conditioned images: https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/sampling.py#L52
        # conditioned text:   https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/sampling.py#L141
        # RoPE embedding has 4 components:
        # - time: main image is at t=0, conditioning images are at t=10, 20, ...
        # - height: all text embeds stay at pos=0
        # - width: all text embeds stay at pos=0
        # - text length
        # technically conditioning images can have sizes different from the main image's
        img_rope = self.pos_embed.create((0, 0, 0, 0), (1, H, W, 1))
        txt_rope = self.pos_embed.create((0, 0, 0, 0), (1, 1, 1, L))
        return torch.cat([txt_rope, img_rope], dim=0)

    def forward(
        self,
        img: Tensor,
        time: Tensor,
        txt: Tensor,
        guidance: Tensor | None = None,
        rope: Tensor | None = None,
    ) -> Tensor:
        _, _, H, W = img.shape
        L = txt.shape[1]
        img = img.to(self.img_in.weight.dtype).flatten(-2).transpose(1, 2)

        img = self.img_in(img)
        txt = self.txt_in(txt)
        vec = self.time_in(timestep_embedding(time, 256).to(img.dtype))

        if guidance is not None:  # allow no guidance_embed
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if rope is None:
            rope = self.make_rope(H, W, L)

        mod_img = self.double_stream_modulation_img(vec)
        mod_txt = self.double_stream_modulation_txt(vec)
        for block in self.double_blocks:
            img, txt = block(img, txt, rope, mod_img, mod_txt)

        joint = torch.cat([txt, img], dim=1)
        mod = self.single_stream_modulation(vec)
        for block in self.single_blocks:
            joint = block(joint, rope, mod)
        img = joint[:, L:]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        img = img.transpose(1, 2).unflatten(-1, (H, W))
        return img


def _load_flux2(repo_id: str, filename: str):
    state_dict = load_hf_state_dict(repo_id, filename)

    num_double_blocks = 0
    num_single_blocks = 0

    for key in state_dict.keys():
        if key.startswith("double_blocks."):
            num_double_blocks = max(num_double_blocks, int(key.split(".")[1]) + 1)
        elif key.startswith("single_blocks."):
            num_single_blocks = max(num_single_blocks, int(key.split(".")[1]) + 1)

    dim, txt_dim = state_dict["txt_in.weight"].shape
    cfg_distill = "guidance_in.in_layer.weight" in state_dict

    cfg = Flux2Config(
        txt_dim=txt_dim,
        dim=dim,
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
        cfg_distill=cfg_distill,
    )
    with torch.device("meta"):
        model = Flux2(cfg)

    model.load_state_dict(state_dict, assign=True)
    return model


def load_flux2(name: str = "klein-4B"):
    repo_id, filename = {
        "dev": ("black-forest-labs/FLUX.2-dev", "flux2-dev.safetensors"),
        "klein-4B": ("black-forest-labs/FLUX.2-klein-4B", "flux-2-klein-4b.safetensors"),
        "klein-9B": ("black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors"),
        "klein-base-4B": ("black-forest-labs/FLUX.2-klein-base-4B", "flux-2-klein-base-4b.safetensors"),
        "klein-base-9B": ("black-forest-labs/FLUX.2-klein-base-9B", "flux-2-klein-base-9b.safetensors"),
    }[name]

    return _load_flux2(repo_id, filename)
