# https://github.com/black-forest-labs/flux/blob/7e14a05ed7280f7a34ece612f7324fcc2ec9efbb/src/flux/model.py

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..utils import load_hf_state_dict
from .layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding


@dataclass
class FluxConfig:
    in_channels: int = 64
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
        self.config = config
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
        self.final_layer = LastLayer(config.hidden_size, 1, config.in_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


def load_flux(name: str = "dev", dtype=torch.bfloat16):
    assert name in ("dev", "schnell")
    with torch.device("meta"):
        model = Flux()
    state_dict = load_hf_state_dict(f"black-forest-labs/FLUX.1-{name}", f"flux1-{name}.safetensors")
    model.load_state_dict(state_dict, assign=True)
    return model.to(dtype=dtype)
