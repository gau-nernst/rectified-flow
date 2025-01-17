# https://github.com/Stability-AI/generative-models/blob/1659a1c09b0953ad9cc0d480f42e4526c5575b37/sgm/modules/diffusionmodules/openaimodel.py

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .autoencoder import Upsample
from .flux import timestep_embedding
from .utils import load_hf_state_dict


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Sequential):
    def __init__(self, dim: int, out_dim: int | None = None, mult: float = 4.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        out_dim = out_dim or dim
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Identity(), nn.Linear(inner_dim, out_dim))


class Attention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8, head_dim: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        inner_dim = head_dim * num_heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim))

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        if context is None:
            context = x
        q = self.to_q(x).unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        k = self.to_k(context).unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        v = self.to_v(context).unflatten(2, (self.num_heads, -1)).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(out.transpose(1, 2).flatten(2))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.attn1 = Attention(dim, dim, num_heads=n_heads, head_dim=d_head)
        self.ff = FeedForward(dim)
        self.attn2 = Attention(dim, context_dim, num_heads=n_heads, head_dim=d_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int, context_dim: int, n_heads: int, head_dim: int, depth: int) -> None:
        super().__init__()
        inner_dim = n_heads * head_dim
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [CrossAttentionBlock(inner_dim, context_dim, n_heads, head_dim) for _ in range(depth)]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        _, _, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        h = self.proj_in(h)
        for block in self.transformer_blocks:
            h = block(h, context=context)
        h = self.proj_out(h).transpose(1, 2).unflatten(2, (H, W))  # (N, H*W, C) -> (N, C, H, W)
        return x + h


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float,
        up: bool = False,
        down: bool = False,
    ) -> None:
        super().__init__()
        assert not (up and down)
        self.resample = (
            nn.Upsample(scale_factor=2.0, mode="nearest") if up else nn.AvgPool2d(2) if down else nn.Identity()
        )
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.Sequential(nn.SiLU(), self.resample),
            nn.Conv2d(channels, out_channels, 3, 1, 1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.skip_connection = nn.Conv2d(channels, out_channels, 1) if out_channels != channels else nn.Identity()

        for p in self.out_layers.parameters():
            nn.init.zeros_(p)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        emb_out = self.emb_layers(emb).type(x.dtype).view(x.shape[0], -1, 1, 1)
        h = self.out_layers(self.in_layers(x) + emb_out)
        return self.skip_connection(self.resample(x)) + h


class CustomSequential(nn.Sequential):
    def forward(self, x: Tensor, emb: Tensor, context: Tensor) -> Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class MlpEmbedder(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))


class UNet(nn.Module):
    def __init__(
        self,
        model_channels: int = 320,
        transformer_depths: tuple[int, ...] = (2, 10),
        attention_resolutions: tuple[int, ...] = (2, 4),
        in_channels: int = 4,
        out_channels: int = 4,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        head_dim: int = 64,
        context_dim: int = 2048,
        adm_in_channels: int = 2816,
    ) -> None:
        super().__init__()
        self.model_channels = model_channels

        time_embed_dim = model_channels * 4
        self.time_embed = MlpEmbedder(model_channels, time_embed_dim)
        self.label_emb = nn.Sequential(MlpEmbedder(adm_in_channels, time_embed_dim))

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(CustomSequential(nn.Conv2d(in_channels, model_channels, 3, 1, 1)))

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = model_channels * mult
                block = CustomSequential(ResBlock(ch, out_ch, time_embed_dim, dropout))
                ch = out_ch
                if ds in attention_resolutions:
                    depth = transformer_depths[attention_resolutions.index(ds)]
                    block.append(SpatialTransformer(ch, context_dim, ch // head_dim, head_dim, depth))

                self.input_blocks.append(block)
                input_block_chans.append(ch)

            if level < len(channel_mult) - 1:
                layer = nn.Sequential()
                layer.op = nn.Conv2d(ch, ch, 3, 2, 1)
                self.input_blocks.append(CustomSequential(layer))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = CustomSequential(
            ResBlock(ch, ch, time_embed_dim, dropout),
            SpatialTransformer(ch, context_dim, ch // head_dim, head_dim, transformer_depths[-1]),
            ResBlock(ch, ch, time_embed_dim, dropout),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                out_ch = model_channels * mult
                block = CustomSequential(ResBlock(ch + ich, out_ch, time_embed_dim, dropout))
                ch = out_ch
                if ds in attention_resolutions:
                    depth = transformer_depths[attention_resolutions.index(ds)]
                    block.append(SpatialTransformer(ch, context_dim, ch // head_dim, head_dim, depth))

                if level < len(channel_mult) - 1 and i == num_res_blocks:
                    block.append(Upsample(ch))
                    ds //= 2

                self.output_blocks.append(block)

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, 1, 1),
        )

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: Tensor, timesteps: Tensor, context: Tensor, y: Tensor) -> Tensor:
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb.to(self.time_embed[0].weight.dtype))
        emb = t_emb + self.label_emb(y)

        hs = [x]
        for module in self.input_blocks:
            hs.append(module(hs[-1], emb, context))

        h = self.middle_block(hs[-1], emb)

        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb, context)

        return self.out(h.type(x.dtype))


def load_unet(repo_id: str, filename: str, prefix: str | None = None):
    state_dict = load_hf_state_dict(repo_id, filename, prefix=prefix)
    with torch.device("meta"):
        unet = UNet()
    unet.load_state_dict(state_dict, assign=True)
    return unet


def load_sdxl():
    return load_unet(
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sd_xl_base_1.0_0.9vae.safetensors",
        prefix="model.diffusion_model.",
    )
