# https://github.com/black-forest-labs/flux/blob/7e14a05ed7280f7a34ece612f7324fcc2ec9efbb/src/flux/modules/autoencoder.py
# https://github.com/Stability-AI/sd3.5/blob/4e484e05308d83fb77ae6f680028e6c313f9da54/sd3_impls.py

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import load_hf_state_dict


@dataclass
class AutoEncoderConfig:
    in_ch: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 1.0
    shift_factor: float = 0.0


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            for name in ("q", "k", "v", "proj_out"):
                key = f"{prefix}{name}.weight"
                state_dict[key] = state_dict[key].squeeze()

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).transpose(-1, -2)  # (N, C, H, W) -> (N, HW, C)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        h = F.scaled_dot_product_attention(q, k, v)
        h = self.proj_out(h)
        h = h.transpose(-1, -2).view(B, C, H, W)

        return x + h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.swish1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.swish2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.swish1(self.norm1(x)))
        h = self.conv2(self.swish2(self.norm2(h)))
        return self.nin_shortcut(x) + h


class Downsample(nn.Sequential):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)


class Upsample(nn.Sequential):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)


class Encoder(nn.Module):
    def __init__(self, in_ch: int, ch: int, ch_mult: tuple[int], num_res_blocks: int, z_channels: int) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = nn.Conv2d(in_ch, ch, 3, 1, 1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # end
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, 1, 1)
        self.swish = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, ch: int, out_ch: int, ch_mult: tuple[int], num_res_blocks: int, z_channels: int) -> None:
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)
        self.swish = nn.SiLU()

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


def diagonal_gaussian(z: Tensor, sample: bool) -> Tensor:
    mean, logvar = torch.chunk(z, 2, dim=1)
    if sample:
        # SD clamps logvar to [-30,20]
        std = (0.5 * logvar).exp()
        return mean + std * torch.randn_like(mean)
    else:
        return mean


class AutoEncoder(nn.Module):
    def __init__(self, config: AutoEncoderConfig | None = None) -> None:
        super().__init__()
        config = config or AutoEncoderConfig()
        self.encoder = Encoder(
            config.in_ch,
            config.ch,
            config.ch_mult,
            config.num_res_blocks,
            config.z_channels,
        )
        self.decoder = Decoder(
            config.ch,
            config.out_ch,
            config.ch_mult,
            config.num_res_blocks,
            config.z_channels,
        )
        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

    def encode(self, x: Tensor, sample: bool = False) -> Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 127.5 - 1
        x = x.to(self.encoder.conv_in.weight.dtype)

        z = diagonal_gaussian(self.encoder(x), sample)
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor, uint8: bool = False) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        x = self.decoder(z)

        if uint8:
            x = x.float().add(1).mul(127.5).clip(0, 255).to(torch.uint8)
        return x

    def forward(self, x: Tensor, sample: bool = False) -> Tensor:
        return self.decode(self.encode(x, sample), x.dtype == torch.uint8)


def load_autoencoder(
    repo_id: str,
    filename: str,
    scale_factor: float,
    shift_factor: float,
    prefix: str | None = None,
):
    config = AutoEncoderConfig(scale_factor=scale_factor, shift_factor=shift_factor)
    with torch.device("meta"):
        ae = AutoEncoder(config)
    state_dict = load_hf_state_dict(repo_id, filename, prefix=prefix)
    ae.load_state_dict(state_dict, assign=True)
    return ae


def load_flux_autoencoder():
    # original weight is FP32
    return load_autoencoder(
        "black-forest-labs/FLUX.1-dev",
        "ae.safetensors",
        scale_factor=0.3611,
        shift_factor=0.1159,
    ).bfloat16()


def load_sd3_autoencoder():
    # official SD3.5 inference code uses FP16 VAE, even though the provided weights are in BF16
    # https://github.com/Stability-AI/sd3.5/blob/fbf8f483f992d8d6ad4eaaeb23b1dc5f523c3b3a/sd3_infer.py#L195-L202
    return load_autoencoder(
        "stabilityai/stable-diffusion-3.5-medium",
        "sd3.5_medium.safetensors",
        # https://github.com/Stability-AI/sd3.5/blob/fbf8f483f992d8d6ad4eaaeb23b1dc5f523c3b3a/sd3_impls.py#L275-L276
        scale_factor=1.5305,
        shift_factor=0.0609,
        prefix="first_stage_model.",
    )
