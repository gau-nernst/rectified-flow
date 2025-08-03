# https://github.com/Wan-Video/Wan2.2/blob/388807310646ed5f318a99f8e8d9ad28c5b65373/wan/modules/vae2_1.py
# https://github.com/Wan-Video/Wan2.2/blob/388807310646ed5f318a99f8e8d9ad28c5b65373/wan/modules/vae2_2.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import load_hf_state_dict

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x: Tensor, cache_x: Tensor | None = None) -> Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)

    def cache_forward(
        self,
        x: Tensor,
        feat_cache: list[Tensor | None] | None = None,
        feat_idx: list[int] = [0],
    ) -> Tensor:
        # TODO: understand this
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            out = self(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            out = self(x)

        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, images: int = True) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *broadcastable_dims))

    def forward(self, x: Tensor) -> Tensor:
        out = F.normalize(x.float(), dim=1) * self.gamma * self.scale
        return out.to(x.dtype)


class Resample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mode: str) -> None:
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(in_dim, out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.time_conv = CausalConv3d(in_dim, in_dim * 2, (3, 1, 1), padding=(1, 0, 0))
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(in_dim, out_dim, 3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(in_dim, out_dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(in_dim, out_dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(in_dim, in_dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: Tensor, feat_cache: list[Tensor | None] | None = None, feat_idx: list[int] = [0]) -> Tensor:
        B, C, T, H, W = x.shape

        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(B, 2, C, T, H, W)
                x = x.permute(0, 2, 3, 1, 4, 5)
                x = x.reshape(B, C, T * 2, H, W)

        x = x.transpose(1, 2).flatten(0, 1)  # (B*T, C, H, W)
        x = self.resample(x)
        x = x.unflatten(0, (B, -1)).transpose(1, 2)  # (B, C, T, H, W)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor, feat_cache: list[Tensor | None] | None = None, feat_idx: list[int] = [0]) -> Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d):
                x = layer.cache_forward(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        # TODO: rewrite conv as linear
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).flatten(0, 1)  # (B*T, C, H, W)

        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(B * T, 1, C * 3, -1).transpose(2, 3).contiguous().chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(2, 3).reshape(B * T, C, H, W)

        x = self.proj(x)
        x = x.reshape(B, T, C, H, W).transpose(1, 2)  # (B, C, T, H, W)
        return x + identity


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def avg_down(x: Tensor, out_dim: int, factor_t: int, factor_s: int) -> Tensor:
    pad_t = cdiv(x.shape[2], factor_t) * factor_t - x.shape[2]
    x = F.pad(x, (0, 0, 0, 0, pad_t, 0))

    B, C, T, H, W = x.shape
    new_T = T // factor_t
    new_H = H // factor_s
    new_W = W // factor_s

    x = x.view(B, C, new_T, factor_t, new_H, factor_s, new_W, factor_s)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.reshape(B, out_dim, -1, new_T, new_H, new_W)
    x = x.mean(dim=2)

    return x


def dup_up(x: Tensor, out_dim: int, factor_t: int, factor_s: int, first_chunk: bool) -> Tensor:
    B, C, T, H, W = x.shape
    num_repeats = out_dim * factor_t * factor_s * factor_s // C
    new_T = T * factor_t
    new_H = H * factor_s
    new_W = W * factor_s

    x = x.repeat_interleave(num_repeats, dim=1).unflatten(1, (out_dim, factor_t, factor_s, factor_s))
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, -1, new_T, new_H, new_W)

    # TODO: understand this
    if first_chunk:
        x = x[:, :, factor_t - 1 :]

    return x


class DownResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        down_t: bool = False,
        down_s: bool = False,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.factor_t = 2 if down_t else 1
        self.factor_s = 2 if down_s else 1

        # Main path with residual blocks and downsample
        self.downsamples = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_s:
            mode = "downsample3d" if down_t else "downsample2d"
            self.downsamples.append(Resample(out_dim, out_dim, mode))

    def forward(self, x: Tensor, feat_cache: list[Tensor | None] | None = None, feat_idx: list[int] = [0]) -> Tensor:
        shortcut = avg_down(x, self.out_dim, self.factor_t, self.factor_s)
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)
        return x + shortcut


class UpResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, mult: int, up_t: bool = False, up_s: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.factor_t = 2 if up_t else 1
        self.factor_s = 2 if up_s else 1

        # Main path with residual blocks and upsample
        self.upsamples = nn.ModuleList()
        for _ in range(mult):
            self.upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_s:
            mode = "upsample3d" if up_t else "upsample2d"
            self.upsamples.append(Resample(out_dim, out_dim, mode))

    def forward(
        self,
        x: Tensor,
        feat_cache: list[Tensor | None] | None = None,
        feat_idx: list[int] = [0],
        first_chunk: bool = False,
    ) -> Tensor:
        shortcut = dup_up(x, self.out_dim, self.factor_t, self.factor_s, first_chunk) if self.factor_s > 1 else None
        for module in self.upsamples:
            x = module(x, feat_cache, feat_idx)
        if shortcut is not None:
            x = x + shortcut
        return x


class Encoder3d(nn.Module):
    def __init__(
        self,
        img_dim: int = 3,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temporal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        version: str = "2.1",
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]

        self.conv1 = CausalConv3d(img_dim, dims[0], 3, padding=1)

        self.downsamples = nn.ModuleList()
        for i, (img_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if version == "2.1":
                # residual (+attention) blocks
                for _ in range(num_res_blocks):
                    self.downsamples.append(ResidualBlock(img_dim, out_dim, dropout))
                    img_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                    self.downsamples.append(Resample(out_dim, out_dim, mode))

            elif version == "2.2":
                down_t = temporal_downsample[i] if i < len(temporal_downsample) else False
                down_s = i != len(dim_mult) - 1
                self.downsamples.append(DownResidualBlock(img_dim, out_dim, dropout, num_res_blocks, down_t, down_s))

            else:
                raise ValueError(f"Unsupported {version=}")

        middle = [
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        ]
        self.middle = nn.ModuleList(middle)
        head = [
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        ]
        self.head = nn.ModuleList(head)

    def forward(self, x: Tensor, feat_cache: list[Tensor | None] | None = None, feat_idx: list[int] = [0]) -> Tensor:
        x = self.conv1.cache_forward(x, feat_cache, feat_idx)

        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d):
                x = layer.cache_forward(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        img_dim: int = 3,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temporal_upsample: list[bool] = [True, True, False],
        dropout: float = 0.0,
        version: str = "2.1",
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        middle = [
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        ]
        self.middle = nn.ModuleList(middle)

        self.upsamples = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if version == "2.1":
                # in Wan2.1, upsample reduces dim by 2
                if i > 0:
                    in_dim = in_dim // 2
                for _ in range(num_res_blocks + 1):
                    self.upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                    in_dim = out_dim

                # upsample block
                if i != len(dim_mult) - 1:
                    mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
                    self.upsamples.append(Resample(out_dim, out_dim // 2, mode))

            elif version == "2.2":
                up_t = temporal_upsample[i] if i < len(temporal_upsample) else False
                up_s = i != len(dim_mult) - 1
                self.upsamples.append(UpResidualBlock(in_dim, out_dim, dropout, num_res_blocks + 1, up_t, up_s))

            else:
                raise ValueError(f"Unsupported {version=}")

        head = [
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, img_dim, 3, padding=1),
        ]
        self.head = nn.ModuleList(head)

    def forward(
        self,
        x: Tensor,
        feat_cache: list[Tensor | None] | None = None,
        feat_idx: list[int] = [0],
        first_chunk: bool = False,
    ) -> Tensor:
        x = self.conv1.cache_forward(x, feat_cache, feat_idx)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if isinstance(layer, UpResidualBlock):
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x, feat_cache, feat_idx)

        for layer in self.head:
            if isinstance(layer, CausalConv3d):
                x = layer.cache_forward(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        return x


def count_conv3d(model: nn.Module):
    return sum(1 if isinstance(m, CausalConv3d) else 0 for m in model.modules())


def patchify(x: Tensor, patch_size: int) -> Tensor:
    if patch_size == 1:
        return x

    if x.ndim == 4:
        B, _, H, W = x.shape
        new_H = H // patch_size
        new_W = W // patch_size
        x = x.reshape(B, -1, new_H, patch_size, new_W, patch_size)
        x = x.permute(0, 1, 5, 3, 2, 4)
        x = x.reshape(B, -1, new_H, new_W)
    elif x.ndim == 5:
        B, _, T, H, W = x.shape
        new_H = H // patch_size
        new_W = W // patch_size
        x = x.reshape(B, -1, T, new_H, patch_size, new_W, patch_size)
        x = x.permute(0, 1, 6, 4, 2, 3, 5)
        x = x.reshape(B, -1, T, new_H, new_W)
    else:
        raise ValueError(f"Unsupported {x.ndim=}")

    return x


def unpatchify(x: Tensor, patch_size: int) -> Tensor:
    if patch_size == 1:
        return x

    if x.ndim == 4:
        B, _, H, W = x.shape
        x = x.unflatten(1, (-1, patch_size, patch_size))
        x = x.permute(0, 1, 4, 3, 5, 2)
        x = x.reshape(B, -1, H * patch_size, W * patch_size)
    elif x.ndim == 5:
        B, _, T, H, W = x.shape
        x = x.unflatten(1, (-1, patch_size, patch_size))
        x = x.permute(0, 1, 4, 5, 3, 6, 2)
        x = x.reshape(B, -1, T, H * patch_size, W * patch_size)
    else:
        raise ValueError(f"Unsupported {x.ndim=}")

    return x


class WanVAE(nn.Module):
    def __init__(
        self,
        encode_dim: int = 96,
        decode_dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temporal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        patch_size: int = 1,
        version: str = "2.1",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        img_dim = 3 * patch_size * patch_size

        self.encoder = Encoder3d(
            img_dim, encode_dim, z_dim * 2, dim_mult, num_res_blocks, temporal_downsample, dropout, version
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            img_dim, decode_dim, z_dim, dim_mult, num_res_blocks, temporal_downsample[::-1], dropout, version
        )
        self.register_buffer("mean", torch.zeros(z_dim), persistent=False)
        self.register_buffer("scale", torch.ones(z_dim), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x: Tensor) -> Tensor:
        self.clear_cache()
        x = patchify(x, self.patch_size)

        # first frame is processed separately. no downsampling.
        self._enc_conv_idx = [0]
        out = self.encoder(x[:, :, :1], self._enc_feat_map, self._enc_conv_idx)

        # process 4 frames at a time, downsample to 1 frame
        for i in range((x.shape[2] - 1) // 4):
            self._enc_conv_idx = [0]
            out_ = self.encoder(x[:, :, 1 + 4 * i : 1 + 4 * (i + 1)], self._enc_feat_map, self._enc_conv_idx)
            out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        mu = (mu - self.mean[:, None, None, None]) * self.scale[:, None, None, None]

        self.clear_cache()
        return mu

    def decode(self, z: Tensor) -> Tensor:
        self.clear_cache()

        z = z / self.scale[:, None, None, None] + self.mean[:, None, None, None]
        x = self.conv2(z)

        # process 1st frame, no upsampling
        self._conv_idx = [0]
        out = self.decoder(x[:, :, :1, :, :], self._feat_map, self._conv_idx, first_chunk=True)

        # process other frames, each is upsampled to 4 frames
        for i in range(1, z.shape[2]):
            self._conv_idx = [0]
            out_ = self.decoder(x[:, :, i : i + 1, :, :], self._feat_map, self._conv_idx)
            out = torch.cat([out, out_], 2)

        out = unpatchify(out, self.patch_size)
        self.clear_cache()
        return out

    def reparameterize(self, mu: Tensor, log_var: Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs: Tensor, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def load_wan_vae(version: str = "2.1") -> WanVAE:
    if version == "2.1":
        cfg = dict(encode_dim=96, decode_dim=96, z_dim=16, patch_size=1)
        repo_id = "Wan-AI/Wan2.1-T2V-14B"
        filename = "Wan2.1_VAE.pth"

        # fmt: off
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        # fmt: on

    elif version == "2.2":
        cfg = dict(encode_dim=160, decode_dim=256, z_dim=48, patch_size=2)
        repo_id = "Wan-AI/Wan2.2-TI2V-5B"
        filename = "Wan2.2_VAE.pth"

        # fmt: off
        mean = [
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
            -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
            -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
            0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
        ]
        std = [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ]
        # fmt: on

    else:
        raise ValueError(f"Unsupported {version=}")

    mean = torch.tensor(mean, dtype=torch.float)
    scale = 1.0 / torch.tensor(std, dtype=torch.float)

    with torch.device("meta"):
        model = WanVAE(**cfg, version=version)
    model.register_buffer("mean", mean, persistent=False)
    model.register_buffer("scale", scale, persistent=False)

    state_dict = load_hf_state_dict(repo_id, filename)
    model.load_state_dict(state_dict, assign=True)
    model.eval().requires_grad_(False)

    return model
