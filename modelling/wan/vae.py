# https://github.com/Wan-Video/Wan2.2/blob/388807310646ed5f318a99f8e8d9ad28c5b65373/wan/modules/vae2_1.py
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

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


# TODO: set axis
class RMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: int = True) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))

    def forward(self, x: Tensor) -> Tensor:
        out = F.normalize(x.float(), dim=(1 if self.channel_first else -1))
        out = out * (self.gamma * self.scale)
        return out.to(x.dtype)


# Fix bfloat16 support for nearest neighbor interpolation.
class Upsample(nn.Upsample):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: Tensor, feat_cache=None, feat_idx=[0]) -> Tensor:
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
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
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
                # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                #     # cache last frame of last two chunk
                #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

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

    def forward(self, x: Tensor, feat_cache=None, feat_idx=[0]) -> Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
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


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temperal_downsample=[False, True, True],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        self.downsamples = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                self.downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x: Tensor, feat_cache=None, feat_idx=[0]) -> Tensor:
        if feat_cache is not None:
            # TODO: make a utility function for feat_cache logic?
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temperal_upsample=[True, True, False],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        self.upsamples = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                self.upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                self.upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0

        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x: Tensor, feat_cache=None, feat_idx=[0]) -> Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


def count_conv3d(model: nn.Module):
    return sum(1 if isinstance(m, CausalConv3d) else 0 for m in model.modules())


class WanVAE(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks: int = 2,
        temperal_downsample=[False, True, True],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, temperal_downsample[::-1], dropout)

    def forward(self, x: Tensor) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x: Tensor, scale: list[Tensor]) -> Tensor:
        self.clear_cache()
        T = x.shape[2]

        # first frame is processed separately. no downsampling.
        self._enc_conv_idx = [0]
        out = self.encoder(x[:, :, :1], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)

        # process 4 frames at a time, downsample to 1 frame
        for i in range((T - 1) // 4):
            self._enc_conv_idx = [0]
            out_ = self.encoder(
                x[:, :, 1 + 4 * i : 1 + 4 * (i + 1)],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        mu = (mu - scale[0][:, None, None, None]) * scale[1][:, None, None, None]

        self.clear_cache()
        return mu

    def decode(self, z: Tensor, scale: list[Tensor]) -> Tensor:
        self.clear_cache()

        z = z / scale[1][:, None, None, None] + scale[0][:, None, None, None]
        x = self.conv2(z)

        # process 1st frame, no upsampling
        self._conv_idx = [0]
        out = self.decoder(x[:, :, :1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)

        # process other frames, each is upsampled to 4 frames
        for i in range(1, z.shape[2]):
            self._conv_idx = [0]
            out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)

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


# TODO: merge this to WanVAE
class Wan2_1_VAE:
    def __init__(self, device="cpu"):
        self.device = device

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
        self.mean = torch.tensor(mean, dtype=torch.float, device=device)
        self.std = torch.tensor(std, dtype=torch.float, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        with torch.device("meta"):
            model = WanVAE()

        state_dict = load_hf_state_dict("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth")
        model.load_state_dict(state_dict, assign=True)
        model.eval().requires_grad_(False).to(device)

        self.model = model

    def encode(self, videos: list[Tensor]):
        # each video is [C, T, H, W]
        return [self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0) for u in videos]

    def decode(self, zs: list[Tensor]):
        return [self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0) for u in zs]
