# https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_z_image.py

import dataclasses

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attn import dispatch_attn
from .flux import timestep_embedding
from .utils import load_hf_state_dict, make_merge_hook


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(256, hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        scale = self.adaLN_modulation(c).unsqueeze(1)
        x = (1.0 + scale) * F.layer_norm(x, x.shape[-1:], eps=1e-6)
        return self.linear(x)


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (1536, 512, 512),
    ) -> None:
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = self.precompute_freqs_cis(axes_dims, axes_lens, theta)

    @staticmethod
    def precompute_freqs_cis(dims: tuple[int, ...], lengths: tuple[int, ...], theta: float = 256.0):
        freqs_cis = []
        for dim, length in zip(dims, lengths):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim))
            timestep = torch.arange(length, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
            freqs_cis.append(freqs_cis_i)

        return freqs_cis

    def __call__(self, ids: Tensor) -> Tensor:
        assert ids.ndim == 2, ids.ndim
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        # Ensure freqs_cis are on the same device as ids
        if self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


# TODO: unify with FLUX
def apply_rope(x: Tensor, freqs_cis: Tensor):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(2)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.head_dim = 128
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False))
        self.register_load_state_dict_pre_hook(make_merge_hook(["to_q", "to_k", "to_v"], "qkv"))

    def forward(self, x: Tensor, pe: Tensor):
        q, k, v = self.qkv(x).unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        q = apply_rope(self.norm_q(q), pe)
        k = apply_rope(self.norm_k(k), pe)
        out = dispatch_attn(q, k, v).flatten(2)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w13 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.register_load_state_dict_pre_hook(make_merge_hook(["w1", "w3"], "w13"))

    def forward(self, x: Tensor) -> Tensor:
        w1, w3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(w1) * w3)


class Block(nn.Module):
    def __init__(self, dim: int, mod_dim: int, mlp_ratio: float, eps: float = 1e-5) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.Linear(mod_dim, 4 * dim)) if mod_dim > 0 else None
        self.attention = Attention(dim)
        self.feed_forward = FeedForward(dim, int(dim * mlp_ratio))
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)

    def forward(self, x: Tensor, adaln_input: Tensor | None, pe: Tensor) -> Tensor:
        if self.adaLN_modulation is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)

            res = self.attention(self.attention_norm1(x) * (1.0 + scale_msa), pe)
            x = x + self.attention_norm2(res) * gate_msa.tanh()

            res = self.feed_forward(self.ffn_norm1(x) * (1.0 + scale_mlp))
            x = x + self.ffn_norm2(res) * gate_mlp.tanh()
        else:
            x = x + self.attention_norm2(self.attention(self.attention_norm1(x), pe))
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


@dataclasses.dataclass
class ZImageConfig:
    in_channels: int = 16
    txt_dim: int = 2560
    dim: int = 3840
    n_refiner_layers: int = 2
    n_layers: int = 30
    patch_size: int = 2
    mod_dim: int = 256
    mlp_ratio: float = 8 / 3
    rope_dims: tuple[int, int, int] = (32, 48, 48)


class ZImage(nn.Module):
    def __init__(self, cfg: ZImageConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ZImageConfig()
        self.cfg = cfg

        self.t_embedder = nn.Sequential()
        self.t_embedder.mlp = nn.Sequential(nn.Linear(256, 1024), nn.SiLU(), nn.Linear(1024, cfg.mod_dim))
        self.rope = RopeEmbedder(axes_dims=cfg.rope_dims)
        self.x_pad_token = nn.Parameter(torch.zeros(1, cfg.dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, cfg.dim))

        # image-only processing
        patchified_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels
        self.all_x_embedder = nn.ModuleDict()
        self.all_x_embedder["2-1"] = nn.Linear(patchified_dim, cfg.dim)
        self.noise_refiner = nn.ModuleList(
            [Block(cfg.dim, cfg.mod_dim, cfg.mlp_ratio) for _ in range(cfg.n_refiner_layers)]
        )

        # text-only processing
        self.cap_embedder = nn.Sequential(nn.RMSNorm(cfg.txt_dim, eps=1e-5), nn.Linear(cfg.txt_dim, cfg.dim))
        self.context_refiner = nn.ModuleList([Block(cfg.dim, 0, cfg.mlp_ratio) for _ in range(cfg.n_refiner_layers)])

        # joint processing
        self.layers = nn.ModuleList([Block(cfg.dim, cfg.mod_dim, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.all_final_layer = nn.ModuleDict()
        self.all_final_layer["2-1"] = FinalLayer(cfg.dim, patchified_dim)

    @staticmethod
    def create_coordinate_grid(start_list: tuple[int, ...], size_list: tuple[int, ...], device=None) -> Tensor:
        if start_list is None:
            start_list = (0 for _ in size_list)

        axes = [
            torch.arange(start, start + size, dtype=torch.int32, device=device)
            for start, size in zip(start_list, size_list)
        ]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def forward(self, img: Tensor, timesteps: Tensor, txt: Tensor) -> Tensor:
        B, C, H, W = img.shape
        B, L, D = txt.shape

        t_embeds = self.t_embedder(timestep_embedding(timesteps, 256).to(img.dtype))

        # there are 3 components in a RoPE embedding
        # - text/frequency component: text embeds stay as pos=[1,L+1), img embeds stay at pos=L+1
        # - height component: all text embeds stay at pos=0
        # - width component: all text embeds stay at pos=0

        # patchify
        patch_size = self.cfg.patch_size
        nH = H // patch_size
        nW = W // patch_size
        img = img.view(B, C, nH, patch_size, nW, patch_size)
        img = img.permute(0, 2, 4, 3, 5, 1)  # (B, nH, nW, 2, 2, C)
        img = img.reshape(B, nH * nW, patch_size * patch_size * C)

        # image-only processing
        img = self.all_x_embedder["2-1"](img)
        img_pos_ids = self.create_coordinate_grid((L + 1, 0, 0), (1, nH, nW), device=img.device)
        img_rope = self.rope(img_pos_ids.flatten(0, 2)).unsqueeze(0)
        for layer in self.noise_refiner:
            img = layer(img, t_embeds, img_rope)

        # text-only processing
        txt = self.cap_embedder(txt)
        txt_pos_ids = self.create_coordinate_grid((1, 0, 0), (L, 1, 1), device=txt.device)
        txt_rope = self.rope(txt_pos_ids.flatten(0, 2)).unsqueeze(0)
        for layer in self.context_refiner:
            txt = layer(txt, None, txt_rope)

        # joint processing
        unified = torch.cat([img, txt], dim=1)
        unified_rope = torch.cat([img_rope, txt_rope], dim=1)
        for layer in self.layers:
            unified = layer(unified, t_embeds, unified_rope)

        # unpatchify
        unified = self.all_final_layer["2-1"](unified, t_embeds)
        img = unified[:, : nH * nW].view(B, nH, nW, patch_size, patch_size, C)
        img = img.permute(0, 5, 1, 3, 2, 4)  # (B, C, nH, 2, nW, 2)
        img = img.reshape(B, C, H, W)
        return img


def load_zimage():
    with torch.device("meta"):
        model = ZImage()

    state_dict = load_hf_state_dict(
        "Tongyi-MAI/Z-Image-Turbo",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    )
    model.load_state_dict(state_dict, assign=True)
    return model
