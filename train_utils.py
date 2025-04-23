from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import v2

from flux_infer import FluxTextEmbedder
from modelling import (
    SD3,
    Flux,
    LoRALinear,
    load_flex_alpha,
    load_flux,
    load_flux_autoencoder,
    load_sd3_5,
    load_sd3_autoencoder,
)
from offload import PerLayerOffloadWithBackward
from sd3_infer import SD3TextEmbedder
from time_sampler import TimeSampler


def setup_model(model_name: str, offload: bool, lora: int, use_compile: bool, int8_training: bool):
    if model_name in ("flux-dev", "flex-alpha"):
        model = {
            "flux-dev": load_flux,
            "flex-alpha": load_flex_alpha,
        }[model_name]()
        layers = list(model.double_blocks) + list(model.single_blocks)

        ae = load_flux_autoencoder()
        text_embedder = FluxTextEmbedder(offload_t5=True)

    elif model_name in ("sd3.5-medium", "sd3.5-large"):
        model = load_sd3_5(model_name.removeprefix("sd3.5-"))
        layers = list(model.joint_blocks)

        ae = load_sd3_autoencoder()
        text_embedder = SD3TextEmbedder(offload_t5=True)

    else:
        raise ValueError(f"Unsupported {model_name=}")

    model.bfloat16().train().requires_grad_(False)
    offloader = PerLayerOffloadWithBackward(model, enable=offload).cuda()
    ae.eval().cuda()
    text_embedder.cuda()

    for layer in layers:
        if lora > 0 or int8_training:
            LoRALinear.add_lora(layer, rank=lora, quantization="int8_training", device="cuda")
        layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        if use_compile:  # might not be optimal to compile this way, but required for offloading
            layer.forward = torch.compile(layer.forward)

    return model, offloader, ae, text_embedder


def compute_loss(
    model: Flux | SD3,
    latents: Tensor,
    embeds: Tensor,
    vecs: Tensor,
    time_sampler: TimeSampler,
    model_kwargs: dict,
) -> Tensor:
    bsize = latents.shape[0]
    t_vec = time_sampler(bsize, device=latents.device).bfloat16()
    noise = torch.randn_like(latents)
    interpolate = latents.lerp(noise, t_vec.view(-1, 1, 1, 1))

    v = model(interpolate, t_vec, embeds, vecs, **model_kwargs)

    # rectified flow loss. predict velocity from latents (t=0) to noise (t=1).
    return F.mse_loss(noise.float() - latents.float(), v.float())


def parse_img_size(img_size: str):
    out = [int(x) for x in img_size.split(",")]
    if len(out) == 1:
        out = [out[0], out[0]]
    assert len(out) == 2
    assert out[0] % 16 == 0 and out[1] % 16 == 0, out
    return tuple(out)


def random_resize(img_pil: Image.Image, min_size: int, max_size: int):
    # randomly resize while maintaining aspect ratio.
    long_edge = max(img_pil.size)
    assert long_edge >= min_size
    max_size = min(max_size, long_edge)

    target_long_edge = torch.randint(min_size, max_size + 1, size=()).item()
    scale = target_long_edge / long_edge
    target_height = round(img_pil.height * scale)
    target_width = round(img_pil.width * scale)
    img_pil = img_pil.resize((target_width, target_height), Image.Resampling.BICUBIC)

    # slightly crop the image so that each side is divisible by 16.
    # to reduce fragmentation, make it in 32 increment.
    height = target_height // 32 * 32
    width = target_width // 32 * 32
    img_pt = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)
    img_pt = v2.RandomCrop((height, width))(img_pt)
    return img_pt


class EMA:
    def __init__(self, model: nn.Module, beta: float = 0.999, num_warmup: int = 500):
        self.model = model
        self.ema_params = {
            name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad
        }
        self.beta = beta
        self.num_warmup = num_warmup

    @torch.no_grad()
    def update(self, step: int):
        if step < self.num_warmup:
            return

        ema_params, online_params = [], []
        for name, param in self.model.named_parameters():
            if name in self.ema_params:
                ema_params.append(self.ema_params[name])
                online_params.append(param)

        if step == self.num_warmup:
            torch._foreach_copy_(ema_params, online_params)
        else:
            torch._foreach_lerp_(ema_params, online_params, 1 - self.beta)

    def swap_params(self):
        for name, param in self.model.named_parameters():
            if name in self.ema_params:
                ema_p = self.ema_params[name]
                param.data, ema_p.data = ema_p.data, param.data

    def state_dict(self):
        return dict(self.ema_params)  # shallow copy

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        state_dict = dict(state_dict)  # shallow copy
        for name, param in self.ema_params.items():
            param.copy_(state_dict.pop(name))
        assert len(state_dict) == 0
