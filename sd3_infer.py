import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from modelling import SD3, load_clip_l, load_openclip_bigg, load_sd3_5, load_sd3_autoencoder, load_t5
from offload import PerLayerOffloadCUDAStream

logger = logging.getLogger(__name__)


class SD3TextEmbedder:
    def __init__(self, offload_t5: bool = False, offload_clip_g: bool = False, dtype: torch.dtype = torch.bfloat16):
        self.t5 = load_t5(max_length=256).to(dtype)  # 9.5 GB in BF16
        self.clip_l = load_clip_l(output_key=["pooler_output", -2]).to(dtype)  # 246 MB in BF16
        self.clip_g = load_openclip_bigg().to(dtype)  # 1.4 GB in BF16
        self.t5_offloader = PerLayerOffloadCUDAStream(self.t5, enable=offload_t5)
        self.clip_g_offloader = PerLayerOffloadCUDAStream(self.clip_g, enable=offload_clip_g)

    def cpu(self):
        for m in (self.clip_l, self.t5_offloader, self.clip_g_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.clip_l, self.t5_offloader, self.clip_g_offloader):
            m.cuda()
        return self

    def __call__(self, prompt: list[str]):
        t5_embeds = self.t5(prompt)  # (B, 256, 4096)
        clip_l_vecs, clip_l_embeds = self.clip_l(prompt)  # (B, 768) and (B, 77, 768)
        clip_g_vecs, clip_g_embeds = self.clip_g(prompt)  # (B, 1280) and (B, 77, 1280)

        clip_vecs = torch.cat([clip_l_vecs, clip_g_vecs], dim=-1)  # (B, 2048)
        clip_embeds = torch.cat([clip_l_embeds, clip_g_embeds], dim=-1)  # (B, 77, 2048)
        clip_embeds = F.pad(clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1]))  # (B, 77, 4096)
        embeds = torch.cat([clip_embeds, t5_embeds], dim=1)  # (B, 77+256, 4096)
        return embeds, clip_vecs


# https://github.com/Stability-AI/sd3.5/blob/fbf8f483f992d8d6ad4eaaeb23b1dc5f523c3b3a/sd3_infer.py#L509-L520
class SkipLayerConfig(NamedTuple):
    scale: float = 0.0
    start: float = 0.01
    end: float = 0.20
    layers: tuple[int, ...] = (7, 8, 9)


class SD3Generator:
    def __init__(
        self,
        sd3: SD3 | None = None,
        offload_sd3: bool = False,
        offload_t5: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.sd3 = (sd3 or load_sd3_5()).to(dtype)  # 4.5 GB for medium in BF16
        self.ae = load_sd3_autoencoder().to(dtype)  # 168 MB in BF16
        self.text_embedder = SD3TextEmbedder(offload_t5, dtype=dtype)

        # autoencoder and clip are small, don't need to offload
        # NOTE: offload SD3 is not compatible with skip layer
        self.sd3_offloader = PerLayerOffloadCUDAStream(self.sd3, enable=offload_sd3)

    def cpu(self):
        for m in (self.ae, self.text_embedder, self.sd3_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.ae, self.text_embedder, self.sd3_offloader):
            m.cuda()
        return self

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        img_size: int | tuple[int, int] = 512,
        latents: Tensor | None = None,
        denoise: float = 1.0,
        num_steps: int = 50,
        cfg_scale: int = 5.0,
        slg_config: SkipLayerConfig = SkipLayerConfig(),
        seed: int | None = None,
        pbar: bool = False,
        compile: bool = False,
        solver: str = "dpm++2m",
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        bsize = len(prompt)

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * bsize

        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size

        embeds, clip_vecs = self.text_embedder(prompt)
        if cfg_scale != 1.0:
            neg_embeds, neg_clip_vecs = self.text_embedder(negative_prompt)
        else:
            neg_embeds = neg_clip_vecs = None

        rng = torch.Generator("cuda")
        rng.manual_seed(seed) if seed is not None else logger.info(f"Using seed={rng.seed()}")
        dtype = self.sd3.context_embedder.weight.dtype
        noise = torch.randn(bsize, 16, height // 8, width // 8, device="cuda", dtype=dtype, generator=rng)
        # this is basically SDEdit https://arxiv.org/abs/2108.01073
        latents = latents.lerp(noise, denoise) if latents is not None else noise

        # denoise from t=1 (noise) to t=0 (latents)
        timesteps = sd3_timesteps(denoise, 0.0, num_steps)

        if not self.sd3_offloader.enable:
            self.sd3.cuda()  # model offload
        solver_fn = {
            "euler": sd3_euler_generate,
            "dpm++2m": sd3_dpmpp2m_generate,
        }[solver]
        latents = solver_fn(
            self.sd3,
            latents,
            timesteps,
            context=embeds,
            vec=clip_vecs,
            neg_context=neg_embeds,
            neg_vec=neg_clip_vecs,
            cfg_scale=cfg_scale,
            slg_config=slg_config,
            pbar=pbar,
            compile=compile,
        )
        if not self.sd3_offloader.enable:
            self.sd3.cpu()
        return self.ae.decode(latents, uint8=True)


def sd3_timesteps(start: float = 1.0, end: float = 0.0, num_steps: int = 50):
    timesteps = torch.linspace(start, end, num_steps + 1)
    timesteps = 3.0 / (3.0 + 1 / timesteps - 1)  # static shift
    return timesteps.tolist()


def sd3_forward(
    sd3: SD3,
    latents: Tensor,
    t: Tensor,
    context: Tensor,
    vec: Tensor,
    neg_context: Tensor | None,
    neg_vec: Tensor | None,
    cfg_scale: float,
    slg_scale: float = 0.0,
    skip_layers: tuple[int, ...] = (),
):
    # t must be Tensor (cpu is fine) to avoid recompilation
    t_vec = t.view(1).cuda()
    if cfg_scale != 1.0 and neg_context is not None and neg_vec is not None:
        pos_v, neg_v = sd3(
            latents.repeat(2, 1, 1, 1),
            t_vec,
            torch.cat([context, neg_context], dim=0),
            torch.cat([vec, neg_vec], dim=0),
        ).chunk(2, dim=0)
        v = neg_v.lerp(pos_v, cfg_scale)  # classifier-free guidance

    else:
        pos_v = sd3(latents, t_vec, context, vec)
        v = pos_v

    # https://github.com/Stability-AI/sd3.5/blob/fbf8f483f992d8d6ad4eaaeb23b1dc5f523c3b3a/sd3_impls.py#L219
    if slg_scale > 0.0:
        skip_layer_v = sd3(latents, t_vec, context, vec, skip_layers)
        v.add_(pos_v.float() - skip_layer_v.float(), alpha=slg_scale)

    return v


@torch.no_grad()
def sd3_euler_generate(
    sd3: SD3,
    latents: Tensor,
    timesteps: list[float],
    context: Tensor,
    vec: Tensor,
    neg_context: Tensor | None = None,
    neg_vec: Tensor | None = None,
    cfg_scale: float = 5.0,
    slg_config: SkipLayerConfig = SkipLayerConfig(),
    pbar: bool = False,
    compile: bool = False,
):
    num_steps = len(timesteps) - 1
    forward = torch.compile(sd3_forward, disable=not compile)
    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = torch.tensor(timesteps[i])
        slg_scale = slg_config.scale if slg_config.start < i / num_steps < slg_config.end else 0.0
        v = forward(sd3, latents, t, context, vec, neg_context, neg_vec, cfg_scale, slg_scale, slg_config.layers)
        latents.add_(v, alpha=timesteps[i + 1] - timesteps[i])  # Euler's method. move from t_curr to t_next
    return latents


@torch.no_grad()
def sd3_dpmpp2m_generate(
    sd3: SD3,
    latents: Tensor,
    timesteps: list[float],
    context: Tensor,
    vec: Tensor,
    neg_context: Tensor | None = None,
    neg_vec: Tensor | None = None,
    cfg_scale: float = 5.0,
    slg_config: SkipLayerConfig = SkipLayerConfig(),
    pbar: bool = False,
    compile: bool = False,
):
    # DPM-Solver++(2M) https://arxiv.org/abs/2211.01095
    # the implementation below has been simplified for flow matching / rectified flow
    # with sigma(t) = t and alpha(t) = 1-t
    # coincidentally (or not?), this results in identical calculations as k-diffusion implementation
    # https://github.com/crowsonkb/k-diffusion/blob/21d12c91ad4550e8fcf3308ff9fe7116b3f19a08/k_diffusion/sampling.py#L585-L607

    num_steps = len(timesteps) - 1
    forward = torch.compile(sd3_forward, disable=not compile)
    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = torch.tensor(timesteps[i])
        slg_scale = slg_config.scale if slg_config.start < i / num_steps < slg_config.end else 0.0
        v = forward(sd3, latents, t, context, vec, neg_context, neg_vec, cfg_scale, slg_scale, slg_config.layers)
        data_pred = latents.add(v, alpha=-timesteps[i])  # data prediction model

        if i == 0:
            latents = data_pred.lerp(latents, timesteps[i + 1] / timesteps[i])
        elif timesteps[i + 1] == 0.0:  # avoid log(0). note that lim x.log(x) when x->0 is 0.
            latents = data_pred
        else:
            lambda_prev = -math.log(timesteps[i - 1])
            lambda_curr = -math.log(timesteps[i])
            lambda_next = -math.log(timesteps[i + 1])
            r = (lambda_curr - lambda_prev) / (lambda_next - lambda_curr)
            D = data_pred.lerp(prev_data_pred, -1 / (2 * r))
            latents = D.lerp(latents, timesteps[i + 1] / timesteps[i])

        prev_data_pred = data_pred

    return latents
