import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from modelling import AutoEncoder, Flux, TextEmbedder, load_clip_text, load_flux, load_flux_autoencoder, load_t5
from offload import PerLayerOffloadCUDAStream

logger = logging.getLogger(__name__)


class FluxGenerator:
    def __init__(
        self,
        ae: AutoEncoder | None = None,
        flux: Flux | None = None,
        t5: TextEmbedder | None = None,
        clip: TextEmbedder | None = None,
        offload_t5: bool = False,
        offload_flux: bool = False,
    ) -> None:
        self.ae = ae or load_flux_autoencoder()  # 168 MB in BF16
        self.flux = flux or load_flux()  # 23.8 GB in BF16

        # TODO: investigate max_length. currently T5 and CLIP have different max_length
        # -> CLIP will truncate long prompts.
        self.t5 = t5 or load_t5()  #  9.5 GB in BF16
        self.clip = clip or load_clip_text()  #  246 MB in BF16

        # autoencoder and clip are small, don't need to offload
        self.t5_offloader = PerLayerOffloadCUDAStream(self.t5, enable=offload_t5)
        self.flux_offloader = PerLayerOffloadCUDAStream(self.flux, enable=offload_flux)

    def cpu(self):
        self.ae.cpu()
        self.clip.cpu()
        self.t5_offloader.cpu()  # this only move some params
        self.flux_offloader.cpu()
        return self

    def cuda(self):
        self.ae.cuda()
        self.clip.cuda()
        self.t5_offloader.cuda()  # this only move some params
        self.flux_offloader.cuda()
        return self

    @torch.no_grad()
    def generate(
        self,
        t5_prompt: str | list[str],
        clip_prompt: str | list[str] | None = None,
        img_size: int | tuple[int, int] = 512,
        latents: Tensor | None = None,
        denoise: float = 1.0,
        guidance: float = 3.5,
        num_steps: int = 50,
        seed: int | None = None,
        pbar: bool = False,
        compile: bool = False,
    ):
        if isinstance(t5_prompt, str):
            t5_prompt = [t5_prompt]
        bsize = len(t5_prompt)

        if clip_prompt is None:
            clip_prompt = t5_prompt
        elif isinstance(clip_prompt, str):
            clip_prompt = [clip_prompt] * bsize
        assert len(clip_prompt) == bsize

        return flux_generate(
            self.flux,
            self.ae,
            self.t5(t5_prompt).cuda(),  # (B, 512, 4096)
            self.clip(clip_prompt).cuda(),  # (B, 768)
            (img_size, img_size) if isinstance(img_size, int) else img_size,
            latents,
            denoise,
            guidance,
            num_steps,
            seed,
            pbar,
            compile,
        )


@torch.no_grad()
def flux_generate(
    flux: Flux,
    ae: AutoEncoder | None,
    txt: Tensor,
    vec: Tensor,
    img_size: tuple[int, int],
    latents: Tensor | None = None,
    denoise: float = 1.0,
    guidance: Tensor | float = 3.5,
    num_steps: int = 50,
    seed: int | None = None,
    pbar: bool = False,
    compile: bool = False,
):
    height, width = img_size
    bsize = txt.shape[0]

    # TODO: support for initial image
    # NOTE: Flux uses patchify and unpatchify on model's input and output.
    # this is equivalent to pixel unshuffle and pixel shuffle respectively.
    latent_h = height // 16
    latent_w = width // 16
    img_ids = flux_img_ids(bsize, latent_h, latent_w).cuda()  # this inject size info
    txt_ids = torch.zeros(bsize, txt.shape[1], 3, device="cuda")  # TODO: check this

    # only dev version has time shift
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = flux_time_shift(timesteps, latent_h, latent_w)
    if not isinstance(guidance, Tensor):
        guidance = torch.full((bsize,), guidance, device="cuda", dtype=torch.bfloat16)

    rng = torch.Generator("cuda")
    rng.manual_seed(seed) if seed is not None else logger.info(f"Using seed={rng.seed()}")
    noise = torch.randn(bsize, latent_h * latent_w, 64, device="cuda", dtype=torch.bfloat16, generator=rng)

    if latents is not None:
        # timesteps is ordered from 1->0. thus, we need to use (1 - noise_strength)
        timesteps = timesteps[int(num_steps * (1 - denoise)) :]
        latents = latents.lerp(noise, timesteps[0].item())
    else:
        latents = noise

    for i in tqdm(range(timesteps.shape[0] - 1), disable=not pbar, dynamic_ncols=True):
        # t_curr and t_prev must be Tensor (cpu is fine) to avoid recompilation
        t_curr = timesteps[i]
        t_prev = timesteps[i + 1]
        torch.compile(flux_denoise_euler_step, disable=not compile)(
            flux, latents, img_ids, txt, txt_ids, vec, t_curr, t_prev, guidance
        )

    if ae is not None:
        return torch.compile(flux_decode, disable=not compile)(ae, latents, (latent_h, latent_w))
    else:
        return latents


def flux_img_ids(bsize: int, latent_h: int, latent_w: int):
    img_ids = torch.zeros(latent_h, latent_w, 3)
    img_ids[..., 1] = torch.arange(latent_h)[:, None]
    img_ids[..., 2] = torch.arange(latent_w)[None, :]
    return img_ids.view(1, latent_h * latent_w, 3).expand(bsize, -1, -1)


# https://arxiv.org/abs/2403.03206
# Section 5.3.2 - Resolution-dependent shifting of timesteps schedules
def flux_time_shift(timesteps: Tensor, latent_h: int, latent_w: int, base_shift: float = 0.5, max_shift: float = 1.15):
    m = (max_shift - base_shift) / (4096 - 256)
    b = base_shift - m * 256
    mu = m * (latent_h * latent_w) + b
    exp_mu = math.exp(mu)  # this is (m/n) in Equation (23)
    return exp_mu / (exp_mu + timesteps.reciprocal() - 1)


def flux_denoise_euler_step(
    flux: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    t_curr: Tensor,
    t_prev: Tensor,
    guidance: Tensor,
) -> None:
    # NOTE: (latents) 0 <= t_prev < t_curr <= 1 (noise)
    # NOTE: t_curr and t_prev are FP32
    t_vec = t_curr.to(img.dtype).view(1).cuda()
    v = flux(img, img_ids, txt, txt_ids, t_vec, vec, guidance)
    img += (t_prev - t_curr) * v  # Euler's method. move to t=0 direction (latents)


@torch.no_grad()
def flux_encode(ae: AutoEncoder, imgs: Tensor, sample: bool) -> Tensor:
    if imgs.dtype == torch.uint8:
        imgs = imgs.float() / 127.5 - 1
    latents = ae.encode(imgs.bfloat16(), sample=sample)
    return F.pixel_unshuffle(latents, 2).view(imgs.shape[0], 64, -1).transpose(1, 2)


@torch.no_grad()
def flux_decode(ae: AutoEncoder, latents: Tensor, latent_size: tuple[int, int]) -> Tensor:
    # NOTE: original repo uses FP32 AE + cast latents to FP32 + BF16 autocast
    # (B, 64, latent_h, latent_w) -> (B, 16, latent_h * 2, latent_w * 2)
    latents = F.pixel_shuffle(latents.transpose(1, 2).unflatten(-1, latent_size), 2)
    imgs = ae.decode(latents)
    return imgs.float().add(1).mul(127.5).clip(0, 255).to(torch.uint8)


def flux_resize_latents(latents: Tensor, latent_size: tuple[int, int], scale_factor: float, mode: str = "nearest"):
    latents = F.pixel_shuffle(latents.transpose(1, 2).unflatten(-1, latent_size), 2)
    latents = F.interpolate(latents, scale_factor=scale_factor, mode=mode)
    latents = F.pixel_unshuffle(latents, 2).view(latents.shape[0], 64, -1).transpose(1, 2)
    return latents
