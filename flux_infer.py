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

        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size

        # TODO: support for initial image
        # NOTE: Flux uses pixel unshuffle on latent image before passing it to Flux model,
        # and pixel shuffle on Flux outputs before passing it to AE's decoder.
        # prepare inputs and text conditioning
        latent_h = height // 16
        latent_w = width // 16
        rng = torch.Generator("cuda")
        rng.manual_seed(seed) if seed is not None else logger.info(f"Using seed={rng.seed()}")
        img = torch.randn(bsize, latent_h * latent_w, 64, device="cuda", dtype=torch.bfloat16, generator=rng)
        img_ids = flux_img_ids(bsize, latent_h, latent_w).cuda()

        txt = self.t5(t5_prompt).to(img.device)  # (B, 512, 4096)
        txt_ids = torch.zeros(bsize, txt.shape[1], 3, device=img.device)
        vec = self.clip(t5_prompt)  # (B, 768)

        # only dev version has time shift
        timesteps = torch.linspace(1, 0, num_steps + 1)
        timesteps = flux_time_shift(timesteps, latent_h, latent_w)
        guidance_vec = torch.full((bsize,), guidance, device=img.device, dtype=img.dtype)

        for i in tqdm(range(num_steps), disable=not pbar):
            # t_curr and t_prev must be Tensor (cpu is fine) to avoid recompilation
            t_curr = timesteps[i]
            t_prev = timesteps[i + 1]
            torch.compile(flux_denoise_step, disable=not compile)(
                self.flux, img, img_ids, txt, txt_ids, vec, t_curr, t_prev, guidance_vec
            )

        img_u8 = torch.compile(flux_decode, disable=not compile)(self.ae, img, (latent_h, latent_w))
        return img_u8


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


def flux_denoise_step(
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

    # Euler's method. move to t=0 direction (latents)
    img += (t_prev - t_curr) * v


def flux_decode(ae: AutoEncoder, img: Tensor, latent_size: tuple[int, int]) -> Tensor:
    # NOTE: original repo uses FP32 AE + cast latents to FP32 + BF16 autocast
    img = img.transpose(1, 2).unflatten(-1, latent_size)
    img = F.pixel_shuffle(img, 2)  # (B, 64, latent_h, latent_w) -> (B, 16, latent_h * 2, latent_w * 2)
    img = ae.decode(img)
    img_u8 = img.float().add(1).mul(127.5).clip(0, 255).to(torch.uint8)
    return img_u8
