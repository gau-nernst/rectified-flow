import logging
import math

import torch
from torch import Tensor
from tqdm import tqdm

from modelling import AutoEncoder, Flux, TextEmbedder, load_clip_l, load_flux, load_flux_autoencoder, load_t5
from offload import PerLayerOffloadCUDAStream

logger = logging.getLogger(__name__)


class FluxGenerator:
    def __init__(
        self,
        ae: AutoEncoder | None = None,
        flux: Flux | None = None,
        t5: TextEmbedder | None = None,
        clip: TextEmbedder | None = None,
        offload_flux: bool = False,
        offload_t5: bool = False,
    ) -> None:
        self.ae = ae or load_flux_autoencoder()  # 168 MB in BF16
        self.flux = flux or load_flux()  # 23.8 GB in BF16
        self.t5 = t5 or load_t5()  #  9.5 GB in BF16
        self.clip = clip or load_clip_l().bfloat16()  # 246 MB in BF16

        # autoencoder and clip are small, don't need to offload
        self.flux_offloader = PerLayerOffloadCUDAStream(self.flux, enable=offload_flux)
        self.t5_offloader = PerLayerOffloadCUDAStream(self.t5, enable=offload_t5)

    def cpu(self):
        for m in (self.ae, self.clip, self.flux_offloader, self.t5_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.ae, self.clip, self.flux_offloader, self.t5_offloader):
            m.cuda()
        return self

    @torch.no_grad()
    def generate(
        self,
        t5_prompt: str | list[str],
        clip_prompt: str | list[str] | None = None,
        img_size: int | tuple[int, int] = 512,
        latents: Tensor | None = None,
        extra_txt_embeds: Tensor | None = None,
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

        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size

        t5_embeds = self.t5(t5_prompt).cuda()  # (B, 512, 4096)
        clip_vecs = self.clip(clip_prompt).cuda()  # (B, 768)

        if extra_txt_embeds is not None:  # e.g. Flux-Redux
            t5_embeds = torch.cat([t5_embeds, extra_txt_embeds], dim=1)

        rng = torch.Generator("cuda")
        rng.manual_seed(seed) if seed is not None else logger.info(f"Using seed={rng.seed()}")
        noise = torch.randn(bsize, 16, height // 8, width // 8, device="cuda", dtype=torch.bfloat16, generator=rng)
        if latents is not None:
            # this is basically SDEdit https://arxiv.org/abs/2108.01073
            latents = latents.lerp(noise, denoise)
        else:
            latents = noise

        latents = flux_generate(
            flux=self.flux,
            txt=t5_embeds,
            vec=clip_vecs,
            latents=latents,
            start_t=denoise,
            guidance=guidance,
            num_steps=num_steps,
            pbar=pbar,
            compile=compile,
        )
        return self.ae.decode(latents, uint8=True)


@torch.no_grad()
def flux_generate(
    flux: Flux,
    txt: Tensor,
    vec: Tensor,
    latents: Tensor,
    start_t: float = 1.0,
    end_t: float = 0.0,
    guidance: Tensor | float = 3.5,
    num_steps: int = 50,
    pbar: bool = False,
    compile: bool = False,
):
    bsize, _, latent_h, latent_w = latents.shape

    # denoise from t=1 (noise) to t=0 (latents)
    # only dev version has time shift
    # divide by 4 since FLUX patchifies input
    timesteps = torch.linspace(start_t, end_t, num_steps + 1)
    timesteps = time_shift(timesteps, latent_h * latent_w // 4)
    if not isinstance(guidance, Tensor):
        guidance = torch.full((bsize,), guidance, device="cuda", dtype=torch.bfloat16)

    for i in tqdm(range(timesteps.shape[0] - 1), disable=not pbar, dynamic_ncols=True):
        torch.compile(flux_step, disable=not compile)(flux, latents, txt, vec, timesteps[i], timesteps[i + 1], guidance)

    return latents


# https://arxiv.org/abs/2403.03206
# Section 5.3.2 - Resolution-dependent shifting of timesteps schedules
# https://github.com/black-forest-labs/flux/blob/805da8571a0b49b6d4043950bd266a65328c243b/src/flux/sampling.py#L222-L238
def time_shift(timesteps: Tensor | float, img_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15):
    m = (max_shift - base_shift) / (4096 - 256)
    b = base_shift - m * 256

    mu = m * img_seq_len + b
    exp_mu = math.exp(mu)  # this is (m/n) in Equation (23)
    return exp_mu / (exp_mu + 1 / timesteps - 1)


def flux_step(
    flux: Flux,
    latents: Tensor,
    txt: Tensor,
    vec: Tensor,
    t_curr: Tensor,
    t_next: Tensor,
    guidance: Tensor,
) -> None:
    # t_curr and t_next must be Tensor (cpu is fine) to avoid recompilation
    t_vec = t_curr.to(latents.dtype).view(1).cuda()
    v = flux(latents, txt, t_vec, vec, guidance)
    latents.add_(v, alpha=t_next - t_curr)  # Euler's method. move from t_curr to t_next
