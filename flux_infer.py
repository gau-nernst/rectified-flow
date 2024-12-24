import logging
import math

import torch
from torch import Tensor
from tqdm import tqdm

from modelling import Flux, load_clip_l, load_flux, load_flux_autoencoder, load_t5
from offload import PerLayerOffloadCUDAStream

logger = logging.getLogger(__name__)


class FluxTextEmbedder:
    def __init__(self, offload_t5: bool = False):
        self.t5 = load_t5()  #  9.5 GB in BF16
        self.clip = load_clip_l().bfloat16()  # 246 MB in BF16
        self.t5_offloader = PerLayerOffloadCUDAStream(self.t5, enable=offload_t5)

    def cpu(self):
        for m in (self.clip, self.t5_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.clip, self.t5_offloader):
            m.cuda()
        return self

    def __call__(self, prompt: list[str]):
        return self.t5(prompt), self.clip(prompt)


class FluxGenerator:
    def __init__(self, flux: Flux | None = None, offload_flux: bool = False, offload_t5: bool = False) -> None:
        self.flux = flux or load_flux()  # 23.8 GB in BF16
        self.ae = load_flux_autoencoder()  # 168 MB in BF16
        self.text_embedder = FluxTextEmbedder(offload_t5)

        # autoencoder and clip are small, don't need to offload
        self.flux_offloader = PerLayerOffloadCUDAStream(self.flux, enable=offload_flux)

    def cpu(self):
        for m in (self.ae, self.text_embedder, self.flux_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.ae, self.text_embedder, self.flux_offloader):
            m.cuda()
        return self

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
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
        if isinstance(prompt, str):
            prompt = [prompt]
        bsize = len(prompt)

        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size

        t5_embeds, clip_vecs = self.text_embedder(prompt)  # (B, 512, 4096) and (B, 768)
        if extra_txt_embeds is not None:  # e.g. Flux-Redux
            t5_embeds = torch.cat([t5_embeds, extra_txt_embeds], dim=1)

        rng = torch.Generator("cuda")
        rng.manual_seed(seed) if seed is not None else logger.info(f"Using seed={rng.seed()}")
        noise = torch.randn(bsize, 16, height // 8, width // 8, device="cuda", dtype=torch.bfloat16, generator=rng)
        # this is basically SDEdit https://arxiv.org/abs/2108.01073
        latents = latents.lerp(noise, denoise) if latents is not None else noise

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
    neg_txt: Tensor | None = None,
    neg_vec: Tensor | None = None,
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
        torch.compile(flux_step, disable=not compile)(
            flux, latents, txt, vec, neg_txt, neg_vec, timesteps[i], timesteps[i + 1], guidance
        )

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
    neg_txt: Tensor | None,
    neg_vec: Tensor | None,
    t_curr: Tensor,
    t_next: Tensor,
    guidance: Tensor,
) -> None:
    # t_curr and t_next must be Tensor (cpu is fine) to avoid recompilation
    t_vec = t_curr.to(latents.dtype).view(1).cuda()

    if neg_txt is not None and neg_vec is not None:
        # classifier-free guidance
        _g = latents.new_ones(1)
        v = flux(latents, txt, t_vec, vec, _g)
        neg_v = flux(latents, neg_txt, t_vec, neg_vec, _g)
        v = neg_v.lerp(v, guidance)

    else:
        # built-in distilled guidance
        v = flux(latents, txt, t_vec, vec, guidance)

    latents.add_(v, alpha=t_next - t_curr)  # Euler's method. move from t_curr to t_next
