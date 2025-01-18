import math

import torch
from torch import Tensor
from tqdm import tqdm

from modelling import AutoEncoder, Flux, load_clip_l, load_flux, load_flux_autoencoder, load_t5
from offload import PerLayerOffloadCUDAStream


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
        return self.t5(prompt), self.clip(prompt)  # (B, 512, 4096) and (B, 768)


def prepare_inputs(
    ae: AutoEncoder,
    text_embedder: FluxTextEmbedder,
    prompt: str | list[str],
    negative_prompt: str | list[str] = "",
    img_size: int | tuple[int, int] = 512,
    latents: Tensor | None = None,
    denoise: float = 1.0,
    cfg_scale: float = 1.0,
    seed: int | None = None,
):
    if isinstance(prompt, str):
        prompt = [prompt]
    bsize = len(prompt)

    if isinstance(img_size, int):
        height = width = img_size
    else:
        height, width = img_size

    embeds, vecs = text_embedder(prompt)

    if cfg_scale != 1.0:
        neg_embeds, neg_vecs = text_embedder(negative_prompt)
        if neg_embeds.shape[0] == 1:
            neg_embeds = neg_embeds.expand(bsize, -1, -1)
            neg_vecs = neg_vecs.expand(bsize, -1)
    else:
        neg_embeds = neg_vecs = None

    shape = (bsize, ae.z_dim, height // 8, width // 8)
    weight = ae.encoder.conv_in.weight
    rng = torch.Generator("cuda")
    rng.manual_seed(seed) if seed is not None else rng.seed()
    noise = torch.randn(shape, device=weight.device, dtype=weight.dtype, generator=rng)

    # this is basically SDEdit https://arxiv.org/abs/2108.01073
    latents = latents.lerp(noise, denoise) if latents is not None else noise

    return latents, embeds, vecs, neg_embeds, neg_vecs


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
        negative_prompt: str | list[str] = "",
        img_size: int | tuple[int, int] = 512,
        latents: Tensor | None = None,
        extra_embeds: Tensor | None = None,
        denoise: float = 1.0,
        guidance: Tensor | float | None = 3.5,
        cfg_scale: float = 1.0,
        num_steps: int = 50,
        seed: int | None = None,
        pbar: bool = False,
        compile: bool = False,
    ):
        latents, embeds, vecs, neg_embeds, neg_vecs = prepare_inputs(
            self.ae, self.text_embedder, prompt, negative_prompt, img_size, latents, denoise, cfg_scale, seed
        )

        if extra_embeds is not None:  # e.g. Flux-Redux
            embeds = torch.cat([embeds, extra_embeds], dim=1)
            # NOTE: neg_embeds are not extended

        # denoise from t=1 (noise) to t=0 (latents)
        # divide by 4 due to FLUX's input patchification
        height, width = latents.shape[-2:]
        timesteps = flux_timesteps(denoise, 0.0, num_steps, height * width // 4)

        latents = flux_euler_generate(
            flux=self.flux,
            latents=latents,
            timesteps=timesteps,
            txt=embeds,
            vec=vecs,
            neg_txt=neg_embeds,
            neg_vec=neg_vecs,
            guidance=guidance,
            cfg_scale=cfg_scale,
            pbar=pbar,
            compile=compile,
        )
        return self.ae.decode(latents, uint8=True)


@torch.no_grad()
def flux_euler_generate(
    flux: Flux,
    latents: Tensor,
    timesteps: list[float],
    txt: Tensor,
    vec: Tensor,
    neg_txt: Tensor | None = None,
    neg_vec: Tensor | None = None,
    guidance: Tensor | float | None = 3.5,
    cfg_scale: float = 1.0,
    pbar: bool = False,
    compile: bool = False,
):
    if isinstance(guidance, (int, float)):
        guidance = torch.full(latents.shape[:1], guidance, device="cuda", dtype=torch.bfloat16)

    num_steps = len(timesteps) - 1
    timesteps_pt = torch.tensor(timesteps)
    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        torch.compile(flux_step, disable=not compile)(
            flux, latents, txt, vec, neg_txt, neg_vec, timesteps_pt[i], timesteps_pt[i + 1], guidance, cfg_scale
        )

    return latents


def flux_timesteps(start: float = 1.0, end: float = 0.0, num_steps: int = 50, img_seq_len: int = 0):
    # only dev version has time shift
    timesteps = torch.linspace(start, end, num_steps + 1)
    timesteps = flux_time_shift(timesteps, img_seq_len)
    return timesteps.tolist()


# https://arxiv.org/abs/2403.03206
# Section 5.3.2 - Resolution-dependent shifting of timesteps schedules
# https://github.com/black-forest-labs/flux/blob/805da8571a0b49b6d4043950bd266a65328c243b/src/flux/sampling.py#L222-L238
def flux_time_shift(timesteps: Tensor | float, img_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15):
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
    cfg_scale: float,
) -> None:
    # t_curr and t_next must be Tensor (cpu is fine) to avoid recompilation
    t_vec = t_curr.view(1).cuda()

    if cfg_scale != 1.0 and neg_txt is not None and neg_vec is not None:
        # classifier-free guidance
        v, neg_v = flux(
            latents.repeat(2, 1, 1, 1),
            t_vec,
            torch.cat([txt, neg_txt], dim=0),
            torch.cat([vec, neg_vec], dim=0),
            guidance.repeat(2),
        ).chunk(2, dim=0)
        v = neg_v.lerp(v, cfg_scale)

    else:
        # built-in distilled guidance
        v = flux(latents, t_vec, txt, vec, guidance)

    latents.add_(v, alpha=t_next - t_curr)  # Euler's method. move from t_curr to t_next
