# https://arxiv.org/abs/2410.10792

import logging
from typing import Literal

import torch
from torch import Tensor
from tqdm import tqdm

from flux_infer import time_shift
from modelling import Flux

logger = logging.getLogger(__name__)


# Algorithm 1 & 2
def rf_inversion_step(
    flux: Flux,
    latents: Tensor,
    reference: Tensor,
    direction: Literal["forward", "reverse"],
    controller_guidance: float,
    txt: Tensor,
    vec: Tensor,
    t_curr: Tensor,
    t_next: Tensor,
    guidance: Tensor,
):
    if direction == "forward":  # reference is at t=1
        conditional_v = (reference - latents) / (1.0 - t_curr)
    elif direction == "reverse":  # reference is at t=0
        conditional_v = (latents - reference) / t_curr

    # when controller_guidance=0, this is equivalent to flux_step
    t_vec = t_curr.to(latents.dtype).view(1).cuda()
    unconditional_v = flux(latents, txt, t_vec, vec, guidance)
    controlled_v = unconditional_v.lerp(conditional_v, controller_guidance)
    latents += (t_next - t_curr) * controlled_v  # the usual Euler's method


@torch.no_grad()
def rf_inversion_generate(
    flux: Flux,
    latents: Tensor,
    txt: Tensor,
    vec: Tensor,
    img_size: tuple[int, int],
    start_t: float = 0.0,
    end_t: float = 1.0,
    controller_guidance: float = 1.0,
    guidance: Tensor | float = 3.5,
    num_steps: int = 50,
    seed: int | None = None,
    pbar: bool = False,
    compile: bool = False,
    fast: bool = False,
):
    # see Table 4 for possible values of starting_time, stopping_time, and controller_guidance.
    # starting and stopping time from the table must be divided by 28.
    # they are relative to the denoising process, starting from t=1
    height, width = img_size
    bsize = txt.shape[0]

    latent_h = height // 8
    latent_w = width // 8

    timesteps = torch.linspace(1, 0, num_steps + 1)[int(start_t * num_steps) :]
    timesteps = time_shift(timesteps, latent_h * latent_w // 4)
    if not isinstance(guidance, Tensor):
        guidance = torch.full((bsize,), guidance, device="cuda", dtype=torch.bfloat16)

    rng = torch.Generator("cuda")
    rng.manual_seed(seed) if seed is not None else logger.info(f"Using seed={rng.seed()}")
    reference = latents  # we won't modify reference in-place, so it's safe
    noise = torch.randn(bsize, 16, latent_h, latent_w, device="cuda", dtype=torch.bfloat16, generator=rng)

    # inversion
    if fast:
        # jump directly to starting_time, instead of doing forward ODE
        latents = latents.lerp(noise, timesteps[0].item())
    else:
        # there is no reference code so we don't know how the original paper exactly obtains null embeddings.
        # this is different from using empty string. but considering that Flux was probably not trained with
        # embeddings of empty text, it's more natural to use zero embeddings (SD3 paper mentioned that they
        # zero-out embeddings for classifier-free guidance training).
        null_txt = torch.zeros_like(txt)
        null_vec = torch.zeros_like(vec)
        latents = latents.clone()  # clone since we will modify latents in-place

        # forward ODE. controller guidance is fixed at 0.5
        timesteps = timesteps.flip(0)
        for i in tqdm(range(timesteps.shape[0] - 1), disable=not pbar, dynamic_ncols=True):
            torch.compile(rf_inversion_step, disable=not compile)(
                flux,
                latents,
                noise,
                "forward",
                0.5,
                null_txt,
                null_vec,
                timesteps[i],
                timesteps[i + 1],
                guidance,
            )
        timesteps = timesteps.flip(0)

    # editting i.e. reverse ODE (the usual denoising)
    for i in tqdm(range(timesteps.shape[0] - 1), disable=not pbar, dynamic_ncols=True):
        eta = controller_guidance if i < (end_t - start_t) * num_steps else 0.0
        torch.compile(rf_inversion_step, disable=not compile)(
            flux,
            latents,
            reference,
            "reverse",
            eta,
            txt,
            vec,
            timesteps[i],
            timesteps[i + 1],
            guidance,
        )

    return latents
