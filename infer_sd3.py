from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from infer_flux import prepare_inputs
from modelling import SD3, load_clip_l, load_openclip_bigg, load_sd3_5, load_sd3_autoencoder, load_t5
from offload import PerLayerOffloadCUDAStream
from solvers import get_solver


class SD3TextEmbedder:
    def __init__(self, offload_t5: bool = False, dtype: torch.dtype = torch.bfloat16) -> None:
        self.t5 = load_t5(max_length=256).to(dtype)  # 9.5 GB in BF16
        self.clip_l = load_clip_l(output_key=["pooler_output", -2]).to(dtype)  # 246 MB in BF16
        self.clip_g = load_openclip_bigg().to(dtype)  # 1.4 GB in BF16
        self.t5_offloader = PerLayerOffloadCUDAStream(self.t5, enable=offload_t5)

    def cpu(self):
        for m in (self.clip_l, self.clip_g, self.t5_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.clip_l, self.clip_g, self.t5_offloader):
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
        self.sd3 = (sd3 or load_sd3_5()).to(dtype)  # 4.5 GB for medium, 16.1 GB for large in BF16
        self.ae = load_sd3_autoencoder().to(dtype)  # 168 MB in BF16
        self.text_embedder = SD3TextEmbedder(offload_t5, dtype=dtype)
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
        solver: str = "dpm++2m",
        pbar: bool = False,
    ) -> Tensor:
        # NOTE: right now this is broken...
        latents, embeds, vecs, neg_embeds, neg_vecs = prepare_inputs(
            self.ae, self.text_embedder, prompt, negative_prompt, img_size, latents, denoise, cfg_scale, seed
        )
        timesteps = sd3_timesteps(denoise, 0.0, num_steps)  # denoise from t=1 (noise) to t=0 (latents)
        latents = sd3_generate(
            self.sd3,
            latents,
            timesteps,
            context=embeds,
            vec=vecs,
            neg_context=neg_embeds,
            neg_vec=neg_vecs,
            cfg_scale=cfg_scale,
            slg_config=slg_config,
            solver=solver,
            pbar=pbar,
        )
        return self.ae.decode(latents, uint8=True)


def sd3_timesteps(start: float = 1.0, end: float = 0.0, num_steps: int = 50, shift: float = 3.0):
    timesteps = torch.linspace(start, end, num_steps + 1)
    timesteps = shift / (shift + 1 / timesteps - 1)  # static shift
    return timesteps.tolist()


@torch.no_grad()
def sd3_generate(
    sd3: SD3,
    latents: Tensor,
    timesteps: list[float],
    context: Tensor,
    vec: Tensor,
    neg_context: Tensor | None = None,
    neg_vec: Tensor | None = None,
    cfg_scale: float = 5.0,
    slg_config: SkipLayerConfig = SkipLayerConfig(),
    solver: str = "euler",
    pbar: bool = False,
) -> Tensor:
    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver)

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = torch.tensor([timesteps[i]], device="cuda")
        pos_v = sd3(latents, t, context, vec).float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            assert neg_context is not None and neg_vec is not None
            neg_v = sd3(latents, t, neg_context, neg_vec).float()
            v = neg_v.lerp(pos_v, cfg_scale)
        else:
            v = pos_v

        # skip-layer guidance
        # https://github.com/Stability-AI/sd3.5/blob/fbf8f483f992d8d6ad4eaaeb23b1dc5f523c3b3a/sd3_impls.py#L219
        if slg_config.start < i / num_steps < slg_config.end:
            skip_layer_v = sd3(latents, t, context, vec, slg_config.layers).float()
            v = v.add(pos_v - skip_layer_v, alpha=slg_config.scale)

        latents = solver_.step(latents, v, timesteps, i)

    return latents
