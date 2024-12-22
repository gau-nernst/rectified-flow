import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from modelling import (
    SD3,
    AutoEncoder,
    TextEmbedder,
    load_clip_l,
    load_openclip_bigg,
    load_sd3_5,
    load_sd3_autoencoder,
    load_t5,
)
from offload import PerLayerOffloadCUDAStream

logger = logging.getLogger(__name__)


class SD3Generator:
    def __init__(
        self,
        ae: AutoEncoder | None = None,
        sd3: SD3 | None = None,
        t5: TextEmbedder | None = None,
        clip_l: TextEmbedder | None = None,
        clip_g: TextEmbedder | None = None,
        offload_sd3: bool = False,
        offload_t5: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.ae = (ae or load_sd3_autoencoder()).to(dtype)  # 168 MB in BF16
        self.sd3 = (sd3 or load_sd3_5()).to(dtype)  # 4.5 GB for medium in BF16
        self.t5 = (t5 or load_t5(max_length=256)).to(dtype)  # 9.5 GB in BF16
        self.clip_l = (clip_l or load_clip_l(output_key=["pooler_output", -2])).to(dtype)  # 246 MB in BF16
        self.clip_g = (clip_g or load_openclip_bigg()).to(dtype)  # 1.4 GB in BF16

        # autoencoder and clip are small, don't need to offload
        self.sd3_offloader = PerLayerOffloadCUDAStream(self.sd3, enable=offload_sd3)
        self.t5_offloader = PerLayerOffloadCUDAStream(self.t5, enable=offload_t5)

    def cpu(self):
        for m in (self.ae, self.clip_l, self.clip_g, self.sd3_offloader, self.t5_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.ae, self.clip_l, self.clip_g, self.sd3_offloader, self.t5_offloader):
            m.cuda()
        return self

    def embed_prompt(self, t5_prompt: list[str], clip_prompt: list[str]):
        t5_embeds = self.t5(t5_prompt).cuda()  # (B, 256, 4096)
        clip_l_vecs, clip_l_embeds = self.clip_l(clip_prompt)  # (B, 768) and (B, 77, 768)
        clip_g_vecs, clip_g_embeds = self.clip_g(clip_prompt)  # (B, 1280) and (B, 77, 1280)

        clip_vecs = torch.cat([clip_l_vecs.cuda(), clip_g_vecs.cuda()], dim=-1)  # (B, 2048)
        clip_embeds = torch.cat([clip_l_embeds.cuda(), clip_g_embeds.cuda()], dim=-1)  # (B, 77, 2048)
        clip_embeds = F.pad(clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1]))  # (B, 77, 4096)
        embeds = torch.cat([clip_embeds, t5_embeds], dim=1)  # (B, 77+256, 4096)
        return embeds, clip_vecs

    @torch.no_grad()
    def generate(
        self,
        t5_prompt: str | list[str],
        clip_prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] = "",
        img_size: int | tuple[int, int] = 512,
        latents: Tensor | None = None,
        denoise: float = 1.0,
        num_steps: int = 50,
        guidance: int = 5.0,
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

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * bsize

        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size
        latent_h = height // 8
        latent_w = width // 8

        embeds, clip_vecs = self.embed_prompt(t5_prompt, clip_prompt)
        if guidance != 1.0:
            neg_embeds, neg_clip_vecs = self.embed_prompt(negative_prompt, negative_prompt)

        rng = torch.Generator("cuda")
        rng.manual_seed(seed) if seed is not None else logger.info(f"Using seed={rng.seed()}")
        dtype = self.sd3.context_embedder.weight.dtype
        noise = torch.randn(bsize, 16, latent_h, latent_w, device="cuda", dtype=dtype, generator=rng)
        if latents is not None:
            # this is basically SDEdit https://arxiv.org/abs/2108.01073
            latents = latents.lerp(noise, denoise)
        else:
            latents = noise

        latents = sd3_generate(
            sd3=self.sd3,
            context=embeds,
            vec=clip_vecs,
            neg_context=neg_embeds,
            neg_vec=neg_clip_vecs,
            latents=latents,
            start_t=denoise,
            num_steps=num_steps,
            guidance=guidance,
            pbar=pbar,
            compile=compile,
        )
        return self.ae.decode(latents, uint8=True)


@torch.no_grad()
def sd3_generate(
    sd3: SD3,
    context: Tensor,
    vec: Tensor,
    latents: Tensor,
    neg_context: Tensor | None = None,
    neg_vec: Tensor | None = None,
    start_t: float = 1.0,
    end_t: float = 0.0,
    num_steps: int = 50,
    guidance: float = 4.5,
    pbar: bool = False,
    compile: bool = False,
):
    # denoise from t=1 (noise) to t=0 (latents)
    # NOTE: SD3 uses discrete timesteps (1000), but we use continuous timesteps.
    timesteps = torch.linspace(start_t, end_t, num_steps + 1)
    timesteps = 3.0 / (3.0 + 1 / timesteps - 1)  # static shift

    for i in tqdm(range(timesteps.shape[0] - 1), disable=not pbar, dynamic_ncols=True):
        torch.compile(sd3_step, disable=not compile)(
            sd3, latents, context, vec, neg_context, neg_vec, guidance, timesteps[i], timesteps[i + 1]
        )

    return latents


def sd3_step(
    sd3: SD3,
    latents: Tensor,
    context: Tensor,
    vec: Tensor,
    neg_context: Tensor | None,
    neg_vec: Tensor | None,
    guidance: float,
    t_curr: Tensor,
    t_next: Tensor,
) -> None:
    # t_curr and t_next must be Tensor (cpu is fine) to avoid recompilation
    t_vec = t_curr.to(latents.dtype).view(1).cuda()

    if neg_context is not None and neg_vec is not None and guidance != 1.0:
        v, neg_v = sd3(
            latents.repeat(2, 1, 1, 1),
            t_vec,
            torch.cat([vec, neg_vec], dim=0),
            torch.cat([context, neg_context], dim=0),
        ).chunk(2, dim=0)
        v = neg_v.lerp(v, guidance)

    else:
        v = sd3(latents, t_vec, vec, context)

    latents.add_(v, alpha=t_next - t_curr)  # Euler's method. move from t_curr to t_next
