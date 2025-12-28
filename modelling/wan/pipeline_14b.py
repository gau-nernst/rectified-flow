import torch
from torch import Tensor
from tqdm import tqdm

from ..offload import PerLayerOffloadCUDAStream
from ..solvers import get_solver
from ..text_embedder import load_umt5_xxl
from .model import WanModel, load_wan
from .pipeline_5b import NEGATIVE_PROMPT, prepare_inputs, wan_timesteps
from .vae import load_wan_vae


class Wan14BPipeline:
    def __init__(self, offload_wan: bool = False, offload_umt5: bool = False) -> None:
        self.wan_high = load_wan("wan2.2-t2v-a14b-high")
        self.wan_low = load_wan("wan2.2-t2v-a14b-low")
        self.vae = load_wan_vae("2.1")
        self.umt5 = load_umt5_xxl()

        self.wan_high_offloader = PerLayerOffloadCUDAStream(self.wan_high, enable=offload_wan)
        self.wan_low_offloader = PerLayerOffloadCUDAStream(self.wan_low, enable=offload_wan)
        self.umt5_offloader = PerLayerOffloadCUDAStream(self.umt5, enable=offload_umt5)

    def cpu(self):
        for m in (self.vae, self.wan_high_offloader, self.wan_low_offloader, self.umt5_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.vae, self.wan_high_offloader, self.wan_low_offloader, self.umt5_offloader):
            m.cuda()
        return self

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = NEGATIVE_PROMPT,
        img_size: tuple[int, int] = (720, 1280),
        num_frames: int = 1,
        cfg_scale: float = 5.0,
        boundary: float = 0.875,
        num_steps: int = 50,
        seed: int | None = None,
        pbar: bool = False,
        solver: str = "unipc",
    ):
        device = self.vae.conv1.weight.device
        latents, txt, neg_txt = prepare_inputs(
            self.vae.cfg,
            self.umt5,
            prompt,
            negative_prompt if cfg_scale != 1.0 else None,
            img_size,
            num_frames,
            device,
            seed,
        )
        timesteps = wan_timesteps(num_steps)
        latents = wan_generate(
            self.wan_high,
            self.wan_low,
            latents,
            timesteps,
            txt,
            neg_txt,
            cfg_scale=cfg_scale,
            boundary=boundary,
            solver=solver,
            pbar=pbar,
        )
        return self.vae.decode(latents)


@torch.no_grad()
def wan_generate(
    wan_high: WanModel,
    wan_low: WanModel,
    latents: Tensor,
    timesteps: list[int],
    txt: Tensor,
    neg_txt: Tensor | None = None,
    cfg_scale: float = 5.0,
    boundary: float = 0.875,
    solver: str = "unipc",
    pbar: bool = False,
) -> Tensor:
    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        # Wan2.2 uses discrete timestep. this is truncated, not rounded.
        # TODO: fold 1000 into modelling code
        t_discrete = int(timesteps[i] * 1000)
        t = torch.tensor([t_discrete], dtype=torch.float, device="cuda")

        # select model based on `boundary``
        wan = wan_high if timesteps[i] >= boundary else wan_low
        v = wan(latents, t, txt)

        # classifier-free guidance
        if cfg_scale != 1.0:
            neg_v = wan(latents, t, neg_txt)
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)

    return latents
