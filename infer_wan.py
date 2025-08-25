import torch
from torch import Tensor
from tqdm import tqdm

from modelling import TextEmbedder, WanModel, WanVAEConfig, load_umt5_xxl, load_wan, load_wan_vae
from offload import PerLayerOffloadCUDAStream
from solvers import get_solver

NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def prepare_inputs(
    vae_config: WanVAEConfig,
    umt5: TextEmbedder,
    prompt: str | list[str],
    negative_prompt: str | list[str] | None = None,
    img_size: tuple[int, int] = (1280, 704),
    num_frames: int = 1,
    device: torch.types.Device = None,
    seed: int | None = None,
):
    # https://github.com/Wan-Video/Wan2.2/blob/031a9be5/wan/modules/tokenizers.py
    # Wan2.2 does some text cleaning. try ignore for now
    if isinstance(prompt, str):
        prompt = [prompt]
    txt_embeds, _ = umt5(prompt)

    if negative_prompt is not None:
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        neg_txt_embeds, _ = umt5(negative_prompt)
    else:
        neg_txt_embeds = None

    bs = len(prompt)
    height, width = img_size
    sT, sH, sW = vae_config.get_stride()
    assert num_frames % 4 == 1
    shape = (
        bs,
        vae_config.z_dim,
        1 + (num_frames - 1) // sT,
        height // sH,
        width // sW,
    )
    rng = torch.Generator(device).manual_seed(seed) if seed is not None else None
    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=rng)

    return noise, txt_embeds, neg_txt_embeds


class Wan5BGenerator:
    def __init__(self, offload_wan: bool = False, offload_umt5: bool = False) -> None:
        self.wan = load_wan("wan2.2-ti2v-5b")
        self.vae = load_wan_vae("2.2")
        self.umt5 = load_umt5_xxl()

        self.wan_offloader = PerLayerOffloadCUDAStream(self.wan, enable=offload_wan)
        self.umt5_offloader = PerLayerOffloadCUDAStream(self.umt5, enable=offload_umt5)

    def cpu(self):
        for m in (self.vae, self.wan_offloader, self.umt5_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.vae, self.wan_offloader, self.umt5_offloader):
            m.cuda()
        return self

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = NEGATIVE_PROMPT,
        img_size: tuple[int, int] = (1280, 704),
        num_frames: int = 1,
        cfg_scale: float = 5.0,
        num_steps: int = 50,
        seed: int | None = None,
        pbar: bool = False,
        solver: str = "unipc",
    ):
        device = self.vae.conv1.weight.device
        latents, context, neg_context = prepare_inputs(
            self.vae.cfg,
            self.umt5,
            prompt,
            negative_prompt,
            img_size,
            num_frames,
            device,
            seed,
        )
        timesteps = wan_timesteps(num_steps)
        latents = wan_generate(
            self.wan,
            latents,
            timesteps,
            context,
            neg_context,
            cfg_scale=cfg_scale,
            solver=solver,
            pbar=pbar,
        )
        return self.vae.decode(latents)


# https://github.com/Wan-Video/Wan2.2/blob/031a9be5/wan/utils/fm_solvers_unipc.py
# not sure why they use discrete timesteps
# NOTE: we keep continuous timestep (but discretized to 1/1000) and keep the last t=0
def wan_timesteps(num_steps: int = 50, shift: float = 5.0, num_train_steps: int = 1000):
    sigmas = torch.linspace(1 - 1 / num_train_steps, 0, num_steps + 1)
    sigmas = shift / (shift + 1 / sigmas - 1)
    return sigmas.mul(num_train_steps).round().div(num_train_steps).tolist()


@torch.no_grad()
def wan_generate(
    wan: WanModel,
    latents: Tensor,
    timesteps: list[int],
    context: Tensor,
    neg_context: Tensor | None = None,
    cfg_scale: float = 5.0,
    solver: str = "unipc",
    pbar: bool = False,
) -> Tensor:
    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver)

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        # TODO: fold 1000 into modelling code
        t = torch.tensor([timesteps[i] * 1000], device="cuda")
        v = wan(latents, t, context)

        # classifier-free guidance
        if cfg_scale != 1.0:
            neg_v = wan(latents, t, neg_context)
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, timesteps, i)

    return latents
