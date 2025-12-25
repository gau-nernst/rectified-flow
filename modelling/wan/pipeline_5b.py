import torch
from torch import Tensor
from tqdm import tqdm

from ..offload import PerLayerOffloadCUDAStream
from ..solvers import get_solver
from ..text_embedder import TextEmbedder, load_umt5_xxl
from .model import WanModel, load_wan
from .vae import WanVAEConfig, load_wan_vae

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
    txt_embeds = umt5(prompt)
    neg_txt_embeds = umt5(negative_prompt) if negative_prompt is not None else None

    bs = txt_embeds.shape[0]
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


class Wan5BPipeline:
    def __init__(self, offload_wan: bool = False, offload_umt5: bool = False) -> None:
        self.wan = load_wan("wan2.2-ti2v-5b").bfloat16()
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
        img: Tensor | None = None,
        img_size: tuple[int, int] = (704, 1280),
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
            negative_prompt if cfg_scale != 1.0 else None,
            img_size,
            num_frames,
            device,
            seed,
        )
        if img is not None:
            img = self.vae.encode(img.unsqueeze(-3).to(device))  # add time dim

        timesteps = wan_timesteps(num_steps)
        latents = wan_generate(
            self.wan,
            latents,
            timesteps,
            context,
            neg_context,
            cfg_scale=cfg_scale,
            first_frame=img,
            solver=solver,
            pbar=pbar,
        )
        return self.vae.decode(latents)


# https://github.com/Wan-Video/Wan2.2/blob/031a9be5/wan/utils/fm_solvers_unipc.py
# NOTE: we keep the last t=0
def wan_timesteps(num_steps: int = 50, shift: float = 5.0, num_train_steps: int = 1000):
    sigmas = torch.linspace(1 - 1 / num_train_steps, 0, num_steps + 1)
    sigmas = shift / (shift + 1 / sigmas - 1)
    return sigmas.tolist()


@torch.no_grad()
def wan_generate(
    wan: WanModel,
    latents: Tensor,
    timesteps: list[int],
    context: Tensor,
    neg_context: Tensor | None = None,
    cfg_scale: float = 5.0,
    first_frame: Tensor | None = None,
    solver: str = "unipc",
    pbar: bool = False,
) -> Tensor:
    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)

    F = latents.shape[2] // wan.cfg.patch_size[0]
    H = latents.shape[3] // wan.cfg.patch_size[1]
    W = latents.shape[4] // wan.cfg.patch_size[2]

    if first_frame is not None:
        latents[:, :, :1] = first_frame  # fill the first frame

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        # Wan2.2 uses discrete timestep. this is truncated, not rounded.
        # TODO: fold 1000 into modelling code
        t_discrete = int(timesteps[i] * 1000)
        if first_frame is not None:  # set time of 1st frame to zeros
            t = torch.tensor([[0] * (H * W) + [t_discrete] * ((F - 1) * H * W)], dtype=torch.float, device="cuda")
        else:
            t = torch.tensor([t_discrete], dtype=torch.float, device="cuda")

        v = wan(latents, t, context)

        # classifier-free guidance
        if cfg_scale != 1.0:
            neg_v = wan(latents, t, neg_context)
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)

        if first_frame is not None:
            latents[:, :, :1] = first_frame

    return latents


if __name__ == "__main__":
    import argparse

    import av
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input_img")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    fps = 24  # Wan2.2-5B is 24. Wan2.2-14B is 16
    num_frames = int(args.duration * fps) // 4 * 4 + 1

    gen = Wan5BPipeline(offload_wan=True, offload_umt5=True)
    gen.cuda()

    if args.input_img is not None:
        img_pil = Image.open(args.input_img)
        img_pil = img_pil.resize((1280, 704), resample=Image.Resampling.LANCZOS)
        img_pil = img_pil.convert("RGB")
        img_pt = torch.frombuffer(img_pil.tobytes(), dtype=torch.uint8).view(704, 1280, 3)
        img_pt = img_pt.permute(2, 0, 1).unsqueeze(0)
        img_pt = img_pt.float().sub(127.5).div(127.5)
    else:
        img_pt = None

    # [3, T, H, W]
    video = gen.generate(
        args.prompt,
        img=img_pt,
        num_frames=num_frames,
        num_steps=args.num_steps,
        seed=args.seed,
        pbar=True,
    ).squeeze(0)

    # TODO: merge uint8 conversion to VAE
    video_np = video.add(1).mul(127.5).round().clip(0, 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()

    container = av.open(args.output, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 1280
    stream.height = 704

    for frame_id in range(video_np.shape[0]):
        frame = av.VideoFrame.from_ndarray(video_np[frame_id])
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
