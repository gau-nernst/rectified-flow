import torch
from torch import Tensor
from tqdm import tqdm

from infer_wan5b import NEGATIVE_PROMPT, prepare_inputs, wan_timesteps
from modelling import WanModel, load_umt5_xxl, load_wan, load_wan_vae
from offload import PerLayerOffloadCUDAStream
from solvers import get_solver


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
        timesteps = wan_timesteps(num_steps)
        latents = wan_generate(
            self.wan_high,
            self.wan_low,
            latents,
            timesteps,
            context,
            neg_context,
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
    context: Tensor,
    neg_context: Tensor | None = None,
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
        v = wan(latents, t, context)

        # classifier-free guidance
        if cfg_scale != 1.0:
            neg_v = wan(latents, t, neg_context)
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)

    return latents


if __name__ == "__main__":
    import argparse

    import av

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    fps = 16  # Wan2.2-5B is 24. Wan2.2-14B is 16
    num_frames = int(args.duration * fps) // 4 * 4 + 1

    gen = Wan14BPipeline(offload_wan=True, offload_umt5=True)
    gen.cuda()

    # [3, T, H, W]
    video = gen.generate(
        args.prompt,
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
