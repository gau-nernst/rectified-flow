import argparse

import av
import torch
from PIL import Image

from modelling import Wan5BPipeline, Wan14BPipeline


def main(args: argparse.Namespace):
    # Wan2.2-5B is 24. Wan2.2-14B is 16
    if args.model == "5b":
        gen = Wan5BPipeline(offload_wan=True, offload_umt5=True)
        fps = 24

    elif args.model == "14b":
        gen = Wan14BPipeline(offload_wan=True, offload_umt5=True)
        fps = 16

    else:
        raise ValueError

    num_frames = int(args.duration * fps) // 4 * 4 + 1
    gen.cuda()

    kwargs = dict(
        prompt=args.prompt,
        num_frames=num_frames,
        num_steps=args.num_steps,
        seed=args.seed,
        pbar=True,
    )

    if args.input_img is not None:
        assert args.model == "5b", "Only 5B model supports image input"
        img_pil = Image.open(args.input_img)
        img_pil = img_pil.resize((1280, 704), resample=Image.Resampling.LANCZOS)
        img_pil = img_pil.convert("RGB")
        img_pt = torch.frombuffer(img_pil.tobytes(), dtype=torch.uint8).view(704, 1280, 3)
        img_pt = img_pt.permute(2, 0, 1).unsqueeze(0)
        img_pt = img_pt.float().sub(127.5).div(127.5)

        kwargs.update(img=img_pt)

    # [3, T, H, W]
    video = gen.generate(**kwargs).squeeze(0)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["5b", "14b"], default="5b")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input_img")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    main(args)
