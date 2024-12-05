import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from flux_infer import flux_generate
from modelling import load_clip_text, load_flux, load_t5
from offload import PerLayerOffloadCUDAStream

if __name__ == "__main__":

    def parse_img_size(size):
        size = tuple(int(x) for x in size.split(","))
        assert len(size) == 2
        return size

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=parse_img_size, default=(1024, 1024))
    args = parser.parse_args()

    args.save_path.parent.mkdir(exist_ok=True, parents=True)
    prompts = [line.rstrip() for line in open(args.prompt, encoding="utf-8")]
    N = len(prompts)
    latent_h = args.img_size[0] // 16
    latent_w = args.img_size[1] // 16

    # size for 1 latent: (64, 64, 64) -> 0.5 MB  (1024x1024 image)
    latents = torch.empty(N, latent_h * latent_w, 64, dtype=torch.bfloat16)

    flux = load_flux()
    t5 = load_t5()
    clip = load_clip_text().cuda()

    PerLayerOffloadCUDAStream(flux).cuda()
    PerLayerOffloadCUDAStream(t5).cuda()

    for offset in tqdm(range(0, N, args.batch_size), "Generate", dynamic_ncols=True):
        s = slice(offset, min(offset + args.batch_size, N))
        batch = prompts[s]
        txt = t5(batch)
        vec = clip(batch)
        noise = torch.randn(txt.shape[0], latent_h * latent_w, 64, device="cuda", dtype=torch.bfloat16)
        latents[s] = flux_generate(flux, txt, vec, args.img_size, noise, compile=True).cpu()

    data = dict(
        latents=latents,
        prompts=prompts,
        img_size=args.img_size,
    )
    torch.save(data, args.save_path)
