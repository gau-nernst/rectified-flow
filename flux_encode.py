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

    t5_embeds = torch.empty(N, 512, 4096, dtype=torch.bfloat16)
    clip_embeds = torch.empty(N, 768, dtype=torch.bfloat16)
    latents = torch.empty(N, latent_h * latent_w, 64, dtype=torch.bfloat16)

    t5 = load_t5().cuda()
    clip = load_clip_text().cuda()
    for offset in tqdm(range(0, N, args.batch_size), "Embed text", dynamic_ncols=True):
        s = slice(offset, min(offset + args.batch_size, N))
        batch = prompts[s]
        t5_embeds[s] = t5(batch).cpu()
        clip_embeds[s] = clip(batch).cpu()
    del t5
    del clip

    flux = load_flux()
    PerLayerOffloadCUDAStream(flux).cuda()
    for offset in tqdm(range(0, N, args.batch_size), "Generate", dynamic_ncols=True):
        s = slice(offset, min(offset + args.batch_size, N))
        txt = t5_embeds[s].cuda()
        vec = clip_embeds[s].cuda()
        latents[s] = flux_generate(flux, None, txt, vec, args.img_size, guidance=3.5, compile=True).cpu()
    del flux

    # T5 embeds are the largest. might want to compress T5 embeds somehow.
    data = dict(
        t5_embeds=t5_embeds,
        clip_embeds=clip_embeds,
        latents=latents,
        img_size=args.img_size,
    )
    torch.save(data, args.save_path)
