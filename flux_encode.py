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
    parser.add_argument("--t5_length", type=int, default=512)
    parser.add_argument("--img_size", type=parse_img_size, default=(1024, 1024))
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    args = parser.parse_args()

    args.save_path.parent.mkdir(exist_ok=True, parents=True)
    prompts = [line.rstrip() for line in open(args.prompt, encoding="utf-8")]
    N = len(prompts)
    latent_h = args.img_size[0] // 16
    latent_w = args.img_size[1] // 16

    # size for 1 sample
    # T5 embeds: (512, 4096) -> 4 MB
    # CLIP embeds: (768) -> 1.5 KB
    # latents: (64, 64, 64) -> 0.5 MB  (1024x1024 image)

    # we generate multiple images for 1 prompt to save on space occupied by T5 embeddings.
    # possibly reduce T5 seq_len to reduce T5 embeddings size.
    t5_embeds = torch.empty(N, args.t5_length, 4096, dtype=torch.bfloat16)
    clip_embeds = torch.empty(N, 768, dtype=torch.bfloat16)
    latents = torch.empty(N, args.num_images_per_prompt, latent_h * latent_w, 64, dtype=torch.bfloat16)

    t5 = load_t5(max_length=args.t5_length).cuda()
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
        for i in range(args.num_images_per_prompt):
            noise = torch.randn(txt.shape[0], latent_h * latent_w, 64, device="cuda", dtype=torch.bfloat16)
            latents[s, i] = flux_generate(flux, txt, vec, args.img_size, noise, compile=True).cpu()
    del flux

    # T5 embeds are the largest. might want to compress T5 embeds somehow.
    data = dict(
        t5_embeds=t5_embeds,
        clip_embeds=clip_embeds,
        latents=latents,
        img_size=args.img_size,
    )
    torch.save(data, args.save_path)
