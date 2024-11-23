import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # improve memory usage

from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import ImageReadMode, decode_image, write_png
from torchvision.transforms import v2
from tqdm import tqdm

from flux_infer import FluxGenerator, flux_img_ids
from modelling import (
    AutoEncoder,
    Flux,
    LoRALinear,
    TextEmbedder,
    load_clip_text,
    load_flux,
    load_flux_autoencoder,
    load_t5,
)
from subclass import NF4Tensor, quantize_

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def compute_and_cache_text_embeds(
    model: TextEmbedder,
    texts: list[str],
    shape: tuple[int, ...],
    save_path: str,
    batch_size: int = 32,
):
    # numpy doesn't have BF16, so we use uint16 instead.
    embeds = np.memmap(save_path, dtype=np.uint16, mode="w+", shape=shape)
    for i in tqdm(range(0, len(texts), batch_size)):
        s = slice(i, min(i + batch_size, len(texts)))
        embeds[s] = model(texts[s]).cpu().view(torch.uint16).numpy()
    embeds.flush()


class FluxDataset(Dataset):
    def __init__(self, meta_path: str, data_dir: str, img_size: tuple[int, int]) -> None:
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.random_crop = v2.RandomCrop(img_size)

        df = pd.read_csv(meta_path)
        self.img_paths = df["img_path"].tolist()

        t5_embeds_path = f"{meta_path}.t5_embeds"
        t5_shape = (len(df), 512, 4096)
        if not Path(t5_embeds_path).exists():
            logger.info("Precompute T5 embeddings")
            t5 = load_t5().cuda()
            compute_and_cache_text_embeds(t5, df["prompt"].tolist(), t5_shape, t5_embeds_path)
            del t5
        t5_embeds = np.memmap(t5_embeds_path, dtype=np.uint16, mode="r", shape=t5_shape)
        self.t5_embeds = torch.from_numpy(t5_embeds).view(torch.bfloat16)

        clip_embeds_path = f"{meta_path}.clip_embeds"
        clip_shape = (len(df), 768)
        if not Path(clip_embeds_path).exists():
            logger.info("Precompute CLIP text embeddings")
            clip = load_clip_text().cuda()
            compute_and_cache_text_embeds(clip, df["prompt"].tolist(), clip_shape, clip_embeds_path)
            del clip
        clip_embeds = np.memmap(clip_embeds_path, dtype=np.uint16, mode="r", shape=clip_shape)
        self.clip_embeds = torch.from_numpy(clip_embeds).view(torch.bfloat16)

    def __getitem__(self, idx: int):
        # resize while maintaining aspect ratio, then make a random crop spanning 1 dimension.
        # all images should have similar aspect ratios.
        img = decode_image(self.data_dir / self.img_paths[idx], mode=ImageReadMode.RGB)
        scale = max(self.img_size[0] / img.shape[1], self.img_size[1] / img.shape[2])
        img = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode="bicubic", antialias=True).squeeze(0)
        img = self.random_crop(img)

        t5_embed = self.t5_embeds[idx]
        clip_embed = self.clip_embeds[idx]
        return img, t5_embed, clip_embed

    def __len__(self):
        return len(self.img_paths)


def save_images(
    flux: Flux,
    ae: AutoEncoder,
    prompt_path: str,
    save_dir: Path,
    img_size: tuple[int, int],
    batch_size: int = 4,
):
    # NOTE: this has memory leak.
    # TODO: pre-compute text embeddings like we did for training data.
    gen = FluxGenerator(flux=flux, ae=ae, offload_t5=True)
    # when tensor subclass + torch.compile() are used together, we can't do .cpu() or .cuda(), even if it doesn't
    # change device. hence, manually move sub-models to CUDA
    gen.clip.cuda()
    gen.t5_offloader.cuda()

    prompts = [line.rstrip() for line in open(prompt_path)]
    save_dir.mkdir(parents=True, exist_ok=True)

    for offset in tqdm(range(0, len(prompts), batch_size), "Generating images"):
        batch = prompts[offset : min(offset + batch_size, len(prompts))]
        imgs = gen.generate(batch, img_size=img_size, seed=2024, compile=True).cpu()

        for img_idx in range(imgs.shape[0]):
            # TODO: investigate saving with webp
            write_png(imgs[img_idx], save_dir / f"{offset + img_idx:04d}.png")


class InfiniteSampler(Sampler):
    def __init__(self, size: int):
        self.size = size

    def __iter__(self):
        while True:
            yield from torch.randperm(self.size).tolist()


def compute_loss(
    flux: Flux,
    ae: AutoEncoder,
    imgs: Tensor,
    t5_embeds: Tensor,
    clip_embeds: Tensor,
    img_ids: Tensor,
    txt_ids: Tensor,
    guidance: Tensor,
) -> Tensor:
    imgs = imgs.float() / 127.5 - 1
    latents = ae.encode(imgs.bfloat16(), sample=True)
    latents = F.pixel_unshuffle(latents, 2).view(imgs.shape[0], 64, -1).transpose(1, 2)
    latents = latents.float()  # FP32

    # uniform [0,1). TODO: change this
    t_vec = torch.rand(imgs.shape[0], device=latents.device)
    noise = torch.randn_like(latents)
    interpolate = latents.lerp(noise, t_vec.view(-1, 1, 1))

    v = flux(interpolate.bfloat16(), img_ids, t5_embeds, txt_ids, t_vec.bfloat16(), clip_embeds, guidance)

    # TODO: add loss weight based as t (effectively same as distribution transform)
    # rectified flow loss. predict velocity from latents (t=0) to noise (t=1).
    return F.mse_loss(noise - latents, v.float())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def parse_img_size(img_size: str):
        out = json.loads(img_size)
        if isinstance(out, int):
            out = [out, out]
        return tuple(out)

    parser.add_argument("--img_size", type=parse_img_size, default=(512, 512))
    parser.add_argument("--lora", type=int, default=8)
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--data_dir", required=True)

    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--test_prompt_path", required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--run_name", default="Debug")
    args = parser.parse_args()

    wandb.init(project="Flux finetune", name=args.run_name, dir="/tmp")

    ds = FluxDataset(args.meta_path, args.data_dir, args.img_size)
    dloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=InfiniteSampler(len(ds)),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dloader_iter = iter(dloader)

    # QLoRA
    flux = load_flux().requires_grad_(False)
    quantize_(flux, NF4Tensor, "cuda")
    LoRALinear.to_lora(flux.double_blocks, rank=args.lora)
    LoRALinear.to_lora(flux.single_blocks, rank=args.lora)
    flux.cuda()
    ae = load_flux_autoencoder().cuda()
    optim = torch.optim.AdamW(flux.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    logger.info(flux)

    # activation checkpointing
    for layer in list(flux.double_blocks) + list(flux.single_blocks):
        layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)

    latent_h = args.img_size[0] // 16
    latent_w = args.img_size[1] // 16
    img_ids = flux_img_ids(args.batch_size, latent_h, latent_w).cuda()
    txt_ids = torch.zeros(args.batch_size, 512, 3, device="cuda")
    guidance_vec = torch.full((args.batch_size,), 3.5, device="cuda", dtype=torch.bfloat16)  # sample guidance?

    log_dir = args.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    img_dir = log_dir / "images"

    # inference before any training
    step = 0
    pbar = tqdm(total=args.num_steps)
    torch.cuda.reset_peak_memory_stats()
    time0 = time.perf_counter()
    # save_images(flux, ae, args.test_prompt_path, img_dir / f"step{step:06d}", args.img_size)
    while step < args.num_steps:
        images, t5_embeds, clip_embeds = next(dloader_iter)
        loss = torch.compile(compute_loss, disable=not args.compile)(
            flux,
            ae,
            images.cuda(),
            t5_embeds.cuda(),
            clip_embeds.cuda(),
            img_ids,
            txt_ids,
            guidance_vec,
        )
        loss.backward()
        optim.step()
        optim.zero_grad()

        if step % args.log_interval == 0:
            wandb.log(dict(loss=loss.item()), step)

        step += 1
        pbar.update()

        if step % args.log_interval == 0:
            time1 = time.perf_counter()
            log_dict = dict(
                imgs_per_second=args.batch_size * args.log_interval / (time1 - time0),
                max_memory_allocated=torch.cuda.max_memory_allocated(),
                memory_allocated=torch.cuda.memory_allocated(),
            )
            wandb.log(log_dict, step=step)
            time0 = time1

        if step % args.eval_interval == 0:
            # only save LoRA weights. Flux doesn't have buffers.
            state_dict = {name: p.detach() for name, p in flux.named_parameters() if p.requires_grad}
            torch.save(state_dict, ckpt_dir / f"step{step:06d}.pth")

            # infer with test prompts
            # save_images(flux, ae, args.test_prompt_path, img_dir / f"step{step:06d}", args.img_size)

    wandb.finish()
