import argparse
import dataclasses
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
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import ImageReadMode, decode_image, write_png
from torchvision.transforms import v2
from tqdm import tqdm

from flux_infer import flux_encode, flux_generate, flux_img_ids, flux_decode
from modelling import AutoEncoder, Flux, LoRALinear, load_clip_text, load_flux, load_flux_autoencoder, load_t5
from subclass import NF4Tensor, quantize_

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def compute_and_cache_text_embeds(model_name: str, texts: list[str], save_path: str, batch_size: int = 32):
    if model_name == "t5":
        model_loader = load_t5
        shape = (len(texts), 512, 4096)
    elif model_name == "clip":
        model_loader = load_clip_text
        shape = (len(texts), 768)
    else:
        raise ValueError(f"Unsupported {model_name=}")

    if not Path(save_path).exists():
        logger.info(f"Precompute {model_name} embeddings")
        model = model_loader().cuda()

        # numpy doesn't have BF16, so we use uint16 instead.
        embeds = np.memmap(save_path, dtype=np.uint16, mode="w+", shape=shape)
        for i in tqdm(range(0, len(texts), batch_size), dynamic_ncols=True):
            s = slice(i, min(i + batch_size, len(texts)))
            embeds[s] = model(texts[s]).cpu().view(torch.uint16).numpy()
        embeds.flush()
        del model

    embeds = np.memmap(save_path, dtype=np.uint16, mode="r", shape=shape)
    return torch.from_numpy(embeds).view(torch.bfloat16)


class FluxImageDataset(Dataset):
    def __init__(self, meta_path: str, data_dir: str, img_size: tuple[int, int]) -> None:
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.random_crop = v2.RandomCrop(img_size)

        df = pd.read_csv(meta_path)
        self.img_paths = df["img_path"].tolist()
        prompts = df["prompt"].tolist()

        self.t5_embeds = compute_and_cache_text_embeds("t5", prompts, f"{meta_path}.t5_embeds")
        self.clip_embeds = compute_and_cache_text_embeds("clip", prompts, f"{meta_path}.clip_embeds")

    def __getitem__(self, idx: int):
        # resize while maintaining aspect ratio, then make a random crop spanning 1 dimension.
        # all images should have similar aspect ratios.
        img = decode_image(self.data_dir / self.img_paths[idx], mode=ImageReadMode.RGB)
        h, w = img.shape[1:]
        scale = max(self.img_size[0] / h, self.img_size[1] / w)
        new_h = round(h * scale)
        new_w = round(w * scale)
        img = F.interpolate(img.unsqueeze(0), (new_h, new_w), mode="bicubic", antialias=True).squeeze(0)
        img = self.random_crop(img)

        t5_embed = self.t5_embeds[idx]
        clip_embed = self.clip_embeds[idx]
        return img, t5_embed, clip_embed

    def __len__(self):
        return len(self.img_paths)


class FluxLatentDataset(Dataset):
    def __init__(self, data_path: str, img_size: tuple[int, int]) -> None:
        self.data = torch.load(data_path, map_location="cpu", weights_only=True, mmap=True)
        self.img_size = self.data["img_size"]
        assert self.img_size == img_size, (self.img_size, img_size)

    def __getitem__(self, idx: int):
        return tuple(self.data[key][idx] for key in ("latents", "t5_embeds", "clip_embeds"))

    def __len__(self):
        return self.data["latents"].shape[0]


def save_images(
    flux: Flux,
    ae: AutoEncoder,
    prompt_path: str,
    save_dir: Path,
    img_size: tuple[int, int],
    batch_size: int = 4,
):
    prompts = [line.rstrip() for line in open(prompt_path, encoding="utf-8")]
    t5_embeds = compute_and_cache_text_embeds("t5", prompts, f"{prompt_path}.t5_embeds")
    clip_embeds = compute_and_cache_text_embeds("clip", prompts, f"{prompt_path}.clip_embeds")

    save_dir.mkdir(parents=True, exist_ok=True)
    latent_size = (img_size[0] // 16, img_size[1] // 16)

    for offset in tqdm(range(0, len(prompts), batch_size), "Generating images", dynamic_ncols=True):
        s = slice(offset, min(offset + batch_size, len(prompts)))
        latents = flux_generate(flux, t5_embeds[s].cuda(), clip_embeds[s].cuda(), img_size, seed=2024, compile=True)
        imgs = flux_decode(ae, latents, latent_size)
        imgs = imgs.cpu()

        for img_idx in range(imgs.shape[0]):
            # TODO: investigate saving with webp to save storage
            write_png(imgs[img_idx], save_dir / f"{offset + img_idx:04d}.png")


class InfiniteSampler(Sampler):
    def __init__(self, size: int):
        self.size = size

    def __iter__(self):
        while True:
            yield from torch.randperm(self.size).tolist()


@dataclasses.dataclass
class uniform:
    def __call__(self, n: int, device: torch.device):
        return torch.rand(n, device=device)


@dataclasses.dataclass
class logit_normal:
    """Section 3.1 in https://arxiv.org/abs/2403.03206"""

    mean: float = 0.0
    std: float = 1.0

    def __call__(self, n: int, device: torch.device):
        return torch.normal(self.mean, self.std, size=(n,), device=device).sigmoid()


def compute_loss(
    flux: Flux,
    latents: Tensor,
    t5_embeds: Tensor,
    clip_embeds: Tensor,
    img_ids: Tensor,
    txt_ids: Tensor,
    guidance: Tensor,
    time_sampler: uniform | logit_normal,
) -> Tensor:
    latents = latents.float()  # FP32
    t_vec = time_sampler(latents.shape[0], device=latents.device)
    noise = torch.randn_like(latents)
    interpolate = latents.lerp(noise, t_vec.view(-1, 1, 1))

    # NOTE: this is a guidance-distilled model. using guidance for finetuning might be "problematic".
    # best is to fix the guidance for finetuning and later inference.
    v = flux(interpolate.bfloat16(), img_ids, t5_embeds, txt_ids, t_vec.bfloat16(), clip_embeds, guidance)

    # rectified flow loss. predict velocity from latents (t=0) to noise (t=1).
    # TODO: check if we use logit-normal sampling, whether we need to also apply loss weight
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
    parser.add_argument("--time_sampler", default="uniform()")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--latents_path")  # for DreamBooth prior preservation loss

    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--test_prompt_path", required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--run_name", default="Debug")
    args = parser.parse_args()

    assert args.batch_size % args.gradient_accumulation == 0
    batch_size = args.batch_size // args.gradient_accumulation

    wandb.init(project="Flux finetune", name=args.run_name, dir="/tmp")

    if args.latents_path is not None:
        assert batch_size % 2 == 0
        batch_size = batch_size // 2

        ds_latent = FluxLatentDataset(args.latents_path, args.img_size)
        dloader_latent = DataLoader(
            ds_latent,
            batch_size=batch_size,
            sampler=InfiniteSampler(len(ds_latent)),
            num_workers=1,  # just read from disk, don't need a lot of workers
            pin_memory=True,
        )
        dloader_latent_iter = iter(dloader_latent)
    else:
        dloader_latent_iter = None

    ds_img = FluxImageDataset(args.meta_path, args.data_dir, args.img_size)
    dloader_img = DataLoader(
        ds_img,
        batch_size=batch_size,
        sampler=InfiniteSampler(len(ds_img)),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dloader_img_iter = iter(dloader_img)

    # QLoRA
    flux = load_flux().requires_grad_(False)
    for module_list in [flux.double_blocks, flux.single_blocks]:
        quantize_(module_list, NF4Tensor, "cuda").cpu()
        LoRALinear.to_lora(module_list, rank=args.lora)
    ae = load_flux_autoencoder()
    optim = torch.optim.AdamW(flux.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    logger.info(flux)

    # activation checkpointing
    for layer in list(flux.double_blocks) + list(flux.single_blocks):
        layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)

    latent_h = args.img_size[0] // 16
    latent_w = args.img_size[1] // 16
    img_ids = flux_img_ids(args.batch_size, latent_h, latent_w).cuda()
    txt_ids = torch.zeros(args.batch_size, 512, 3, device="cuda")
    # FLUX-dev default
    guidance = torch.full((args.batch_size,), 3.5, device="cuda", dtype=torch.bfloat16)

    time_sampler = eval(args.time_sampler, dict(uniform=uniform, logit_normal=logit_normal))

    log_dir = args.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    img_dir = log_dir / "images"

    # inference before any training
    step = 0

    # due to torch.compile + tensor subclass bug, we can't move compiled FLUX to CPU.
    # thus, manually compute test prompt embeddings and move FLUX to CUDA outside of save_images()
    prompts = [line.rstrip() for line in open(args.test_prompt_path, encoding="utf-8")]
    t5_embeds = compute_and_cache_text_embeds("t5", prompts, f"{args.test_prompt_path}.t5_embeds")
    clip_embeds = compute_and_cache_text_embeds("clip", prompts, f"{args.test_prompt_path}.clip_embeds")

    flux.cuda()
    ae.cuda()
    save_images(flux, ae, args.test_prompt_path, img_dir / f"step{step:06d}", args.img_size)

    pbar = tqdm(total=args.num_steps, dynamic_ncols=True)
    torch.cuda.reset_peak_memory_stats()
    time0 = time.perf_counter()
    while step < args.num_steps:
        for _ in range(args.gradient_accumulation):
            imgs, t5_embeds, clip_embeds = next(dloader_img_iter)
            latents = torch.compile(flux_encode, disable=not args.compile)(ae, imgs.cuda(), sample=True)

            if dloader_latent_iter is not None:
                latents2, t5_embeds2, clip_embeds2 = next(dloader_latent_iter)
                latents = torch.cat([latents, latents2.cuda()])
                t5_embeds = torch.cat([t5_embeds.cuda(), t5_embeds2.cuda()])
                clip_embeds = torch.cat([clip_embeds.cuda(), clip_embeds2.cuda()])

            loss = torch.compile(compute_loss, disable=not args.compile)(
                flux,
                latents,
                t5_embeds.cuda(),
                clip_embeds.cuda(),
                img_ids,
                txt_ids,
                guidance,
                time_sampler,
            )
            loss.backward()

        if step % args.log_interval == 0:
            grad_norm = sum(p.grad.square().sum() for p in flux.parameters() if p.grad is not None) ** 0.5
            log_dict = dict(
                loss=loss.item(),
                grad_norm=grad_norm,
            )
            wandb.log(log_dict, step)

        optim.step()
        optim.zero_grad()
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
            state_dict = {name: p.detach().bfloat16() for name, p in flux.named_parameters() if p.requires_grad}
            torch.save(state_dict, ckpt_dir / f"step{step:06d}.pth")

            # infer with test prompts
            save_images(flux, ae, args.test_prompt_path, img_dir / f"step{step:06d}", args.img_size)

    wandb.finish()
