import argparse
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # improve memory usage

import json
from functools import partial

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate, get_worker_info
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from flux_infer import flux_decode, flux_encode, flux_generate
from modelling import (
    AutoEncoder,
    Flux,
    IResNet,
    TextEmbedder,
    load_adaface_ir101,
    load_clip_text,
    load_flux,
    load_flux_autoencoder,
    load_t5,
)
from modelling.face_embedder import arcface_crop
from offload import PerLayerOffloadCUDAStream, PerLayerOffloadWithBackward
from train_qlora import compute_loss, logit_normal, uniform

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FaceTrainDataset(IterableDataset):
    def __init__(self, meta_path: str, img_dir: str, img_sizes: list[tuple[int, int]], batch_size: int) -> None:
        self.img_dir = Path(img_dir)
        self.img_sizes = img_sizes
        self.log_ratios = [math.log(width / height) for height, width in img_sizes]
        self.batch_size = batch_size

        meta = pd.read_csv(meta_path)
        self.filenames = meta["filename"].tolist()
        self.kpts = np.array([x[0] for x in meta["keypoints"].map(json.loads)])
        self.prompts = meta["prompt"].tolist()

        self.shuffle_rng = torch.Generator()
        self.shuffle_rng.seed()

    def __iter__(self):
        buckets = [[] for _ in range(len(self.log_ratios))]

        counter = -1
        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        while True:
            # indices sequence is identical across workers
            indices = torch.randperm(len(self.filenames), generator=self.shuffle_rng)
            for idx in indices:
                counter = (counter + 1) % num_workers
                if counter != worker_id:
                    continue

                img_path = self.img_dir / self.filenames[idx]
                img = decode_image(img_path, mode=ImageReadMode.RGB, apply_exif_orientation=True)
                log_ratio = math.log(img.shape[2] / img.shape[1])
                _, bucket_idx = min((abs(log_ratio - x), _i) for _i, x in enumerate(self.log_ratios))

                kpt = self.kpts[idx].copy()
                kpt[..., 0] *= img.shape[2]
                kpt[..., 1] *= img.shape[1]
                img_np = img.permute(1, 2, 0).contiguous().numpy()
                crop = arcface_crop(img_np, kpt)
                crop = torch.from_numpy(crop).permute(2, 0, 1)
                crop = F.interpolate(crop.unsqueeze(0), 112, mode="bicubic", antialias=True).squeeze(0)
                crop = crop.float() / 127.5 - 1

                height, width = self.img_sizes[bucket_idx]
                scale = max(height / img.shape[1], width / img.shape[2])
                target_size = (round(img.shape[1] * scale), round(img.shape[2] * scale))
                img = F.interpolate(img.unsqueeze(0), target_size, mode="bicubic", antialias=True).squeeze(0)
                img = v2.RandomCrop((height, width))(img)

                buckets[bucket_idx].append((img, crop, self.prompts[idx]))
                if len(buckets[bucket_idx]) == self.batch_size:
                    yield default_collate(buckets[bucket_idx])
                    buckets[bucket_idx] = []


class AlternateDataset(IterableDataset):
    def __init__(self, ds_list: list[IterableDataset]):
        self.ds_list = ds_list

    def __iter__(self):
        ds_iter_list = [iter(ds) for ds in self.ds_list]
        while True:
            for ds_iter in ds_iter_list:
                yield next(ds_iter)


class FaceTestDataset(Dataset):
    def __init__(self, meta_path: str, img_dir: str) -> None:
        self.img_dir = Path(img_dir)

        meta = pd.read_csv(meta_path)
        self.filenames = meta["filename"].tolist()
        self.kpts = np.array([x[0] for x in meta["keypoints"].map(json.loads)])
        self.prompts = meta["prompt"].tolist()

    def __getitem__(self, idx: int):
        img_path = self.img_dir / self.filenames[idx]
        img = decode_image(img_path, mode=ImageReadMode.RGB, apply_exif_orientation=True)

        kpt = self.kpts[idx].copy()
        kpt[..., 0] *= img.shape[2]
        kpt[..., 1] *= img.shape[1]
        img_np = img.permute(1, 2, 0).contiguous().numpy()
        crop = arcface_crop(img_np, kpt)
        crop = torch.from_numpy(crop).permute(2, 0, 1)
        crop = F.interpolate(crop.unsqueeze(0), 112, mode="bicubic", antialias=True).squeeze(0)
        crop = crop.float() / 127.5 - 1

        return crop, self.prompts[idx]

    def __len__(self):
        return len(self.filenames)


@torch.no_grad()
def save_images(
    flux: Flux,
    ae: AutoEncoder,
    t5: TextEmbedder,
    clip: TextEmbedder,
    adaface: IResNet,
    face_redux: nn.Module,
    meta_path: str,
    img_dir: str,
    save_dir: Path,
    img_size: tuple[int, int],
    batch_size: int = 4,
):
    ds = FaceTestDataset(meta_path, img_dir)
    dloader = DataLoader(ds, batch_size=batch_size)

    save_dir.mkdir(parents=True, exist_ok=True)
    latent_h = img_size[0] // 16
    latent_w = img_size[1] // 16
    rng = torch.Generator("cuda").manual_seed(2024)

    img_idx = 0
    for crops, prompts in tqdm(dloader, "Generating images", dynamic_ncols=True):
        t5_embeds = t5(prompts)
        clip_embeds = clip(prompts)
        face_embeds = adaface.forward_features(crops.cuda()).flatten(-2).transpose(1, 2)
        with torch.autocast("cuda", torch.bfloat16):
            face_embeds = face_redux(face_embeds.bfloat16())
        embeds = torch.cat([t5_embeds, face_embeds], dim=1)

        noise = torch.randn(crops.shape[0], latent_h * latent_w, 64, device="cuda", dtype=torch.bfloat16, generator=rng)
        latents = flux_generate(flux, embeds, clip_embeds, img_size, noise)
        imgs = flux_decode(ae, latents, (latent_h, latent_w))
        imgs = imgs.cpu().permute(0, 2, 3, 1).contiguous().numpy()

        for img in imgs:
            Image.fromarray(img).save(save_dir / f"{img_idx:04d}.webp", lossless=True)
            img_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def parse_img_size(img_size: str):
        out = [int(x) for x in img_size.split(",")]
        if len(out) == 1:
            out = [out[0], out[0]]
        assert len(out) == 2
        return tuple(out)

    parser.add_argument("--img_sizes", type=parse_img_size, nargs="+", default=(512, 512))
    parser.add_argument("--time_sampler", default="uniform()")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", type=json.loads, required=True)

    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--test_meta_path", required=True)
    parser.add_argument("--test_img_dir", required=True)
    parser.add_argument("--test_img_size", type=parse_img_size, default=(512, 512))

    parser.add_argument("--log_dir", type=Path, default="logs")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--run_name", default="Debug")
    args = parser.parse_args()

    assert args.batch_size % args.gradient_accumulation == 0
    batch_size = args.batch_size // args.gradient_accumulation
    torch._dynamo.config.cache_size_limit = 10_000

    wandb.init(project="Flux face", name=args.run_name, dir="/tmp")

    ds_list = []
    for ds_config in args.datasets:
        ds_list.append(FaceTrainDataset(ds_config["meta_path"], ds_config["img_dir"], args.img_sizes, args.batch_size))
    ds = AlternateDataset(ds_list)

    dloader = DataLoader(
        ds,
        batch_size=None,
        num_workers=4,
        pin_memory=True,
    )
    dloader_iter = iter(dloader)

    flux = load_flux().eval().requires_grad_(False)
    flux_offloader = PerLayerOffloadWithBackward(flux).cuda()
    ae = load_flux_autoencoder().eval().cuda()

    # activation checkpointing
    for layer in list(flux.double_blocks) + list(flux.single_blocks):
        layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        if args.compile:
            layer.forward = torch.compile(layer.forward)

    t5 = load_t5().eval()
    PerLayerOffloadCUDAStream(t5).cuda()
    clip = load_clip_text().eval().cuda()

    # same MLP as redux
    adaface = load_adaface_ir101().cuda().eval()
    face_redux = nn.Sequential(
        nn.Linear(512, 4096 * 3),
        nn.SiLU(),
        nn.Linear(4096 * 3, 4096),
    ).cuda()
    # nn.init.zeros_(face_redux[-1].weight)
    # nn.init.zeros_(face_redux[-1].bias)

    optim = torch.optim.AdamW(face_redux.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)

    time_sampler = eval(args.time_sampler, dict(uniform=uniform, logit_normal=logit_normal))

    log_dir = args.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    img_dir = log_dir / "images"

    # inference before any training
    step = 0
    save_images(
        flux,
        ae,
        t5,
        clip,
        adaface,
        face_redux,
        args.test_meta_path,
        args.test_img_dir,
        img_dir / f"step{step:06d}",
        args.test_img_size,
    )

    pbar = tqdm(total=args.num_steps, dynamic_ncols=True)
    torch.cuda.reset_peak_memory_stats()
    time0 = time.perf_counter()
    while step < args.num_steps:
        for _ in range(args.gradient_accumulation):
            imgs, crops, prompts = next(dloader_iter)
            latents = flux_encode(ae, imgs.cuda(), sample=True)

            try:
                t5_embeds = t5(prompts)
                clip_embeds = clip(prompts)
            except:
                print(prompts)
                raise
            # TODO: drop t5_embeds, similar to CFG training
            with torch.no_grad():
                face_embeds = adaface.forward_features(crops.cuda()).flatten(-2).transpose(1, 2)
            with torch.autocast("cuda", torch.bfloat16):
                face_embeds = face_redux(face_embeds.bfloat16())
            embeds = torch.cat([t5_embeds, face_embeds], dim=1)

            loss = compute_loss(flux, latents, embeds, clip_embeds, imgs.shape[2:], time_sampler)
            with flux_offloader.disable_forward_hook():
                loss.backward()

        if step % args.log_interval == 0:
            grad_norm = sum(p.grad.square().sum() for p in face_redux.parameters()) ** 0.5
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
            )
            wandb.log(log_dict, step=step)
            time0 = time1

        if step % args.eval_interval == 0:
            torch.save(face_redux.state_dict(), ckpt_dir / f"step{step:06d}.pth")

            # infer with test prompts
            save_images(
                flux,
                ae,
                t5,
                clip,
                adaface,
                face_redux,
                args.test_meta_path,
                args.test_img_dir,
                img_dir / f"step{step:06d}",
                args.test_img_size,
            )

    wandb.finish()
