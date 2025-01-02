import argparse
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # improve memory usage

import json
from functools import partial

import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate, get_worker_info
from tqdm import tqdm

from flux_infer import FluxTextEmbedder, flux_euler_generate, flux_timesteps
from modelling import (
    SD3,
    AutoEncoder,
    Flux,
    IResNet,
    load_adaface_ir101,
    load_flux,
    load_flux_autoencoder,
    load_sd3_5,
    load_sd3_autoencoder,
)
from modelling.face_embedder import arcface_crop
from offload import PerLayerOffloadWithBackward
from sd3_infer import SD3TextEmbedder, sd3_euler_generate, sd3_timesteps
from time_sampler import LogitNormal, Uniform
from train_qlora import bucket_resize_crop, compute_loss, parse_img_size

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FaceTrainDataset(IterableDataset):
    def __init__(self, meta_path: str, img_dir: str, img_sizes: list[tuple[int, int]], batch_size: int) -> None:
        self.img_dir = Path(img_dir)
        self.img_sizes = img_sizes
        self.batch_size = batch_size

        if str(meta_path).endswith(".tsv"):
            meta = pd.read_csv(meta_path, sep="\t")
        else:
            meta = pd.read_csv(meta_path)
        self.filenames = meta["filename"].tolist()
        self.kpts = np.array([x[0] for x in meta["keypoints"].map(json.loads)])
        self.prompts = meta["prompt"].tolist()

        self.shuffle_rng = torch.Generator()
        self.shuffle_rng.seed()

    def __iter__(self):
        buckets = [[] for _ in range(len(self.img_sizes))]

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

                # torchvision 0.20 has memory leak when decoding webp. use PIL for robustness
                img_path = self.img_dir / self.filenames[idx]
                img_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")

                kpt = self.kpts[idx].copy()
                kpt[..., 0] *= img_pil.width
                kpt[..., 1] *= img_pil.height
                img_np = np.array(img_pil)
                crop = arcface_crop(img_np, kpt)
                crop = torch.from_numpy(crop).permute(2, 0, 1)
                crop = F.interpolate(crop.unsqueeze(0), 112, mode="bicubic", antialias=True).squeeze(0)
                crop = crop.float() / 127.5 - 1

                img_pt, bucket_idx = bucket_resize_crop(img_pil, self.img_sizes)
                buckets[bucket_idx].append((img_pt, crop, self.prompts[idx]))
                if len(buckets[bucket_idx]) == self.batch_size:
                    yield default_collate(buckets[bucket_idx])
                    buckets[bucket_idx] = []


class FaceTestDataset(Dataset):
    def __init__(self, meta_path: str, img_dir: str) -> None:
        self.img_dir = Path(img_dir)

        meta = pd.read_csv(meta_path)
        self.filenames = meta["filename"].tolist()
        self.kpts = np.array([x[0] for x in meta["keypoints"].map(json.loads)])
        self.prompts = meta["prompt"].tolist()

    def __getitem__(self, idx: int):
        img_path = self.img_dir / self.filenames[idx]
        img_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")

        kpt = self.kpts[idx].copy()
        kpt[..., 0] *= img_pil.width
        kpt[..., 1] *= img_pil.height
        img_np = np.array(img_pil)
        crop = arcface_crop(img_np, kpt)
        crop = torch.from_numpy(crop).permute(2, 0, 1)
        crop = F.interpolate(crop.unsqueeze(0), 112, mode="bicubic", antialias=True).squeeze(0)
        crop = crop.float() / 127.5 - 1

        return crop, self.prompts[idx]

    def __len__(self):
        return len(self.filenames)


@torch.no_grad()
def save_images(
    model: Flux | SD3,
    ae: AutoEncoder,
    text_embedder: FluxTextEmbedder | SD3TextEmbedder,
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
    rng = torch.Generator("cuda").manual_seed(2024)

    img_idx = 0
    for crops, prompts in tqdm(dloader, "Generating images", dynamic_ncols=True):
        embeds, vecs = text_embedder(prompts)
        face_embeds = adaface.forward_features(crops.cuda()).flatten(-2).transpose(1, 2)
        with torch.autocast("cuda", torch.bfloat16):
            face_embeds = face_redux(face_embeds.bfloat16())
        embeds = torch.cat([embeds, face_embeds], dim=1)

        shape = crops.shape[0], 16, img_size[0] // 8, img_size[1] // 8
        noise = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)

        if isinstance(model, Flux):
            timesteps = flux_timesteps(img_seq_len=shape[2] * shape[3] // 4)
            latents = flux_euler_generate(model, noise, timesteps, embeds, vecs)
        elif isinstance(model, SD3):
            neg_embeds, neg_vecs = text_embedder([""] * shape[0])
            neg_embeds = torch.cat([neg_embeds, torch.zeros_like(face_embeds)], dim=1)
            timesteps = sd3_timesteps()
            latents = sd3_euler_generate(model, noise, timesteps, embeds, vecs, neg_embeds, neg_vecs)
        else:
            raise RuntimeError

        imgs = ae.decode(latents, uint8=True).cpu().permute(0, 2, 3, 1).contiguous().numpy()
        for img in imgs:
            Image.fromarray(img).save(save_dir / f"{img_idx:04d}.webp", lossless=True)
            img_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["flux-dev", "sd3.5-medium"], default="flux-dev")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--img_sizes", type=parse_img_size, nargs="+", default=(512, 512))
    parser.add_argument("--time_sampler", default="Uniform()")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ds", type=json.loads, required=True)
    parser.add_argument("--distill_ds", type=json.loads, required=True)

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

    assert args.batch_size % (args.gradient_accumulation * 2) == 0
    batch_size = args.batch_size // args.gradient_accumulation
    torch._dynamo.config.cache_size_limit = 10_000

    wandb.init(project="Flux face", config=vars(args), name=args.run_name, dir="/tmp")

    def create_dloader(ds_config: dict):
        ds = FaceTrainDataset(ds_config["meta_path"], ds_config["img_dir"], args.img_sizes, batch_size // 2)
        dloader = DataLoader(ds, batch_size=None, num_workers=args.num_workers, pin_memory=True)
        return iter(dloader)

    train_dloader = create_dloader(args.train_ds)
    distill_dloader = create_dloader(args.distill_ds)

    if args.model == "flux-dev":
        model = load_flux()
        layers = list(model.double_blocks) + list(model.single_blocks)

        ae = load_flux_autoencoder().eval().cuda()
        text_embedder = FluxTextEmbedder(offload_t5=True).cuda()

    elif args.model == "sd3.5-medium":
        model = load_sd3_5()
        layers = list(model.joint_blocks)

        ae = load_sd3_autoencoder().eval().cuda()
        text_embedder = SD3TextEmbedder(offload_t5=True, offload_clip_g=True).cuda()

    else:
        raise ValueError(f"Unsupported {args.model=}")

    model.bfloat16().eval().requires_grad_(False)
    offloader = PerLayerOffloadWithBackward(model, enable=args.offload).cuda()

    # activation checkpointing
    for layer in layers:
        layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        if args.compile:  # might not be optimal to compile this way, but required for offloading
            layer.forward = torch.compile(layer.forward)

    # same MLP as redux
    adaface = load_adaface_ir101().cuda().eval()
    face_redux = nn.Sequential(
        nn.Linear(512, 4096 * 3),
        nn.SiLU(),
        nn.Linear(4096 * 3, 4096),
    ).cuda()
    # nn.init.zeros_(face_redux[-1].weight)
    # nn.init.zeros_(face_redux[-1].bias)

    # TODO: try other optims e.g. muon
    optim = torch.optim.AdamW(face_redux.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)

    time_sampler = eval(args.time_sampler, dict(Uniform=Uniform, LogitNormal=LogitNormal))

    log_dir = args.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    img_dir = log_dir / "images"

    # inference before any training
    step = 0
    save_images(
        model,
        ae,
        text_embedder,
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
            imgs, crops, prompts = next(train_dloader)
            with torch.no_grad():
                latents = ae.encode(imgs.cuda(), sample=True)
                embeds, vecs = text_embedder(prompts)
                face_embeds = adaface.forward_features(crops.cuda()).flatten(-2).transpose(1, 2)
            with torch.autocast("cuda", torch.bfloat16):
                face_embeds = face_redux(face_embeds.bfloat16())

            # randomly drop t5_embeds, similar to CFG training
            mask = torch.empty(embeds.shape[0], 1, 1, device="cuda").bernoulli_(p=0.5)
            embeds *= mask
            embeds = torch.cat([embeds, face_embeds], dim=1)
            train_loss = compute_loss(model, latents, embeds, vecs, time_sampler)

            # distill loss. student
            imgs, crops, prompts = next(distill_dloader)
            with torch.no_grad():
                latents = ae.encode(imgs.cuda(), sample=True)
                embeds, vecs = text_embedder(prompts)
                face_embeds = adaface.forward_features(crops.cuda()).flatten(-2).transpose(1, 2)
            with torch.autocast("cuda", torch.bfloat16):
                face_embeds = face_redux(face_embeds.bfloat16())

            bsize = latents.shape[0]
            t_vec = time_sampler(bsize, device=latents.device).bfloat16()
            noise = torch.randn_like(latents)
            interpolate = latents.lerp(noise, t_vec.view(-1, 1, 1, 1))

            # TODO: add guidance
            teacher_outputs = model(interpolate, t_vec, embeds, vecs)
            student_outputs = model(interpolate, t_vec, torch.cat([embeds, face_embeds], dim=1), vecs)
            distill_loss = F.mse_loss(student_outputs.float(), teacher_outputs.float())

            loss = (train_loss + distill_loss) * 0.5
            with offloader.disable_forward_hook():
                loss.backward()

        if step % args.log_interval == 0:
            grad_norm = sum(p.grad.square().sum() for p in face_redux.parameters()) ** 0.5
            log_dict = dict(
                lr=args.lr,
                loss=loss.item(),
                train_loss=train_loss.item(),
                distill_loss=distill_loss.item(),
                grad_norm=grad_norm,
            )
            wandb.log(log_dict, step)

        # TODO: LR schedule
        optim.step()
        optim.zero_grad()
        step += 1
        pbar.update()

        if step % args.log_interval == 0:
            memory_info = psutil.virtual_memory()
            time1 = time.perf_counter()
            log_dict = dict(
                imgs_per_second=args.batch_size * args.log_interval / (time1 - time0),
                max_memory_allocated=torch.cuda.max_memory_allocated() / 1e9,
                cpu_mem_active=memory_info.active / 1e9,
                cpu_mem_used=memory_info.used / 1e9,
            )
            wandb.log(log_dict, step=step)
            time0 = time1

        if step % args.eval_interval == 0:
            torch.save(face_redux.state_dict(), ckpt_dir / f"step{step:06d}.pth")

            # infer with test prompts
            save_images(
                model,
                ae,
                text_embedder,
                adaface,
                face_redux,
                args.test_meta_path,
                args.test_img_dir,
                img_dir / f"step{step:06d}",
                args.test_img_size,
            )

    wandb.finish()
