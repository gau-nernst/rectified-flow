import argparse
import json
import logging
import math
import os
import time
from datetime import datetime
from functools import partial
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # improve memory usage

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import wandb
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, IterableDataset, default_collate, get_worker_info
from torchvision.transforms import v2
from tqdm import tqdm

from flux_infer import FluxTextEmbedder, flux_euler_generate, flux_timesteps
from modelling import (
    SD3,
    AutoEncoder,
    Flux,
    LoRALinear,
    load_flex_alpha,
    load_flux,
    load_flux_autoencoder,
    load_sd3_5,
    load_sd3_autoencoder,
)
from offload import PerLayerOffloadWithBackward
from sd3_infer import SD3TextEmbedder, sd3_euler_generate, sd3_timesteps
from time_sampler import LogitNormal, TimeSampler, Uniform

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def bucket_resize_crop(img_pil: Image.Image, img_sizes: list[tuple[int, int]]):
    ratio = img_pil.width / img_pil.height
    ratios = [width / height for height, width in img_sizes]
    _, bucket_idx = min((abs(math.log(ratio / x)), _i) for _i, x in enumerate(ratios))
    height, width = img_sizes[bucket_idx]

    # resize while maintaining aspect ratio, then make a random crop spanning 1 dimension.
    scale = max(height / img_pil.height, width / img_pil.width)
    target_height = round(img_pil.height * scale)
    target_width = round(img_pil.width * scale)
    img_pil = img_pil.resize((target_width, target_height), Image.Resampling.BICUBIC)
    img_pt = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)
    img_pt = v2.RandomCrop((height, width))(img_pt)
    return img_pt, bucket_idx


class ImageDataset(IterableDataset):
    def __init__(
        self,
        meta_path: str,
        img_dir: str,
        img_sizes: list[tuple[int, int]],
        batch_size: int,
        drop_tag_rate: float = 0.0,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.img_sizes = img_sizes
        self.batch_size = batch_size
        self.drop_tag_rate = drop_tag_rate

        if str(meta_path).endswith(".tsv"):
            meta = pd.read_csv(meta_path, sep="\t")
        else:
            meta = pd.read_csv(meta_path)
        self.filenames = meta["filename"].tolist()
        self.prompts = meta["prompt"].tolist()

        self.shuffle_rng = torch.Generator()
        self.shuffle_rng.seed()

    def __iter__(self):
        buckets = [[] for _ in range(len(self.img_sizes))]
        counter = -1

        # NOTE: not handle distributed training for now
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
                img_pt, bucket_idx = bucket_resize_crop(img_pil, self.img_sizes)

                prompt = self.prompts[idx]
                if self.drop_tag_rate > 0.0 and torch.rand(()).item() < self.drop_tag_rate:
                    SEPARATOR = ", "
                    tags = prompt.split(SEPARATOR)
                    if len(tags) > 5:
                        tags.pop(torch.randint(len(tags), size=()).item())
                        prompt = SEPARATOR.join(tags)

                buckets[bucket_idx].append((img_pt, prompt))
                if len(buckets[bucket_idx]) == self.batch_size:
                    yield default_collate(buckets[bucket_idx])
                    buckets[bucket_idx] = []


@torch.no_grad()
def save_images(
    model: Flux | SD3,
    ae: AutoEncoder,
    text_embedder: FluxTextEmbedder | SD3TextEmbedder,
    prompt_path: str,
    save_dir: Path,
    img_size: tuple[int, int],
    batch_size: int = 4,
):
    prompts = [line.rstrip() for line in open(prompt_path, encoding="utf-8")]
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator("cuda").manual_seed(2024)

    neg_embeds, neg_vecs = text_embedder([""])

    for offset in tqdm(range(0, len(prompts), batch_size), "Generating images", dynamic_ncols=True):
        embeds, vecs = text_embedder(prompts[offset : offset + batch_size])

        shape = (embeds.shape[0], 16, img_size[0] // 8, img_size[1] // 8)
        noise = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)

        if isinstance(model, Flux):
            timesteps = flux_timesteps(img_seq_len=shape[2] * shape[3] // 4)

            if len(model.double_blocks) == 19:  # flux-dev
                guidance_list = [(3.5, 1.0), (1.0, 3.5), (2.0, 2.0)]
            elif len(model.double_blocks) == 8:  # flex-alpha
                guidance_list = [(3.5, 1.0), (None, 3.5), (2.0, 2.0)]
            else:
                raise ValueError

            for guidance, cfg_scale in guidance_list:
                latents = flux_euler_generate(
                    model,
                    noise.clone(),
                    timesteps,
                    embeds,
                    vecs,
                    neg_embeds.expand_as(embeds),
                    neg_vecs.expand_as(vecs),
                    guidance=guidance,
                    cfg_scale=cfg_scale,
                )
                imgs = ae.decode(latents, uint8=True).permute(0, 2, 3, 1).cpu()
                for img_idx in range(imgs.shape[0]):
                    save_path = save_dir / f"{offset + img_idx:04d}_{guidance}-{cfg_scale}.webp"
                    Image.fromarray(imgs[img_idx].numpy()).save(save_path, lossless=True)

        elif isinstance(model, SD3):
            timesteps = sd3_timesteps()
            latents = sd3_euler_generate(
                model,
                noise,
                timesteps,
                embeds,
                vecs,
                neg_embeds.expand_as(embeds),
                neg_vecs.expand_as(vecs),
            )
            imgs = ae.decode(latents, uint8=True).permute(0, 2, 3, 1).cpu()
            for img_idx in range(imgs.shape[0]):
                save_path = save_dir / f"{offset + img_idx:04d}.webp"
                Image.fromarray(imgs[img_idx].numpy()).save(save_path, lossless=True)


def compute_loss(
    model: Flux | SD3,
    latents: Tensor,
    embeds: Tensor,
    vecs: Tensor,
    time_sampler: TimeSampler,
    model_kwargs: dict,
) -> Tensor:
    bsize = latents.shape[0]
    t_vec = time_sampler(bsize, device=latents.device).bfloat16()
    noise = torch.randn_like(latents)
    interpolate = latents.lerp(noise, t_vec.view(-1, 1, 1, 1))

    v = model(interpolate, t_vec, embeds, vecs, **model_kwargs)

    # rectified flow loss. predict velocity from latents (t=0) to noise (t=1).
    return F.mse_loss(noise.float() - latents.float(), v.float())


def setup_model(model_name: str, offload: bool, lora: int):
    if model_name in ("flux-dev", "flex-alpha"):
        model = {
            "flux-dev": load_flux,
            "flex-alpha": load_flex_alpha,
        }[model_name]()
        layers = list(model.double_blocks) + list(model.single_blocks)

        ae = load_flux_autoencoder()
        text_embedder = FluxTextEmbedder(offload_t5=True)

    elif model_name in ("sd3.5-medium", "sd3.5-large"):
        model = load_sd3_5(model_name.removeprefix("sd3.5-"))
        layers = list(model.joint_blocks)

        ae = load_sd3_autoencoder()
        text_embedder = SD3TextEmbedder(offload_t5=True)

    else:
        raise ValueError(f"Unsupported {model_name=}")

    model.bfloat16().eval().requires_grad_(False)
    offloader = PerLayerOffloadWithBackward(model, enable=offload).cuda()
    ae.eval().cuda()
    text_embedder.cuda()

    for layer in layers:
        if lora > 0:
            LoRALinear.add_lora(layer, rank=lora, device="cuda")
        layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        if args.compile:  # might not be optimal to compile this way, but required for offloading
            layer.forward = torch.compile(layer.forward)

    return model, offloader, ae, text_embedder


def parse_img_size(img_size: str):
    out = [int(x) for x in img_size.split(",")]
    if len(out) == 1:
        out = [out[0], out[0]]
    assert len(out) == 2
    assert out[0] % 16 == 0 and out[1] % 16 == 0, out
    return tuple(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="flux-dev")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--img_sizes", nargs="+", type=parse_img_size, default=(512, 512))
    parser.add_argument("--lora", type=int, default=8)
    parser.add_argument("--time_sampler", default="LogitNormal()")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ds", type=json.loads, required=True)
    parser.add_argument("--distill_ds", type=json.loads)
    parser.add_argument("--drop_tag_rate", type=float, default=0.0)

    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--test_img_size", type=parse_img_size, default=(512, 512))
    parser.add_argument("--test_prompt_path", required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--run_name", default="Debug")
    parser.add_argument("--resume")
    args = parser.parse_args()

    assert args.batch_size % args.gradient_accumulation == 0
    batch_size = args.batch_size // args.gradient_accumulation
    torch._dynamo.config.cache_size_limit = 1000
    torch._dynamo.config.accumulated_cache_size_limit = 1000

    wandb.init(project="Flux finetune", name=args.run_name, dir="/tmp")

    def create_dloader(ds_config: dict, batch_size: int):
        ds = ImageDataset(ds_config["meta_path"], ds_config["img_dir"], args.img_sizes, batch_size, args.drop_tag_rate)
        dloader = DataLoader(ds, batch_size=None, num_workers=args.num_workers, pin_memory=True)
        return iter(dloader)

    if args.distill_ds is not None:
        assert batch_size % 2 == 0
        train_dloader = create_dloader(args.train_ds, batch_size // 2)
        distill_dloader = create_dloader(args.distill_ds, batch_size // 2)
    else:
        train_dloader = create_dloader(args.train_ds, batch_size)
        distill_dloader = None

    model, offloader, ae, text_embedder = setup_model(args.model, args.offload, args.lora)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    logger.info(model)
    time_sampler = eval(args.time_sampler, dict(Uniform=Uniform, LogitNormal=LogitNormal))

    log_dir = args.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    img_dir = log_dir / "images"

    step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)

        if "model" in ckpt:  # training checkpoint
            print("Load model: ", model.load_state_dict(ckpt["model"], strict=False))
            optim.load_state_dict(ckpt["optim"])
            step = ckpt["step"]

        else:  # model only
            print("Load model: ", model.load_state_dict(ckpt, strict=False))

    # inference before any training
    if step == 0:
        save_images(model, ae, text_embedder, args.test_prompt_path, img_dir / f"step{step:06d}", args.test_img_size)

    pbar = tqdm(initial=step, total=args.num_steps, dynamic_ncols=True)
    torch.cuda.reset_peak_memory_stats()
    time0 = time.perf_counter()
    while step < args.num_steps:
        for _ in range(args.gradient_accumulation):
            imgs, prompts = next(train_dloader)

            if distill_dloader is not None:
                distill_imgs, distill_prompts = next(distill_dloader)
                imgs = torch.cat([imgs, distill_imgs], dim=0)
                prompts = prompts + distill_prompts

            with torch.no_grad():
                latents = ae.encode(imgs.cuda(), sample=True)

            if args.model == "flux-dev":
                # finetune at guidance=1.0 is better than at 3.5
                model_kwargs = dict(guidance=torch.full((imgs.shape[0],), 1.0, device="cuda", dtype=torch.bfloat16))
            else:
                model_kwargs = dict()

            loss = compute_loss(model, latents, *text_embedder(prompts), time_sampler, model_kwargs)
            with offloader.disable_forward_hook():
                loss.backward()

        if step % args.log_interval == 0:
            grad_norm = sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None) ** 0.5
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
            # only save LoRA weights. Flux doesn't have buffers.
            ckpt = dict(
                model={name: p.detach().bfloat16() for name, p in model.named_parameters() if p.requires_grad},
                optim=optim.state_dict(),
                step=step,
            )
            torch.save(ckpt, ckpt_dir / f"step{step:06d}.pth")

            # infer with test prompts
            save_images(
                model, ae, text_embedder, args.test_prompt_path, img_dir / f"step{step:06d}", args.test_img_size
            )

    wandb.finish()
