import argparse
import os
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # improve memory usage

from PIL import Image
from tqdm import tqdm

from flux_infer import FluxGenerator
from sd3_infer import SD3Generator, SkipLayerConfig

if __name__ == "__main__":

    def parse_img_size(size):
        size = tuple(int(x) for x in size.split(","))
        assert len(size) == 2
        return size

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=parse_img_size, default=(1024, 1024))
    args = parser.parse_args()

    args.save_dir.mkdir(exist_ok=True, parents=True)
    all_prompts = [line.rstrip() for line in open(args.prompt, encoding="utf-8")]
    N = len(all_prompts)

    if args.model == "flux-dev":
        gen = FluxGenerator(offload_flux=True, offload_t5=True)
        extra_kwargs = dict()
    elif args.model == "sd3.5-medium":
        gen = SD3Generator(offload_t5=True)
        extra_kwargs = dict(slg_config=SkipLayerConfig(scale=0.2))  # default for sd3.5-medium
    else:
        raise ValueError(f"{args.model=}")
    gen.cuda()

    for offset in tqdm(range(0, N, args.batch_size), "Generate", dynamic_ncols=True):
        prompts = all_prompts[offset : offset + args.batch_size]
        imgs = gen.generate(prompts, img_size=args.img_size, compile=True, **extra_kwargs)

        imgs = imgs.permute(0, 2, 3, 1).cpu().contiguous()
        for sub_idx in range(imgs.shape[0]):
            save_path = args.save_dir / f"{offset + sub_idx:06d}.webp"
            Image.fromarray(imgs[sub_idx].numpy()).save(save_path, lossless=True)
