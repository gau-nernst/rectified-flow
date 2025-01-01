# rectified-flow

Supported models:

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev)
- SD3.5 [medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)

Supported features:

- Training: QLoRA fine-tuning (NF4), logit-normal sampler
- [RF-inversion](https://arxiv.org/abs/2410.10792) (FLUX-Redux is simpler and superior)
- INT8 matmul (2x inference speedup for consumer cards)
- Aspect ratio bucketing for training with multiple aspect ratios.
- Single-GPU model offloading for inference and training.
- [DPM-Solver++(2M)](https://arxiv.org/abs/2211.01095) for SD3.5 (TODO: support FLUX?)
- Skip-layer guidance for SD3.5

TODO:
- Investigate the impact of T5 embeddings. SD3 paper says it's more important for text rendering.

Resources:

- Rectified-Flow paper: https://arxiv.org/abs/2209.03003
- Stable Diffusion 3 paper: https://arxiv.org/abs/2403.03206
- DreamBooth: https://arxiv.org/abs/2208.12242

Notes:

- Finetune with logit-normal sampler seems to be more stable than uniform sampler. The changes are less abrupt. Not necessarily mean the final result is better.
- If you only have low-res images, you can train in low-res and FLUX can extrapolate to hi-res.
- FLUX-Redux
  - To reduce influence of image guidance, we can just scale the embeddings down e.g. `* 0.2`. Similarly, to increase influence of text guidance, we can scale T5 embeddings up.
  - We can use multiple guidance images (>=2). Just concat the embeddings.
