# rectified-flow

Supported models:

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

Supported features:

- Training: QLoRA fine-tuning (NF4), logit-normal sampler
- [RF-inversion](https://arxiv.org/abs/2410.10792)
- INT8 matmul

TODO:
- DreamBooth for subject-based fine-tuning (prior preservation loss).
- (maybe) more schedules?
- Investigate the impact of T5 embeddings. SD3 paper says it's more important for text rendering.

Resources:

- Rectified-Flow paper: https://arxiv.org/abs/2209.03003
- Stable Diffusion 3 paper: https://arxiv.org/abs/2403.03206
- DreamBooth: https://arxiv.org/abs/2208.12242

Notes:

- Finetune with logit-normal sampler seems to be more stable than uniform sampler. The changes are less abrupt. Not necessarily mean the final result is better.
