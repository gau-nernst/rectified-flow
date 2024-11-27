# rectified-flow

Supported models:

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

Supported features:

- Training: QLoRA fine-tuning, logit-normal sampler

TODO:
- DreamBooth for subject-based fine-tuning (prior preservation loss).
- INT8 inference and training.
- (maybe) more schedules?

Resources:

- Rectified-Flow paper: https://arxiv.org/abs/2209.03003
- Stable Diffusion 3 paper: https://arxiv.org/abs/2403.03206
- DreamBooth: https://arxiv.org/abs/2208.12242

Notes:

- Finetune with logit-normal sampler seems to be more stable than uniform sampler. The changes are less abrupt. Not necessarily mean the final result is better.
