# https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/text_encoder.py

import math

import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, Qwen3Model

from ..autoencoder import load_autoencoder
from ..offload import PerLayerOffloadCUDAStream
from ..solvers import get_solver
from .model import Flux2, load_flux2


# TODO: might unify with Z-Image
class Flux2Qwen3TextEncoder(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = Qwen3Model.from_pretrained(model_id, dtype="auto")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

        # TODO: we can truncate Qwen3 to 27 layers
        self.model.eval()
        self.output_indices = [9, 18, 27]

    @torch.no_grad()
    def forward(self, texts: str | list[str]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]

        texts = [
            self.tokenizer.apply_chat_template(
                [dict(role="user", content=txt)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for txt in texts
        ]
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # TODO: use varlen
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

        outputs = [outputs.hidden_states[idx] for idx in self.output_indices]
        return torch.cat(outputs, dim=-1)


class Flux2Pipeline:
    DEV_DEFAULTS = dict(guidance=4.0, cfg_scale=1.0, num_steps=50)
    KLEIN_DEFAULTS = dict(guidance=None, cfg_scale=1.0, num_steps=4)
    KLEIN_BASE_DEFAULTS = dict(guidance=None, cfg_scale=4.0, num_steps=50)

    def __init__(
        self,
        flux: Flux2 | None = None,
        text_encoder_id: str = "Qwen/Qwen3-4B-FP8",
        offload_flux: bool = False,
    ) -> None:
        self.flux = flux or load_flux2()
        self.ae = load_autoencoder("flux2").bfloat16()
        self.text_encoder = Flux2Qwen3TextEncoder(text_encoder_id)

        self.flux_offloader = PerLayerOffloadCUDAStream(self.flux, enable=offload_flux)

    def cpu(self):
        for m in (self.ae, self.text_encoder, self.flux_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.ae, self.text_encoder, self.flux_offloader):
            m.cuda()
        return self

    # default is for klein
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str | list[str] = "",
        img_size: int | tuple[int, int] = 512,
        guidance: Tensor | float | None = None,
        cfg_scale: float = 1.0,
        num_steps: int = 4,
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
    ) -> Tensor:
        txt = self.text_encoder(prompt)

        if cfg_scale != 1.0:
            neg_txt = self.text_encoder(neg_prompt)
            if neg_txt.shape[0] == 1:
                neg_txt = neg_txt.expand(txt.shape[0], -1, -1)
        else:
            neg_txt = None

        bsize = txt.shape[0]
        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size

        ae = self.ae
        shape = (bsize, self.ae.cfg.z_dim * 4, height // ae.downsample // 2, width // ae.downsample // 2)
        device = ae.encoder.conv_in.weight.device

        # keep latents in FP32 for accurate .lerp()
        rng = torch.Generator(device).manual_seed(seed) if seed is not None else None
        noise = torch.randn(shape, device=device, dtype=torch.float32, generator=rng)

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
        timesteps = flux2_time_shift(timesteps, num_steps, shape[2] * shape[3])
        timesteps = timesteps.tolist()

        latents = flux2_generate(
            flux=self.flux,
            latents=noise,
            timesteps=timesteps,
            txt=txt,
            neg_txt=neg_txt,
            guidance=guidance,
            cfg_scale=cfg_scale,
            solver=solver,
            pbar=pbar,
        )
        return self.ae.decode(latents, uint8=True)


# https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/sampling.py#L244
def flux2_time_shift(timesteps: Tensor | float, num_steps: int, img_seq_len: int):
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    m_10 = a1 * img_seq_len + b1
    m_200 = a2 * img_seq_len + b2

    if img_seq_len > 4300:
        mu = m_200

    else:
        a = (m_200 - m_10) / (200 - 10)
        b = m_200 - 200 * a
        mu = a * num_steps + b

    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + 1 / timesteps - 1)


@torch.no_grad()
def flux2_generate(
    flux: Flux2,
    latents: Tensor,
    timesteps: list[float],
    txt: Tensor,
    neg_txt: Tensor | None = None,
    guidance: Tensor | float | None = 3.5,
    cfg_scale: float = 1.0,
    solver: str = "euler",
    pbar: bool = False,
) -> Tensor:
    B, _, H, W = latents.shape

    if isinstance(guidance, (int, float)):
        guidance = torch.full((B,), guidance, device=latents.device)

    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)

    img_rope = flux.make_img_rope(H, W)
    txt_rope = flux.make_txt_rope(txt.shape[1])
    rope = torch.cat([img_rope, txt_rope], dim=0)

    latents = latents.flatten(-2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = torch.tensor([timesteps[i]], device="cuda")
        v = flux(latents, t, txt, rope, guidance).float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            assert neg_txt is not None
            neg_v = flux(latents, t, neg_txt, rope, guidance).float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)

    latents = latents.transpose(1, 2).unflatten(-1, (H, W))
    return latents
