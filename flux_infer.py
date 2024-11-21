import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from modelling import AutoEncoder, Flux, load_autoencoder, load_clip_text, load_flux, load_t5
from offload import PerLayerOffloadCUDAStream


class FluxGenerator:
    def __init__(self, offload_t5: bool = False, offload_flux: bool = False) -> None:
        super().__init__()
        self.ae = load_autoencoder(
            "black-forest-labs/FLUX.1-schnell",
            "ae.safetensors",
            scale_factor=0.3611,
            shift_factor=0.1159,
        )  #  168 MB in BF16
        self.flux = load_flux()  # 23.8 GB in BF16
        self.t5 = load_t5()  #  9.5 GB in BF16
        self.clip = load_clip_text()  #  246 MB in BF16

        # autoencoder and clip are small, don't need to offload
        self.t5_offloader = PerLayerOffloadCUDAStream(self.t5, enable=offload_t5)
        self.flux_offloader = PerLayerOffloadCUDAStream(self.flux, enable=offload_flux)

    def cpu(self):
        self.ae.cpu()
        self.clip.cpu()
        self.t5_offloader.cpu()  # this only move some params
        self.flux_offloader.cpu()
        return self

    def cuda(self):
        self.ae.cuda()
        self.clip.cuda()
        self.t5_offloader.cuda()  # this only move some params
        self.flux_offloader.cuda()
        return self

    @torch.no_grad()
    def generate(
        self,
        t5_prompt: str,
        clip_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        guidance: float = 4.0,
        num_steps: int = 50,
        pbar: bool = False,
        compile: bool = False,
    ):
        if clip_prompt is None:
            clip_prompt = t5_prompt

        # TODO: support for initial image
        # NOTE: Flux uses pixel unshuffle on latent image before passing it to Flux model,
        # and pixel shuffle on Flux outputs before passing it to AE's decoder.
        # prepare inputs and text conditioning
        latent_h = height // 16
        latent_w = width // 16
        img = torch.randn(1, latent_h * latent_w, 64, device="cuda", dtype=torch.bfloat16)

        img_ids = torch.zeros(latent_h, latent_w, 3)
        img_ids[..., 1] = torch.arange(latent_h)[:, None]
        img_ids[..., 2] = torch.arange(latent_w)[None, :]
        img_ids = img_ids.view(1, latent_h * latent_w, 3).cuda()

        # TODO: might need to restrict txt length to avoid recompilation
        txt = self.t5([t5_prompt]).to(img.device)  # (B, L, 4096)
        txt_ids = torch.zeros(1, txt.shape[1], 3, device=img.device)

        vec = self.clip([t5_prompt])  # (B, 768)

        timesteps = torch.linspace(1, 0, num_steps + 1)
        guidance_vec = torch.full((1,), guidance, device=img.device, dtype=img.dtype)

        for i in tqdm(range(num_steps), disable=not pbar):
            # t_curr and t_prev must be Tensor (cpu is fine) to avoid recompilation
            t_curr = timesteps[i]
            t_prev = timesteps[i + 1]
            torch.compile(flux_denoise_step, disable=not compile)(
                self.flux, img, img_ids, txt, txt_ids, vec, t_curr, t_prev, guidance_vec
            )

        img_u8 = torch.compile(flux_decode, disable=not compile)(self.ae, img, latent_h, latent_w)
        return img_u8


def flux_denoise_step(
    flux: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids,
    vec: Tensor,
    t_curr: Tensor,
    t_prev: Tensor,
    guidance: Tensor,
) -> None:
    # NOTE: t_prev > t_curr
    # NOTE: t_curr and t_prev is FP32
    t_vec = t_curr.to(img.dtype).view(1).cuda()
    v = flux(img, img_ids, txt, txt_ids, t_vec, vec, guidance)
    img += (t_prev - t_curr) * v  # Euler's method


def flux_decode(ae: AutoEncoder, img: Tensor, latent_h: int, latent_w: int) -> Tensor:
    # NOTE: original repo uses FP32 AE + cast latents to FP32 + BF16 autocast
    img = img.transpose(1, 2).unflatten(-1, (latent_h, latent_w))
    img = F.pixel_shuffle(img, 2)  # (B, 64, latent_h, latent_w) -> (B, 16, latent_h * 2, latent_w * 2)
    img = ae.decode(img)
    img_u8 = img.clip(-1, 1).add(1).mul(127.5).to(torch.uint8)
    return img_u8
