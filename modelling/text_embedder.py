# https://github.com/black-forest-labs/flux/blob/7e14a05ed7280f7a34ece612f7324fcc2ec9efbb/src/flux/modules/conditioner.py#L5

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection, PreTrainedModel, T5EncoderModel


class TextEmbedder(nn.Module):
    def __init__(
        self,
        model_id: str,
        max_length: int,
        model_class: type[PreTrainedModel],
        output_key: str | int | list[str | int],
        tokenizer_id: str | None = None,
        subfolder: str = "",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
        self.max_length = max_length  # allow overrides
        self.output_key = output_key
        self.model = model_class.from_pretrained(model_id, torch_dtype=dtype, subfolder=subfolder)
        self.model.eval().requires_grad_(False)

    @staticmethod
    def unwrap_output(out, key: str | int | list[str | int]):
        if isinstance(key, str):
            return out[key]
        elif isinstance(key, int):
            return out.hidden_states[key]
        else:
            return [TextEmbedder.unwrap_output(out, k) for k in key]

    @torch.no_grad()
    def forward(self, texts: list[str]) -> Tensor:
        # following Flux, we will always pad or truncate to max_length
        tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]
        out = self.model(tokens.to(self.model.device), output_hidden_states=True)
        return self.unwrap_output(out, self.output_key)


def load_t5(model_id: str = "mcmonkey/google_t5-v1_1-xxl_encoderonly", max_length: int = 512):
    return TextEmbedder(model_id, max_length, T5EncoderModel, "last_hidden_state", dtype=torch.bfloat16)


def load_clip_l(output_key: str | int | list[str | int] = "pooler_output"):
    # NOTE: OpenAI CLIP weights use FP16 for Linear, and FP32 for the rest (Embeddings, LayerNorm, pos_embed, ...)
    # but Flux loads it in BF16. TODO: investigate FP16 vs BF16
    # NOTE: FLUX only uses pooler_output. SD3/3.5 use pooler_output and hidden_states[-2]
    # NOTE: use weights from Stability AI to reduce download time, since it is FP16 and doesn't include ViT's weights.
    return TextEmbedder(
        "stabilityai/stable-diffusion-3.5-large",
        max_length=77,
        model_class=CLIPTextModel,
        output_key=output_key,
        tokenizer_id="openai/clip-vit-large-patch14",
        subfolder="text_encoder",
        dtype=torch.float16,
    )


def load_openclip_bigg():
    # not sure which dtype was used for training
    # NOTE: use weights from Stability AI to reduce download time, since it is FP16 and doesn't include ViT's weights.
    # NOTE: SD3 uses this model with the projection layer
    return TextEmbedder(
        "stabilityai/stable-diffusion-3.5-large",
        max_length=77,
        model_class=CLIPTextModelWithProjection,
        output_key=["text_embeds", -2],
        tokenizer_id="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        subfolder="text_encoder_2",
        dtype=torch.float16,
    )
