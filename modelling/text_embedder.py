# https://github.com/black-forest-labs/flux/blob/7e14a05ed7280f7a34ece612f7324fcc2ec9efbb/src/flux/modules/conditioner.py#L5

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel


class TextEmbedder(nn.Module):
    def __init__(self, model_id: str, max_length: int, model_class, output_key: str, dtype=torch.float32) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=max_length)
        self.model = model_class.from_pretrained(model_id, torch_dtype=dtype)
        self.model.eval().requires_grad_(False)
        self.output_key = output_key

    @torch.no_grad()
    def forward(self, texts: list[str]) -> Tensor:
        out = self.tokenizer(texts, return_tensors="pt", return_attention_mask=False)["input_ids"]
        out = self.model(out.to(self.model.device), output_hidden_states=False)
        return out[self.output_key]


class T5Embedder(TextEmbedder):
    def __init__(self, model_id: str = "mcmonkey/google_t5-v1_1-xxl_encoderonly", max_length: int = 512) -> None:
        super().__init__(model_id, max_length, T5EncoderModel, "last_hidden_state", dtype=torch.bfloat16)


class ClipTextEmbedder(TextEmbedder):
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", max_length: int = 77) -> None:
        # NOTE: OpenAI CLIP was trained with FP16
        super().__init__(model_id, max_length, CLIPTextModel, "pooler_output", dtype=torch.bfloat16)
