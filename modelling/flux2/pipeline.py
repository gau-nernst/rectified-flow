# https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/text_encoder.py

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer


# TODO: might unify with Z-Image
class Flux2KleinTextEmbedder(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, dtype="auto")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

        # TODO: we can truncate Qwen3 to 27 layers
        self.model.eval()
        self.output_indices = [9, 18, 27]

    @torch.no_grad()
    def forward(self, texts: list[str]) -> Tensor:
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

        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

        outputs = [outputs.hidden_states[idx] for idx in self.output_indices]
        return torch.cat(outputs, dim=-1)
