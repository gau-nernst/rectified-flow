import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Linear):
    def init_adapter(self, rank: int = 8, alpha: float = 8.0, dtype: torch.dtype = torch.float32) -> None:
        """By default, use FP32 for LoRA weights."""
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scale = self.alpha / self.rank

        if rank > 0:
            dtype = dtype or self.weight.dtype
            self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype))
            self.lora_b = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype))

            nn.init.kaiming_normal_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)

    def extra_repr(self):
        return f"{super().extra_repr()}, rank={self.rank}, alpha={self.alpha}"

    def forward(self, x: Tensor):
        out = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            out = out + x @ self.lora_a.T @ self.lora_b.T * self.scale
        return out

    @staticmethod
    def convert_model(model: nn.Module, rank: int = 8, alpha: float = 8.0, dtype: torch.dtype = torch.float32):
        if rank == 0:
            return model

        if type(model) == nn.Linear:  # exact match, no subclass
            model.__class__ = LoRALinear
            model.init_adapter(rank=rank, alpha=alpha, dtype=dtype)
        else:
            for child in model.children():
                LoRALinear.convert_model(child, rank=rank, alpha=alpha, dtype=dtype)
        return model
