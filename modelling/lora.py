import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Linear):
    def init_lora(self, rank: int = 8, scale: float = 1.0, dtype: torch.dtype = torch.float32) -> None:
        """By default, use FP32 for LoRA weights."""
        assert rank > 0
        self.rank = rank
        # NOTE: there is a problem with torch.compile when self.scale is a Python float.
        # use tensor buffer as a workaround.
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32), persistent=False)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        dtype = dtype or self.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype))
        self.lora_b = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype))

        nn.init.kaiming_normal_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def extra_repr(self):
        return f"{super().extra_repr()}, rank={self.rank}, scale={self.scale.item()}"

    def forward(self, x: Tensor):
        out = F.linear(x, self.weight, self.bias)
        out = out + x @ self.lora_a.to(x.dtype).T @ self.lora_b.to(x.dtype).T * self.scale
        return out

    @staticmethod
    def to_lora(model: nn.Module, rank: int = 8, dtype: torch.dtype = torch.float32):
        if rank == 0:
            return model

        if type(model) == nn.Linear:  # exact match, no subclass
            model.__class__ = LoRALinear
            model.init_lora(rank=rank, dtype=dtype)
        else:
            for child in model.children():
                LoRALinear.to_lora(child, rank=rank, dtype=dtype)
        return model

    def merge_lora(model: nn.Module):
        if type(model) == LoRALinear:
            d_weight = model.lora_b.detach() @ model.lora_a.detach() * model.scale
            weight = model.weight.detach() + d_weight
            model.__class__ = nn.Linear
            model.weight = nn.Parameter(weight, requires_grad=False)
            for attr in ("rank", "scale", "lora_a", "lora_b"):
                delattr(model, attr)
        else:
            for child in model.children():
                LoRALinear.merge_lora(child)
        return model

    def set_scale(model: nn.Module, scale: float):
        """This method can be used both as instance method and static method
        E.g. `linear.set_scale(1.0)` and `LoRALinear.set_scale(model, 1.0)`"""
        if type(model) == LoRALinear:
            model.scale.copy_(scale)
        else:
            for child in model.children():
                LoRALinear.set_scale(child, scale)
        return model
