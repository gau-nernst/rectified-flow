import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from gn_kernels import triton_mm
except ImportError:
    pass


def quantize_row_wise(x: Tensor):
    assert x.ndim == 2
    abs_max = x.abs().amax(dim=1, keepdim=True).float()
    scales = abs_max / 127
    x_i8 = (x / scales.clip(1e-5)).clip(-128, 127).round().to(torch.int8)
    return x_i8, scales.to(x.dtype)  # check if FP32 scales is much better?


@torch.no_grad()
def w8a8_dynamic(a: Tensor, b: Tensor, bias: Tensor | None):
    a_i8, a_scales = quantize_row_wise(a)
    bt_i8, bt_scales = quantize_row_wise(b.T)
    return triton_mm(
        a_i8,
        bt_i8.contiguous().T,
        bias,
        scale_A=a_scales,
        scale_B=bt_scales.T,
        out_dtype=a.dtype,
    )


class LoRALinear(nn.Linear):
    def init_lora(
        self,
        rank: int = 8,
        scale: float = 1.0,
        quantization: str = "",
        dtype: torch.dtype | None = torch.float32,
        device: torch.types.Device = None,
    ) -> None:
        """By default, use FP32 for LoRA weights."""
        assert quantization in ("", "int8_training", "int8_inference")
        self.rank = rank
        self.quantization = quantization
        # NOTE: there is a problem with torch.compile when self.scale is a Python float.
        # use tensor buffer as a workaround.
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32), persistent=False)

        if quantization == "int8_inference":
            weight_i8, weight_scales = quantize_row_wise(self.weight.detach())
            self.weight = nn.Parameter(weight_i8, requires_grad=False)
            self.register_buffer("weight_scales", weight_scales, persistent=False)
        else:
            self.weight.requires_grad_(False)

        # TODO: should we make bias trainable?
        if self.bias is not None:
            self.bias.requires_grad_(False)

        if rank > 0:
            dtype = dtype or self.weight.dtype
            self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype, device=device))
            self.lora_b = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype, device=device))
            nn.init.kaiming_normal_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)
        else:
            self.lora_a = self.lora_b = None

    def extra_repr(self):
        return f"{super().extra_repr()}, rank={self.rank}, scale={self.scale.item()}"

    def forward(self, x: Tensor):
        if self.quantization == "int8_training":
            out = LinearW8A8Dynamic.apply(x, self.weight, self.bias)
        elif self.quantization == "int8_inference":
            x_i8, x_scales = quantize_row_wise(x.flatten(0, -2))
            out = triton_mm(
                x_i8,
                self.weight.T,
                self.bias,
                scale_A=x_scales,
                scale_B=self.weight_scales.T,
                out_dtype=x.dtype,
            )
            out = out.unflatten(0, x.shape[:-1])
        else:
            out = F.linear(x, self.weight, self.bias)

        if self.rank > 0:
            out = out + x @ (self.lora_a * self.scale / self.rank).to(x.dtype).T @ self.lora_b.to(x.dtype).T
        return out

    @staticmethod
    def add_lora(
        model: nn.Module,
        rank: int = 8,
        quantization: str = "",
        dtype: torch.dtype = torch.float32,
        device: torch.types.Device = None,
    ):
        if type(model) is nn.Linear:  # exact match, no subclass
            model.__class__ = LoRALinear
            model.init_lora(rank=rank, quantization=quantization, dtype=dtype, device=device)
        else:
            for child in model.children():
                LoRALinear.add_lora(child, rank=rank, quantization=quantization, dtype=dtype, device=device)
        return model

    def merge_lora(model: nn.Module):
        if type(model) is LoRALinear:
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
        if type(model) is LoRALinear:
            model.scale.copy_(scale)
        else:
            for child in model.children():
                LoRALinear.set_scale(child, scale)
        return model


class LinearW8A8Dynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None):
        ctx.save_for_backward(weight)
        ctx.has_bias = bias is not None
        return w8a8_dynamic(x.flatten(0, -2), weight.T, bias).unflatten(0, x.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # NOTE: we never calculate grad_weight
        (weight,) = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        grad_x = w8a8_dynamic(grad_output.flatten(0, -2), weight, None).unflatten(0, grad_output.shape[:-1])
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.flatten(0, -2).sum(0)
        return grad_x, grad_weight, grad_bias
