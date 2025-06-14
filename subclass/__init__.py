import torch
from torch import nn

from .int8 import Int8W8A8Tensor
from .nf4 import NF4Tensor


def quantize_(
    model: nn.Module,
    subclass: type[NF4Tensor | Int8W8A8Tensor],
    device: torch.device | None = None,
    **kwargs,
):
    if isinstance(model, nn.Linear):
        # move to CUDA before quantization for faster quantization
        weight = subclass.from_float(model.weight.detach().to(device=device), **kwargs)
        model.weight = nn.Parameter(weight, requires_grad=model.weight.requires_grad)
    else:
        for m in model.children():
            quantize_(m, subclass, device=device, **kwargs)
    return model


def dequantize_(model: nn.Module, device: torch.device | None = None):
    if isinstance(model, nn.Linear):
        if isinstance(model.weight.data, (NF4Tensor, Int8W8A8Tensor)):
            # move to CPU if CUDA cannot hold all dequantized weights
            weight = model.weight.data.dequantize().to(device=device)
            model.weight = nn.Parameter(weight, requires_grad=model.weight.requires_grad)
    else:
        for m in model.children():
            dequantize_(m, device=device)
    return model
