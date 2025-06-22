import torch
from torch import Tensor

from .int8 import scaled_int8_mm


def scaled_fp8_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor, bias: Tensor | None) -> Tensor:
    return torch._scaled_mm(A, B, scale_A, scale_B, bias, out_dtype=torch.bfloat16)
