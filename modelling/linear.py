import torch
import torch.nn.functional as F
from gn_kernels import quantize_nvfp4_triton
from torch import Tensor, nn


def nvfp4_mm(x: Tensor, xs: Tensor, xs2: Tensor, w: Tensor, ws: Tensor, ws2: Tensor, bias: Tensor | None = None):
    w = w.view(torch.float4_e2m1fn_x2)
    scale_type = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
    return F.scaled_mm(x, w.T, [xs, xs2], scale_type, [ws, ws2], scale_type, swizzle, swizzle, bias)


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if f"{prefix}weight_scale_2" in state_dict:
                # nvfp4 ckpt
                del self.weight
                self.register_buffer("input_scale", torch.empty((), dtype=torch.float32))
                self.register_buffer("weight", torch.empty(out_dim, in_dim // 2, dtype=torch.uint8))
                self.register_buffer("weight_scale", torch.empty(out_dim, in_dim // 16, dtype=torch.float8_e4m3fn))
                self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32))

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor) -> Tensor:
        dims = x.shape[:-1]
        x = x.view(-1, x.shape[-1])

        if hasattr(self, "weight_scale_2"):
            # nvfp4
            xq, xs = quantize_nvfp4_triton(x, self.input_scale)
            out = nvfp4_mm(xq, xs, self.input_scale, self.weight, self.weight_scale, self.weight_scale_2, self.bias)

        else:
            # bf16
            out = F.linear(x, self.weight, self.bias)

        return out.view(*dims, -1)
