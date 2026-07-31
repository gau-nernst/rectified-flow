import torch
from gn_kernels import permute_nv_sf, quantize_mx
from torch import Tensor, nn


class MXLinear(nn.Module):
    @staticmethod
    def convert(linear: nn.Linear, dtype: torch.dtype, compute_scale_method: str = "ocp"):
        if linear.in_features % 128 != 0 or linear.out_features % 128 != 0:
            return

        linear.__class__ = MXLinear
        linear.compute_scale_method = compute_scale_method
        wq, ws = quantize_mx(
            linear.weight.detach(),
            dtype,
            compute_scale_method=compute_scale_method,
        )
        del linear.weight
        linear.register_buffer("weight", wq)
        linear.register_buffer("weight_scale", permute_nv_sf(ws))

    def forward(self, x: Tensor):
        x_2d = x.reshape(-1, x.shape[-1])
        xq, xs = quantize_mx(x_2d, self.wq.dtype, compute_scale_method=self.compute_scale_method)
        xs = permute_nv_sf(xs)
        out = torch._scaled_mm(xq, self.weight.T, xs, self.weight_scale, self.bias, out_dtype=torch.bfloat16)
        return out.reshape(*x.shape[:-1], out.shape[-1])
