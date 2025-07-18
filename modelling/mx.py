import torch
from torch import nn, Tensor

from gn_kernels import quantize_mx, pack_block_scales_nv, cutlass_mxfp4_mm


class MXLinear(nn.Module):
    @staticmethod
    def convert(linear: nn.Linear, dtype: torch.dtype, compute_scale_method: str = "ocp"):
        if linear.in_features % 128 != 0 or linear.out_features % 128 != 0:
            return

        linear.__class__ = MXLinear
        linear.compute_scale_method = compute_scale_method
        wq, ws = quantize_mx(linear.weight.detach(), dtype, compute_scale_method=compute_scale_method)
        linear.register_buffer("wq", wq)
        linear.register_buffer("ws", pack_block_scales_nv(ws))
        del linear.weight

    def forward(self, x: Tensor):
        x_2d = x.reshape(-1, x.shape[-1])
        xq, xs = quantize_mx(x_2d, self.wq.dtype, compute_scale_method=self.compute_scale_method)
        xs = pack_block_scales_nv(xs)

        if self.wq.dtype == torch.float4_e2m1fn_x2:
            out = cutlass_mxfp4_mm(xq, self.wq.T, xs, self.ws, self.bias)
        else:
            out = torch._scaled_mm(xq, self.wq.T, xs, self.ws, self.bias, out_dtype=torch.bfloat16)

        return out.reshape(*x.shape[:-1], out.shape[-1])
