import torch
from torch import nn, Tensor

from gn_kernels import quantize_mx, pack_block_scales_nv, mxfp4_mm


class MXLinear(nn.Module):
    @staticmethod
    def convert(linear: nn.Linear, dtype: torch.dtype):
        if linear.in_features % 128 != 0 or linear.out_features % 128 != 0:
            return

        linear.__class__ = MXLinear
        wq, w_scales = quantize_mx(linear.weight.detach(), dtype)
        linear.register_buffer("wq", wq)
        linear.register_buffer("w_scales", pack_block_scales_nv(w_scales))
        del linear.weight

    def forward(self, x: Tensor):
        x_2d = x.reshape(-1, x.shape[-1])
        xq, x_scales = quantize_mx(x_2d, self.wq.dtype)
        x_scales = pack_block_scales_nv(x_scales)

        if self.wq.dtype == torch.float4_e2m1fn_x2:
            out = mxfp4_mm(xq, self.wq.T, x_scales, self.w_scales)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = torch._scaled_mm(xq, self.wq.T, x_scales, self.w_scales, self.bias, out_dtype=torch.bfloat16)

        return out.reshape(*x.shape[:-1], out.shape[-1])
