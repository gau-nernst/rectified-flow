import torch
import torch.nn.functional as F
from gn_kernels import attn_mxfp8, quantize_mx, triton_scaled_qk_attn
from torch import Tensor


def dispatch_attention(q: Tensor, k: Tensor, v: Tensor, impl: str = "pt") -> Tensor:
    if impl == "pt":
        return F.scaled_dot_product_attention(q, k, v)

    elif impl == "mxfp8":
        q_f8, scale_q = _quantize_mxfp8(q)
        k_f8, scale_k = _quantize_mxfp8(k)
        return attn_mxfp8(q_f8, k_f8, v, scale_q, scale_k)

    elif impl == "int8":
        q_i8, scale_q = _quantize_i8(q)
        k_i8, scale_k = _quantize_i8(k)
        return triton_scaled_qk_attn(q_i8.contiguous(), k_i8.contiguous(), v.contiguous(), scale_q, scale_k)

    raise NotImplementedError(f"{impl=}")


def _quantize_mxfp8(x: Tensor):
    x_f8, scale_x = quantize_mx(x.flatten(0, -2), torch.float8_e4m3fn, compute_scale_method="nv")
    return x_f8.view(x.shape), scale_x


def _quantize_i8(x: Tensor):
    amax = x.float().abs().amax(dim=-1)
    scale = amax / 127
    x_i8 = (x / scale.unsqueeze(-1)).clip(-128, 127).round().to(torch.int8)
    return x_i8, scale
