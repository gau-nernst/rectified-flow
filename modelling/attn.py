import torch
import torch.nn.functional as F
from torch import Tensor

from gn_kernels import attn_int8_qk, attn_mxfp8_qk, quantize_mx, triton_attn


def dispatch_attn(q: Tensor, k: Tensor, v: Tensor, impl: str = "pt") -> Tensor:
    if impl == "pt":
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)

    elif impl == "mxfp8_qk":
        q_f8, scale_q = quantize_mx(q.contiguous(), torch.float8_e4m3fn, compute_scale_method="nv")
        k_f8, scale_k = quantize_mx(k.contiguous(), torch.float8_e4m3fn, compute_scale_method="nv")
        return attn_mxfp8_qk(q_f8, k_f8, v, scale_q, scale_k)

    elif impl == "gn_cuda_qk_int8":
        q_i8, scale_q = _quantize_i8(q)
        k_i8, scale_k = _quantize_i8(k)
        return attn_int8_qk(q_i8, k_i8, v, scale_q, scale_k)

    elif impl == "gn_triton_qk_int8":
        q_i8, scale_q = _quantize_i8(q)
        k_i8, scale_k = _quantize_i8(k)
        return triton_attn(q_i8, k_i8, v, scale_q, scale_k)

    raise NotImplementedError(f"{impl=}")


def _quantize_i8(x: Tensor):
    scale = x.float().abs().amax(dim=-1) / 127.5
    x_i8 = (x / scale.clip(1e-4).unsqueeze(-1)).clip(-128, 127).round().to(torch.int8)
    scale = scale.transpose(1, 2).contiguous()  # (bs, num_heads, len)
    return x_i8, scale
