import torch
from gn_kernels import quantize_nvfp4_triton
from gn_kernels.cutedsl import sm120_mm_nvfp4
from torch import Tensor, nn


def nvfp4_calibration_hook(module: nn.Module, args):
    module.input_amax_list.append(args[0].abs().amax())


class NVFP4Linear(nn.Module):
    @staticmethod
    def install_calibration_hook(model: nn.Module):
        # pre-order traversal
        # assuming Linear is a leaf node
        if isinstance(model, nn.Linear):
            if model.in_features % 128 != 0 or model.out_features % 128 != 0:
                return

            model.input_amax_list = []
            handle = model.register_forward_pre_hook(nvfp4_calibration_hook)
            model.nvfp4_handle = handle
            return

        for child in model.children():
            NVFP4Linear.install_calibration_hook(child)

    @staticmethod
    def convert(m: nn.Module):
        # pre-order traversal
        # assuming Linear is a leaf node
        if isinstance(m, nn.Linear):
            input_amax_list = getattr(m, "input_amax_list", None)
            if not input_amax_list:
                return

            input_scale = torch.stack(input_amax_list).amax().float() / (448.0 * 6.0)
            m.nvfp4_handle.remove()
            del m.input_amax_list
            del m.nvfp4_handle

            m.__class__ = NVFP4Linear
            w = m.weight.detach()
            del m.weight
            w_tensor_scale = w.abs().amax().float() / (448.0 * 6.0)
            wq, ws = quantize_nvfp4_triton(w, w_tensor_scale)
            m.register_buffer("weight", wq)
            m.register_buffer("weight_scale", ws)
            m.register_buffer("input_scale", input_scale)
            m.output_scale = input_scale.item() * w_tensor_scale.item()

            return

        for child in m.children():
            NVFP4Linear.convert(child)

    def forward(self, x: Tensor):
        x_2d = x.reshape(-1, x.shape[-1])
        xq, xs = quantize_nvfp4_triton(x_2d, self.input_scale)
        out = sm120_mm_nvfp4.mm(xq, self.weight, xs, self.weight_scale, self.output_scale)
        # TODO: support bias
        if self.bias is not None:
            out += self.bias
        return out.view(*x.shape[:-1], out.shape[-1])
