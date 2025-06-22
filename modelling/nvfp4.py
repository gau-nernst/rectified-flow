from torch import nn, Tensor
import torch

from gn_kernels import quantize_nvfp4, pack_block_scales_nv, nvfp4_mm


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
    def convert(model: nn.Module):
        # pre-order traversal
        # assuming Linear is a leaf node
        if isinstance(model, nn.Linear):
            input_amax_list = getattr(model, "input_amax_list", None)
            if not input_amax_list:
                return

            x_tensor_scale = torch.stack(input_amax_list).amax() / (448.0 * 6.0)
            model.nvfp4_handle.remove()
            del model.input_amax_list
            del model.nvfp4_handle

            model.__class__ = NVFP4Linear
            wq, ws, w_tensor_scale = quantize_nvfp4(model.weight.detach())
            model.register_buffer("wq", wq)
            model.register_buffer("ws", pack_block_scales_nv(ws))
            model.register_buffer("x_tensor_scale", x_tensor_scale)
            model.register_buffer("output_scale", x_tensor_scale * w_tensor_scale)
            del model.weight

            return

        for child in model.children():
            NVFP4Linear.convert(child)

    def forward(self, x: Tensor):
        x_2d = x.reshape(-1, x.shape[-1])
        xq, xs, _ = quantize_nvfp4(x_2d, self.x_tensor_scale)
        xs = pack_block_scales_nv(xs)

        out = nvfp4_mm(xq, self.wq.T, xs, self.ws, self.output_scale, self.bias)
        return out.view(*x.shape[:-1], out.shape[-1])
