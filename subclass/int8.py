from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .int8_mm import scaled_int8_mm

aten = torch.ops.aten


def rowwise_quantize(x: Tensor, eps: float = 1e-12):
    absmax = x.abs().amax(-1, keepdim=True)
    scale = absmax.float() / 127
    int_data = (x / scale.clip(eps)).to(torch.int8)  # scale x to [-127,127]
    return int_data, scale


class ScaledInt8Config(NamedTuple):
    int8_output: bool = False
    int8_grad_weight: bool = False


class ScaledInt8Tensor(Tensor):
    tensor_attrs = ["i8_data", "scale"]

    def __new__(cls, i8_data: Tensor, scale: Tensor, dtype, config: ScaledInt8Config):
        return Tensor._make_wrapper_subclass(cls, i8_data.shape, device=i8_data.device, dtype=dtype)

    def __init__(self, i8_data: Tensor, scale: Tensor, dtype, config: ScaledInt8Config):
        assert i8_data.dtype is torch.int8
        assert scale.dtype is torch.float32
        assert i8_data.device == scale.device
        assert scale.ndim == i8_data.ndim
        self.i8_data = i8_data
        self.scale = scale
        self.config = config

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.dtype, self.config]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"config={self.config}, "
            f"shape={tuple(self.shape)}, "
            f"device={self.device})"
        )

    def dequantize(self):
        return (self.i8_data.float() * self.scale).to(self.dtype)

    @classmethod
    def from_float(cls, x: Tensor, config: ScaledInt8Config = ScaledInt8Config()):
        int_data, scale = rowwise_quantize(x)
        return cls(int_data, scale, x.dtype, config)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            input, weight = args[:2]
            bias = args[2] if len(args) > 2 else None
            out = _Int8LinearFunction.apply(input, weight)
            if bias is not None:  # autograd still works under torch_function.
                out = out + bias  # let autograd handles bias.
            return out

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in (aten.detach.default, aten.clone.default):
            data_names, attrs = args[0].__tensor_flatten__()
            data = [func(getattr(args[0], name), *args[1:], **kwargs) for name in data_names]
            return cls(*data, *attrs)

        elif func in (aten._to_copy.default,):
            device = kwargs.get("device", None)
            dtype = kwargs.get("dtype", args[0].dtype)

            data_names, _ = args[0].__tensor_flatten__()
            data = [getattr(args[0], name).to(device=device) for name in data_names]
            return cls(*data, dtype, args[0].config)  # only change appearance dtype

        elif func is aten.copy_.default:
            if isinstance(args[0], cls) and isinstance(args[1], cls):
                args[0].i8_data.copy_(args[1].i8_data)
                args[0].scale.copy_(args[1].scale)

            elif isinstance(args[0], cls):
                i8_data, scale = rowwise_quantize(args[1])
                args[0].i8_data.copy_(i8_data)
                args[0].scale.copy_(scale)

            else:
                args[0].copy_(args[1].dequantize())

            return args[0]

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


class _Int8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: ScaledInt8Tensor):
        ctx.save_for_backward(input, weight)
        if weight.config.int8_output:
            input_i8, input_scale = rowwise_quantize(input.view(-1, weight.shape[1]))
            out = scaled_int8_mm(input_i8, weight.i8_data.T, input_scale.view(-1), weight.scale.view(-1))
            return out.view(*input.shape[:-1], weight.shape[0])
        else:
            # mixed matmul
            return (input @ weight.i8_data.to(input.dtype).T) * weight.scale.T

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight.dequantize()

        if ctx.needs_input_grad[1]:
            grad_output = grad_output.view(-1, weight.shape[0])
            input = input.view(-1, weight.shape[1])

            if weight.config.int8_grad_weight:
                grad_output_i8_t, grad_output_scale_t = rowwise_quantize(grad_output.T)
                input_i8_t, input_scale_t = rowwise_quantize(input.T)
                grad_weight = scaled_int8_mm(
                    grad_output_i8_t, input_i8_t.T, grad_output_scale_t.view(-1), input_scale_t.view(-1)
                )
            else:
                grad_weight = grad_output.T @ input

        return grad_input, grad_weight
