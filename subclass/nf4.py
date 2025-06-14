import torch
import torch.nn.functional as F
from torch import Tensor, nn

aten = torch.ops.aten


# https://arxiv.org/abs/2305.14314, appendix E
NF4_LUT = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def scale_absmax(input: Tensor, block_size: int):
    input = input.view(-1, block_size)
    scale = input.abs().amax(-1)
    input = input / scale.view(-1, 1).clip(1e-12)  # input is in [-1,1] now
    return input.view(-1), scale


def quantize_4bit_with_qmap(input: Tensor, qmap: Tensor):
    # GPU-friendly binary search
    # https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
    codes = torch.where(input >= qmap[8], 8, 0)
    codes += torch.where(input >= qmap[codes + 4], 4, 0)
    codes += torch.where(input >= qmap[codes + 2], 2, 0)
    codes += torch.where(input >= qmap[codes + 1], 1, 0)

    # rounding
    codes_up = (codes + 1).clip(max=15)
    val_down = qmap[codes]
    val_up = qmap[codes_up]
    residual = input - val_down
    codes = torch.where(residual >= (val_up - val_down) * 0.5, codes_up, codes)

    return codes.to(torch.uint8)


class NF4Tensor(Tensor):
    """A simple NF4 tensor subclass implementation without double quantization."""

    tensor_attrs = ["codes", "scale", "qmap"]

    def __new__(cls, codes: Tensor, scale: Tensor, qmap: Tensor, shape, dtype):
        return Tensor._make_wrapper_subclass(cls, shape, device=codes.device, dtype=dtype)

    def __init__(self, codes: Tensor, scale: Tensor, qmap: Tensor, shape, dtype):
        assert codes.dtype is torch.uint8
        assert codes.ndim == 1
        assert scale.dtype is torch.float32
        assert scale.ndim == 1
        assert qmap.dtype is torch.float32
        assert codes.device == scale.device == qmap.device
        self.codes = codes
        self.scale = scale
        self.qmap = qmap
        self.block_size = codes.numel() * 2 // scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.shape, self.dtype]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(block_size={self.block_size}, shape={tuple(self.shape)}, device={self.device})"
        )

    def dequantize(self):
        codes = torch.stack([self.codes >> 4, self.codes & 0b1111], dim=-1)  # unpack

        # torch.compile() cannot use uint8 as index
        out = self.qmap[codes.int()].view(self.scale.shape[0], -1) * self.scale.view(-1, 1)
        out = out.to(self.dtype).view(self.shape)
        return out

    @classmethod
    def from_float(cls, x: Tensor, block_size: int = 64):
        qmap = torch.tensor(NF4_LUT, device=x.device)  # FP32
        scaled_x, scale = scale_absmax(x.float(), block_size)  # scale is FP32
        codes = quantize_4bit_with_qmap(scaled_x, qmap)
        codes = (codes[::2] << 4) | codes[1::2]  # pack
        return cls(codes, scale, qmap, x.shape, x.dtype)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            input, weight = args[:2]
            bias = args[2] if len(args) > 2 else None
            # NOTE: we can also do F.linear(input, weight.dequantize(), bias)
            # however, this will save dequantized weight for backward.
            # will not matter if we use activation checkpointing.
            out = _NF4LinearFunction.apply(input, weight)
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
            return cls(*data, args[0].shape, dtype)  # only change appearance dtype

        elif func is aten.copy_.default:
            if isinstance(args[0], cls) and isinstance(args[1], cls):
                args[0].codes.copy_(args[1].codes)
                args[0].scale.copy_(args[1].scale)
                # qmap should be the same

            elif isinstance(args[0], cls):
                scaled_x, scale = scale_absmax(args[1].float(), args[0].block_size)  # scale is FP32
                codes = quantize_4bit_with_qmap(scaled_x, args[0].qmap)
                args[0].codes.copy_(codes)
                args[0].scale.copy_(scale)

            else:
                args[0].copy_(args[1].dequantize())

            return args[0]

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


class _NF4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: NF4Tensor):
        ctx.save_for_backward(input, weight)
        return F.linear(input, weight.dequantize())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight.dequantize()
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.view(-1, weight.shape[0]).T @ input.view(-1, weight.shape[1])

        return grad_input, grad_weight


class NF4Linear(nn.Linear):
    @staticmethod
    def from_float(linear: nn.Linear):
        linear.__class__ = NF4Linear
        weight = linear.weight.detach()
        del linear.weight

        weight_nf4 = NF4Tensor.from_float(weight)
        linear.register_buffer("codes", weight_nf4.codes)
        linear.register_buffer("scale", weight_nf4.scale)
        linear.register_buffer("qmap", weight_nf4.qmap)
        linear.shape = weight.shape
        linear.dtype = weight.dtype
        return linear

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: this will save dequantized weight for backward.
        # will not matter if we use activation checkpointing.
        return F.linear(input, NF4Tensor.dequantize(self), self.bias)
