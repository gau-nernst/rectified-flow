# https://github.com/mk-minchul/CVLface/blob/d3d09c93e53c5d5db9ec5c5fa96e5d77fe7a78d8/cvlface/research/recognition/code/run_v1/models/iresnet/model.py

from typing import NamedTuple

import torch
from torch import Tensor, nn

from .utils import load_hf_state_dict


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlockIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x: Tensor):
        return self.res_layer(x) + self.shortcut_layer(x)


class LayerConfig(NamedTuple):
    in_channel: int
    depth: int
    stride: int


def get_block(in_channel, depth, num_units, stride=2):
    return [LayerConfig(in_channel, depth, stride)] + [LayerConfig(depth, depth, 1) for _ in range(num_units - 1)]


class IResNet(nn.Module):
    def __init__(
        self,
        blocks: list[list[LayerConfig]],
        input_size: tuple[int, int] = (112, 112),
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        output_channel = 512

        total_stride = int(2 ** len(blocks))
        feat_h = input_size[0] // total_stride
        feat_w = input_size[1] // total_stride

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        self.body = nn.Sequential()
        for block in blocks:
            for bottleneck in block:
                self.body.append(BasicBlockIR(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(output_channel),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(output_channel * feat_h * feat_w, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor):
        x = self.input_layer(x)
        x = self.body(x)
        return x

    def forward(self, x: Tensor):
        x = self.forward_features(x)
        x = self.output_layer(x)
        return x


def load_adaface_ir101(dtype: torch.dtype = torch.float32):
    blocks = [
        get_block(in_channel=64, depth=64, num_units=3),
        get_block(in_channel=64, depth=128, num_units=13),
        get_block(in_channel=128, depth=256, num_units=30),
        get_block(in_channel=256, depth=512, num_units=3),
    ]
    with torch.device("meta"):
        model = IResNet(blocks)

    state_dict = load_hf_state_dict("minchul/cvlface_adaface_ir101_webface12m", "model.safetensors")
    state_dict = {k.removeprefix("model.net."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, assign=True)
    return model.to(dtype)
