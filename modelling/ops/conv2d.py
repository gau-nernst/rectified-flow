import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Conv2d(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, kernel_size, kernel_size, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.stride = stride
        self.padding = padding

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            key = f"{prefix}weight"
            state_dict[key] = state_dict[key].permute(0, 2, 3, 1)

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor):
        return F.conv2d(
            x.permute(0, 3, 1, 2),
            self.weight.permute(0, 3, 1, 2),
            self.bias,
            self.stride,
            self.padding,
        ).permute(0, 2, 3, 1)
