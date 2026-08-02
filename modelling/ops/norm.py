import torch.nn.functional as F
from torch import Tensor, nn


class GroupNorm(nn.GroupNorm):
    def forward(self, x: Tensor):
        return super().forward(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, x: Tensor):
        return super().forward(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
