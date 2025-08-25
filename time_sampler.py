from dataclasses import dataclass

import torch


@dataclass
class TimeSampler:
    def __call__(self, n: int, device: torch.types.Device) -> torch.Tensor: ...


class Uniform(TimeSampler):
    def __call__(self, n: int, device: torch.device):
        return torch.rand(n, device=device)


class LogitNormal(TimeSampler):
    """Section 3.1 in https://arxiv.org/abs/2403.03206"""

    mean: float = 0.0
    std: float = 1.0

    def __call__(self, n: int, device: torch.device):
        return torch.normal(self.mean, self.std, size=(n,), device=device).sigmoid()
