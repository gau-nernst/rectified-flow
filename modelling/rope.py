import torch
from torch import Tensor, nn


def compute_rope(length: int, dim: int, theta: float, device: torch.types.Device = None):
    omega = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim))
    timestep = torch.arange(length, device=device, dtype=torch.float64)
    freqs = (timestep[:, None] * omega).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    # x: [B, L, nH, D] in real
    # rope: [L, D/2] in complex
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))  # [B, L, nH, D/2]
    x_out = torch.view_as_real(x_complex * rope.unsqueeze(1)).flatten(-2)  # [B, L, nH, D]
    return x_out.type_as(x)


class RopeND(nn.Module):
    def __init__(self, dims: tuple[int, ...], max_lens: tuple[int, ...], theta: float) -> None:
        super().__init__()
        self.dims = dims
        self.max_lens = max_lens
        self.theta = theta
        self.precompute_rope(torch.get_default_device())

    def precompute_rope(self, device: torch.types.Device = None):
        # don't create things on meta device to avoid weird cases...
        if torch.device(device) == torch.device("meta"):
            device = "cpu"

        for i, (dim, length) in enumerate(zip(self.dims, self.max_lens)):
            # always compute on CPU, then move to the requested device
            rope = compute_rope(length, dim, self.theta, device="cpu")
            assert rope.dtype == torch.complex64
            self.register_buffer(f"rope{i}", rope.to(device), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)

        # recompute rope if dtype is changed
        if any(getattr(self, f"rope{i}").dtype != torch.float32 for i in range(len(self.dims))):
            self.precompute_rope(self.rope0.device)

        return self

    def create(self, start_list: tuple[int, ...], length_list: tuple[int, ...]) -> Tensor:
        pos_list = [
            torch.arange(start, start + length, device=self.rope0.device)
            for start, length in zip(start_list, length_list)
        ]
        grids = torch.meshgrid(pos_list, indexing="ij")  # this returns list[Tensor]

        rope_list = [getattr(self, f"rope{i}")[grid.flatten()] for i, grid in enumerate(grids)]
        return torch.cat(rope_list, dim=-1)
