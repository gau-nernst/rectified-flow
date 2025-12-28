import torch
from torch import Tensor, nn


def compute_rope(
    length: int,
    dim: int,
    theta: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.types.Device = None,
) -> Tensor:
    # initial computations in fp64
    omega = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim))
    timestep = torch.arange(length, device=device, dtype=torch.float64)
    freqs = (timestep[:, None] * omega).to(dtype)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    # x: [B, L, nH, D] in real
    # rope: [L, D/2] in complex
    dtype = rope.dtype.to_real()
    x_ = torch.view_as_complex(x.to(dtype).unflatten(-1, (-1, 2)))  # [B, L, nH, D/2]
    out = torch.view_as_real(x_ * rope.unsqueeze(-2)).flatten(-2)  # [B, L, nH, D]
    return out.type_as(x)


class RopeND(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        max_lens: tuple[int, ...],
        theta: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.max_lens = max_lens
        self.theta = theta
        self.dtype = dtype
        self.precompute_rope()

    def precompute_rope(self, device: torch.types.Device = None) -> None:
        # don't create things on meta device to avoid weird cases...
        device = device or torch.get_default_device()
        if torch.device(device) == torch.device("meta"):
            device = "cpu"

        for i, (dim, length) in enumerate(zip(self.dims, self.max_lens)):
            # always compute on CPU, then move to the requested device
            rope = compute_rope(length, dim, self.theta, dtype=self.dtype, device="cpu")
            self.register_buffer(f"rope{i}", rope.to(device), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)

        # recompute rope if dtype is changed
        dtype = self.dtype.to_complex()
        if any(getattr(self, f"rope{i}").dtype != dtype for i in range(len(self.dims))):
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
