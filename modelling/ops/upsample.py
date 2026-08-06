import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit(do_not_specialize=["W"])
def _upsample_nn2x_kernel(
    x_ptr,  # [N,H,W,C]
    y_ptr,  # [N,2H,2W,C]
    W,
    stride_xn,
    stride_xh,
    stride_xw,
    stride_yn,
    stride_yh,
    stride_yw,
    BLOCK_W: tl.constexpr,
    C: tl.constexpr,
):
    tile_id = tl.program_id(0)
    h = tl.program_id(1)
    batch_id = tl.program_id(2)

    x_ptr += batch_id * stride_xn + h * stride_xh
    y_ptr += batch_id * stride_yn + h * 2 * stride_yh

    offs_w = tile_id * BLOCK_W + tl.arange(0, BLOCK_W)[:, None]
    offs_c = tl.arange(0, C)
    mask = offs_w < W
    x_ptrs = x_ptr + offs_w * stride_xw + offs_c
    x = tl.load(x_ptrs, mask=mask)  # [BLOCK_W, C]

    for i in range(2):
        for j in range(2):
            y_ptrs = y_ptr + i * stride_yh + (offs_w * 2 + j) * stride_yw + offs_c
            tl.store(y_ptrs, x, mask)


def upsample_nn2x(x: Tensor):
    if x.requires_grad:
        return F.interpolate(x.permute(0, 3, 1, 2), scale_factor=2.0, mode="nearest").permute(0, 2, 3, 1)

    assert x.stride(-1) == 1
    N, H, W, C = x.shape
    y = x.new_empty(N, H * 2, W * 2, C)

    BLOCK_W = min(64, triton.next_power_of_2(W))
    num_tiles = triton.cdiv(W, BLOCK_W)
    _upsample_nn2x_kernel[(num_tiles, H, N)](x, y, W, *x.stride()[:-1], *y.stride()[:-1], BLOCK_W, C)
    return y
