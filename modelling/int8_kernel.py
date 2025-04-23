import torch
import triton
import triton.language as tl
from torch import Tensor

try:
    import triton_dejavu

    autotune = triton_dejavu.autotune
except ImportError:
    autotune = triton.autotune

lib = torch.library.Library("my_rf", "DEF")
lib_ops = torch.ops.my_rf

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
configs = [
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    (128, 256, 64, 3, 8),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 5, 2),
    (32, 64, 32, 5, 2),
    # Good config for fp8 inputs
    (128, 256, 128, 3, 8),
    (256, 128, 128, 3, 8),
    (256, 64, 128, 4, 4),
    (64, 256, 128, 4, 4),
    (128, 128, 128, 4, 4),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (128, 32, 64, 4, 4),
    # https://github.com/pytorch/pytorch/blob/7868b65c4d4f34133607b0166f08e9fbf3b257c4/torch/_inductor/kernel/mm_common.py#L172
    (64, 64, 32, 2, 4),
    (64, 128, 32, 3, 4),
    (128, 64, 32, 3, 4),
    (64, 128, 32, 4, 8),
    (128, 64, 32, 4, 8),
    (64, 32, 32, 5, 8),
    (32, 64, 32, 5, 8),
    (128, 128, 32, 2, 8),
    (64, 64, 64, 3, 8),
]

configs = [
    triton.Config(dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K), num_stages=num_stages, num_warps=num_warps)
    for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in configs
]


def _grid(meta):
    return (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)


@autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _scaled_mm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    row_scale_ptr,
    col_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
    HAS_BIAS: tl.constexpr = False,
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    row_scale = tl.load(row_scale_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    col_scale = tl.load(col_scale_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    acc = acc.to(tl.float32) * row_scale * col_scale
    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + idx_n, mask=idx_n < N).to(tl.float32)

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define("scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B, Tensor? bias) -> Tensor")


def scaled_int8_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor, bias: Tensor | None) -> Tensor:
    """Matmul for tile-wise quantized A and B. `A` and `B` are both INT8 or FP8 to utilize
    INT8/FP8 tensor cores. `scale_A` and `scaled_B` are quantization scales for A and B
    respectively with appropriate shapes.

    E.g.
      - if `A` is quantized with tile shape (128, 64), `scale_A`'s shape will be
    `(A.shape[0] / 128, A.shape[1] / 64)`.
      - if `A` is row-wise quantized, `scale_A`'s shape will be `(A.shape[0], 1)`.
    """
    assert A.dtype == B.dtype == torch.int8
    assert scale_A.dtype == scale_B.dtype
    assert A.ndim == B.ndim == scale_A.ndim == scale_B.ndim == 2
    assert A.shape[1] == B.shape[0]
    assert scale_A.shape == (A.shape[0], 1)
    assert scale_B.shape == (1, B.shape[1])
    assert scale_A.is_contiguous()
    assert scale_B.is_contiguous()
    if bias is not None:
        assert bias.is_contiguous()
    return lib_ops.scaled_mm(A, B, scale_A, scale_B, bias)


@torch.library.impl(lib, "scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor, bias: Tensor | None):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=row_scale.dtype)


@torch.library.impl(lib, "scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor, bias: Tensor | None):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=row_scale.dtype)
    _scaled_mm_kernel[_grid](
        A,
        B,
        C,
        row_scale,
        col_scale,
        bias,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        ACC_DTYPE=tl.int32 if A.dtype == torch.int8 else tl.float32,
        EVEN_K=K % 2 == 0,
        HAS_BIAS=bias is not None,
    )
    return C
