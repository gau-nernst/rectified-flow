import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit(do_not_specialize=["L"])
def _group_norm_kernel(
    x_ptr,  # [B, L, num_groups, DIM]
    w_ptr,  # [num_groups, DIM]
    b_ptr,  # [num_groups, DIM]
    y_ptr,  # [B, L, num_groups, DIM]
    L,
    NUM_GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_L: tl.constexpr,
    eps: float = 1e-6,
    act: tl.constexpr = "none",
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    batch_id = pid // NUM_GROUPS

    x_ptr += (batch_id * L * NUM_GROUPS + group_id) * DIM
    w_ptr += group_id * DIM
    b_ptr += group_id * DIM
    y_ptr += (batch_id * L * NUM_GROUPS + group_id) * DIM

    count = tl.zeros((BLOCK_L, 1), tl.int32)
    mean = tl.zeros((BLOCK_L, 1), tl.float32)
    M2 = tl.zeros((BLOCK_L, 1), tl.float32)

    offs_l = tl.arange(0, BLOCK_L)[:, None]
    offs_d = tl.arange(0, DIM)
    x_ptrs = x_ptr + offs_l * NUM_GROUPS * DIM + offs_d

    for i in tl.range(tl.cdiv(L, BLOCK_L), num_stages=num_stages):
        mask = offs_l < L - i * BLOCK_L
        x = tl.load(x_ptrs, mask, other=0.0).to(tl.float32)

        # state of the new tile
        new_count = mask * DIM
        new_mean = tl.sum(x, axis=1, keep_dims=True) / DIM
        x2 = tl.where(mask, (x - new_mean) * (x - new_mean), 0.0)
        new_M2 = tl.sum(x2, axis=1, keep_dims=True)

        # welford update
        count += new_count
        delta = new_mean - mean
        mean += delta * new_count / tl.maximum(count, 1)
        delta2 = new_mean - mean
        M2 += new_M2 + delta * delta2 * new_count

        x_ptrs += BLOCK_L * NUM_GROUPS * DIM

    count_all = tl.sum(count)
    mean_all = tl.sum(mean * count) / count_all
    delta_all = mean - mean_all
    M2_all = tl.sum(M2 + delta_all * delta_all * count)
    rrms = tl.rsqrt(M2_all / count_all + eps)

    w = tl.load(w_ptr + offs_d).to(tl.float32)
    b = tl.load(b_ptr + offs_d).to(tl.float32)

    offs_l = tl.arange(0, BLOCK_L)[:, None]
    offs_d = tl.arange(0, DIM)
    x_ptrs = x_ptr + offs_l * NUM_GROUPS * DIM + offs_d
    y_ptrs = y_ptr + offs_l * NUM_GROUPS * DIM + offs_d

    for i in tl.range(tl.cdiv(L, BLOCK_L), num_stages=num_stages):
        mask = offs_l < L - i * BLOCK_L
        x = tl.load(x_ptrs, mask).to(tl.float32)
        y = (x - mean_all) * rrms * w + b

        if act == "silu":
            y *= tl.sigmoid(y)
        else:
            assert act == "none"
        tl.store(y_ptrs, y, mask)

        x_ptrs += BLOCK_L * NUM_GROUPS * DIM
        y_ptrs += BLOCK_L * NUM_GROUPS * DIM


def group_norm(x: Tensor, w: Tensor, b: Tensor, num_groups: int, eps: float = 1e-6, act: str = "none"):
    assert x.is_contiguous() and w.is_contiguous() and b.is_contiguous()
    y = torch.empty_like(x)
    B, H, W, C = x.shape
    assert C % num_groups == 0
    DIM = C // num_groups

    # heuristic
    if DIM == 4:
        BLOCK_L, num_stages = 128, 4
    elif DIM == 8:
        BLOCK_L, num_stages = 512, 4
    else:
        BLOCK_L, num_stages = 256, 6
    _group_norm_kernel[(B * num_groups,)](
        x, w, b, y, H * W, num_groups, DIM, BLOCK_L, eps, act, num_stages, num_warps=8
    )
    return y


class GroupNorm(nn.GroupNorm):
    def forward(self, x: Tensor, act: str = "none"):
        if not torch.is_grad_enabled():
            return group_norm(x, self.weight, self.bias, self.num_groups, self.eps, act)

        out = super().forward(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if act == "silu":
            out = F.silu(out)
        else:
            assert act == "none"
        return out


class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, x: Tensor):
        return super().forward(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
