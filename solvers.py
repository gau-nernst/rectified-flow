import math
import typing
import dataclasses

import torch
from torch import Tensor


# make solver a class so that it can store extra data i.e. stateful
@dataclasses.dataclass
class Solver:
    timesteps: tuple[float]

    def step(self, latents: Tensor, v: Tensor, i: int): ...


@dataclasses.dataclass
class EulerSolver(Solver):
    def step(self, latents: Tensor, v: Tensor, i: int):
        # move from timesteps[i] to timesteps[i+1]
        timesteps = self.timesteps
        return latents.add(v, alpha=timesteps[i + 1] - timesteps[i])


@dataclasses.dataclass
class Dpmpp2mSolver(Solver):
    """DPM-Solver++(2M) https://arxiv.org/abs/2211.01095"""

    prev_data_pred: Tensor | None = None

    def step(self, latents: Tensor, v: Tensor, i: int):
        # the implementation below has been simplified for flow matching / rectified flow
        # with sigma(t) = t and alpha(t) = 1-t
        # coincidentally (or not?), this results in identical calculations as k-diffusion implementation
        # https://github.com/crowsonkb/k-diffusion/blob/21d12c91/k_diffusion/sampling.py#L585-L607
        timesteps = self.timesteps
        data_pred = latents.add(v, alpha=-timesteps[i])  # data prediction model

        if i == 0:
            latents = data_pred.lerp(latents, timesteps[i + 1] / timesteps[i])
        elif timesteps[i + 1] == 0.0:  # avoid log(0). note that lim x*log(x) when x->0 is 0.
            latents = data_pred
        else:
            lambda_prev = -math.log(timesteps[i - 1])
            lambda_curr = -math.log(timesteps[i])
            lambda_next = -math.log(timesteps[i + 1])
            r = (lambda_curr - lambda_prev) / (lambda_next - lambda_curr)
            D = data_pred.lerp(self.prev_data_pred, -1 / (2 * r))
            latents = D.lerp(latents, timesteps[i + 1] / timesteps[i])

        self.prev_data_pred = data_pred
        return latents


@dataclasses.dataclass
class UniPCSolver(Solver):
    """UniPC https://arxiv.org/abs/2302.04867"""

    order: int = 2
    # holds history of data_pred. size=len(timesteps)
    data_pred_history: list[Tensor | None] = dataclasses.field(init=False)
    prev_corrected_latents: Tensor | None = None

    def __post_init__(self):
        self.data_pred_history = [None] * len(self.timesteps)

    def step(self, latents: Tensor, v: Tensor, i: int):
        # Wan2.2: https://github.com/Wan-Video/Wan2.2/blob/031a9be5/wan/utils/fm_solvers_unipc.py
        # official: https://github.com/wl-zhao/UniPC/blob/main/uni_pc.py
        self.data_pred_history[i] = latents.add(v, alpha=-self.timesteps[i])

        if i > 0:
            latents = self.c_step(i)
        self.prev_corrected_latents = latents

        # clear history
        if i - self.order >= 0:
            self.data_pred_history[i - self.order] = None

        return self.p_step(latents, i)

    def get_lambda(self, i: int):
        # avoid log(0)
        return math.log(1.0 - self.timesteps[i]) - math.log(max(self.timesteps[i], 1e-8))

    def get_order(self, i: int):
        "Order warmup from 1 and cooldown to 1"
        return min(self.order, i + 1, len(self.timesteps) - i)

    def c_step(self, i: int):
        order = self.get_order(i - 1)  # c-step uses previous step's order

        data_pred_curr = self.data_pred_history[i]
        data_pred_prev = self.data_pred_history[i - 1]
        lambda_curr = self.get_lambda(i)
        lambda_prev = self.get_lambda(i - 1)
        device = data_pred_curr.device

        # iterate from [i-2] to [i-order]
        h = lambda_curr - lambda_prev
        rks = []
        D1s = []
        for prev_offset in range(2, order + 1):
            prev_i = i - prev_offset
            lambd = self.get_lambda(prev_i)
            data_pred = self.data_pred_history[prev_i]
            rk = (lambd - lambda_prev) / h
            rks.append(rk)
            D1s.append((data_pred - data_pred_prev) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        D1s = torch.stack(D1s, dim=1) if len(D1s) > 0 else None

        hh = -h
        h_phi_1 = math.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        B_h = h_phi_1  # bh2

        factorial_i = 1
        R = []
        b = []
        for factor in range(1, order + 1):
            R.append(rks ** (factor - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= factor + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)
        rhos_c = torch.tensor([0.5], device=device) if order == 1 else torch.linalg.solve(R, b)

        sigma_curr = self.timesteps[i]
        sigma_prev = self.timesteps[i - 1]
        alpha_curr = 1.0 - sigma_curr
        x_t_ = (sigma_curr / sigma_prev) * self.prev_corrected_latents - alpha_curr * h_phi_1 * data_pred_prev

        # TODO: looks like D1_t can be combined with D1s?
        D1_t = data_pred_curr - data_pred_prev
        corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s) if D1s is not None else 0
        x_t = x_t_ - alpha_curr * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t

    def p_step(self, latents: Tensor, i: int):
        order = self.get_order(i)

        # TODO: simplify math for the last step, since lambda_next will be inf
        data_pred_curr = self.data_pred_history[i]
        lambda_next = self.get_lambda(i + 1)
        lambda_curr = self.get_lambda(i)
        device = data_pred_curr.device

        # iterate from [i-1] to [i-(order-1)]
        h = lambda_next - lambda_curr
        rks = []
        D1s = []
        for prev_offset in range(1, order):
            prev_i = i - prev_offset
            lambd = self.get_lambda(prev_i)
            data_pred = self.data_pred_history[prev_i]
            rk = (lambd - lambda_curr) / h
            rks.append(rk)
            D1s.append((data_pred - data_pred_curr) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        D1s = torch.stack(D1s, dim=1) if len(D1s) > 0 else None

        hh = -h
        h_phi_1 = math.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        B_h = h_phi_1

        factorial_i = 1
        R = []
        b = []
        for factor in range(1, order + 1):
            R.append(rks ** (factor - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= factor + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        sigma_next = self.timesteps[i + 1]
        sigma_curr = self.timesteps[i]
        alpha_next = 1.0 - sigma_next
        x_t = (sigma_next / sigma_curr) * latents - alpha_next * h_phi_1 * data_pred_curr

        if D1s is not None:
            rhos_p = torch.tensor([0.5], device=device) if order == 1 else torch.linalg.solve(R[:-1, :-1], b[:-1])
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            x_t = x_t - sigma_next * B_h * pred_res

        return x_t


def get_solver(solver: str, timesteps: typing.Sequence[float]) -> Solver:
    lookup = {
        "euler": EulerSolver,
        "dpm++2m": Dpmpp2mSolver,
        "unipc": UniPCSolver,
    }
    return lookup[solver](tuple(timesteps))  # shallow copy
