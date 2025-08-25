import math
from dataclasses import dataclass

from torch import Tensor


# make solver a class so that it can store extra data i.e. stateful
@dataclass
class Solver:
    def step(self, latents: Tensor, v: Tensor, timesteps: list[float], i: int): ...


class EulerSolver(Solver):
    def step(self, latents: Tensor, v: Tensor, timesteps: list[float], i: int):
        # move from timesteps[i] to timesteps[i+1]
        return latents.add(v, alpha=timesteps[i + 1] - timesteps[i])


class Dpmpp2mSolver(Solver):
    """DPM-Solver++(2M) https://arxiv.org/abs/2211.01095"""

    prev_data_pred: Tensor | None = None

    def step(self, latents: Tensor, v: Tensor, timesteps: list[float], i: int):
        # the implementation below has been simplified for flow matching / rectified flow
        # with sigma(t) = t and alpha(t) = 1-t
        # coincidentally (or not?), this results in identical calculations as k-diffusion implementation
        # https://github.com/crowsonkb/k-diffusion/blob/21d12c91ad4550e8fcf3308ff9fe7116b3f19a08/k_diffusion/sampling.py#L585-L607
        data_pred = latents.add(v, alpha=-timesteps[i])  # data prediction model

        if i == 0:
            latents = data_pred.lerp(latents, timesteps[i + 1] / timesteps[i])
        elif timesteps[i + 1] == 0.0:  # avoid log(0). note that lim x.log(x) when x->0 is 0.
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


class UnipcSolver(Solver):
    def step(self, latents, v, timesteps, i):
        raise NotImplementedError()


def get_solver(solver: str) -> Solver:
    lookup = {
        "euler": EulerSolver,
        "dpm++2m": Dpmpp2mSolver,
        "unipc": UnipcSolver,
    }
    return lookup[solver]()
