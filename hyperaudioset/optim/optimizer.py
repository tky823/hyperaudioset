from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, ParamsT


class RiemannSGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        proj: Callable | None = None,
    ) -> None:
        defaults = dict(
            lr=lr,
        )

        super().__init__(params, defaults)

        self.proj = proj

    def step(self, closure: Callable = None) -> Any:
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            lr = param_group["lr"]
            params: list[nn.Parameter] = param_group["params"]

            for param in params:
                grad = param.grad.data
                projected = self.proj(param.data - lr * grad)
                param.data.copy_(projected)

        return loss
