from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, ParamsT


class RiemannSGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        expmap: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        defaults = dict(
            lr=lr,
        )

        super().__init__(params, defaults)

        self.expmap = expmap

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
                projected = self.expmap(-lr * grad, origin=param.data)
                param.data.copy_(projected)

        return loss
