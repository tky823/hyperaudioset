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
        proj: Callable[[torch.Tensor], torch.Tensor] | None = None,
        use_retmap: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr,
        )

        super().__init__(params, defaults)

        self.expmap = expmap
        self.proj = proj
        self.use_retmap = use_retmap

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

                if self.use_retmap or self.expmap is None:
                    updated = -lr * grad + param.data
                else:
                    updated = self.expmap(-lr * grad, root=param.data)

                if self.proj is not None:
                    updated = self.proj(updated)

                param.data.copy_(updated)

        return loss
