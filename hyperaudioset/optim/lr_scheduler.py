from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

__all__ = [
    "DummyLRScheduler",
    "BurnInLRScheduler",
]


class DummyLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        super().__init__(optimizer, last_epoch=last_epoch, **kwargs)

    def step(self, *args, **kwargs) -> None:
        pass


class BurnInLRScheduler(LambdaLR):
    def __init__(
        self, optimizer: Optimizer, burnin_step: int, burnin_scale: float, **kwargs
    ) -> None:
        def lr_lambda(epoch: int) -> float:
            if epoch < burnin_step:
                return burnin_scale
            else:
                return 1

        super().__init__(optimizer, lr_lambda, **kwargs)
