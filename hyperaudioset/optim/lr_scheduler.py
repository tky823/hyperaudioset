from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
    "DummyLRScheduler",
]


class DummyLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        super().__init__(optimizer, last_epoch=last_epoch, **kwargs)

    def step(self, *args, **kwargs) -> None:
        pass
