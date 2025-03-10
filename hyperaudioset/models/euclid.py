import torch

from .manifold import ManifoldEmbedding


class EuclidEmbedding(ManifoldEmbedding):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def sub(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        return input - other
