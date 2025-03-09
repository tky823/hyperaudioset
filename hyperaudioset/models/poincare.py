import torch
import torch.nn as nn


class PoincareEmbedding(nn.Embedding):
    """Poincare embedding.

    Args:
        range (tuple, optional): Range of weight in initialization.
            Default: ``(-0.0001, 0.0001)``.

    """

    def __init__(
        self, *args, range: tuple[float] | None = None, eps: float = 1e-3, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.eps = eps

        self._reset_parameters(range)

    def _reset_parameters(self, range: tuple[float] | None = None) -> None:
        if range is None:
            range = (-0.0001, 0.0001)

        _min, _max = range

        self.weight.data.uniform_(_min, _max)

    def proj(self, embedding: torch.Tensor) -> torch.Tensor:
        eps = self.eps

        assert embedding.dim() == 2

        norm = torch.linalg.vector_norm(embedding, dim=-1, keepdim=True)
        projected_embedding = (1 - eps) * embedding / norm
        condition = norm > 1 - self.eps
        embedding = embedding.where(condition, projected_embedding)

        return embedding
