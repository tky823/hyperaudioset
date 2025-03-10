import math
from typing import Any

import torch

from ..functional.hyperbolic import mobius_add
from .manifold import ManifoldEmbedding


class PoincareEmbedding(ManifoldEmbedding):
    """Poincare embedding.

    Args:
        curvature (float): Nonnegative curvature.
        range (tuple, optional): Range of weight in initialization.
            Default: ``(-0.0001, 0.0001)``.

    """

    def __init__(
        self,
        *args,
        curvature: float = 1,
        range: tuple[float] | None = None,
        eps: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert curvature > 0, "curvature should be positive."

        self.curvature = curvature
        self.eps = eps

        self._reset_parameters(range)

    def _reset_parameters(self, range: tuple[float] | None = None) -> None:
        curvature = self.curvature

        if range is None:
            range = (-0.0001, 0.0001)

        _min, _max = range

        assert -1 / math.sqrt(curvature) < _min
        assert _max < 1 / math.sqrt(curvature)

        self.weight.data.uniform_(_min, _max)

    def sub(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        curvature = self.curvature

        return mobius_add(input, -other, curvature=curvature)

    def proj(self, embedding: torch.Tensor) -> torch.Tensor:
        curvature = self.curvature
        eps = self.eps

        assert embedding.dim() == 2

        allowed_norm = 1 / math.sqrt(curvature) - eps
        norm = torch.linalg.vector_norm(embedding, dim=-1, keepdim=True)
        projected_embedding = allowed_norm * embedding / norm
        condition = norm > allowed_norm
        embedding = torch.where(condition, projected_embedding, embedding)

        return embedding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = super().forward(input)
        output = RiemannGradientFunction.apply(x)

        return output


class RiemannGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)

        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors

        num = 1 - torch.sum(input**2, dim=-1)
        scale = (num**2) / 4
        grad_input = scale.unsqueeze(dim=-1) * grad_output

        return grad_input
