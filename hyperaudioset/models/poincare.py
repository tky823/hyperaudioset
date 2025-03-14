import math
from typing import Any

import torch

from ..functional.hyperbolic import mobius_sub
from .manifold import ManifoldEmbedding


class PoincareEmbedding(ManifoldEmbedding):
    """Poincare embedding.

    Args:
        curvature (float): Negative curvature. Default: ``-1``.
        range (tuple, optional): Range of weight in initialization.
            Default: ``(-0.001, 0.001)``.

    """

    def __init__(
        self,
        *args,
        curvature: float = -1,
        range: tuple[float] | None = None,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert curvature < 0, "curvature should be negative."

        self.curvature = curvature
        self.eps = eps

        self._reset_parameters(range)

    def _reset_parameters(self, range: tuple[float] | None = None) -> None:
        curvature = self.curvature

        if range is None:
            range = (-0.001, 0.001)

        _min, _max = range

        assert -1 / math.sqrt(-curvature) < _min
        assert _max < 1 / math.sqrt(-curvature)

        self.weight.data.uniform_(_min, _max)

    def sub(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        curvature = self.curvature
        eps = self.eps

        return mobius_sub(input, other, curvature=curvature, eps=eps)

    def retmap(
        self, input: torch.Tensor, point: torch.Tensor | float = 0
    ) -> torch.Tensor:
        """Retraction map, which is a first-order of approximation of exponential map."""
        output = input + point

        return output

    def expmap(
        self, input: torch.Tensor, point: torch.Tensor | float = 0
    ) -> torch.Tensor:
        """Exponential map, which is approximated by retraction map here."""
        output = self.retmap(input, point=point)

        return output

    def proj(self, input: torch.Tensor) -> torch.Tensor:
        """Projection, which ensures input should be on Poincare disk."""
        curvature = self.curvature
        eps = self.eps

        *batch_shape, embedding_dim = input.size()
        maxnorm = 1 / math.sqrt(-curvature) - eps

        x = input.view(-1, embedding_dim)
        x = torch.renorm(x, p=2, dim=0, maxnorm=maxnorm)
        output = x.view(*batch_shape, embedding_dim)

        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        curvature = self.curvature

        x = super().forward(input)
        output = RiemannGradientFunction.apply(x, curvature)

        return output


class RiemannGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, curvature: float) -> torch.Tensor:
        assert not isinstance(curvature, torch.Tensor)

        ctx.save_for_backward(input)
        ctx.curvature = curvature

        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        curvature = ctx.curvature

        num = 1 + curvature * torch.sum(input**2, dim=-1)
        scale = (num**2) / 4
        grad_input = scale.unsqueeze(dim=-1) * grad_output

        return grad_input, None
