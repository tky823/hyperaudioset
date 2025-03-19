from typing import Any

import torch

from ..functional.hyperbolic import mobius_sub
from .manifold import ManifoldEmbedding


class PoincareEmbedding(ManifoldEmbedding):
    """Poincare embedding.

    Args:
        radius (float): Radius of Poincare ball. Default: ``1``.
        range (tuple, optional): Range of weight in initialization.
            Default: ``(-0.001, 0.001)``.

    """

    def __init__(
        self,
        *args,
        radius: float = 1,
        range: tuple[float] | None = None,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert radius > 0, "radius should be positive."

        self.radius = radius
        self.eps = eps

        self._reset_parameters(range)

    def _reset_parameters(self, range: tuple[float] | None = None) -> None:
        radius = self.radius

        if range is None:
            range = (-0.001, 0.001)

        _min, _max = range

        assert -radius < _min, f"Minimum ({_min}) should be greater than {-radius}."
        assert _max < radius, f"Maximum ({_max}) should be less than {radius}."

        self.weight.data.uniform_(_min, _max)

    def sub(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        radius = self.radius
        eps = self.eps

        return mobius_sub(input, other, radius=radius, eps=eps)

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
        radius = self.radius
        eps = self.eps

        *batch_shape, embedding_dim = input.size()
        maxnorm = radius - eps

        x = input.view(-1, embedding_dim)
        x = torch.renorm(x, p=2, dim=0, maxnorm=maxnorm)
        output = x.view(*batch_shape, embedding_dim)

        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        radius = self.radius

        x = super().forward(input)
        output = RiemannGradientFunction.apply(x, radius)

        return output


class RiemannGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, radius: float) -> torch.Tensor:
        assert not isinstance(radius, torch.Tensor)

        ctx.save_for_backward(input)
        ctx.radius = radius

        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        radius = ctx.radius

        num = 1 - torch.sum(input**2, dim=-1) / (radius**2)
        scale = (num**2) / 4
        grad_input = scale.unsqueeze(dim=-1) * grad_output

        return grad_input, None
