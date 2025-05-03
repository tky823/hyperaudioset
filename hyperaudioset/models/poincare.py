from typing import Any, Optional, Union

import torch

from ..functional.hyperbolic import mobius_add, mobius_sub
from .manifold import ManifoldEmbedding


class PoincareEmbedding(ManifoldEmbedding):
    """Poincare embedding.

    Args:
        curvature (float): Negative curvature of Poincare ball. Default: ``-1``.
        dim (int): Dimension of Poincare ball. Default: ``-1``.
        range (tuple, optional): Range of weight in initialization.
            Default: ``(-0.001, 0.001)``.

    """

    def __init__(
        self,
        *args,
        curvature: float = -1,
        dim: int = -1,
        range: Optional[tuple[float]] = None,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert curvature < 0, "curvature should be negative."

        self.dim = dim
        self.curvature = curvature
        self.eps = eps

        self._reset_parameters(range)

    def _reset_parameters(self, range: Optional[tuple[float]] = None) -> None:
        curvature = self.curvature

        if range is None:
            range = (-0.001, 0.001)

        radius = 1 / ((-curvature) ** 0.5)
        _min, _max = range

        assert -radius < _min, f"Minimum ({_min}) should be greater than {-radius}."
        assert _max < radius, f"Maximum ({_max}) should be less than {radius}."

        self.weight.data.uniform_(_min, _max)

    def add(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        curvature = self.curvature
        dim = self.dim
        eps = self.eps

        return mobius_add(input, other, curvature=curvature, dim=dim, eps=eps)

    def sub(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        curvature = self.curvature
        dim = self.dim
        eps = self.eps

        return mobius_sub(input, other, curvature=curvature, dim=dim, eps=eps)

    def expmap(
        self, input: torch.Tensor, root: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        """Exponential map, which can be approximated by retraction map."""
        curvature = self.curvature
        dim = self.dim
        eps = self.eps

        _curvature = (-curvature) ** 0.5
        norm = torch.linalg.vector_norm(input, dim=dim, keepdim=True)
        norm = torch.clamp(norm, min=eps)
        normalized_input = input / norm
        conformal_factor = 2 / (
            1 + curvature * torch.sum(input**2, dim=dim, keepdim=True)
        )
        y = (
            torch.tanh(_curvature * conformal_factor * norm / 2)
            * normalized_input
            / _curvature
        )
        output = mobius_add(root, y, curvature=curvature, dim=dim, eps=eps)

        return output

    def proj(self, input: torch.Tensor) -> torch.Tensor:
        """Projection, which ensures input should be on Poincare disk."""
        curvature = self.curvature
        dim = self.dim
        eps = self.eps

        assert dim == -1, "Only dim = -1 is supported"

        *batch_shape, embedding_dim = input.size()

        radius = 1 / ((-curvature) ** 0.5)

        x = input.view(-1, embedding_dim)
        x = torch.renorm(x, p=2, dim=0, maxnorm=radius - eps)
        output = x.view(*batch_shape, embedding_dim)

        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        curvature = self.curvature
        dim = self.dim

        x = super().forward(input)
        output = PoincareGradientFunction.apply(x, curvature, dim)

        return output


class PoincareGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, curvature: float, dim: int
    ) -> torch.Tensor:
        assert not isinstance(curvature, torch.Tensor)

        ctx.save_for_backward(input)
        ctx.curvature = curvature
        ctx.dim = dim

        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        curvature = ctx.curvature
        dim = ctx.dim

        assert dim == -1, "Only dim = -1 is supported."

        conformal_factor = 2 / (1 + curvature * torch.sum(input**2, dim=dim))
        scale = 1 / (conformal_factor**2)
        grad_input = scale.unsqueeze(dim=dim) * grad_output

        return grad_input, None, None
