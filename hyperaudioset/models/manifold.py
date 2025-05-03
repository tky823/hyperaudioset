from abc import abstractmethod
from typing import Union

import torch
import torch.nn as nn


class ManifoldEmbedding(nn.Embedding):
    """Base class of manifold embedding."""

    @abstractmethod
    def add(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Add other to input."""

    @abstractmethod
    def sub(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Subtract other from input."""

    def retmap(
        self, input: torch.Tensor, root: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        """Retraction map, which is a first-order of approximation of exponential map."""
        output = input + root

        return output

    def expmap(
        self, input: torch.Tensor, root: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        """Exponential map, which can be approximated by retraction map."""
        output = self.retmap(input, root=root)

        return output

    @abstractmethod
    def proj(self, input: torch.Tensor) -> torch.Tensor:
        """Projection, which ensures input should be on manifold."""
