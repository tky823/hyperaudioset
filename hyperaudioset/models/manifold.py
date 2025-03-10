from abc import abstractmethod

import torch
import torch.nn as nn


class ManifoldEmbedding(nn.Embedding):
    @abstractmethod
    def sub(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Subtract other from input."""
        pass
