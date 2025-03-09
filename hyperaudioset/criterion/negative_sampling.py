import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclidNegativeSamplingLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        reduction = self.reduction

        positive_distance = self.compute_distance(anchor, positive, dim=-1)
        negative_distance = self.compute_distance(
            anchor.unsqueeze(dim=-2), negative, dim=-1
        )
        positive_distance = positive_distance.unsqueeze(dim=-1)
        distance = torch.cat([positive_distance, negative_distance], dim=-1)
        target = torch.zeros(
            positive.size()[:-1], dtype=torch.long, device=distance.device
        )

        loss = F.cross_entropy(-distance, target, reduction=reduction)

        return loss

    def compute_distance(
        self, input: torch.Tensor, other: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        return torch.linalg.vector_norm((input - other), dim=dim)
