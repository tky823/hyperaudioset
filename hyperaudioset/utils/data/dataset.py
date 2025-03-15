from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset


class TrainingDataset(IterableDataset):
    length: int
    burnin: bool | None

    def set_burnin(self, burnin: bool) -> None:
        self.burnin = burnin

    @staticmethod
    def sample(
        candidates: list[str],
        num_samples: int = 1,
        weights: dict[str, float] | None = None,
        dampening: float = 1,
        replacement: bool = False,
        generator: torch.Generator | None = None,
    ) -> list[str]:
        """Sample from candidates based on weights.

        Args:
            weights (dict): Dictionary that maps candidate to weight.
            dampening (float): Dampening parameter for negative sampling.

        Returns:
            list: List of sampled candidates.

        """
        candidate_weights = []

        for candidate in candidates:
            if weights is None:
                weight = 1
            else:
                weight = weights[candidate]

            candidate_weights.append(weight**dampening)

        candidate_weights = torch.tensor(candidate_weights, dtype=torch.float)
        indices = torch.multinomial(
            candidate_weights,
            num_samples,
            replacement=replacement,
            generator=generator,
        )
        indices = indices.tolist()

        samples = []

        for index in indices:
            _sample = candidates[index]
            samples.append(_sample)

        return indices

    def __len__(self) -> int:
        return self.length


class EvaluationDataset(Dataset):
    tags: list[str]
    hierarchy: list[dict[str, Any]]
    is_symmetric: bool

    def __getitem__(self, index: int) -> tuple[str, str, list[str]]:
        tags = self.tags
        hierarchy = self.hierarchy
        is_symmetric = self.is_symmetric

        anchor_index = index
        anchor = hierarchy[anchor_index]["name"]
        parent = hierarchy[anchor_index]["parent"]
        child = hierarchy[anchor_index]["child"]

        if is_symmetric:
            positive_candidates = set(parent) | set(child)
        else:
            positive_candidates = set(parent)

        negative_candidates = set(tags) - set(positive_candidates) - {anchor}

        positive = sorted(list(positive_candidates))
        negative = sorted(list(negative_candidates))

        return anchor, positive, negative

    def __len__(self) -> int:
        return len(self.hierarchy)
