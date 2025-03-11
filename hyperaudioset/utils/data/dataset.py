from typing import Any

from torch.utils.data import Dataset


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
