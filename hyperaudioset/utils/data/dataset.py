from typing import Any

from torch.utils.data import Dataset


class EvaluationDataset(Dataset):
    tags: list[str]
    hierarchy: list[dict[str, Any]]
    parent_as_positive: bool
    child_as_positive: bool

    def __getitem__(self, index: int) -> tuple[str, str, list[str]]:
        tags = self.tags
        hierarchy = self.hierarchy
        parent_as_positive = self.parent_as_positive
        child_as_positive = self.child_as_positive

        anchor_index = index
        anchor = hierarchy[anchor_index]["name"]
        parent = hierarchy[anchor_index]["parent"]
        child = hierarchy[anchor_index]["child"]

        positive_candidates = set()

        if parent_as_positive:
            positive_candidates |= set(parent)

        if child_as_positive:
            positive_candidates |= set(child)

        negative_candidates = set(tags) - set(positive_candidates) - {anchor}

        positive = sorted(list(positive_candidates))
        negative = sorted(list(negative_candidates))

        return anchor, positive, negative

    def __len__(self) -> int:
        return len(self.hierarchy)
