import json
import os
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from ._download import download_wordnet_hierarchy


class TrainingMammalDataset(IterableDataset):
    def __init__(
        self,
        num_neg_samples: int = 1,
        parent_as_positive: bool = True,
        child_as_positive: bool = False,
        length: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()

        from ... import hyperaudioset_cache_dir

        wordnet_root = os.path.join(hyperaudioset_cache_dir, "data", "WordNet")

        url = "https://github.com/tky823/hyperaudioset/releases/download/v0.0.0/wordnet_mammal.json"

        if wordnet_root:
            os.makedirs(wordnet_root, exist_ok=True)

        filename = os.path.basename(url)
        path = os.path.join(wordnet_root, filename)
        chunk_size = 8192

        if not os.path.exists(path):
            download_wordnet_hierarchy(url, path, chunk_size=chunk_size)

        with open(path) as f:
            hierarchy: list[dict[str, str]] = json.load(f)

        tags = []
        pair_list = []

        for sample in hierarchy:
            name = sample["name"]
            tags.append(name)

            for child_name in sample["child"]:
                pair_list.append({"self": name, "child": child_name})

        if length is None:
            length = len(pair_list)

        if not parent_as_positive and not child_as_positive:
            raise ValueError(
                "Pair of parent_as_positive=False and child_as_positive=False "
                "is not supported."
            )

        self.tags = tags
        self.hierarchy = hierarchy
        self.pair_list = pair_list

        self.parent_as_positive = parent_as_positive
        self.child_as_positive = child_as_positive

        self.num_neg_samples = num_neg_samples
        self.length = length

        self.generator = None
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[str, str, list[str]]]:
        tags = self.tags
        hierarchy = self.hierarchy
        pair_list = self.pair_list
        parent_as_positive = self.parent_as_positive
        child_as_positive = self.child_as_positive
        num_neg_samples = self.num_neg_samples
        length = self.length
        seed = self.seed

        if self.generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

        indices = torch.randint(
            0,
            len(pair_list),
            (length,),
            generator=self.generator,
        )
        indices = indices.tolist()

        for pair_index in indices:
            pair = pair_list[pair_index]

            if pair["self"] == "mammal.n.01":
                # to avoid empty negative candidates
                if parent_as_positive:
                    anchor = pair["child"]
                    positive = pair["self"]
                else:
                    if child_as_positive:
                        raise ValueError(
                            "If parent_as_positive=False and child_as_positive=True, "
                            "mammal.n.01 should be excluded from self in pairs."
                        )
                    else:
                        raise ValueError(
                            "Pair of parent_as_positive=False and child_as_positive=False "
                            "is not supported."
                        )
            else:
                if parent_as_positive:
                    if child_as_positive:
                        if torch.rand((), generator=self.generator) < 0.5:
                            anchor = pair["self"]
                            positive = pair["child"]
                        else:
                            anchor = pair["child"]
                            positive = pair["self"]
                    else:
                        anchor = pair["child"]
                        positive = pair["self"]
                else:
                    if child_as_positive:
                        anchor = pair["self"]
                        positive = pair["child"]
                    else:
                        raise ValueError(
                            "Pair of parent_as_positive=False and child_as_positive=False "
                            "is not supported."
                        )

            anchor_index = tags.index(anchor)
            parent = hierarchy[anchor_index]["parent"]
            child = hierarchy[anchor_index]["child"]

            positive_candidates = set()

            if parent_as_positive:
                positive_candidates |= set(parent)

            if child_as_positive:
                positive_candidates |= set(child)

            negative_candidates = set(tags) - set(positive_candidates) - {anchor}

            negative_indices = torch.randint(
                0, len(negative_candidates), (num_neg_samples,)
            )
            negative_indices = negative_indices.tolist()

            negative = []

            for negative_index in negative_indices:
                _negative = hierarchy[negative_index]["name"]
                negative.append(_negative)

            yield anchor, positive, negative

    def __len__(self) -> int:
        return self.length


class EvaluationMammalDataset(Dataset):
    def __init__(
        self,
        parent_as_positive: bool = True,
        child_as_positive: bool = False,
    ) -> None:
        super().__init__()

        from ... import hyperaudioset_cache_dir

        wordnet_root = os.path.join(hyperaudioset_cache_dir, "data", "WordNet")

        url = "https://github.com/tky823/hyperaudioset/releases/download/v0.0.0/wordnet_mammal.json"

        if wordnet_root:
            os.makedirs(wordnet_root, exist_ok=True)

        filename = os.path.basename(url)
        path = os.path.join(wordnet_root, filename)
        chunk_size = 8192

        if not os.path.exists(path):
            download_wordnet_hierarchy(url, path, chunk_size=chunk_size)

        with open(path) as f:
            hierarchy: list[dict[str, str]] = json.load(f)

        tags = []
        pair_list = []

        for sample in hierarchy:
            name = sample["name"]
            tags.append(name)

            for child_name in sample["child"]:
                pair_list.append({"self": name, "child": child_name})

        if not parent_as_positive and not child_as_positive:
            raise ValueError(
                "Pair of parent_as_positive=False and child_as_positive=False "
                "is not supported."
            )

        self.tags = tags
        self.hierarchy = hierarchy
        self.pair_list = pair_list

        self.parent_as_positive = parent_as_positive
        self.child_as_positive = child_as_positive

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
