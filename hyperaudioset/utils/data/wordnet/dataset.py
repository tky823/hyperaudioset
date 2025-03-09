import json
import os
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from ._download import download_wordnet_hierarchy


class TrainingMammalDadataset(IterableDataset):
    def __init__(
        self,
        num_neg_samples: int = 1,
        length: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()

        from ... import hyperaudioset_cache_dir

        wordnet_root = os.path.join(hyperaudioset_cache_dir, "data", "WordNet")

        url = "https://github.com/tky823/Audyn/releases/download/v0.0.5/wordnet_mammal.json"

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

        self.tags = tags
        self.hierarchy = hierarchy
        self.pair_list = pair_list

        self.num_neg_samples = num_neg_samples
        self.length = length

        self.generator = None
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[int, int, list[int]]]:
        tags = self.tags
        hierarchy = self.hierarchy
        pair_list = self.pair_list
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

            if torch.rand((), generator=self.generator) < 0.5:
                anchor = pair["self"]
                positive = pair["child"]
            else:
                anchor = pair["child"]
                positive = pair["self"]

            anchor_index = tags.index(anchor)
            parent = hierarchy[anchor_index]["parent"]
            child = hierarchy[anchor_index]["child"]
            positive_candidates = set(parent) | set(child)
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


class EvaluationMammalDadataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        from ... import hyperaudioset_cache_dir

        wordnet_root = os.path.join(hyperaudioset_cache_dir, "data", "WordNet")

        url = "https://github.com/tky823/Audyn/releases/download/v0.0.5/wordnet_mammal.json"

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

        self.tags = tags
        self.hierarchy = hierarchy
        self.pair_list = pair_list

    def __getitem__(self, index: int) -> tuple[int, int, list[int]]:
        tags = self.tags
        hierarchy = self.hierarchy
        pair_list = self.pair_list

        pair = pair_list[index]

        anchor = pair["self"]

        anchor_index = tags.index(anchor)
        parent = hierarchy[anchor_index]["parent"]
        child = hierarchy[anchor_index]["child"]
        positive_candidates = set(parent) | set(child)
        negative_candidates = set(tags) - set(positive_candidates) - {anchor}

        positive = []

        for positive_index in range(len(positive_candidates)):
            _positive = hierarchy[positive_index]["name"]
            positive.append(_positive)

        negative = []

        for negative_index in range(len(negative_candidates)):
            _negative = hierarchy[negative_index]["name"]
            negative.append(_negative)

        return anchor, positive, negative

    def __len__(self) -> int:
        return len(self.pair_list)
