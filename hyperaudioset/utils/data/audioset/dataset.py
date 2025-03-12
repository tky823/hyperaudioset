import json
import os
from typing import Iterator

import torch

from ..dataset import EvaluationDataset, TrainingDataset
from ._download import download_audioset_hierarchy


class TrainingAudioSetDataset(TrainingDataset):
    def __init__(
        self,
        num_neg_samples: int = 1,
        length: int | None = None,
        burnin_dampening: float = 1,
        is_symmetric: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if is_symmetric:
            raise NotImplementedError("is_symmetric=True is not fully implemented.")

        from ... import hyperaudioset_cache_dir

        audioset_root = os.path.join(hyperaudioset_cache_dir, "data", "AudioSet")

        url = "https://github.com/tky823/hyperaudioset/releases/download/v0.0.0/audioset.json"

        if audioset_root:
            os.makedirs(audioset_root, exist_ok=True)

        filename = os.path.basename(url)
        path = os.path.join(audioset_root, filename)
        chunk_size = 8192

        if not os.path.exists(path):
            download_audioset_hierarchy(url, path, chunk_size=chunk_size)

        with open(path) as f:
            hierarchy: list[dict[str, str]] = json.load(f)

        tags = []
        pair_list = []
        weights = {}

        for sample in hierarchy:
            name = sample["name"]
            tags.append(name)

            if name not in weights:
                weights[name] = 0

            for child_name in sample["child"]:
                pair_list.append({"self": name, "child": child_name})

                weights[name] += 1

                if child_name not in weights:
                    weights[child_name] = 0

        if length is None:
            length = len(pair_list)

        self.tags = tags
        self.hierarchy = hierarchy
        self.pair_list = pair_list

        self.num_neg_samples = num_neg_samples
        self.length = length
        self.burnin = None
        self.burnin_dampening = burnin_dampening
        self.weights = weights

        self.is_symmetric = is_symmetric

        self.generator = None
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[str, str, list[str]]]:
        tags = self.tags
        hierarchy = self.hierarchy
        pair_list = self.pair_list
        num_neg_samples = self.num_neg_samples
        length = self.length
        burnin = self.burnin
        is_symmetric = self.is_symmetric
        seed = self.seed

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            pass
        else:
            assert worker_info.num_workers == 1, "Multiple workers are not supported."

        if burnin is None:
            raise ValueError(
                "Set burnin by calling set_burnin(True) or set_burnin(False) before iteration."
            )

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

            if is_symmetric:
                if torch.rand((), generator=self.generator) < 0.5:
                    anchor = pair["self"]
                    positive = pair["child"]
                else:
                    anchor = pair["child"]
                    positive = pair["self"]
            else:
                anchor = pair["child"]
                positive = pair["self"]

            anchor_index = tags.index(anchor)
            parent = hierarchy[anchor_index]["parent"]
            child = hierarchy[anchor_index]["child"]

            if is_symmetric:
                positive_candidates = set(parent) | set(child)
            else:
                positive_candidates = set(parent)

            negative_candidates = set(tags) - positive_candidates - {anchor}
            negative_candidates = sorted(list(negative_candidates))

            if burnin:
                weights = self.weights
                burnin_dampening = self.burnin_dampening

                negative_candidate_weights = []

                for _negative in negative_candidates:
                    _weight = weights[_negative] ** burnin_dampening
                    negative_candidate_weights.append(_weight)

                negative_candidate_weights = torch.tensor(
                    negative_candidate_weights, dtype=torch.float
                )
                negative_indices = torch.multinomial(
                    negative_candidate_weights,
                    num_neg_samples,
                    replacement=False,
                    generator=self.generator,
                )
            else:
                negative_indices = torch.randperm(
                    len(negative_candidates), generator=self.generator
                )
                negative_indices = negative_indices[:num_neg_samples]

            negative_indices = negative_indices.tolist()

            negative = []

            for negative_index in negative_indices:
                _negative = negative_candidates[negative_index]
                negative.append(_negative)

            yield anchor, positive, negative


class EvaluationAudioSetDataset(EvaluationDataset):
    def __init__(
        self,
        is_symmetric: bool = False,
    ) -> None:
        super().__init__()

        from ... import hyperaudioset_cache_dir

        audioset_root = os.path.join(hyperaudioset_cache_dir, "data", "AudioSet")

        url = "https://github.com/tky823/hyperaudioset/releases/download/v0.0.0/audioset.json"

        if audioset_root:
            os.makedirs(audioset_root, exist_ok=True)

        filename = os.path.basename(url)
        path = os.path.join(audioset_root, filename)
        chunk_size = 8192

        if not os.path.exists(path):
            download_audioset_hierarchy(url, path, chunk_size=chunk_size)

        with open(path) as f:
            hierarchy: list[dict[str, str]] = json.load(f)

        tags = []

        for sample in hierarchy:
            name = sample["name"]
            tags.append(name)

        self.tags = tags
        self.hierarchy = hierarchy

        self.is_symmetric = is_symmetric
