import json
import os
from typing import Iterator

import torch
import torch.distributed as dist

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

                if is_symmetric:
                    weights[child_name] += 1

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
        self.epoch_index = 0  # to share random state among workers

    def __iter__(self) -> Iterator[tuple[str, str, list[str]]]:
        tags = self.tags
        hierarchy = self.hierarchy
        pair_list = self.pair_list
        num_neg_samples = self.num_neg_samples
        length = self.length
        burnin = self.burnin
        is_symmetric = self.is_symmetric
        seed = self.seed
        epoch_index = self.epoch_index

        worker_info = torch.utils.data.get_worker_info()
        is_distributed = dist.is_available() and dist.is_initialized()

        if worker_info is None:
            local_worker_id = 0
            num_local_workers = 1
        else:
            local_worker_id = worker_info.id
            num_local_workers = worker_info.num_workers

        if is_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        global_worker_id = rank * num_local_workers + local_worker_id
        num_global_workers = world_size * num_local_workers

        if burnin is None:
            raise ValueError(
                "Set burnin by calling set_burnin(True) or set_burnin(False) before iteration."
            )

        if self.generator is None:
            self.generator = torch.Generator()

        # share random state among workers to random sampling
        self.generator.manual_seed(seed + epoch_index)

        indices = torch.randint(
            0,
            len(pair_list),
            (length,),
            generator=self.generator,
        )
        indices = indices.tolist()
        indices = indices[global_worker_id::num_global_workers]

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
                negative = self.sample(
                    negative_candidates,
                    num_samples=num_neg_samples,
                    weights=self.weights,
                    dampening=self.burnin_dampening,
                    replacement=self.generator,
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

        self.epoch_index = epoch_index + 1


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
