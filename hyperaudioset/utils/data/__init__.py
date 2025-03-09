from typing import Any

from .indexer import Indexer

__all__ = [
    "Indexer",
    "NegativeSamplingCollator",
]


class NegativeSamplingCollator:
    def __init__(self) -> None:
        pass

    def __call__(
        self, list_batch: list[tuple[str, str, list[str]]]
    ) -> tuple[list[str], list[str], list[list[str]]]:
        anchors = []
        positives = []
        negatives = []

        for anchor, positive, negative in list_batch:
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return anchors, positives, negatives
