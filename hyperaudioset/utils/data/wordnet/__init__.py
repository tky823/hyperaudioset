from ._download import load_mammal_name_to_index
from .dataset import EvaluationMammalDataset, TrainingMammalDataset

__all__ = [
    "TrainingMammalDataset",
    "EvaluationMammalDataset",
    "load_mammal_name_to_index",
]
