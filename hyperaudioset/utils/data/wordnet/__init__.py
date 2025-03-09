from ._download import load_mammal_name_to_index
from .dataset import EvaluationMammalDadataset, TrainingMammalDadataset

__all__ = [
    "TrainingMammalDadataset",
    "EvaluationMammalDadataset",
    "load_mammal_name_to_index",
]
