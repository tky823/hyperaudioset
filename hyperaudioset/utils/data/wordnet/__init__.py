from ._download import load_mammal_name_to_index
from .dataset import MammalEvaluationDadataset, MammalTrainingDadataset

__all__ = [
    "MammalTrainingDadataset",
    "MammalEvaluationDadataset",
    "load_mammal_name_to_index",
]
