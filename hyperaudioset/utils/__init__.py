import os

import torch
from omegaconf import DictConfig, OmegaConf

from ..configs import Config

__all__ = [
    "hyperaudioset_cache_dir",
    "setup",
]

_home_dir = os.path.expanduser("~")
hyperaudioset_cache_dir = os.getenv("HYPERAUDIOSET_CACHE_DIR") or os.path.join(
    _home_dir, ".cache", "hyperaudioset"
)


def setup(config: DictConfig | Config) -> None:
    if config.system.accelerator is None:
        if torch.cuda.is_available():
            OmegaConf.update(config.system, "accelerator", "cuda")
        else:
            OmegaConf.update(config.system, "accelerator", "cpu")

    torch.manual_seed(config.system.seed)
