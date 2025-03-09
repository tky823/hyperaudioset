import os

import hydra
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from ..configs import Config, _OptimizerConfig

__all__ = [
    "hyperaudioset_cache_dir",
    "setup",
    "instantiate_optimizer",
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

    output_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(output_dir, ".hydra", "resolved_config.yaml")
    OmegaConf.save(config, path, resolve=True)

    torch.manual_seed(config.system.seed)


def instantiate_optimizer(
    config: DictConfig | _OptimizerConfig, module_or_params: nn.Module | ParamsT
) -> Optimizer:
    if isinstance(module_or_params, nn.Module):
        params = module_or_params.parameters()
    else:
        params = module_or_params

    optimizer: Optimizer = hydra.utils.instantiate(config, params)

    return optimizer
