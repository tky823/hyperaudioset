import importlib
import os

import hydra
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from ..configs import Config, _OptimizerConfig
from ..models.poincare import PoincareEmbedding
from ..optim.optimizer import RiemannSGD

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
    """Instantiate optimizer to support Riemann SGD.

    Args:
        config (DictConfig): Optimizer config.
        module_or_params (nn.Module or ParamsT): Parameters to be optimized.

    """
    target = config._target_
    module, name = target.rsplit(".", maxsplit=1)
    module = importlib.import_module(module)
    optimizer_class = getattr(module, name)
    optimizer_kwargs = {}

    is_rsgd = optimizer_class is RiemannSGD

    if is_rsgd:
        if isinstance(module_or_params, nn.Module):
            module = module_or_params

            assert isinstance(module, PoincareEmbedding), (
                "PoincareEmbedding should be used for RiemannSGD."
            )

            params = module.parameters()
            optimizer_kwargs["expmap"] = module.expmap
        else:
            raise ValueError(
                "Module should be given to instantiate_optimizer if optimizer is RiemannSGD."
            )
    else:
        if isinstance(module_or_params, nn.Module):
            params = module_or_params.parameters()
        else:
            params = module_or_params

    optimizer: Optimizer = hydra.utils.instantiate(config, params, **optimizer_kwargs)

    return optimizer
