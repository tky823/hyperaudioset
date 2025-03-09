import os
from logging import Logger, getLogger
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard.writer import SummaryWriter

import hyperaudioset
import hyperaudioset.utils
from hyperaudioset.configs import Config
from hyperaudioset.criterion.negative_sampling import _NegativeSamplingLoss
from hyperaudioset.utils import setup
from hyperaudioset.utils.data import Indexer


@hyperaudioset.main()
def main(config: Config | DictConfig) -> None:
    setup(config)

    accelerator = config.system.accelerator

    logger = getLogger(__name__)
    logger.info(config)

    epochs = config.epochs
    exp_dir = config.exp_dir
    writer = SummaryWriter(config.tensorboard_dir)

    indexer: Indexer = hydra.utils.instantiate(config.data.indexer)
    training_dataset: IterableDataset = hydra.utils.instantiate(
        config.data.dataset.train
    )
    evaluation_dataset: IterableDataset = hydra.utils.instantiate(
        config.data.dataset.evaluate
    )
    training_dataloader: DataLoader = hydra.utils.instantiate(
        config.data.dataloader.train, training_dataset
    )
    evaluation_dataloader: DataLoader = hydra.utils.instantiate(
        config.data.dataloader.evaluate, evaluation_dataset
    )
    model: nn.Module = hydra.utils.instantiate(config.model)
    criterion: _NegativeSamplingLoss = hydra.utils.instantiate(config.criterion)

    model = model.to(accelerator)
    criterion = criterion.to(accelerator)

    optimizer: Optimizer = hyperaudioset.utils.instantiate_optimizer(
        config.optimizer.optimizer, model
    )
    lr_scheduer: _LRScheduler = hydra.utils.instantiate(
        config.optimizer.lr_scheduler, optimizer
    )

    best_training_loss = float("inf")
    iteration = 0

    state = {
        "accelerator": accelerator,
        "logger": logger,
        "writer": writer,
        "iteration": iteration,
        "epoch": 0,
        "epochs": epochs,
    }

    for epoch in range(epochs):
        state["epoch"] = epoch
        training_loss = []
        evaluation_mean_rank = []

        training_loss = train_for_one_epoch(
            indexer,
            training_dataloader,
            model,
            criterion,
            optimizer,
            state,
        )
        evaluation_mean_rank = evaluate_for_one_epoch(
            indexer, evaluation_dataloader, model, criterion, state
        )

        writer.add_scalar("training_loss (epoch)", training_loss, global_step=epoch + 1)
        writer.add_scalar(
            "evaluation_mean_rank (epoch)", evaluation_mean_rank, global_step=epoch + 1
        )

        lr_scheduer.step()

        msg = f"[Epoch {epoch + 1}/{epochs}] training_loss: {training_loss}"
        logger.info(msg)
        msg = (
            f"[Epoch {epoch + 1}/{epochs}] evaluation_mean_rank: {evaluation_mean_rank}"
        )
        logger.info(msg)

        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            unwrapped_model = model.module
        else:
            unwrapped_model = model

        state_dict = unwrapped_model.state_dict()

        path = os.path.join(exp_dir, "last.pth")
        torch.save(state_dict, path)

        if training_loss < best_training_loss:
            path = os.path.join(exp_dir, "best.pth")
            torch.save(state_dict, path)


def train_for_one_epoch(
    indexer: Indexer,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: _NegativeSamplingLoss,
    optimizer: Optimizer,
    state: dict[str, Any],
) -> float:
    accelerator: str = state["accelerator"]
    logger: Logger = state["logger"]
    writer: SummaryWriter = state["writer"]
    iteration: int = state["iteration"]
    epoch: int = state["epoch"]
    epochs: int = state["epochs"]

    training_loss = []

    model.train()

    for anchor, positive, negative in dataloader:
        anchor = indexer(anchor)
        positive = indexer(positive)
        negative = indexer(negative)

        anchor = torch.tensor(anchor)
        positive = torch.tensor(positive)
        negative = torch.tensor(negative)

        anchor = anchor.to(accelerator)
        positive = positive.to(accelerator)
        negative = negative.to(accelerator)

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())

        writer.add_scalar(
            "training_loss (iteration)", loss.item(), global_step=iteration + 1
        )

        msg = f"[Epoch {epoch + 1}/{epochs}, Iter {iteration + 1}] training_loss: {loss.item()}"
        logger.info(msg)

        iteration += 1

    state["iteration"] = iteration
    state["epoch"] = epoch + 1

    training_loss = sum(training_loss) / len(training_loss)

    return training_loss


@torch.no_grad()
def evaluate_for_one_epoch(
    indexer: Indexer,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: _NegativeSamplingLoss,
    state: dict[str, Any],
) -> float:
    accelerator: str = state["accelerator"]

    evaluation_mean_rank = []

    model.eval()

    for anchor, positive, negative in dataloader:
        anchor = indexer(anchor)
        positive = indexer(positive)
        negative = indexer(negative)

        anchor = torch.tensor(anchor)
        positive = torch.tensor(positive)
        negative = torch.tensor(negative)

        anchor = anchor.to(accelerator)
        positive = positive.to(accelerator)
        negative = negative.to(accelerator)

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        anchor = anchor.squeeze(dim=0)
        positive = positive.squeeze(dim=0)
        negative = negative.squeeze(dim=0)
        anchor_embedding = anchor_embedding.squeeze(dim=0)
        positive_embedding = positive_embedding.squeeze(dim=0)
        negative_embedding = negative_embedding.squeeze(dim=0)

        _positive_distance = criterion.compute_distance(
            anchor_embedding, positive_embedding
        )
        _negative_distance = criterion.compute_distance(
            anchor_embedding, negative_embedding
        )
        distance = torch.cat([_positive_distance, _negative_distance], dim=-1)
        positive_indices = torch.arange(
            positive.size(-1), dtype=torch.long, device=positive.device
        )

        indices = torch.argsort(distance)
        positive_condition = torch.isin(indices, positive_indices)
        (positive_ranks,) = torch.where(positive_condition)
        positive_ranks = positive_ranks.tolist()

        mean_rank = sum(positive_ranks) / len(positive_ranks)
        evaluation_mean_rank.append(mean_rank)

    evaluation_mean_rank = sum(evaluation_mean_rank) / len(evaluation_mean_rank)

    return evaluation_mean_rank


if __name__ == "__main__":
    main()
