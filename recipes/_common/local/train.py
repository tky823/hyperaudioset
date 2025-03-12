import os
from logging import Logger, getLogger
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import hyperaudioset
import hyperaudioset.utils
from hyperaudioset.configs import Config
from hyperaudioset.criterion.negative_sampling import _NegativeSamplingLoss
from hyperaudioset.models.manifold import ManifoldEmbedding
from hyperaudioset.optim.lr_scheduler import BurnInLRScheduler
from hyperaudioset.utils import setup
from hyperaudioset.utils.data import Indexer
from hyperaudioset.utils.data.audioset import (
    EvaluationAudioSetDataset,
    TrainingAudioSetDataset,
)
from hyperaudioset.utils.data.wordnet import (
    EvaluationMammalDataset,
    TrainingMammalDataset,
)


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
    training_dataset: TrainingMammalDataset | TrainingAudioSetDataset = (
        hydra.utils.instantiate(config.data.dataset.train)
    )
    evaluation_dataset: EvaluationMammalDataset | EvaluationAudioSetDataset = (
        hydra.utils.instantiate(config.data.dataset.evaluate)
    )
    training_dataloader: DataLoader = hydra.utils.instantiate(
        config.data.dataloader.train, training_dataset
    )
    evaluation_dataloader: DataLoader = hydra.utils.instantiate(
        config.data.dataloader.evaluate, evaluation_dataset
    )
    model: ManifoldEmbedding = hydra.utils.instantiate(config.model)
    criterion: _NegativeSamplingLoss = hydra.utils.instantiate(config.criterion)

    model = model.to(accelerator)
    criterion = criterion.to(accelerator)

    optimizer: Optimizer = hyperaudioset.utils.instantiate_optimizer(
        config.optimizer.optimizer, model
    )
    lr_scheduer: _LRScheduler = hydra.utils.instantiate(
        config.optimizer.lr_scheduler, optimizer
    )

    training_dataset: TrainingMammalDataset | TrainingAudioSetDataset = (
        training_dataloader.dataset
    )

    if isinstance(lr_scheduer, BurnInLRScheduler):
        training_dataset.set_burnin(True)

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

        if isinstance(lr_scheduer, BurnInLRScheduler):
            burnin_step = lr_scheduer.burnin_step

            if epoch >= burnin_step:
                training_dataset.set_burnin(False)

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
    model: ManifoldEmbedding,
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
        anchor_index = indexer(anchor)
        positive_index = indexer(positive)
        negative_index = indexer(negative)

        anchor_index = torch.tensor(anchor_index, dtype=torch.long)
        positive_index = torch.tensor(positive_index, dtype=torch.long)
        negative_index = torch.tensor(negative_index, dtype=torch.long)

        anchor_index = anchor_index.to(accelerator)
        positive_index = positive_index.to(accelerator)
        negative_index = negative_index.to(accelerator)

        anchor_embedding = model(anchor_index)
        positive_embedding = model(positive_index)
        negative_embedding = model(negative_index)

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
    model: ManifoldEmbedding,
    criterion: _NegativeSamplingLoss,
    state: dict[str, Any],
) -> float:
    accelerator: str = state["accelerator"]

    evaluation_ranks = []
    evaluation_positive_samples = []

    model.eval()

    for anchor, positive, negative in dataloader:
        anchor_index = indexer(anchor)
        positive_index = indexer(positive)
        negative_index = indexer(negative)

        anchor_index = torch.tensor(anchor_index, dtype=torch.long)
        positive_index = torch.tensor(positive_index, dtype=torch.long)
        negative_index = torch.tensor(negative_index, dtype=torch.long)

        anchor_index = anchor_index.to(accelerator)
        positive_index = positive_index.to(accelerator)
        negative_index = negative_index.to(accelerator)

        num_positive_samples = positive_index.size(-1)
        num_negative_samples = negative_index.size(-1)

        if num_positive_samples == 0:
            # all samples are negative
            continue
        elif num_negative_samples == 0:
            # all samples are positive
            anchor_index = anchor_index.squeeze(dim=0)
            positive_index = positive_index.squeeze(dim=0)
            negative_index = negative_index.squeeze(dim=0)

            positive_ranks = torch.arange(
                num_positive_samples,
                dtype=torch.long,
                device=accelerator,
            )
        else:
            anchor_embedding = model(anchor_index)
            positive_embedding = model(positive_index)
            negative_embedding = model(negative_index)

            anchor_index = anchor_index.squeeze(dim=0)
            positive_index = positive_index.squeeze(dim=0)
            negative_index = negative_index.squeeze(dim=0)
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
                num_positive_samples,
                dtype=torch.long,
                device=accelerator,
            )
            indices = torch.argsort(distance)
            positive_condition = torch.isin(indices, positive_indices)
            (positive_ranks,) = torch.where(positive_condition)

        positive_ranks = positive_ranks.tolist()
        evaluation_rank = (
            sum(positive_ranks) - num_positive_samples * (num_positive_samples - 1) / 2
        )
        evaluation_ranks.append(evaluation_rank)
        evaluation_positive_samples.append(num_positive_samples)

    evaluation_mean_rank = sum(evaluation_ranks) / sum(evaluation_positive_samples)

    return evaluation_mean_rank


if __name__ == "__main__":
    main()
