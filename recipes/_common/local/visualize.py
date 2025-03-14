import os
from logging import getLogger
from typing import Any, Callable

import hydra
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import hyperaudioset
import hyperaudioset.utils
from hyperaudioset.configs import Config
from hyperaudioset.models.manifold import ManifoldEmbedding
from hyperaudioset.utils import setup
from hyperaudioset.utils.data import Indexer
from hyperaudioset.utils.data.audioset import (
    EvaluationAudioSetDataset,
)
from hyperaudioset.utils.data.wordnet import (
    EvaluationMammalDataset,
)


@hyperaudioset.main()
def main(config: Config | DictConfig) -> None:
    setup(config)

    accelerator = config.system.accelerator

    logger = getLogger(__name__)
    logger.info(config)

    exp_dir = config.exp_dir
    root = config.root
    depth = config.depth

    indexer: Indexer = hydra.utils.instantiate(config.data.indexer)
    evaluation_dataset: EvaluationMammalDataset | EvaluationAudioSetDataset = (
        hydra.utils.instantiate(config.data.dataset.evaluate)
    )
    evaluation_dataloader: DataLoader = hydra.utils.instantiate(
        config.data.dataloader.evaluate, evaluation_dataset
    )
    model: ManifoldEmbedding = hydra.utils.instantiate(config.model)

    path = os.path.join(exp_dir, "best.pth")
    state_dict = torch.load(
        path, map_location=lambda storage, loc: storage, weights_only=True
    )
    model.load_state_dict(state_dict)
    model = model.to(accelerator)

    state = {
        "accelerator": accelerator,
        "logger": logger,
    }

    embeddings = embed_for_one_epoch(indexer, evaluation_dataloader, model, state)
    shifted_embeddings = shift_embeddings(embeddings, root=root, sub=model.sub)

    fig = go.Figure()
    _ = visualize(
        fig,
        evaluation_dataset.hierarchy,
        indexer,
        shifted_embeddings,
        root=root,
        depth=depth,
        colors=px.colors.qualitative.Plotly,
    )
    fig.update_layout(showlegend=False)
    fig.show()


@torch.no_grad()
def embed_for_one_epoch(
    indexer: Indexer,
    dataloader: DataLoader,
    model: nn.Module,
    state: dict[str, Any],
) -> dict[str, torch.Tensor]:
    accelerator: str = state["accelerator"]

    embeddings = {}

    model.eval()

    for anchor, _, _ in dataloader:
        anchor_index = indexer(anchor)
        anchor_index = torch.tensor(anchor_index)
        anchor_index = anchor_index.to(accelerator)

        anchor_embedding = model(anchor_index)

        anchor = anchor[0]
        anchor_embedding = anchor_embedding.squeeze(dim=0)

        embeddings[anchor] = anchor_embedding

    return embeddings


def shift_embeddings(
    embeddings: dict[str, torch.Tensor],
    root: str,
    sub: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    shifted_embeddings = {}

    root_embedding = embeddings[root]

    for name, embedding in embeddings.items():
        shifted_embeddings[name] = sub(embedding, root_embedding)

    return shifted_embeddings


@torch.no_grad()
def visualize(
    fig: go.Figure,
    hierarchy: dict[str, Any],
    indexer: Indexer,
    embeddings: dict[str, torch.Tensor],
    root: str,
    depth: int = 1,
    colors: list[str] | None = None,
) -> list[float]:
    if colors is None:
        colors = px.colors.qualitative.Plotly

    root_index = indexer.encode(root)
    root_embedding = embeddings[root]
    root_embedding = root_embedding.to("cpu")
    root_coordinate = root_embedding.tolist()

    sample = hierarchy[root_index]

    if depth > 0:
        for child in sample["child"]:
            child_coordinate = visualize(
                fig,
                hierarchy,
                indexer,
                embeddings,
                child,
                depth=depth - 1,
                colors=colors[1:],
            )

            x0, y0 = root_coordinate
            x1, y1 = child_coordinate
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    opacity=0.3,
                    marker=dict(
                        color="gray",
                        size=10,
                        symbol="arrow-bar-up",
                        angleref="previous",
                    ),
                    hoverinfo="none",
                )
            )

    x, y = root_coordinate
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            zorder=1,
            name=root,
            marker=dict(
                color=colors[0],
                size=10,
            ),
            hoverinfo="name",
        )
    )

    return root_coordinate


if __name__ == "__main__":
    main()
