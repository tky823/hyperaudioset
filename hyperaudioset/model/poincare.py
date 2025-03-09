import torch.nn as nn


class PoincareEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
