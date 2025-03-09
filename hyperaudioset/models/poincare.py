import torch.nn as nn


class PoincareEmbedding(nn.Embedding):
    """Poincare embedding.

    Args:
        range (tuple, optional): Range of weight in initialization.
            Default: ``(-0.0001, 0.0001)``.

    """

    def __init__(self, *args, range: tuple[float] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if range is None:
            range = (-0.0001, 0.0001)

        _min, _max = range

        self.weight.data.uniform_(_min, _max)
