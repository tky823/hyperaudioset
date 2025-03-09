class Indexer:
    """Indexer to map name into index.

    Args:
        name_to_index (dict): Dictionary to map name to index.

    Examples:

        >>> from hyperaudioset.utils.data.wordnet import load_mammal_name_to_index
        >>> from hyperaudioset.utils.data import Indexer
        >>> name_to_index = load_mammal_name_to_index()
        >>> indexer = Indexer(name_to_index)
        >>> indexer("mammal.n.01")
        639
        >>> indexer(["mammal.n.01", "dog.n.01"])
        [639, 305]

    """

    def __init__(self, name_to_index: dict[str, int]) -> None:
        super().__init__()

        self.name_to_index = name_to_index

    def __call__(self, name_or_names: str | list[str]) -> int | list[int]:
        index = self._name_to_index(name_or_names)

        return index

    def __len__(self) -> int:
        return len(self.name_to_index)

    def _name_to_index(self, name_or_names: str | list[str]) -> int | list[int]:
        if isinstance(name_or_names, str):
            index = self.name_to_index[name_or_names]

            return index
        else:
            indices = []

            for name in name_or_names:
                index = self._name_to_index(name)
                indices.append(index)

            return indices
