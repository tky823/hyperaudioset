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
        637
        >>> indexer(["mammal.n.01", "dog.n.01"])
        [637, 305]

    """

    def __init__(self, name_to_index: dict[str, int]) -> None:
        super().__init__()

        index_to_name = {}

        for name, index in name_to_index.items():
            assert index not in index_to_name

            index_to_name[index] = name

        self.name_to_index = name_to_index
        self.index_to_name = index_to_name

    def __call__(self, name_or_names: str | list[str]) -> int | list[int]:
        index = self.encode(name_or_names)

        return index

    def __len__(self) -> int:
        return len(self.name_to_index)

    def encode(self, name_or_names: str | list[str]) -> int | list[int]:
        index = self._name_to_index(name_or_names)

        return index

    def decode(self, index_or_indices: int | list[int]) -> str | list[str]:
        name = self._index_to_name(index_or_indices)

        return name

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

    def _index_to_name(self, index_or_indices: int | list[int]) -> str | list[str]:
        if isinstance(index_or_indices, int):
            name = self.index_to_name[index_or_indices]

            return name
        else:
            names = []

            for index in index_or_indices:
                name = self._index_to_name(index)
                names.append(name)

            return names
