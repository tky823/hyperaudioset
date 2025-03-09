class Indexer:
    """Indexer to map name into index.

    Args:
        name_to_index (dict): Dictionary to map name to index.

    """

    def __init__(self, name_to_index: dict[str, int]) -> None:
        super().__init__()

        self.name_to_index = name_to_index

    def __call__(self, name_or_names: str | list[str]) -> int | list[int]:
        if isinstance(name_or_names, str):
            index = self._name_to_index(name_or_names)

            return index
        else:
            indices = []

            for name in name_or_names:
                index = self._name_to_index(name)
                indices.append(index)

            return indices

    def __len__(self) -> int:
        return len(self.name_to_index)

    def _name_to_index(self, name: str) -> int:
        index = self.name_to_index[name]

        return index
