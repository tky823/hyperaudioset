import os

from ._hydra import main

__all__ = [
    "hyperaudioset_cache_dir",
    "main",
]

_home_dir = os.path.expanduser("~")
hyperaudioset_cache_dir = os.getenv("HYPERAUDIOSET_CACHE_DIR") or os.path.join(
    _home_dir, ".cache", "hyperaudioset"
)
