import os

__all__ = [
    "hyperaudioset",
]

_home_dir = os.path.expanduser("~")
hyperaudioset_cache_dir = os.getenv("HYPERAUDIOSET_CACHE_DIR") or os.path.join(
    _home_dir, ".cache", "hyperaudioset"
)
