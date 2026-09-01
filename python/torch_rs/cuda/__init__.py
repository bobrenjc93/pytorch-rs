r"""
CPU-build CUDA compatibility probes.
"""

from . import memory as memory
from .memory import empty_cache

__all__ = ["device_count", "empty_cache", "is_available", "is_initialized"]


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    return False


def device_count() -> int:
    r"""Returns the number of GPUs available."""
    return 0


def is_initialized():
    r"""Return whether PyTorch's CUDA state has been initialized."""
    return False
