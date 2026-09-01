r"""
CPU-build CUDA compatibility probes and memory query no-ops.
"""

from . import memory as memory
from .memory import (
    empty_cache,
    max_memory_allocated,
    max_memory_reserved,
    memory_allocated,
    memory_reserved,
    memory_stats,
    reset_accumulated_memory_stats,
    reset_peak_memory_stats,
)


__all__ = [
    "device_count",
    "empty_cache",
    "is_available",
    "is_initialized",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    return False


def device_count() -> int:
    r"""Returns the number of GPUs available."""
    return 0


def is_initialized():
    r"""Return whether PyTorch's CUDA state has been initialized."""
    return False
