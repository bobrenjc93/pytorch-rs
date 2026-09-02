r"""
CPU-build CUDA compatibility probes.
"""

__all__ = [
    "device_count",
    "is_available",
    "is_initialized",
    "max_memory_allocated",
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


# PyTorch re-exports this callable from torch.cuda.memory; keep that owner for
# pickle/module metadata while limiting the top-level CUDA namespace.
from .memory import max_memory_allocated as max_memory_allocated

try:
    del memory
except NameError:
    pass
