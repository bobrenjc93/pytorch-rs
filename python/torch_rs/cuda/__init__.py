r"""
CPU-build CUDA compatibility probes.
"""

__all__ = ["is_available", "device_count"]


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    return False


def device_count() -> int:
    r"""Returns the number of GPUs available."""
    return 0
