r"""
CUDA availability probes for the narrow public tensor storage path.
"""

__all__ = ["device_count", "is_available", "is_initialized"]


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    from torch_rs import _cuda_public_storage

    return _cuda_public_storage.is_available()


def device_count() -> int:
    r"""Returns the number of GPUs available."""
    from torch_rs import _cuda_public_storage

    return _cuda_public_storage.device_count()


def is_initialized():
    r"""Return whether PyTorch's CUDA state has been initialized."""
    from torch_rs import _cuda_public_storage

    return _cuda_public_storage.is_initialized()

# Discover optional wheel runtime paths before the first native allocation.
from torch_rs import _cuda_public_storage as _storage
