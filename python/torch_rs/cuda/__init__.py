r"""CPU-build CUDA compatibility probes.

This module intentionally exposes only availability metadata. CUDA tensors,
streams, events, synchronization, memory APIs, device selection, runtime
initialization, and `torch.compile` CUDA execution remain unsupported.
"""

__all__ = ["device_count", "is_available"]


def is_available() -> bool:
    r"""
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """
    return False


def device_count() -> int:
    r"""
    Return the number of GPUs available.

    .. note:: This API will NOT poison fork if NVML discovery succeeds.
        See :ref:`multiprocessing-poison-fork-note` for more details.
    """
    return 0
