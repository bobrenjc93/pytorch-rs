"""CUDA availability for the CPU-only backend."""

__all__ = ["is_available"]


def is_available() -> bool:
    r"""
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """
    return False
