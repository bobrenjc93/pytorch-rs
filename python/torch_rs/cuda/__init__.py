r"""
CPU-build CUDA compatibility probes.
"""

__all__ = ["device_count", "is_available", "is_initialized", "memory_reserved"]


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    return False


def device_count() -> int:
    r"""Returns the number of GPUs available."""
    return 0


def is_initialized():
    r"""Return whether PyTorch's CUDA state has been initialized."""
    return False


def memory_reserved(device: "Device" = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return 0
