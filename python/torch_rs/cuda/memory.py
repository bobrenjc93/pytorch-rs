r"""
CPU-build CUDA memory compatibility probes.
"""

__all__ = ["max_memory_allocated"]


def max_memory_allocated(device: "Device" = None) -> int:
    r"""Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return 0
