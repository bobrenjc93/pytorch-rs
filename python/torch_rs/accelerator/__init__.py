r"""
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""

from .. import device as _device


__all__ = [
    "current_accelerator",
    "current_device_index",
    "device_count",
    "is_available",
]


def _discover_accelerator() -> tuple[_device | None, bool, int, int | None]:
    """Return the native backend's accelerator build and runtime metadata."""
    # The native Device enum currently contains only CPU. Keep the four public
    # discovery values on one boundary so a future accelerator backend can
    # replace this result without scattering hardware probes through the API.
    return None, False, 0, None


def device_count() -> int:
    r"""Return the number of current :ref:`accelerator<accelerators>` available.

    Returns:
        int: the number of the current :ref:`accelerator<accelerators>` available.
            If there is no available accelerators, return 0.

    .. note:: This API delegates to the device-specific version of `device_count`.
        On CUDA, this API will NOT poison fork if NVML discovery succeeds.
        Otherwise, it will. For more details, see :ref:`multiprocessing-poison-fork-note`.
    """
    _, _, count, _ = _discover_accelerator()
    return count


def is_available() -> bool:
    r"""Check if the current accelerator is available at runtime: it was built, all the
    required drivers are available and at least one device is visible.
    See :ref:`accelerator<accelerators>` for details.

    Returns:
        bool: A boolean indicating if there is an available :ref:`accelerator<accelerators>`.

    .. note:: This API delegates to the device-specific version of `is_available`.
        On CUDA, when the environment variable ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set,
        this function will NOT poison fork. Otherwise, it will. For more details, see
        :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> assert torch.accelerator.is_available() "No available accelerators detected."
    """
    _, available, _, _ = _discover_accelerator()
    return available


def current_accelerator(check_available: bool = False) -> _device | None:
    r"""Return the device of the accelerator available at compilation time.
    If no accelerator were available at compilation time, returns None.
    See :ref:`accelerator<accelerators>` for details.

    Args:
        check_available (bool, optional): if True, will also do a runtime check to see
            if the device :func:`torch.accelerator.is_available` on top of the compile-time
            check.
            Default: ``False``

    Returns:
        torch.device: return the current accelerator as :class:`torch.device`.

    .. note:: The index of the returned :class:`torch.device` will be ``None``, please use
        :func:`torch.accelerator.current_device_index` to know the current index being used.
        This API does NOT poison fork. For more details, see :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> # xdoctest:
        >>> # If an accelerator is available, sent the model to it
        >>> model = torch.nn.Linear(2, 2)
        >>> if (current_device := current_accelerator(check_available=True)) is not None:
        >>>     model.to(current_device)
    """
    accelerator, available, _, _ = _discover_accelerator()
    if accelerator is not None and ((not check_available) or available):
        return accelerator
    return None


def current_device_index() -> int:
    r"""Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

    Returns:
        int: the index of a currently selected device.
    """
    _, _, _, index = _discover_accelerator()
    return index


# PyTorch's deprecated ``current_device_idx`` alias tags the shared underlying
# callable. Preserve that metadata without exposing the deprecated alias.
current_device_index.__deprecated__ = "Use `current_device_index` instead."
