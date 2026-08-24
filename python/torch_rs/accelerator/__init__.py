r"""
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""

from .. import device as _device


_DEVICE_TYPE_NAMES = (
    "cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, "
    "ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone"
)
_DEVICE_TYPES = frozenset(_DEVICE_TYPE_NAMES.split(", "))
_DEVICE_TYPE_ERROR = (
    f"Expected one of {_DEVICE_TYPE_NAMES} device type at start of device string: "
)


__all__ = [
    "current_accelerator",
    "device_count",
    "is_available",
    "synchronize",
]


def _discover_accelerator() -> tuple[_device | None, bool, int]:
    """Return the native backend's accelerator, availability, and device count."""
    # The native Device enum currently contains only CPU. Keep the public
    # discovery queries and no-device synchronization on one boundary so a
    # future accelerator backend can replace this result without scattering
    # hardware probes through the API.
    return None, False, 0


def _normalize_synchronize_device_type(device: object) -> str | None:
    """Validate descriptor forms before accelerator synchronization."""
    if isinstance(device, str):
        specification = str.__str__(device)
        try:
            parsed_device = _device(specification)
        except RuntimeError as error:
            unsupported = (
                f"device(): device '{specification}' is not supported; "
                "only 'cpu' is implemented"
            )
            if str(error) != unsupported:
                raise
            device_type = str.partition(specification, ":")[0]
            if device_type not in _DEVICE_TYPES:
                raise RuntimeError(f"{_DEVICE_TYPE_ERROR}{specification}") from None
            return device_type
        return parsed_device.type
    if isinstance(device, _device):
        return device.type
    return None


def device_count() -> int:
    r"""Return the number of current :ref:`accelerator<accelerators>` available.

    Returns:
        int: the number of the current :ref:`accelerator<accelerators>` available.
            If there is no available accelerators, return 0.

    .. note:: This API delegates to the device-specific version of `device_count`.
        On CUDA, this API will NOT poison fork if NVML discovery succeeds.
        Otherwise, it will. For more details, see :ref:`multiprocessing-poison-fork-note`.
    """
    _, _, count = _discover_accelerator()
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
    _, available, _ = _discover_accelerator()
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
    accelerator, available, _ = _discover_accelerator()
    if accelerator is not None and ((not check_available) or available):
        return accelerator
    return None


def synchronize(device: _device | str | int | None = None, /) -> None:
    r"""Wait for all kernels in all streams on the given device to complete.

    Args:
        device (:class:`torch.device`, str, int, optional): device for which to synchronize. It must match
            the current :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    .. note:: This function is a no-op if the current :ref:`accelerator<accelerators>` is not initialized.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> assert torch.accelerator.is_available() "No available accelerators detected."
        >>> start_event = torch.Event(enable_timing=True)
        >>> end_event = torch.Event(enable_timing=True)
        >>> start_event.record()
        >>> tensor = torch.randn(100, device=torch.accelerator.current_accelerator())
        >>> sum = torch.sum(tensor)
        >>> end_event.record()
        >>> torch.accelerator.synchronize()
        >>> elapsed_time_ms = start_event.elapsed_time(end_event)
    """
    device_type = _normalize_synchronize_device_type(device)
    if isinstance(device, int):
        device_index = int.__index__(device)
        if not -128 <= device_index <= 127:
            raise TypeError(
                "_accelerator_synchronizeDevice(): incompatible function arguments. "
                "The following argument types are supported:\n"
                "    1. (arg0: typing.SupportsInt | typing.SupportsIndex) -> None\n\n"
                f"Invoked with: {device!r}"
            )
    accelerator, _, _ = _discover_accelerator()
    if device_type is not None:
        if accelerator is None:
            raise RuntimeError("Accelerator expected")
        if accelerator.type != device_type:
            raise ValueError(
                f"{device_type} doesn't match the current accelerator {accelerator}."
            )
    if accelerator is None:
        raise RuntimeError("Cannot access accelerator device when none is available.")
    raise RuntimeError("Accelerator synchronization is not implemented")
