r"""
This package implements abstractions found in ``torch.cuda``
to facilitate writing device-agnostic code.
"""

from .. import device as _device

__all__ = [
    "is_available",
    "is_initialized",
    "synchronize",
    "current_device",
    "set_device",
    "device_count",
]


def is_available() -> bool:
    r"""Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return True


def is_initialized() -> bool:
    r"""Returns True if the CPU is initialized. Always True.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return True


def synchronize(device: _device | str | int | None = None) -> None:
    r"""Waits for all kernels in all streams on the CPU device to complete.

    Args:
        device (torch.device or int, optional): ignored, there's only one CPU device.

    N.B. This function only exists to facilitate device-agnostic code.
    """


def current_device() -> str:
    r"""Returns current device for cpu. Always 'cpu'.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return "cpu"


def set_device(device: _device | str | int | None) -> None:
    r"""Sets the current device, in CPU we do nothing.

    N.B. This function only exists to facilitate device-agnostic code
    """


def device_count() -> int:
    r"""Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return 1
