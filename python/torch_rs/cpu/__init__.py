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
    "device_count",
    "Event",
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


class Event:
    def query(self) -> bool:
        return True

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass


def current_device() -> str:
    r"""Returns current device for cpu. Always 'cpu'.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return "cpu"


def device_count() -> int:
    r"""Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return 1
