r"""
This package implements abstractions found in ``torch.cuda``
to facilitate writing device-agnostic code.
"""

__all__ = ["is_available", "device_count"]


def is_available() -> bool:
    r"""Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return True


def device_count() -> int:
    r"""Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return 1
