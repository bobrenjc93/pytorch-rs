r"""
This package implements abstractions found in ``torch.cuda``
to facilitate writing device-agnostic code.
"""

__all__ = ["is_available"]


def is_available() -> bool:
    r"""Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return True
