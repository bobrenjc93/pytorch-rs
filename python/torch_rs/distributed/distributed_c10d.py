"""Distributed Collective Communication (c10d)."""

__all__ = ["is_initialized"]


def is_initialized() -> bool:
    """Check if the default process group has been initialized."""
    return False
