"""Distributed Collective Communication (c10d)."""

import os as _os

__all__ = [
    "is_initialized",
    "is_nccl_available",
    "is_torchelastic_launched",
]


def is_initialized() -> bool:
    """Check if the default process group has been initialized."""
    return False


def is_nccl_available() -> bool:
    """Check if the NCCL backend is available."""
    return False


def is_torchelastic_launched() -> bool:
    """
    Check whether this process was launched with ``torch.distributed.elastic`` (aka torchelastic).

    The existence of ``TORCHELASTIC_RUN_ID`` environment
    variable is used as a proxy to determine whether the current process
    was launched with torchelastic. This is a reasonable proxy since
    ``TORCHELASTIC_RUN_ID`` maps to the rendezvous id which is always a
    non-null value indicating the job id for peer discovery purposes..
    """
    return _os.getenv("TORCHELASTIC_RUN_ID") is not None
