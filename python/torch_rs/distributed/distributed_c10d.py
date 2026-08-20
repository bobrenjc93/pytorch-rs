"""Distributed Collective Communication (c10d)."""

__all__ = ["is_initialized", "is_mpi_available", "is_nccl_available"]


def is_initialized() -> bool:
    """Check if the default process group has been initialized."""
    return False


def is_mpi_available() -> bool:
    """Check if the MPI backend is available."""
    return False


def is_nccl_available() -> bool:
    """Check if the NCCL backend is available."""
    return False
