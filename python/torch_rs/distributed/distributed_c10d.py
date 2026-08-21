"""Distributed Collective Communication (c10d)."""

__all__ = [
    "get_pg_count",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_nccl_available",
    "is_ucc_available",
]


def get_pg_count() -> int:
    """
    Return the number of process groups.

    """
    return 0


def is_gloo_available() -> bool:
    """Check if the Gloo backend is available."""
    return False


def is_initialized() -> bool:
    """Check if the default process group has been initialized."""
    return False


def is_mpi_available() -> bool:
    """Check if the MPI backend is available."""
    return False


def is_nccl_available() -> bool:
    """Check if the NCCL backend is available."""
    return False


def is_ucc_available() -> bool:
    """Check if the UCC backend is available."""
    return False
