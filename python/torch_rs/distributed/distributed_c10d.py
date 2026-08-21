"""Distributed Collective Communication (c10d)."""

import os as _os

__all__ = [
    "get_pg_count",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_nccl_available",
    "is_ucc_available",
    "get_node_local_rank",
]


def get_pg_count() -> int:
    """
    Return the number of process groups.

    """
    return 0


def get_node_local_rank(fallback_rank: int | None = None) -> int:
    """
    Return the local rank of the current process relative to the node.

    Semantically, this is a useful concept for mapping processes to devices.
    For example, on a node with 8 accelerator you could use the node local rank to decide
    which accelerator device to bind the process to.

    In practice, the actual assignment of node local ranks is handled by the process launcher outside of pytorch,
    and communicated via the `LOCAL_RANK` environment variable.

    Torchrun will automatically populate `LOCAL_RANK`, but other launchers may not.  If `LOCAL_RANK` is unspecified,
    this API will fall back to the provided kwarg 'fallback_rank' if specified, otherwise it will raise an error. The
    intent is to allow writing an application that runs either in single or multi device contexts without error.

    """
    if "LOCAL_RANK" in _os.environ:
        return int(_os.environ["LOCAL_RANK"])
    elif fallback_rank is not None:
        return int(fallback_rank)
    raise RuntimeError(
        "LOCAL_RANK is not in the environment. Consider passing fallback_rank to allow `get_node_local_rank` to work, "
        "assuming you are not running in a multi-device context and want the code to run locally instead."
    )


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
