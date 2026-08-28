"""Distributed Collective Communication (c10d)."""

import os as _os

from ..torch_rs import Tensor as _Tensor

__all__ = [
    "destroy_process_group",
    "get_backend_config",
    "get_backend",
    "get_rank",
    "get_world_size",
    "get_pg_count",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_nccl_available",
    "is_ucc_available",
    "is_xccl_available",
    "get_group_rank",
    "get_process_group_ranks",
    "get_node_local_rank",
]


# Preserve PyTorch's annotation spelling without exporting unsupported types.
_ProcessGroup = type("ProcessGroup", (), {"__module__": __name__})
_Backend = type("Backend", (), {"__module__": __name__})


def destroy_process_group(group: _ProcessGroup | None = None):
    """
    Destroy a given process group, and deinitialize the distributed package.

    Args:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
    if type(group) is _Tensor:
        if group.numel() != 1:
            bool(group)
        if group.item() == -100:
            return
    elif group == -100:
        return
    if group is None:
        raise AssertionError("Process group cannot be None")
    if {}.get(group, None) is None:
        raise ValueError("Invalid process group specified")


def get_backend_config(group: _ProcessGroup | None = None) -> str:
    """
    Return the backend configuration of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend configuration of the given process group as a lower case string.

    """
    if group is not None:
        raise NotImplementedError(
            "torch_rs.distributed.get_backend_config() does not support "
            "non-None process groups"
        )
    raise ValueError(
        "Default process group has not been initialized, please make sure to "
        "call init_process_group."
    )


def get_backend(group: _ProcessGroup | None = None) -> _Backend:
    """
    Return the backend of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend of the given process group as a lower case string.

    """
    if group is not None:
        raise NotImplementedError(
            "torch_rs.distributed.get_backend() does not support non-None "
            "process groups"
        )
    raise ValueError(
        "Default process group has not been initialized, please make sure to "
        "call init_process_group."
    )


def get_rank(group: _ProcessGroup | None = None) -> int:
    """
    Return the rank of the current process in the provided ``group``, default otherwise.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    if group is not None:
        raise NotImplementedError(
            "torch_rs.distributed.get_rank() does not support non-None "
            "process groups"
        )
    raise ValueError(
        "Default process group has not been initialized, please make sure to "
        "call init_process_group."
    )


def get_group_rank(group: _ProcessGroup, global_rank: int) -> int:
    """
    Translate a global rank into a group rank.

    ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the relative rank.
        global_rank (int): Global rank to query.

    Returns:
        Group rank of ``global_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    if group is None:
        return global_rank
    if group not in {}:
        raise ValueError(
            f"Group {group} is not registered, please create group with "
            "torch.distributed.new_group API"
        )


def get_process_group_ranks(group: _ProcessGroup | None) -> list[int]:
    """
    Get all ranks associated with ``group``.

    Args:
        group (Optional[ProcessGroup]): ProcessGroup to get all ranks from.
            If None, the default process group will be used.

    Returns:
        List of global ranks ordered by group rank.
    """
    if not group:
        raise ValueError(
            "Default process group has not been initialized, please make sure "
            "to call init_process_group."
        )
    return list({}[group].keys())


def get_world_size(group: _ProcessGroup | None = None) -> int:
    """
    Return the number of processes in the current process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The world size of the process group
        -1, if not part of the group

    """
    if group is not None:
        raise NotImplementedError(
            "torch_rs.distributed.get_world_size() does not support non-None "
            "process groups"
        )
    raise ValueError(
        "Default process group has not been initialized, please make sure to "
        "call init_process_group."
    )


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


def is_xccl_available() -> bool:
    """Check if the XCCL backend is available."""
    return False


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
