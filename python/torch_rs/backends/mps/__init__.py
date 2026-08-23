import torch_rs as torch


__all__ = ["is_built"]


def is_built() -> bool:
    r"""Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run a machine with working MPS drivers
    and devices, we would be able to use it.
    """
    return torch._C._has_mps
