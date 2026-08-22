# mypy: allow-untyped-defs
import torch_rs as torch


__all__ = ["is_available"]


def is_available():
    r"""Return whether PyTorch is built with NNPACK support."""
    return torch._nnpack_available()
