# mypy: allow-untyped-defs
import torch_rs as torch


__all__ = ["is_available", "set_flags"]


def is_available():
    r"""Return whether PyTorch is built with NNPACK support."""
    return torch._nnpack_available()


def set_flags(_enabled):
    r"""Set if nnpack is enabled globally"""
    orig_flags = (torch._C._get_nnpack_enabled(),)
    torch._C._set_nnpack_enabled(_enabled)
    return orig_flags
