# mypy: allow-untyped-defs
import torch_rs as torch


def is_available():
    r"""Return whether PyTorch is built with MKL support."""
    return torch._C.has_mkl
