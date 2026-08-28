# mypy: allow-untyped-defs
import torch_rs as torch


def is_available():
    r"""Return whether PyTorch is built with KleidiAI support."""
    return torch._C._has_kleidiai
