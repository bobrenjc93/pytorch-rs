# mypy: allow-untyped-defs
import torch_rs as torch


__all__ = [
    "is_available",
]


def is_available() -> bool:
    r"""Return a bool indicating if cuSPARSELt is currently available."""
    return torch._C._has_cusparselt
