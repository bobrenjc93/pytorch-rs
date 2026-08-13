"""PyTorch-compatible public package backed by the native extension."""

from . import torch_rs as _native
from .torch_rs import *

# PyTorch exposes ``strided`` as an attribute without including it in
# ``torch.__all__``. Bind it explicitly instead of widening wildcard imports.
strided = _native.strided

__doc__ = _native.__doc__
__all__ = _native.__all__

del _native
