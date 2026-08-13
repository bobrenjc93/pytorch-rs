"""PyTorch-compatible public package backed by the native extension."""

import copyreg as _copyreg

from . import torch_rs as _native
from .torch_rs import *

# PyTorch exposes ``strided`` as an attribute without including it in
# ``torch.__all__``. Bind it explicitly instead of widening wildcard imports.
strided = _native.strided


def _get_layout(name):
    """Return the canonical layout identified by its string representation."""
    if name == "torch.strided":
        return strided
    raise KeyError(name)


def _reduce_layout(value):
    return _get_layout, (str(value),)


_copyreg.pickle(layout, _reduce_layout)

__doc__ = _native.__doc__
__all__ = _native.__all__

del _copyreg, _native
