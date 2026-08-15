"""PyTorch-compatible public package backed by the native extension."""

import copyreg as _copyreg
import sys as _sys

from . import torch_rs as _native
from .torch_rs import *

# PyTorch's built-in variable functions reduce through owners in ``torch._C``.
# Expose the native extension under the equivalent package-local name so those
# owners remain importable without creating or modifying a top-level ``torch``.
_C = _native
_sys.modules[f"{__name__}._C"] = _C

# PyTorch's memory-format reducers use dotted public names such as
# ``torch.channels_last``. Mirror its module self-alias so those names resolve
# from this package without adding ``torch`` to wildcard imports.
torch = _sys.modules[__name__]

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

from . import nn as nn

del _copyreg, _native, _sys
