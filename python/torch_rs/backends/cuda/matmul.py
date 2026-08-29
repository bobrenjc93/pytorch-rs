# mypy: allow-untyped-defs
import sys as _sys

import torch_rs as torch
from torch_rs.backends import ContextProp as _ContextProp
from torch_rs.backends import PropModule as _PropModule


class _MatmulModule(_PropModule):
    allow_tf32 = _ContextProp(
        torch._C._get_cublas_allow_tf32,
        torch._C._set_cublas_allow_tf32,
    )

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            raise AttributeError("Unknown attribute " + attr) from None

    def __setattr__(self, attr, value):
        descriptor = type(self).__dict__.get(attr)
        if descriptor is not None and hasattr(descriptor, "__set__"):
            return descriptor.setter(value)
        if attr == "m" or (attr.startswith("__") and attr.endswith("__")):
            return super().__setattr__(attr, value)
        raise AttributeError("Unknown attribute " + attr)


# Expose CUDA matmul preferences without exposing CUDA tensors, cuBLAS
# execution, or reduced-precision reduction controls.
_sys.modules[__name__] = _MatmulModule(_sys.modules[__name__], __name__)

allow_tf32: bool

del _ContextProp, _MatmulModule, _PropModule, _sys
