# mypy: allow-untyped-defs
import sys as _sys
import types as _types

import torch_rs as torch


__cudnn_version: int | None = None


def _init():
    return torch._C._has_cudnn


def version():
    """Return the version of cuDNN."""
    if not _init():
        return None
    return __cudnn_version


def is_available():
    r"""Return a bool indicating if CUDNN is currently available."""
    return torch._C._has_cudnn


class _ContextProp:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, value):
        self.setter(value)


class CudnnModule(_types.ModuleType):
    enabled = _ContextProp(
        torch._C._get_cudnn_enabled,
        torch._C._set_cudnn_enabled,
    )
    benchmark = _ContextProp(
        torch._C._get_cudnn_benchmark,
        torch._C._set_cudnn_benchmark,
    )
    benchmark_limit = _ContextProp(
        torch._C._cuda_get_cudnn_benchmark_limit,
        torch._C._cuda_set_cudnn_benchmark_limit,
    )
    deterministic = _ContextProp(
        torch._C._get_cudnn_deterministic,
        torch._C._set_cudnn_deterministic,
    )
    allow_tf32 = _ContextProp(
        torch._C._get_cudnn_allow_tf32,
        torch._C._set_cudnn_allow_tf32,
    )

    def __init__(self, module, name):
        super().__init__(name)
        self.m = module

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)


# Match PyTorch's module-proxy identity and reload behavior without exposing
# any of its cuDNN execution surface.
_sys.modules[__name__] = CudnnModule(_sys.modules[__name__], __name__)

enabled: bool
benchmark: bool
benchmark_limit: int
deterministic: bool
allow_tf32: bool
