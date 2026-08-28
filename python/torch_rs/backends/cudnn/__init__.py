# mypy: allow-untyped-defs
import sys as _sys
import types as _types
from contextlib import contextmanager

import torch_rs as torch
from torch_rs.backends import __allow_nonbracketed_mutation


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


def _is_default_mode(value, default):
    if value is None:
        return True
    if isinstance(value, str):
        return str.__str__(value) == default
    if isinstance(value, bytes):
        return bytes.decode(value, "utf-8") == default
    return False


def _set_flags(
    _enabled=None,
    _benchmark=None,
    _benchmark_limit=None,
    _deterministic=None,
    _allow_tf32=None,
    _fp32_precision="none",
    _depthwise_kernel="auto",
):
    orig_flags = (
        torch._C._get_cudnn_enabled(),
        torch._C._get_cudnn_benchmark(),
        torch._C._cuda_get_cudnn_benchmark_limit(),
        torch._C._get_cudnn_deterministic(),
        torch._C._get_cudnn_allow_tf32(),
    )
    if _enabled is not None:
        torch._C._set_cudnn_enabled(_enabled)
    if _benchmark is not None:
        torch._C._set_cudnn_benchmark(_benchmark)
    if _benchmark_limit is not None:
        torch._C._cuda_set_cudnn_benchmark_limit(_benchmark_limit)
    if _deterministic is not None:
        torch._C._set_cudnn_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch._C._set_cudnn_allow_tf32(_allow_tf32)
    if not _is_default_mode(_fp32_precision, "none"):
        raise NotImplementedError(
            "torch.backends.cudnn.flags() only supports "
            "fp32_precision='none'"
        )
    if not _is_default_mode(_depthwise_kernel, "auto"):
        raise NotImplementedError(
            "torch.backends.cudnn.flags() only supports "
            "depthwise_kernel='auto'"
        )
    return orig_flags


@contextmanager
def flags(
    enabled=False,
    benchmark=False,
    benchmark_limit=10,
    deterministic=False,
    allow_tf32=True,
    fp32_precision="none",
    depthwise_kernel="auto",
):
    with __allow_nonbracketed_mutation():
        orig_flags = _set_flags(
            enabled,
            benchmark,
            benchmark_limit,
            deterministic,
            allow_tf32,
            fp32_precision,
            depthwise_kernel,
        )
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            _set_flags(*orig_flags)


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
