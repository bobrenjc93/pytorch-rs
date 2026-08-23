"""PyTorch-compatible public package backed by the native extension."""

import builtins as _builtins
import copyreg as _copyreg
import functools as _functools
import multiprocessing.reduction as _multiprocessing_reduction
import sys as _sys
from math import e, inf, nan, pi

from . import torch_rs as _native
from .torch_rs import *

# PyTorch's built-in variable functions reduce through owners in ``torch._C``.
# Expose the native extension under the equivalent package-local name so those
# owners remain importable without creating or modifying a top-level ``torch``.
_C = _native
_sys.modules[f"{__name__}._C"] = _C


def _make_method_descriptor_reducer(previous_reducer):
    if getattr(previous_reducer, "_torch_rs_tensorbase_reducer", False):
        previous_reducer = previous_reducer._torch_rs_previous_reducer

    def reduce_method_descriptor(descriptor):
        if descriptor.__objclass__ is Tensor.__base__:
            return _builtins.getattr, (Tensor, descriptor.__name__)
        if previous_reducer is not None:
            return previous_reducer(descriptor)
        return descriptor.__reduce__()

    reduce_method_descriptor._torch_rs_tensorbase_reducer = True
    reduce_method_descriptor._torch_rs_previous_reducer = previous_reducer
    return reduce_method_descriptor


# TensorBase keeps PyTorch's public ``torch._C`` owner metadata, while this
# package must not install a competing top-level ``torch._C`` module. Resolve
# its inherited method descriptors through the public Tensor class instead,
# while preserving process-wide reducers installed before this package.
_method_descriptor_type = type(Tensor.relu)
_reduce_method_descriptor = _make_method_descriptor_reducer(
    _copyreg.dispatch_table.get(_method_descriptor_type)
)
_copyreg.pickle(_method_descriptor_type, _reduce_method_descriptor)
_reduce_forking_method_descriptor = _make_method_descriptor_reducer(
    _multiprocessing_reduction.ForkingPickler._extra_reducers.get(
        _method_descriptor_type
    )
)
_multiprocessing_reduction.ForkingPickler.register(
    _method_descriptor_type, _reduce_forking_method_descriptor
)

# PyTorch's memory-format reducers use dotted public names such as
# ``torch.channels_last``. Mirror its module self-alias so those names resolve
# from this package without adding ``torch`` to wildcard imports.
torch = _sys.modules[__name__]

# Match PyTorch's NumPy-compatible spelling for inserting a singleton axis.
# This remains a package-level alias; ``torch._C`` does not expose it.
newaxis = None


# PyTorch exposes this private build probe through its immutable variable-
# function owner, but not through ``torch._C`` or wildcard imports.
_nnpack_available = _native._VariableFunctionsClass._nnpack_available


def are_deterministic_algorithms_enabled() -> _builtins.bool:
    r"""Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """
    return False


def get_deterministic_debug_mode() -> _builtins.int:
    r"""Returns the current value of the debug mode for deterministic
    operations. Refer to :func:`torch.set_deterministic_debug_mode`
    documentation for more details.
    """
    return 0


def is_deterministic_algorithms_warn_only_enabled() -> _builtins.bool:
    r"""Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`torch.use_deterministic_algorithms` documentation for more
    details.
    """
    return False


def get_default_device() -> "torch.device":
    r"""Gets the default ``torch.Tensor`` to be allocated on ``device``"""
    return torch.device("cpu")


@_functools.cache
def get_device_module(device: torch.device | str | None = None):
    """
    Returns the module associated with a given device(e.g., torch.device('cuda'), "mtia:0", "xpu", ...).
    If no device is given, return the module for the current accelerator or CPU if none is present.
    """
    if isinstance(device, torch.device):
        device_module_name = device.type
    elif isinstance(device, str):
        device_module_name = torch.device(device).type
    elif device is None:
        # CPU is the only execution device implemented by the native backend.
        device_module_name = "cpu"
    else:
        raise RuntimeError(
            f"Invalid value of device '{device}', expect torch.device, str, or None"
        )
    device_module = getattr(torch, device_module_name, None)
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_module_name}' does not have a corresponding module registered as 'torch.{device_module_name}'."
        )
    return device_module


def get_float32_matmul_precision() -> str:
    r"""Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """
    return "highest"


def set_warn_always(b: _builtins.bool, /) -> None:
    r"""When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """
    _C._set_warnAlways(b)


def is_warn_always_enabled() -> _builtins.bool:
    r"""Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """
    return _C._get_warnAlways()


def set_default_dtype(d: "torch.dtype", /) -> None:
    r"""

    Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
    as inputs. Other dtypes will cause torch to raise an exception.

    When PyTorch is initialized its default floating point dtype is torch.float32,
    and the intent of set_default_dtype(torch.float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating type is float16,
       the default complex dtype is complex32. For float32, the default complex dtype is complex64.
       For float64, it is complex128. For bfloat16, an exception will be raised because
       there is no corresponding complex type for bfloat16.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default.

    Example:
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> # initial default for floating point is torch.float32
        >>> # Python floats are interpreted as float32
        >>> torch.tensor([1.2, 3]).dtype
        torch.float32
        >>> # initial default for floating point is torch.complex64
        >>> # Complex Python numbers are interpreted as complex64
        >>> torch.tensor([1.2, 3j]).dtype
        torch.complex64

        >>> torch.set_default_dtype(torch.float64)
        >>> # Python floats are now interpreted as float64
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float64
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex128

        >>> torch.set_default_dtype(torch.float16)
        >>> # Python floats are now interpreted as float16
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float16
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex32

    """
    _C._set_default_dtype(d)


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
# Keep package-only exports out of the native module's list, just as PyTorch's
# numeric constants live on ``torch`` rather than ``torch._C``.  A separate
# list also keeps reloading safe: the native wildcard import must only name
# attributes that the extension itself owns.
__all__ = [
    *_native.__all__,
    "are_deterministic_algorithms_enabled",
    "get_deterministic_debug_mode",
    "is_deterministic_algorithms_warn_only_enabled",
    "get_default_device",
    "get_device_module",
    "get_float32_matmul_precision",
    "set_warn_always",
    "is_warn_always_enabled",
    "e",
    "pi",
    "nan",
    "inf",
    "newaxis",
]
# PyTorch lists ``matmul`` once among its hand-written package exports and once
# among generated variable functions. Preserve that observable duplicate while
# the native module continues to own the callable itself.
if "matmul" in _native.__all__:
    __all__.insert(0, "matmul")

from . import accelerator as accelerator
from . import autograd as autograd
from . import backends as backends
from . import compiler as compiler
from . import cpu as cpu
from . import distributed as distributed
from . import functional as functional
from . import jit as jit
from . import nn as nn
from . import overrides as overrides
from . import _tensor as _tensor
from . import serialization as serialization
from . import utils as utils
from .functional import atleast_1d as atleast_1d
from .functional import atleast_2d as atleast_2d
from .functional import atleast_3d as atleast_3d
from .functional import broadcast_shapes as broadcast_shapes

del _copyreg, _functools, _method_descriptor_type, _multiprocessing_reduction
del _native, _sys
