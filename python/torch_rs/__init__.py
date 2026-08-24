"""PyTorch-compatible public package backed by the native extension."""

import builtins as _builtins
import copyreg as _copyreg
import functools as _functools
import multiprocessing.reduction as _multiprocessing_reduction
import sys as _sys
import types as _types
from math import e, inf, nan, pi

from . import torch_rs as _native
from .torch_rs import *

# PyTorch's built-in variable functions reduce through owners in ``torch._C``.
# Expose the native extension under the equivalent package-local name so those
# owners remain importable without creating or modifying a top-level ``torch``.
_C = _native
_sys.modules[f"{__name__}._C"] = _C
# TensorBase reports PyTorch's ``torch._C`` metadata, so retain its actual
# native class privately for package-local descriptor reconstruction.
_TensorBase = Tensor.__base__

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


def use_deterministic_algorithms(
    mode: _builtins.bool,
    *,
    warn_only: _builtins.bool = False,
) -> None:
    r"""Sets whether PyTorch operations must use "deterministic"
    algorithms. That is, algorithms which, given the same input, and when
    run on the same software and hardware, always produce the same output.
    When enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw a
    :class:`RuntimeError` when called.

    .. note:: This setting alone is not always enough to make an application
        reproducible. Refer to :ref:`reproducibility` for more information.

    .. note:: :func:`torch.set_deterministic_debug_mode` offers an alternative
        interface for this feature.

    The following normally-nondeterministic operations will act
    deterministically when ``mode=True``:

        * :class:`torch.nn.Conv1d` when called on CUDA tensor
        * :class:`torch.nn.Conv2d` when called on CUDA tensor
        * :class:`torch.nn.Conv3d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose1d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose2d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose3d` when called on CUDA tensor
        * :class:`torch.nn.ReplicationPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad3d` when attempting to differentiate a CUDA tensor
        * :func:`torch.bmm` when called on sparse-dense CUDA tensors
        * :func:`torch.Tensor.__getitem__` when attempting to differentiate a CPU tensor
          and the index is a list of tensors
        * :func:`torch.Tensor.index_put` with ``accumulate=False``
        * :func:`torch.Tensor.index_put` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.put_` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.scatter_add_` when called on a CUDA tensor
        * :func:`torch.gather` when called on a CUDA tensor that requires grad
        * :func:`torch.index_add` when called on CUDA tensor
        * :func:`torch.index_select` when attempting to differentiate a CUDA tensor
        * :func:`torch.repeat_interleave` when attempting to differentiate a CUDA tensor
        * :func:`torch.Tensor.index_copy` when called on a CPU or CUDA tensor
        * :func:`torch.Tensor.scatter` when `src` type is Tensor and called on CUDA tensor
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='sum'`` or ``reduce='mean'`` and called on CUDA tensor
        * :class:`torch.nn.MaxPool3d` when attempting to differentiate a CUDA tensor

    The following normally-nondeterministic operations will throw a
    :class:`RuntimeError` when ``mode=True``:

        * :class:`torch.nn.AvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.MaxUnpool1d`
        * :class:`torch.nn.MaxUnpool2d`
        * :class:`torch.nn.MaxUnpool3d`
        * :func:`torch.nn.functional.interpolate` when attempting to differentiate a CUDA tensor
          and one of the following modes is used:

          - ``linear``
          - ``bilinear``
          - ``bicubic``
          - ``trilinear``

        * :class:`torch.nn.ReflectionPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.NLLLoss` when called on a CUDA tensor
        * :class:`torch.nn.CTCLoss` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.EmbeddingBag` when attempting to differentiate a CUDA tensor when
          ``mode='max'``
        * :func:`torch.Tensor.put_` when ``accumulate=False``
        * :func:`torch.Tensor.put_` when ``accumulate=True`` and called on a CUDA tensor
        * :func:`torch.histc` when called on a CUDA tensor
        * :func:`torch.bincount` when called on a CUDA tensor and ``weights``
          tensor is given
        * :func:`torch.median` with indices output when called on a CUDA tensor
        * :func:`torch.nn.functional.grid_sample` when attempting to differentiate a CUDA tensor
        * :func:`torch.cumsum` when called on a CUDA tensor when dtype is floating point or complex
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='prod'`` and called on CUDA tensor
        * :func:`torch.Tensor.resize_` when called with a quantized tensor

    In addition, several operations fill uninitialized memory when this setting
    is turned on and when
    :attr:`torch.utils.deterministic.fill_uninitialized_memory` is turned on.
    See the documentation for that attribute for more information.

    Note that deterministic operations tend to have worse performance than
    nondeterministic operations.


    When this setting is turned on, the Inductor deterministic mode is also tuned on
    automatically. In deterministic mode, Inductor would avoid doing on device benchmarking
    that affect numerics. This includes:

      - don't pad matmul input shapes. Without enabling deterministic mode, Inductor would do
        benchmarking to check if padding matmul shape is beneficial.
      - don't autotune templates. Inductor has templates for kernels like matmul/conv/attention.
        Without enabling deterministic mode, Inductor would do autotuning to
        pick the best configs for those templates and adopt it if it's faster
        than the kernel in eager mode. In deterministic mode, we pick the eager kernel.
      - don't autotune triton configs for reduction. Reduction numerics are
        very sensitive to triton configs. In deterministic mode, Inductor
        will use some heuristics to pick the most promising configs rather
        than do autotuning.
      - Skip autotuning for reduction in coordinate descent tuning.
      - Don't benchmark for the computation/communication reordering pass
      - Disable the feature that dynamically scale down RBLOCK triton config for higher
        occupancy.


    .. note::

        This flag does not detect or prevent nondeterministic behavior caused
        by calling an inplace operation on a tensor with an internal memory
        overlap or by giving such a tensor as the :attr:`out` argument for an
        operation. In these cases, multiple writes of different data may target
        a single memory location, and the order of writes is not guaranteed.

    Args:
        mode (:class:`bool`): If True, makes potentially nondeterministic
            operations switch to a deterministic algorithm or throw a runtime
            error. If False, allows nondeterministic operations.

    Keyword args:
        warn_only (:class:`bool`, optional): If True, operations that do not
            have a deterministic implementation will throw a warning instead of
            an error. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> torch.use_deterministic_algorithms(True)

        # Backward mode nondeterministic error
        >>> torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True).cuda()).sum().backward()
        ...
        RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation...
    """
    _C._set_deterministic_algorithms(mode, warn_only=warn_only)


def are_deterministic_algorithms_enabled() -> _builtins.bool:
    r"""Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """
    return _C._get_deterministic_algorithms()


def get_deterministic_debug_mode() -> _builtins.int:
    r"""Returns the current value of the debug mode for deterministic
    operations. Refer to :func:`torch.set_deterministic_debug_mode`
    documentation for more details.
    """
    return _C._get_deterministic_debug_mode()


def is_deterministic_algorithms_warn_only_enabled() -> _builtins.bool:
    r"""Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`torch.use_deterministic_algorithms` documentation for more
    details.
    """
    return _C._get_deterministic_algorithms_warn_only()


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


# The default method-descriptor reduction tries to import ``torch._C`` and
# cannot resolve torch-rs's metadata-compatible TensorBase. Handle only that
# owner and delegate every other descriptor to the reducer already in place.
def _get_tensorbase_method_descriptor(name):
    """Return a native TensorBase method descriptor by name."""
    return getattr(_TensorBase, name)


def _make_tensorbase_method_descriptor_reducer(previous):
    if getattr(
        previous,
        "_torch_rs_tensorbase_method_descriptor_reducer",
        False,
    ):
        previous = getattr(
            previous,
            "_torch_rs_previous_method_descriptor_reducer",
            None,
        )

    def reducer(descriptor):
        if descriptor.__objclass__ is _TensorBase:
            return _get_tensorbase_method_descriptor, (descriptor.__name__,)
        if previous is not None:
            return previous(descriptor)
        return descriptor.__reduce__()

    reducer._torch_rs_tensorbase_method_descriptor_reducer = True
    reducer._torch_rs_previous_method_descriptor_reducer = previous
    return reducer


_reduce_tensorbase_method_descriptor = _make_tensorbase_method_descriptor_reducer(
    _copyreg.dispatch_table.get(_types.MethodDescriptorType)
)
_copyreg.pickle(_types.MethodDescriptorType, _reduce_tensorbase_method_descriptor)

_reduce_tensorbase_method_descriptor_for_forking = (
    _make_tensorbase_method_descriptor_reducer(
        _multiprocessing_reduction.ForkingPickler._extra_reducers.get(
            _types.MethodDescriptorType
        )
    )
)
_multiprocessing_reduction.ForkingPickler.register(
    _types.MethodDescriptorType,
    _reduce_tensorbase_method_descriptor_for_forking,
)

__doc__ = _native.__doc__
# Keep package-only exports out of the native module's list, just as PyTorch's
# numeric constants live on ``torch`` rather than ``torch._C``.  A separate
# list also keeps reloading safe: the native wildcard import must only name
# attributes that the extension itself owns.
__all__ = [
    *_native.__all__,
    "use_deterministic_algorithms",
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

del (
    _copyreg,
    _functools,
    _multiprocessing_reduction,
    _native,
    _sys,
    _types,
)
