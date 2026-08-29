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


# Keep the public answer as the immutable build-time fact. Like PyTorch's
# generated package function, it must not observe reassignment or deletion of
# the writable compatibility metadata on ``torch._C``.
def compiled_with_cxx11_abi() -> _builtins.bool:
    r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
    return False


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


def _deterministic_bool_type_name(value):
    value_type = _builtins.type(value)
    if value_type is Tensor:
        return "Tensor"
    if value_type is dtype:
        return "torch.dtype"
    if value_type is device:
        return "torch.device"
    if value_type is memory_format:
        return "torch.memory_format"
    if value_type is layout:
        return "torch.layout"
    if value_type is Size:
        return "torch.Size"
    if value_type is finfo:
        return "torch.finfo"
    type_name = _builtins.object.__getattribute__(value_type, "__name__")
    if _builtins.object.__getattribute__(value_type, "__module__") == "numpy":
        return f"numpy.{type_name}"
    return type_name


def _is_false_bool_or_int(value):
    return _builtins.isinstance(value, _builtins.int) and _builtins.int.__eq__(
        value,
        0,
    )


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
    if not _builtins.isinstance(mode, _builtins.int):
        type_name = _deterministic_bool_type_name(mode)
        raise TypeError(
            "_set_deterministic_algorithms(): argument 'mode' (position 1) "
            f"must be bool, not {type_name}"
        )
    if not _builtins.isinstance(warn_only, _builtins.int):
        type_name = _deterministic_bool_type_name(warn_only)
        raise TypeError(
            "_set_deterministic_algorithms(): argument 'warn_only' "
            f"must be bool, not {type_name}"
        )

    if not _is_false_bool_or_int(warn_only):
        raise NotImplementedError(
            "use_deterministic_algorithms(): warn_only "
            f"{warn_only!r} is not supported; only False and 0 are implemented"
        )
    if _is_false_bool_or_int(mode):
        return None
    raise NotImplementedError(
        "use_deterministic_algorithms(): mode "
        f"{mode!r} is not supported; only False and 0 are implemented"
    )


def set_deterministic_debug_mode(debug_mode: _builtins.int | str) -> None:
    r"""Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`torch.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """
    if not _builtins.isinstance(
        debug_mode,
        (_builtins.int, _builtins.str),
    ):
        debug_mode_type = _builtins.type(debug_mode)
        if debug_mode_type is Tensor:
            type_name = "<class 'torch.Tensor'>"
        elif debug_mode_type is dtype:
            type_name = "<class 'torch.dtype'>"
        elif debug_mode_type is device:
            type_name = "<class 'torch.device'>"
        elif debug_mode_type is memory_format:
            type_name = "<class 'torch.memory_format'>"
        elif debug_mode_type is layout:
            type_name = "<class 'torch.layout'>"
        elif debug_mode_type is Size:
            type_name = "<class 'torch.Size'>"
        elif debug_mode_type is finfo:
            type_name = "<class 'torch.finfo'>"
        else:
            type_name = _builtins.str(debug_mode_type)
        raise TypeError(f"debug_mode must be str or int, but got {type_name}")

    requested_mode = debug_mode
    if _builtins.isinstance(debug_mode, _builtins.str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, "
                f"`warn`, `error`, but got {debug_mode}"
            )

    if debug_mode == 0:
        return None
    if debug_mode == 1 or debug_mode == 2:
        raise NotImplementedError(
            "set_deterministic_debug_mode(): debug_mode "
            f"{requested_mode!r} is not supported; only 0, False, and "
            "'default' are implemented"
        )
    raise RuntimeError(
        f"invalid value of debug_mode, expected 0, 1, or 2, but got {debug_mode}"
    )


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


def set_float32_matmul_precision(precision: str) -> None:
    r"""Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype (24 mantissa
          bits with 23 bits explicitly stored) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    When using "high" precision, float32 multiplications may use a bfloat16-based algorithm
    that is more complicated than simply truncating to some smaller number mantissa bits
    (e.g. 10 for TensorFloat32, 7 for bfloat16 explicitly stored).  Refer to [Henry2019]_ for a complete
    description of this algorithm.  To briefly explain here, the first step is to realize
    that we can perfectly encode a single float32 number as the sum of three bfloat16
    numbers (because float32 has 23 mantissa bits while bfloat16 has 7 explicitly stored, and both have the
    same number of exponent bits).  This means that the product of two float32 numbers can
    be exactly given by the sum of nine products of bfloat16 numbers.  We can then trade
    accuracy for speed by dropping some of these products.  The "high" precision algorithm
    specifically keeps only the three most significant products, which conveniently excludes
    all of the products involving the last 8 mantissa bits of either input.  This means that
    we can represent our inputs as the sum of two bfloat16 numbers rather than three.
    Because bfloat16 fused-multiply-add (FMA) instructions are typically >10x faster than
    float32 ones, it's faster to do three multiplications and 2 additions with bfloat16
    precision than it is to do a single multiplication with float32 precision.

    .. [Henry2019] http://arxiv.org/abs/1904.06376

    .. note::

        This does not change the output dtype of float32 matrix multiplications,
        it controls how the internal computation of the matrix multiplication is performed.

    .. note::

        This does not change the precision of convolution operations. Other flags,
        like `torch.backends.cudnn.allow_tf32`, may control the precision of convolution
        operations.

    .. note::

        This flag currently only affects one native device type: CUDA.
        If "high" or "medium" are set then the TensorFloat32 datatype will be used
        when computing float32 matrix multiplications, equivalent to setting
        `torch.backends.cuda.matmul.allow_tf32 = True`. When "highest" (the default)
        is set then the float32 datatype is used for internal computations, equivalent
        to setting `torch.backends.cuda.matmul.allow_tf32 = False`.

    Args:
        precision(str): can be set to "highest" (default), "high", or "medium" (see above).

    """
    if _builtins.isinstance(precision, _builtins.str):
        try:
            _builtins.str.encode(precision, "utf-8")
        except UnicodeEncodeError:
            raise RuntimeError("error unpacking string as utf-8") from None
        value = _builtins.str.__str__(precision)
    elif _builtins.isinstance(precision, _builtins.bytes):
        value = _builtins.bytes.decode(precision, "utf-8")
    else:
        precision_type = _builtins.type(precision)
        if precision_type is Tensor:
            type_name = "Tensor"
        elif precision_type is dtype:
            type_name = "torch.dtype"
        elif precision_type is device:
            type_name = "torch.device"
        elif precision_type is memory_format:
            type_name = "torch.memory_format"
        elif precision_type is layout:
            type_name = "torch.layout"
        elif precision_type is Size:
            type_name = "torch.Size"
        elif precision_type is finfo:
            type_name = "torch.finfo"
        else:
            type_name = _builtins.object.__getattribute__(precision_type, "__name__")
        raise RuntimeError(
            f"set_float32_matmul_precision expects a str, but got {type_name}"
        )

    if value == "highest":
        return None
    raise NotImplementedError(
        "set_float32_matmul_precision(): precision "
        f"{value!r} is not supported; only 'highest' is implemented"
    )


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
# attributes that the extension itself owns. PyTorch keeps ``__version__``
# directly importable but excludes it from package wildcard imports.
__all__ = [
    *(name for name in _native.__all__ if name != "__version__"),
    "are_deterministic_algorithms_enabled",
    "get_deterministic_debug_mode",
    "use_deterministic_algorithms",
    "set_deterministic_debug_mode",
    "is_deterministic_algorithms_warn_only_enabled",
    "get_default_device",
    "get_device_module",
    "get_float32_matmul_precision",
    "set_float32_matmul_precision",
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

from . import __future__ as __future__
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
from . import version as version
from .functional import atleast_1d as atleast_1d
from .functional import atleast_2d as atleast_2d
from .functional import atleast_3d as atleast_3d
from .functional import broadcast_shapes as broadcast_shapes
from .functional import broadcast_tensors as broadcast_tensors

del (
    _copyreg,
    _functools,
    _multiprocessing_reduction,
    _native,
    _sys,
    _types,
)
