"""Private native-only tracing pieces for future ``torch.compile`` work."""

from __future__ import annotations

import builtins as _builtins
import operator as _operator
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass

from . import torch_rs as _native


class CompileTraceUnsupportedError(NotImplementedError):
    """Raised when the package-local trace prototype reaches an unsupported op."""


@dataclass(frozen=True, slots=True)
class CompileTraceDType:
    name: str

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


float32 = CompileTraceDType("torch.float32")
float = float32


@dataclass(frozen=True, slots=True, eq=False)
class CompileTraceDevice:
    type: str
    index: int | None = None

    def __post_init__(self):
        if _builtins.type(self.type) is not _builtins.str:
            raise TypeError(
                "torch.compile trace device type must be str, "
                f"not {_builtins.type(self.type).__name__}"
            )
        if self.type not in ("cpu", "cuda"):
            raise CompileTraceUnsupportedError(
                "torch.compile trace device metadata only supports CPU and CUDA"
            )
        if self.index is not None:
            if _builtins.type(self.index) is _builtins.bool:
                raise TypeError(
                    "torch.compile trace device index must be int or None, "
                    "not bool"
                )
            try:
                index = _operator.index(self.index)
            except TypeError:
                raise TypeError(
                    "torch.compile trace device index must be int or None, "
                    f"not {_builtins.type(self.index).__name__}"
                ) from None
            if index < 0:
                raise ValueError("torch.compile trace device index must be non-negative")
            object.__setattr__(self, "index", index)
        if self.type == "cpu" and self.index is not None:
            raise CompileTraceUnsupportedError(
                "torch.compile trace CPU device metadata only supports unindexed CPU"
            )

    def __repr__(self):
        return _builtins.repr(_builtins.str(self))

    def __str__(self):
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"

    def __eq__(self, other):
        if _builtins.isinstance(other, CompileTraceDevice):
            return self.type == other.type and self.index == other.index
        if _builtins.isinstance(other, _builtins.str):
            return _builtins.str(self) == other
        return NotImplemented

    def __hash__(self):
        return _builtins.hash(_builtins.str(self))


cpu = CompileTraceDevice("cpu")


def _cuda_device_metadata(index=0):
    return CompileTraceDevice("cuda", index)


def _parse_device_metadata(device):
    if device is None:
        return cpu
    if _builtins.isinstance(device, CompileTraceDevice):
        return device

    specification = _builtins.str(device)
    if device == "cpu" or specification == "cpu":
        return cpu
    if specification == "cuda":
        return CompileTraceDevice("cuda")
    if specification.startswith("cuda:"):
        index_text = specification[5:]
        if (
            index_text
            and index_text.isdecimal()
            and (len(index_text) == 1 or not index_text.startswith("0"))
        ):
            return _cuda_device_metadata(_builtins.int(index_text))

    raise CompileTraceUnsupportedError(
        "torch.compile trace tensor() only supports CPU or private CUDA "
        "metadata inputs"
    )


@dataclass(frozen=True, slots=True)
class CompileTraceTensorMetadata:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: CompileTraceDType
    device: CompileTraceDevice
    requires_grad: bool
    # CPU traces historically accept any storage offset for a shape/stride.
    # CUDA specializations guard the exact offset; None leaves it unspecified.
    storage_offset: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "device", _parse_device_metadata(self.device))


@dataclass(frozen=True, slots=True)
class CompileTraceInput:
    name: str
    index: int
    metadata: CompileTraceTensorMetadata


@dataclass(frozen=True, slots=True)
class CompileTraceCapture:
    name: str
    value: object
    metadata: CompileTraceTensorMetadata


@dataclass(frozen=True, slots=True)
class CompileTraceOperation:
    name: str
    op: str
    target: str
    inputs: tuple[str, ...]
    metadata: CompileTraceTensorMetadata
    scalar: object = None
    reduction: tuple[int, bool] | None = None
    axes: tuple[int, int] | None = None
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class CompileTraceOutputContainer:
    kind: str
    elements: tuple[object, ...]

    def __post_init__(self):
        if self.kind not in ("tuple", "list"):
            raise ValueError(
                "torch.compile trace output container kind must be 'tuple' "
                f"or 'list', got {self.kind!r}"
            )
        object.__setattr__(self, "elements", tuple(self.elements))


@dataclass(frozen=True, slots=True)
class CompileTraceGraph:
    name: str
    inputs: tuple[CompileTraceInput, ...]
    operations: tuple[CompileTraceOperation, ...]
    output: object
    output_metadata: object
    captures: tuple[CompileTraceCapture, ...] = ()
    dynamic: bool = False

    def forward(self, *inputs):
        return execute_compile_trace_graph(self, *inputs)


_SUPPORTED_UNARY_METHODS = (
    "Tensor.t (CUDA, parameterless, rank <= 2)",
    "Tensor.transpose (CUDA, constant integer axes, rank <= 2)",
    "Tensor.reshape (CUDA, constant integer shape, rank <= 2)",
    "Tensor.contiguous (CUDA, parameterless)",
    "Tensor.neg",
    "Tensor.negative",
    "Tensor.abs",
    "Tensor.absolute",
    "Tensor.relu",
    "Tensor.square",
    "Tensor.detach",
    "Tensor.float",
)
_SUPPORTED_VALUE_UNARY_TARGETS = frozenset(("neg", "abs", "relu", "square"))
_SUPPORTED_ALIAS_UNARY_TARGETS = frozenset(("detach",))
_SUPPORTED_IDENTITY_UNARY_TARGETS = frozenset(("float",))
_SUPPORTED_UNARY_TARGETS = (
    frozenset(("contiguous", "t"))
    | _SUPPORTED_VALUE_UNARY_TARGETS
    | _SUPPORTED_ALIAS_UNARY_TARGETS
    | _SUPPORTED_IDENTITY_UNARY_TARGETS
)
_SUPPORTED_BINARY_METHODS = (
    "Tensor.__add__",
    "Tensor.add",
    "Tensor.matmul (CUDA)",
)
_SUPPORTED_BINARY_TARGETS = frozenset(("add", "matmul"))
_SUPPORTED_SCALAR_TARGETS = frozenset(("mul_scalar",))
_SUPPORTED_OPERATION_TARGETS = (
    _SUPPORTED_UNARY_TARGETS | _SUPPORTED_BINARY_TARGETS | _SUPPORTED_SCALAR_TARGETS
)
_SUPPORTED_OPERATION_DESCRIPTION = ", ".join(
    (*_SUPPORTED_UNARY_METHODS, *_SUPPORTED_BINARY_METHODS,
     "Tensor.mul (CUDA scalar)", "Tensor.sum (CUDA rank-two rows)")
)


def _unsupported_operation(operation):
    raise CompileTraceUnsupportedError(
        "torch.compile trace does not support "
        f"{operation}; only {_SUPPORTED_OPERATION_DESCRIPTION} are implemented"
    )


def _normalize_dtype(dtype):
    if dtype is None:
        return float32
    if dtype is float32:
        return float32
    if _builtins.str(dtype) == "torch.float32":
        return float32
    raise CompileTraceUnsupportedError(
        "torch.compile trace tensor() only supports dtype=torch.float32"
    )


def _normalize_device(device):
    return _parse_device_metadata(device)


def _normalize_requires_grad(requires_grad):
    if _builtins.type(requires_grad) is not _builtins.bool:
        raise TypeError(
            "torch.compile trace tensor(): requires_grad must be bool, "
            f"not {_builtins.type(requires_grad).__name__}"
        )
    return requires_grad


def _normalize_shape(shape):
    return tuple(_normalize_dimension(dimension) for dimension in shape)


def _normalize_dimension(dimension):
    value = _operator.index(dimension)
    if value < 0:
        raise ValueError(
            "torch.compile trace tensor(): dimensions must be non-negative"
        )
    return value


def _infer_shape(data):
    if _builtins.isinstance(data, (str, bytes, bytearray)):
        return ()
    if not _builtins.isinstance(data, _Sequence):
        return ()

    length = len(data)
    if length == 0:
        return (0,)

    first_shape = _infer_shape(data[0])
    for item in data[1:]:
        item_shape = _infer_shape(item)
        if item_shape != first_shape:
            raise ValueError(
                "torch.compile trace tensor(): nested input data must be rectangular"
            )
    return (length, *first_shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dimension in reversed(shape):
        stride.append(running)
        running *= max(dimension, 1)
    return tuple(reversed(stride))


def _element_count(shape):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _layout_is_contiguous(shape, stride):
    if _element_count(shape) == 0:
        return True

    expected_stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        dimension = shape[axis]
        if dimension == 1:
            continue
        if stride[axis] != expected_stride:
            return False
        expected_stride *= dimension
    return True


def _layout_is_contiguous_in_order(shape, stride, order):
    if len(shape) != len(order) or len(stride) != len(order):
        return False

    expected_stride = 1
    for axis in order:
        dimension = shape[axis]
        if dimension == 1:
            continue
        if stride[axis] != expected_stride:
            return False
        expected_stride *= dimension
    return True


def _channels_last_stride(shape):
    return _stride_in_physical_order(shape, (1, 3, 2, 0))


def _stride_in_physical_order(shape, order):
    stride = [0] * len(shape)
    running = 1
    for position, axis in enumerate(order):
        stride[axis] = running
        if position + 1 < len(order):
            running *= shape[axis]
    return tuple(stride)


def _layout_is_channels_last_contiguous(shape, stride):
    return _layout_is_contiguous_in_order(shape, stride, (1, 3, 2, 0))


def _layout_is_non_overlapping_and_dense(shape, stride):
    if _element_count(shape) == 0:
        return True

    non_singleton_dimensions = sum(dimension > 1 for dimension in shape)
    matched_dimensions = 0
    expected_stride = 1
    while matched_dimensions < non_singleton_dimensions:
        matching_dimension = None
        for axis, (dimension, axis_stride) in enumerate(zip(shape, stride)):
            if dimension <= 1 or axis_stride != expected_stride:
                continue
            if matching_dimension is not None:
                return False
            matching_dimension = (axis, dimension)
        if matching_dimension is None:
            return False
        _, dimension = matching_dimension
        expected_stride *= dimension
        matched_dimensions += 1
    return True


def _signed_int64(value):
    return (value + (1 << 63)) % (1 << 64) - (1 << 63)


def _elementwise_output_stride(shape, operand_layouts, *, cuda=False):
    # The shared native float32 planner compares signed 64-bit byte strides.
    # Empty CUDA views can wrap even though they allocate no elements. Keep
    # CPU tracing's existing arithmetic separate from this native CUDA path.
    if cuda:
        operand_layouts = tuple(
            (input_shape, tuple(_signed_int64(s * 4) for s in input_stride))
            for input_shape, input_stride in operand_layouts
        )
    rank = len(shape)
    permutation = list(range(rank - 1, -1, -1))

    for index in range(1, rank):
        dimension_1 = index
        for dimension_0 in range(index - 1, -1, -1):
            comparison = _compare_elementwise_dimensions(
                shape,
                operand_layouts,
                permutation[dimension_0],
                permutation[dimension_1],
            )
            if comparison > 0:
                permutation[dimension_0], permutation[dimension_1] = (
                    permutation[dimension_1],
                    permutation[dimension_0],
                )
                dimension_1 = dimension_0
            elif comparison < 0:
                break

    if all(axis == rank - index - 1 for index, axis in enumerate(permutation)):
        return _contiguous_stride(shape)

    stride = [0] * rank
    next_stride = 4 if cuda else 1
    for position, axis in enumerate(permutation):
        stride[axis] = next_stride
        if position + 1 < rank:
            next_stride *= shape[axis]
            if cuda:
                next_stride = _signed_int64(next_stride)
    if cuda:
        # Match elementwise_output_strides in src/tensor.rs, including the
        # recovery of negative wrapped strides in TensorIterator outputs.
        # Byte strides remain multiples of four, so division is exact.
        for axis in range(rank - 1, -1, -1):
            value = stride[axis] // 4
            if value >= 0:
                stride[axis] = value
            else:
                stride[axis] = (
                    1 if axis + 1 == rank
                    else stride[axis + 1] * max(shape[axis + 1], 1)
                )
    return tuple(stride)


def _aligned_broadcast_stride(input_shape, input_stride, output_rank, axis, output_dimension):
    leading_dimensions = output_rank - len(input_shape)
    if axis < leading_dimensions:
        return 0

    input_axis = axis - leading_dimensions
    input_dimension = input_shape[input_axis]
    if input_dimension == 1 and output_dimension != 1:
        return 0
    return input_stride[input_axis]


def _compare_elementwise_dimensions(shape, operand_layouts, dimension_0, dimension_1):
    rank = len(shape)
    for input_shape, input_stride in operand_layouts:
        stride_0 = _aligned_broadcast_stride(
            input_shape,
            input_stride,
            rank,
            dimension_0,
            shape[dimension_0],
        )
        stride_1 = _aligned_broadcast_stride(
            input_shape,
            input_stride,
            rank,
            dimension_1,
            shape[dimension_1],
        )
        if stride_0 == 0 or stride_1 == 0:
            continue
        if stride_0 < stride_1:
            return -1
        if stride_0 > stride_1 or shape[dimension_0] > shape[dimension_1]:
            return 1
    return 0


def _unary_output_stride(shape, input_stride):
    if _layout_is_contiguous(shape, input_stride):
        return _contiguous_stride(shape)
    if _layout_is_channels_last_contiguous(shape, input_stride):
        return _channels_last_stride(shape)
    if _layout_is_non_overlapping_and_dense(shape, input_stride):
        return input_stride
    return _elementwise_output_stride(shape, ((shape, input_stride),))


def _broadcast_shape(left_shape, right_shape):
    rank = max(len(left_shape), len(right_shape))
    output_shape = []
    for axis in range(rank):
        left_axis = axis - (rank - len(left_shape))
        right_axis = axis - (rank - len(right_shape))
        left_dimension = left_shape[left_axis] if left_axis >= 0 else 1
        right_dimension = right_shape[right_axis] if right_axis >= 0 else 1
        if left_dimension == 1:
            output_shape.append(right_dimension)
        elif right_dimension == 1 or left_dimension == right_dimension:
            output_shape.append(left_dimension)
        else:
            raise CompileTraceUnsupportedError(
                "torch.compile trace Tensor.add operands are not broadcastable: "
                f"{left_shape!r} and {right_shape!r}"
            )
    return tuple(output_shape)


def _binary_output_stride(left_metadata, right_metadata, output_shape):
    if left_metadata.shape == right_metadata.shape:
        if _layout_is_contiguous(
            output_shape,
            left_metadata.stride,
        ) and _layout_is_contiguous(output_shape, right_metadata.stride):
            return _contiguous_stride(output_shape)
        if _layout_is_channels_last_contiguous(
            output_shape,
            left_metadata.stride,
        ) and _layout_is_channels_last_contiguous(output_shape, right_metadata.stride):
            return _channels_last_stride(output_shape)
        if (
            _layout_is_non_overlapping_and_dense(output_shape, left_metadata.stride)
            and _layout_is_non_overlapping_and_dense(output_shape, right_metadata.stride)
            and left_metadata.stride == right_metadata.stride
        ):
            return left_metadata.stride

    return _elementwise_output_stride(
        output_shape,
        (
            (left_metadata.shape, left_metadata.stride),
            (right_metadata.shape, right_metadata.stride),
        ),
        cuda=left_metadata.device.type == "cuda",
    )


def _grad_enabled():
    return _native._compile_trace_grad_enabled()


def _t_output_metadata(metadata):
    _validate_cuda_metadata(metadata, require_contiguous=False)
    if metadata.device.type != "cuda" or len(metadata.shape) > 2:
        raise CompileTraceUnsupportedError("torch.compile Tensor.t requires CUDA rank 0, 1 or 2")
    return CompileTraceTensorMetadata(
        shape=metadata.shape[::-1], stride=metadata.stride[::-1],
        dtype=metadata.dtype, device=metadata.device, requires_grad=False,
        storage_offset=metadata.storage_offset,
    )


def _reshape_dimensions(shape):
    if type(shape) is not tuple:
        raise CompileTraceUnsupportedError("torch.compile reshape requires an exact flat tuple")
    # PyTorch's schema checks the first dimension, then unpacks in order.
    # Later booleans/index conversions can be reference-valid, but are outside
    # this exact-integer capture contract. Never use bool/int equality here.
    for index, dimension in enumerate(shape):
        if type(dimension) is not int:
            if ((type(dimension) is bool and index > 0)
                    or (type(dimension) is not bool and hasattr(type(dimension), "__index__"))):
                raise CompileTraceUnsupportedError("torch.compile reshape requires exact integer constants")
            raise TypeError(f"reshape(): shape dimension {index} must be int, not {type(dimension).__name__}")
        if not -(2**63) <= dimension < 2**63:
            raise TypeError(f"reshape(): shape dimension {index} overflows signed int64")
    if len(shape) > 2:
        raise CompileTraceUnsupportedError("torch.compile reshape requires output rank 0, 1 or 2")
    return shape


def _bind_reshape_shape(args, kwargs):
    if not args and "shape" not in kwargs:
        raise TypeError('reshape() missing 1 required positional arguments: "shape"')
    first = args[0] if args else kwargs["shape"]
    if type(first) is int and args:
        shape = args
    elif type(first) is tuple:
        if len(args) > 1:
            raise TypeError("reshape() takes 1 positional argument")
        shape = first
    elif isinstance(first, (tuple, list)) or (type(first) not in (bool, int, _builtins.float, str, type(None))
                               and hasattr(type(first), "__index__")):
        raise CompileTraceUnsupportedError("torch.compile reshape requires exact integer constants or an exact tuple")
    else:
        raise TypeError("reshape(): argument 'shape' must be tuple of ints")
    # Schema validation of the first ordinary dimension precedes keyword
    # binding; remaining dimensions and overflow follow keyword binding.
    if shape and type(shape[0]) is not int:
        _reshape_dimensions((shape[0],))
    if args and "shape" in kwargs:
        raise TypeError("reshape() got multiple values for argument 'shape'")
    for name in kwargs:
        if name != "shape":
            raise TypeError(f"reshape() got an unexpected keyword argument '{name}'")
    return _reshape_dimensions(shape)


def _reshape_output_metadata(metadata, shape):
    shape = _reshape_dimensions(shape)
    _validate_cuda_metadata(metadata, require_contiguous=False)
    if metadata.device.type != "cuda" or len(metadata.shape) > 2:
        raise CompileTraceUnsupportedError("torch.compile Tensor.reshape requires CUDA rank 0, 1 or 2")
    # No tensor storage is allocated here. Both frontend and executor call the
    # shared checked eager resolver/view-stride planner, including empty layouts.
    resolved, stride, offset = _native._compile_trace_cuda_reshape_metadata(
        metadata.shape, metadata.stride, metadata.storage_offset, shape,
    )
    return CompileTraceTensorMetadata(
        shape=tuple(resolved), stride=tuple(stride), dtype=metadata.dtype,
        device=metadata.device, requires_grad=False, storage_offset=offset,
    )


def _transpose_axes(axes, rank):
    if type(axes) is not tuple or len(axes) != 2:
        raise CompileTraceUnsupportedError("torch.compile malformed transpose axes")
    # Match public binding: type-check both arguments before converting either
    # integer or validating dimensions (e.g. transpose(2, True) is TypeError).
    for name, axis in zip(("dim0", "dim1"), axes):
        if type(axis) is not int:
            # Index-like objects and integer subclasses may be valid eager
            # arguments, but capture deliberately accepts exact constants only.
            if hasattr(type(axis), "__index__") and type(axis) is not bool:
                raise CompileTraceUnsupportedError("torch.compile transpose requires exact integer constants")
            raise TypeError(f"transpose(): argument '{name}' must be int, not {type(axis).__name__}")
    for axis in axes:
        if not -(2**63) <= axis < 2**63:
            raise ValueError("Overflow when unpacking long long")
    effective_rank = max(rank, 1)
    normalized = []
    for axis in axes:
        if not -effective_rank <= axis < effective_rank:
            raise IndexError(f"Dimension out of range (expected to be in range of "
                             f"[{-effective_rank}, {effective_rank - 1}], but got {axis})")
        normalized.append(axis % effective_rank)
    return tuple(normalized)


def _transpose_output_metadata(metadata, axes):
    _validate_cuda_metadata(metadata, require_contiguous=False)
    if metadata.device.type != "cuda" or len(metadata.shape) > 2:
        raise CompileTraceUnsupportedError("torch.compile Tensor.transpose requires CUDA rank 0, 1 or 2")
    dim0, dim1 = _transpose_axes(axes, len(metadata.shape))
    shape, stride = list(metadata.shape), list(metadata.stride)
    if shape:
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    return CompileTraceTensorMetadata(
        shape=tuple(shape), stride=tuple(stride), dtype=metadata.dtype,
        device=metadata.device, requires_grad=False, storage_offset=metadata.storage_offset,
    )


def _contiguous_output_metadata(metadata):
    if metadata.device.type != "cuda":
        raise CompileTraceUnsupportedError("torch.compile contiguous only supports CUDA")
    _validate_cuda_metadata(metadata, require_contiguous=False)
    if _layout_is_contiguous(metadata.shape, metadata.stride):
        return metadata
    if len(metadata.shape) not in (1, 2) or any(s == 0 for s in metadata.stride):
        raise CompileTraceUnsupportedError(
            "torch.compile contiguous packing requires positive-stride rank-1 or rank-2 inputs"
        )
    return CompileTraceTensorMetadata(
        shape=metadata.shape, stride=_contiguous_stride(metadata.shape),
        dtype=metadata.dtype, device=metadata.device, requires_grad=False, storage_offset=0,
    )


def _unary_output_metadata(input_metadata, target, *, grad_enabled=None):
    if target == "t":
        return _t_output_metadata(input_metadata)
    if target == "contiguous":
        return _contiguous_output_metadata(input_metadata)
    if input_metadata.device.type == "cuda" and target == "neg":
        _validate_cuda_metadata(input_metadata)
    if target in _SUPPORTED_ALIAS_UNARY_TARGETS:
        return CompileTraceTensorMetadata(
            shape=input_metadata.shape,
            stride=input_metadata.stride,
            dtype=input_metadata.dtype,
            device=input_metadata.device,
            requires_grad=False,
            storage_offset=input_metadata.storage_offset,
        )
    if target in _SUPPORTED_IDENTITY_UNARY_TARGETS:
        return CompileTraceTensorMetadata(
            shape=input_metadata.shape,
            stride=input_metadata.stride,
            dtype=input_metadata.dtype,
            device=input_metadata.device,
            requires_grad=input_metadata.requires_grad,
            storage_offset=input_metadata.storage_offset,
        )
    if target not in _SUPPORTED_VALUE_UNARY_TARGETS:
        _unsupported_operation(f"Tensor.{target}")
    if grad_enabled is None:
        grad_enabled = _grad_enabled()
    return CompileTraceTensorMetadata(
        shape=input_metadata.shape,
        stride=_unary_output_stride(input_metadata.shape, input_metadata.stride),
        dtype=input_metadata.dtype,
        device=input_metadata.device,
        requires_grad=input_metadata.requires_grad and grad_enabled,
        # Value operations allocate fresh CUDA storage, including offset views.
        # CPU traces retain their established offset-polymorphic contract.
        storage_offset=0 if input_metadata.device.type == "cuda" else None,
    )


def _matmul_output_metadata(left, right):
    if left.device.type != "cuda" or left.device != right.device:
        raise CompileTraceUnsupportedError("torch.compile matmul requires the same CUDA device")
    for metadata in (left, right):
        _validate_cuda_metadata(metadata)
        if len(metadata.shape) != 2:
            raise CompileTraceUnsupportedError("torch.compile matmul requires rank-2 inputs")
    if left.shape[1] != right.shape[0]:
        raise RuntimeError("torch.compile matmul inner dimension mismatch")
    shape = (left.shape[0], right.shape[1])
    # Native storage uses checked usize element/byte counts and i64 BLAS
    # dimensions. Reject impossible output metadata before any node launches.
    import sys
    if any(d > sys.maxsize for d in (*left.shape, *right.shape)) or shape[0] * shape[1] > sys.maxsize // 4:
        raise OverflowError("torch.compile matmul dimensions exceed native storage limits")
    return CompileTraceTensorMetadata(
        shape=shape, stride=_contiguous_stride(shape), dtype=float32,
        device=left.device, requires_grad=False, storage_offset=0,
    )


def _binary_output_metadata(left_metadata, right_metadata, *, grad_enabled=None, target="add"):
    if target == "matmul":
        return _matmul_output_metadata(left_metadata, right_metadata)
    if left_metadata.dtype is not right_metadata.dtype:
        raise CompileTraceUnsupportedError(
            "torch.compile trace Tensor.add only supports matching dtypes"
        )
    if left_metadata.device != right_metadata.device:
        raise CompileTraceUnsupportedError(
            "torch.compile trace Tensor.add only supports matching devices"
        )
    if left_metadata.device.type == "cuda":
        _validate_cuda_metadata(left_metadata)
        _validate_cuda_metadata(right_metadata)
        left_shape, right_shape = left_metadata.shape, right_metadata.shape
        if not (
            left_shape == right_shape
            or (len(left_shape) == 2 and right_shape == (left_shape[1],))
            or (len(right_shape) == 2 and left_shape == (right_shape[1],))
        ):
            raise CompileTraceUnsupportedError(
                "torch.compile trace CUDA addition requires the same shape "
                "or shapes (M, N) and (N,); broader broadcasting is unsupported"
            )
    if grad_enabled is None:
        grad_enabled = _grad_enabled()
    shape = _broadcast_shape(left_metadata.shape, right_metadata.shape)
    return CompileTraceTensorMetadata(
        shape=shape,
        stride=_binary_output_stride(left_metadata, right_metadata, shape),
        dtype=left_metadata.dtype,
        device=left_metadata.device,
        requires_grad=(
            left_metadata.requires_grad or right_metadata.requires_grad
        )
        and grad_enabled,
        storage_offset=0 if left_metadata.device.type == "cuda" else None,
    )


def _normalize_mul_scalar(value):
    # Exact immutable Python numbers only: no arbitrary conversion callbacks,
    # NumPy objects, subclasses, complex promotion, or Tensor operands.
    if _builtins.type(value) not in (_builtins.bool, _builtins.int, _builtins.float):
        raise CompileTraceUnsupportedError(
            "torch.compile scalar multiplication requires an exact bool, int or float constant"
        )
    return _native._compile_trace_mul_scalar_value(value)


def _reduction_output_metadata(input_metadata, dim, keepdim):
    if type(dim) is not int or dim not in (1, -1) or type(keepdim) is not bool:
        raise CompileTraceUnsupportedError("torch.compile sum requires constant dim=1 or -1 and boolean keepdim")
    if input_metadata.device.type != "cuda" or len(input_metadata.shape) != 2:
        raise CompileTraceUnsupportedError("torch.compile sum requires rank-2 CUDA inputs")
    _validate_cuda_metadata(input_metadata)
    shape = (input_metadata.shape[0], 1) if keepdim else (input_metadata.shape[0],)
    import sys
    if shape[0] > sys.maxsize // 4:
        raise OverflowError("torch.compile sum output exceeds native storage limits")
    return CompileTraceTensorMetadata(
        shape=shape, stride=_contiguous_stride(shape), dtype=float32,
        device=input_metadata.device, requires_grad=False, storage_offset=0,
    )


def _scalar_output_metadata(input_metadata):
    if input_metadata.device.type != "cuda":
        raise CompileTraceUnsupportedError(
            "torch.compile scalar multiplication only supports CUDA"
        )
    _validate_cuda_metadata(input_metadata)
    return CompileTraceTensorMetadata(
        shape=input_metadata.shape,
        stride=_elementwise_output_stride(
            input_metadata.shape, ((input_metadata.shape, input_metadata.stride),),
            cuda=True,
        ),
        dtype=input_metadata.dtype,
        device=input_metadata.device,
        requires_grad=False,
        storage_offset=0,
    )


def _metadata_from_data(data, *, dtype, device, requires_grad):
    shape = _normalize_shape(_infer_shape(data))
    return CompileTraceTensorMetadata(
        shape=shape,
        stride=_contiguous_stride(shape),
        dtype=_normalize_dtype(dtype),
        device=_normalize_device(device),
        requires_grad=_normalize_requires_grad(requires_grad),
    )


def _type_name(value):
    value_type = _builtins.type(value)
    module = _builtins.object.__getattribute__(value_type, "__module__")
    name = _builtins.object.__getattribute__(value_type, "__qualname__")
    if module == "builtins":
        return name
    return f"{module}.{name}"


def _require_native_tensor(value, value_name):
    if not _builtins.isinstance(value, _native.Tensor):
        raise TypeError(
            "torch.compile trace execution expected native torch_rs Tensor "
            f"for {value_name!r}, got {_type_name(value)}"
        )


def _validate_metadata_types(metadata):
    # Dataclass equality treats bool/int/float substitutions as equal. Check
    # declarations before equality, including cached inputs and output leaves.
    if (type(metadata) is not CompileTraceTensorMetadata
            or type(metadata.shape) is not tuple or type(metadata.stride) is not tuple
            or len(metadata.shape) != len(metadata.stride)
            or any(type(n) is not int for n in (*metadata.shape, *metadata.stride))
            or type(metadata.requires_grad) is not bool
            or (metadata.storage_offset is not None and type(metadata.storage_offset) is not int)
            or type(metadata.dtype) is not CompileTraceDType
            or type(metadata.dtype.name) is not str
            or type(metadata.device) is not CompileTraceDevice
            or type(metadata.device.type) is not str
            or (metadata.device.index is not None and type(metadata.device.index) is not int)):
        raise CompileTraceUnsupportedError("torch.compile malformed metadata field types")


def _validate_cuda_metadata(metadata, *, require_contiguous=True):
    _validate_metadata_types(metadata)
    if metadata.device.type != "cuda":
        return
    if metadata.device.index is None:
        raise CompileTraceUnsupportedError("torch.compile trace CUDA requires a device ordinal")
    if metadata.dtype is not float32:
        raise CompileTraceUnsupportedError("torch.compile trace CUDA only supports float32")
    if metadata.requires_grad:
        raise CompileTraceUnsupportedError("torch.compile trace CUDA gradients are unsupported")
    import sys
    if (type(metadata.requires_grad) is not bool
            or type(metadata.shape) is not tuple or type(metadata.stride) is not tuple
            or len(metadata.shape) != len(metadata.stride)
            or any(type(n) is not int or n < 0 or n > sys.maxsize
                   for n in (*metadata.shape, *metadata.stride))
            or type(metadata.storage_offset) is not int
            or not 0 <= metadata.storage_offset <= sys.maxsize):
        raise CompileTraceUnsupportedError("torch.compile malformed CUDA layout metadata")
    if _element_count(metadata.shape) > sys.maxsize // 4:
        raise CompileTraceUnsupportedError("torch.compile CUDA layout exceeds storage limits")
    if require_contiguous and not _layout_is_contiguous(metadata.shape, metadata.stride):
        raise CompileTraceUnsupportedError("torch.compile trace CUDA requires contiguous layout")


def _metadata_from_native_tensor(tensor):
    shape, stride, requires_grad, dtype, device, offset = (
        _native._compile_trace_tensor_metadata(tensor)
    )
    device = _normalize_device(device)
    metadata = CompileTraceTensorMetadata(
        shape=_normalize_shape(shape),
        stride=_normalize_shape(stride),
        dtype=_normalize_dtype(dtype),
        device=device,
        requires_grad=_normalize_requires_grad(requires_grad),
        storage_offset=offset if device.type == "cuda" else None,
    )
    _validate_cuda_metadata(metadata, require_contiguous=False)
    return metadata


def _require_matching_metadata(
    actual,
    expected,
    *,
    value_name,
    check_requires_grad=True,
    dynamic_shape=False,
):
    _validate_metadata_types(actual)
    _validate_metadata_types(expected)
    if actual == expected and check_requires_grad:
        return

    mismatches = []
    fields = ["stride", "dtype", "device", "storage_offset"]
    if dynamic_shape:
        if len(actual.shape) != len(expected.shape):
            mismatches.append(
                f"shape rank expected {len(expected.shape)}, "
                f"got {len(actual.shape)}"
            )
    else:
        fields = ["shape", *fields]
    if check_requires_grad:
        fields.append("requires_grad")
    for field in fields:
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            mismatches.append(
                f"{field} expected {expected_value!r}, got {actual_value!r}"
            )
    if not mismatches:
        return
    mismatch_details = "; ".join(mismatches)
    raise ValueError(
        "torch.compile trace execution metadata mismatch for "
        f"{value_name!r}: {mismatch_details}"
    )


def _output_container_kind(value):
    if _builtins.type(value) is tuple:
        return "tuple"
    if _builtins.type(value) is list:
        return "list"
    return None


def _output_spec_and_metadata(
    value,
    recorder,
    *,
    role,
    memo=None,
    in_progress=None,
):
    if _builtins.isinstance(value, CompileTraceTensorProxy):
        recorder._require_owned_proxy(value)
        return value.name, value.metadata

    kind = _output_container_kind(value)
    if kind is None:
        raise CompileTraceUnsupportedError(
            "torch.compile trace only supports Tensor proxy return values "
            "or tuple/list containers of Tensor proxy return values, "
            f"got {_type_name(value)} for {role}"
        )

    if memo is None:
        memo = {}
    if in_progress is None:
        in_progress = set()
    value_id = _builtins.id(value)
    if value_id in memo:
        return memo[value_id]
    if value_id in in_progress:
        raise CompileTraceUnsupportedError(
            "torch.compile trace does not support cyclic tuple/list output "
            "containers"
        )

    in_progress.add(value_id)
    output_elements = []
    metadata_elements = []
    try:
        for index, element in enumerate(value):
            output_element, metadata_element = _output_spec_and_metadata(
                element,
                recorder,
                role=f"{role}[{index}]",
                memo=memo,
                in_progress=in_progress,
            )
            output_elements.append(output_element)
            metadata_elements.append(metadata_element)
        result = (
            CompileTraceOutputContainer(kind, tuple(output_elements)),
            CompileTraceOutputContainer(kind, tuple(metadata_elements)),
        )
        memo[value_id] = result
        return result
    finally:
        in_progress.remove(value_id)


def _materialize_graph_output(
    output_spec,
    metadata_spec,
    values,
    *,
    value_name,
    dynamic=False,
    metadata_values=None,
    memo=None,
    metadata_memo=None,
    metadata_only=False,
):
    if _builtins.isinstance(output_spec, _builtins.str):
        try:
            output = values[output_spec]
        except KeyError:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution graph output references unknown "
                f"value {output_spec!r}"
            ) from None
        if not _builtins.isinstance(metadata_spec, CompileTraceTensorMetadata):
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution graph output metadata is "
                "malformed"
            )
        _validate_cuda_metadata(metadata_spec, require_contiguous=False)
        expected_metadata = metadata_spec
        if dynamic:
            if metadata_values is None or output_spec not in metadata_values:
                raise CompileTraceUnsupportedError(
                    "torch.compile trace execution graph output references "
                    f"unknown metadata value {output_spec!r}"
                )
            expected_metadata = metadata_values[output_spec]
        _require_matching_metadata(
            output if metadata_only else _metadata_from_native_tensor(output),
            expected_metadata,
            value_name=output_spec,
            check_requires_grad=False,
        )
        return output

    if memo is None:
        memo = {}
    if not _builtins.isinstance(output_spec, CompileTraceOutputContainer):
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution graph output is malformed"
        )
    if (
        not _builtins.isinstance(metadata_spec, CompileTraceOutputContainer)
        or metadata_spec.kind != output_spec.kind
        or len(metadata_spec.elements) != len(output_spec.elements)
    ):
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution graph output metadata is malformed"
        )

    output_spec_id = _builtins.id(output_spec)
    if metadata_memo is None:
        metadata_memo = set()
    metadata_pair = (output_spec_id, _builtins.id(metadata_spec))
    if metadata_pair in metadata_memo and output_spec_id in memo:
        return memo[output_spec_id]
    metadata_memo.add(metadata_pair)

    # Reuse output objects, but validate each distinct metadata declaration.
    # Identity keys keep equal-valued malformed fields (such as True/1) apart.
    if output_spec_id in memo:
        for index, (child_output, child_metadata) in enumerate(
            zip(output_spec.elements, metadata_spec.elements)
        ):
            _materialize_graph_output(
                child_output,
                child_metadata,
                values,
                value_name=f"{value_name}[{index}]",
                dynamic=dynamic,
                metadata_values=metadata_values,
                memo=memo,
                metadata_memo=metadata_memo,
                metadata_only=metadata_only,
            )
        return memo[output_spec_id]

    if output_spec.kind == "list":
        materialized = []
        memo[output_spec_id] = materialized
        materialized.extend(
            _materialize_graph_output(
                child_output,
                child_metadata,
                values,
                value_name=f"{value_name}[{index}]",
                dynamic=dynamic,
                metadata_values=metadata_values,
                memo=memo,
                metadata_memo=metadata_memo,
                metadata_only=metadata_only,
            )
            for index, (child_output, child_metadata) in enumerate(
                zip(output_spec.elements, metadata_spec.elements)
            )
        )
        return materialized

    materialized = tuple(
        _materialize_graph_output(
            child_output,
            child_metadata,
            values,
            value_name=f"{value_name}[{index}]",
            dynamic=dynamic,
            metadata_values=metadata_values,
            memo=memo,
            metadata_memo=metadata_memo,
            metadata_only=metadata_only,
        )
        for index, (child_output, child_metadata) in enumerate(
            zip(output_spec.elements, metadata_spec.elements)
        )
    )
    memo[output_spec_id] = materialized
    return materialized


def _execute_operation(operation, values):
    if operation.op == "call_reduction":
        return _native._compile_trace_reduction(
            values[operation.inputs[0]], operation.target, *operation.reduction
        )
    if operation.op != "call_method":
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution only supports recorded Tensor "
            f"call_method operations, got {operation.op!r}"
        )
    if operation.target not in _SUPPORTED_OPERATION_TARGETS:
        _unsupported_operation(f"Tensor.{operation.target}")

    if operation.target in _SUPPORTED_SCALAR_TARGETS:
        return _native._compile_trace_scalar(
            values[operation.inputs[0]], operation.scalar, operation.target
        )

    if operation.target in _SUPPORTED_UNARY_TARGETS:
        if len(operation.inputs) != 1:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution only supports unary operations "
                f"with one input, got {len(operation.inputs)} inputs for "
                f"{operation.name!r}"
            )

        (input_name,) = operation.inputs
        try:
            input = values[input_name]
        except KeyError:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution operation "
                f"{operation.name!r} references unknown value {input_name!r}"
            ) from None

        return _native._compile_trace_unary(input, operation.target)

    if len(operation.inputs) != 2:
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution only supports binary operations "
            "with two inputs, "
            f"got {len(operation.inputs)} inputs for {operation.name!r}"
        )

    left_name, right_name = operation.inputs
    try:
        left = values[left_name]
        right = values[right_name]
    except KeyError as error:
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution operation "
            f"{operation.name!r} references unknown value {error.args[0]!r}"
        ) from None

    return _native._compile_trace_binary(left, right, operation.target)


def _expected_operation_metadata(operation, metadata_values, *, grad_enabled, declared_values=None):
    # Validate the complete node before any graph operation can execute, even
    # for graphs built manually or modified with dataclasses.replace().
    _validate_metadata_types(operation.metadata)
    if operation.target != "reshape" and operation.shape is not None:
        raise CompileTraceUnsupportedError("torch.compile non-reshape node has a shape payload")
    if operation.target != "transpose" and operation.axes is not None:
        raise CompileTraceUnsupportedError("torch.compile non-transpose node has axes")
    if operation.op == "call_reduction":
        if (operation.target != "sum" or len(operation.inputs) != 1
                or operation.scalar is not None or type(operation.reduction) is not tuple
                or len(operation.reduction) != 2):
            raise CompileTraceUnsupportedError("torch.compile malformed reduction node")
        expected = _reduction_output_metadata(
            metadata_values[operation.inputs[0]], *operation.reduction
        )
        declared = operation.metadata
        if (not isinstance(declared, CompileTraceTensorMetadata)
                or declared.device != expected.device or declared.storage_offset != 0
                or len(declared.shape) != len(expected.shape)
                or (operation.reduction[1] and declared.shape[1] != 1)):
            raise CompileTraceUnsupportedError("torch.compile malformed reduction metadata")
        _validate_cuda_metadata(declared)
        return expected
    if operation.op != "call_method" or operation.reduction is not None:
        raise CompileTraceUnsupportedError("torch.compile trace requires call_method nodes")
    if operation.target in _SUPPORTED_SCALAR_TARGETS:
        if len(operation.inputs) != 1:
            raise CompileTraceUnsupportedError("torch.compile scalar operation requires one Tensor input")
        _normalize_mul_scalar(operation.scalar)
        expected = _scalar_output_metadata(metadata_values[operation.inputs[0]])
        # Dynamic graphs recompute shapes, but their declared scalar outputs
        # must still describe fresh contiguous CUDA float32 storage. Validate
        # this before any earlier node can launch, including on cache hits.
        declared = operation.metadata
        if not _builtins.isinstance(declared, CompileTraceTensorMetadata):
            raise CompileTraceUnsupportedError("torch.compile scalar output metadata is malformed")
        if declared.device != expected.device or declared.storage_offset != 0:
            raise CompileTraceUnsupportedError(
                "torch.compile scalar output requires matching CUDA device and zero storage offset"
            )
        _validate_cuda_metadata(declared)
        return expected
    if operation.scalar is not None:
        raise CompileTraceUnsupportedError("torch.compile non-scalar operation has a scalar payload")
    if operation.target in ("contiguous", "t", "transpose", "reshape"):
        if len(operation.inputs) != 1:
            raise CompileTraceUnsupportedError(f"torch.compile {operation.target} requires one input")
        name = operation.inputs[0]
        if operation.target == "reshape":
            def infer(metadata):
                return _reshape_output_metadata(metadata, operation.shape)
        elif operation.target == "transpose":
            def infer(metadata):
                return _transpose_output_metadata(metadata, operation.axes)
        else:
            infer = _t_output_metadata if operation.target == "t" else _contiguous_output_metadata
        expected = infer(metadata_values[name])
        declared = operation.metadata
        if not _builtins.isinstance(declared, CompileTraceTensorMetadata):
            raise CompileTraceUnsupportedError(f"torch.compile malformed {operation.target} output metadata")
        _validate_cuda_metadata(declared, require_contiguous=operation.target == "contiguous")
        # Dynamic sizes may change whether this is an alias or a pack. Check
        # the stored declaration against its original input independently.
        declared_input = metadata_values[name] if declared_values is None else declared_values[name]
        if declared != infer(declared_input):
            raise CompileTraceUnsupportedError(f"torch.compile malformed {operation.target} output metadata")
        return expected
    if operation.target in _SUPPORTED_UNARY_TARGETS:
        if len(operation.inputs) != 1:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution only supports unary operations "
                f"with one input, got {len(operation.inputs)} inputs for "
                f"{operation.name!r}"
            )
        (input_name,) = operation.inputs
        if metadata_values[input_name].device.type == "cuda" and operation.target != "neg":
            raise CompileTraceUnsupportedError(
                f"torch.compile trace CUDA unary operation {operation.target!r} is unsupported"
            )
        expected = _unary_output_metadata(
            metadata_values[input_name],
            operation.target,
            grad_enabled=grad_enabled,
        )
        if expected.device.type == "cuda":
            declared = operation.metadata
            if not _builtins.isinstance(declared, CompileTraceTensorMetadata):
                raise CompileTraceUnsupportedError("torch.compile unary output metadata is malformed")
            if (
                declared.device != expected.device
                or declared.storage_offset != 0
                or len(declared.shape) != len(expected.shape)
            ):
                raise CompileTraceUnsupportedError(
                    "torch.compile unary output requires matching CUDA device, rank and zero storage offset"
                )
            _validate_cuda_metadata(declared)
        return expected

    if operation.target not in _SUPPORTED_BINARY_TARGETS:
        _unsupported_operation(f"Tensor.{operation.target}")
    if len(operation.inputs) != 2:
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution only supports binary operations "
            "with two inputs, "
            f"got {len(operation.inputs)} inputs for {operation.name!r}"
        )

    left_name, right_name = operation.inputs
    expected = _binary_output_metadata(
        metadata_values[left_name],
        metadata_values[right_name],
        target=operation.target,
        grad_enabled=grad_enabled,
    )
    if expected.device.type == "cuda":
        # Dynamic execution derives concrete output sizes/strides from current
        # inputs, but even a private/modified graph must declare a valid CUDA
        # result before any earlier operation executes. CPU metadata remains
        # governed by its existing offset and autograd contracts.
        declared = operation.metadata
        if not _builtins.isinstance(declared, CompileTraceTensorMetadata):
            raise CompileTraceUnsupportedError("torch.compile binary output metadata is malformed")
        if (
            declared.device != expected.device
            or declared.storage_offset != 0
            or len(declared.shape) != len(expected.shape)
        ):
            raise CompileTraceUnsupportedError(
                "torch.compile binary output requires matching CUDA device, rank and zero storage offset"
            )
        _validate_cuda_metadata(declared)
    return expected


def execute_compile_trace_graph(graph, *inputs):
    if not _builtins.isinstance(graph, CompileTraceGraph):
        raise TypeError(
            "torch.compile trace execution expected CompileTraceGraph, "
            f"got {_type_name(graph)}"
        )
    if type(graph.dynamic) is not bool:
        raise CompileTraceUnsupportedError("torch.compile malformed dynamic policy")
    if len(graph.inputs) not in (1, 2):
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution currently supports one or two "
            "recorded inputs"
        )
    if len(inputs) != len(graph.inputs):
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution expected "
            f"{len(graph.inputs)} positional Tensor inputs, got {len(inputs)}"
        )

    values = {}
    metadata_values = {}
    for capture in graph.captures:
        if capture.name in values:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution encountered duplicate value "
                f"name {capture.name!r}"
            )
        _require_native_tensor(capture.value, capture.name)
        capture_metadata = _metadata_from_native_tensor(capture.value)
        _require_matching_metadata(
            capture_metadata,
            capture.metadata,
            value_name=capture.name,
        )
        values[capture.name] = capture.value
        metadata_values[capture.name] = capture_metadata

    for graph_input, input in zip(graph.inputs, inputs):
        if graph_input.name in values:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution encountered duplicate value "
                f"name {graph_input.name!r}"
            )
        _require_native_tensor(input, graph_input.name)
        input_metadata = _metadata_from_native_tensor(input)
        _require_matching_metadata(
            input_metadata,
            graph_input.metadata,
            value_name=graph_input.name,
            dynamic_shape=graph.dynamic,
        )
        values[graph_input.name] = input
        metadata_values[graph_input.name] = input_metadata
    # Include unused captures and inputs: they must not divert a CUDA graph
    # into the per-operation executor, even when a cached graph was modified.
    devices = {metadata.device for metadata in metadata_values.values()}
    if any(device.type == "cuda" for device in devices) and len(devices) != 1:
        raise CompileTraceUnsupportedError(
            "torch.compile trace CUDA requires matching devices for all inputs and captures"
        )
    grad_enabled = _grad_enabled()
    declared_values = {item.name: item.metadata for item in (*graph.captures, *graph.inputs)}
    # Validate every operation before launching any work. In particular a
    # dynamic cache hit may have individually valid inputs whose shapes no
    # longer agree at an addition later in the graph.
    for operation in graph.operations:
        if (type(operation) is not CompileTraceOperation
                or any(type(field) is not str for field in (operation.name, operation.op, operation.target))
                or type(operation.inputs) is not tuple
                or any(type(name) is not str for name in operation.inputs)):
            raise CompileTraceUnsupportedError("torch.compile malformed operation field types")
        if operation.name in metadata_values:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution encountered duplicate value "
                f"name {operation.name!r}"
            )
        for input_name in operation.inputs:
            if input_name not in metadata_values:
                raise CompileTraceUnsupportedError(
                    "torch.compile trace execution operation "
                    f"{operation.name!r} references unknown value {input_name!r}"
                )
        expected_metadata = _expected_operation_metadata(
            operation,
            metadata_values,
            grad_enabled=grad_enabled,
            declared_values=declared_values,
        )
        if graph.dynamic and expected_metadata.device.type == "cuda":
            # Runtime sizes may differ, but the cached declarations must still
            # form a valid graph for their recorded inputs. Never let dynamic
            # replanning conceal a malformed downstream shape or stride.
            declared_expected = _expected_operation_metadata(
                operation, declared_values, grad_enabled=grad_enabled,
            )
            _require_matching_metadata(
                declared_expected, operation.metadata, value_name=operation.name,
            )
        declared_values[operation.name] = operation.metadata
        if not graph.dynamic:
            _require_matching_metadata(
                expected_metadata,
                operation.metadata,
                value_name=operation.name,
                check_requires_grad=False,
            )
        metadata_values[operation.name] = expected_metadata

    if (len(graph.operations) > 1 or any(op.target in ("contiguous", "t", "transpose", "reshape") for op in graph.operations)) and all(
        metadata.device.type == "cuda" for metadata in metadata_values.values()
    ):
        # Validate the output tree too before the native bridge can launch.
        # Reuse the materializer's structural/metadata checks without tensors.
        _materialize_graph_output(
            graph.output, graph.output_metadata, declared_values,
            value_name="output", metadata_only=True,
        )
        _materialize_graph_output(
            graph.output, graph.output_metadata, metadata_values,
            value_name="output", dynamic=graph.dynamic,
            metadata_values=metadata_values, metadata_only=True,
        )
        indices = {name: index for index, name in enumerate(metadata_values)}
        nodes = [
            (operation.target, tuple(indices[name] for name in operation.inputs),
             operation.shape if operation.target == "reshape" else
             operation.axes if operation.target == "transpose" else (
                 operation.reduction if operation.op == "call_reduction" else operation.scalar),
             metadata_values[operation.name].shape,
             metadata_values[operation.name].stride)
            for operation in graph.operations
        ]
        # All frontend guards above remain live on every call. Native planning
        # independently checks every node, then validates intermediate results
        # without round trips through Python metadata objects between kernels.
        outputs = _native._compile_trace_cuda_graph(tuple(values.values()), nodes)
        values.update((operation.name, output) for operation, output in zip(graph.operations, outputs))
        return _materialize_graph_output(
            graph.output, graph.output_metadata, values, value_name="output",
            dynamic=graph.dynamic, metadata_values=metadata_values,
        )

    for operation in graph.operations:
        output = _execute_operation(operation, values)
        output_metadata = _metadata_from_native_tensor(output)
        _require_matching_metadata(
            output_metadata,
            metadata_values[operation.name],
            value_name=operation.name,
        )
        if not graph.dynamic:
            _require_matching_metadata(
                output_metadata,
                operation.metadata,
                value_name=operation.name,
                check_requires_grad=False,
            )
        values[operation.name] = output
        metadata_values[operation.name] = output_metadata

    return _materialize_graph_output(
        graph.output,
        graph.output_metadata,
        values,
        value_name="output",
        dynamic=graph.dynamic,
        metadata_values=metadata_values,
    )


@dataclass(frozen=True, slots=True, eq=False)
class CompileTraceTensorProxy:
    _recorder: CompileTraceRecorder
    name: str
    metadata: CompileTraceTensorMetadata

    @property
    def shape(self):
        return self.metadata.shape

    @property
    def dtype(self):
        return self.metadata.dtype

    @property
    def device(self):
        return self.metadata.device

    @property
    def requires_grad(self):
        return self.metadata.requires_grad

    def stride(self):
        return self.metadata.stride

    def dim(self):
        return len(self.metadata.shape)

    def reshape(self, *args, **kwargs):
        return self._recorder.record_reshape(self, _bind_reshape_shape(args, kwargs))

    def transpose(self, dim0, dim1):
        return self._recorder.record_transpose(self, dim0, dim1)

    def t(self):
        return self._recorder.record_unary("t", self)

    def contiguous(self):
        return self._recorder.record_unary("contiguous", self)

    def neg(self):
        return self._recorder.record_unary("neg", self)

    def negative(self):
        return self._recorder.record_unary("neg", self)

    def abs(self):
        return self._recorder.record_unary("abs", self)

    def absolute(self):
        return self._recorder.record_unary("abs", self)

    def relu(self):
        return self._recorder.record_unary("relu", self)

    def square(self):
        return self._recorder.record_unary("square", self)

    def detach(self):
        return self._recorder.record_unary("detach", self)

    def float(self, *args, **kwargs):
        if args:
            raise CompileTraceUnsupportedError(
                "torch.compile trace Tensor.float only supports zero arguments"
            )
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise CompileTraceUnsupportedError(
                "torch.compile trace Tensor.float does not support keyword "
                f"arguments: {names}"
            )
        return self._recorder.record_unary("float", self)

    def __neg__(self):
        return self.neg()

    def __abs__(self):
        return self.abs()

    def __add__(self, other):
        return self._recorder.record_binary("add", self, other, "Tensor.__add__")

    def __radd__(self, other):
        if not _builtins.isinstance(other, CompileTraceTensorProxy):
            raise CompileTraceUnsupportedError(
                "torch.compile trace Tensor.__radd__ only supports Tensor "
                f"operands, got {_type_name(other)}"
            )
        return other._recorder.record_binary("add", other, self, "Tensor.__radd__")

    def add(self, other, *, alpha=1):
        if (
            _builtins.type(alpha) not in (_builtins.int, _builtins.float)
            or alpha != 1
        ):
            raise CompileTraceUnsupportedError(
                "torch.compile trace Tensor.add only supports alpha=1"
            )
        return self._recorder.record_binary("add", self, other, "Tensor.add")

    def sum(self, dim, keepdim=False):
        return self._recorder.record_reduction(self, dim, keepdim)

    def matmul(self, other):
        return self._recorder.record_binary("matmul", self, other, "Tensor.matmul")

    def __matmul__(self, other):
        return self.matmul(other)

    def __iadd__(self, other):
        _unsupported_operation("Tensor.__iadd__")

    def __sub__(self, other):
        _unsupported_operation("Tensor.__sub__")

    def __rsub__(self, other):
        _unsupported_operation("Tensor.__rsub__")

    def mul(self, other):
        return self._recorder.record_scalar("mul_scalar", self, other)

    def multiply(self, other):
        return self.mul(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        _unsupported_operation("Tensor.__truediv__")

    def __rtruediv__(self, other):
        _unsupported_operation("Tensor.__rtruediv__")

    def __eq__(self, other):
        _unsupported_operation("Tensor.__eq__")

    def __ne__(self, other):
        _unsupported_operation("Tensor.__ne__")

    def __lt__(self, other):
        _unsupported_operation("Tensor.__lt__")

    def __le__(self, other):
        _unsupported_operation("Tensor.__le__")

    def __gt__(self, other):
        _unsupported_operation("Tensor.__gt__")

    def __ge__(self, other):
        _unsupported_operation("Tensor.__ge__")

    def __getitem__(self, key):
        _unsupported_operation("Tensor.__getitem__")

    def __len__(self):
        _unsupported_operation("Tensor.__len__")

    def __iter__(self):
        _unsupported_operation("Tensor.__iter__")

    def __bool__(self):
        _unsupported_operation("Tensor.__bool__")

    def positive(self):
        _unsupported_operation("Tensor.positive")

    def __pos__(self):
        _unsupported_operation("Tensor.__pos__")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        _unsupported_operation(f"Tensor.{name}")


class CompileTraceRecorder:
    def __init__(self, name="compile_trace", *, dynamic=False):
        if _builtins.type(dynamic) is not _builtins.bool:
            raise TypeError("compile trace recorder dynamic flag must be bool")
        self._name = _builtins.str(name)
        self._dynamic = dynamic
        self._inputs = []
        self._captures = []
        self._operations = []
        self._closed = False

    def input(
        self,
        *,
        name=None,
        shape,
        stride=None,
        dtype=float32,
        device="cpu",
        requires_grad=False,
        storage_offset=None,
    ):
        self._ensure_open()
        input_name = (
            f"arg{len(self._inputs)}" if name is None else _builtins.str(name)
        )
        if self._has_value(input_name):
            raise ValueError(f"duplicate compile trace value name: {input_name!r}")
        shape = _normalize_shape(shape)
        if stride is None:
            stride = _contiguous_stride(shape)
        else:
            stride = _normalize_shape(stride)
            if len(stride) != len(shape):
                raise ValueError(
                    "torch.compile trace input stride rank must match shape rank"
                )
        device = _normalize_device(device)
        if device.type == "cuda" and storage_offset is None:
            storage_offset = 0
        metadata = CompileTraceTensorMetadata(
            shape=shape,
            stride=stride,
            dtype=_normalize_dtype(dtype),
            device=device,
            requires_grad=_normalize_requires_grad(requires_grad),
            storage_offset=(
                None if storage_offset is None else _normalize_dimension(storage_offset)
            ),
        )
        compile_input = CompileTraceInput(
            name=input_name,
            index=len(self._inputs),
            metadata=metadata,
        )
        self._inputs.append(compile_input)
        return CompileTraceTensorProxy(self, input_name, metadata)

    def capture(self, *, name, value, metadata):
        self._ensure_open()
        capture_name = _builtins.str(name)
        if self._has_value(capture_name):
            raise ValueError(f"duplicate compile trace value name: {capture_name!r}")
        _require_native_tensor(value, capture_name)
        if not _builtins.isinstance(metadata, CompileTraceTensorMetadata):
            raise TypeError(
                "torch.compile trace capture metadata must be "
                "CompileTraceTensorMetadata"
            )
        actual_metadata = _metadata_from_native_tensor(value)
        _require_matching_metadata(
            actual_metadata,
            metadata,
            value_name=capture_name,
        )
        capture = CompileTraceCapture(
            name=capture_name,
            value=value,
            metadata=metadata,
        )
        self._captures.append(capture)
        return CompileTraceTensorProxy(self, capture_name, metadata)

    def record_unary(self, target, input):
        self._ensure_open()
        if target not in _SUPPORTED_UNARY_TARGETS:
            _unsupported_operation(f"Tensor.{target}")
        self._require_owned_proxy(input)
        name = self._next_operation_name(target)
        metadata = _unary_output_metadata(input.metadata, target)
        operation = CompileTraceOperation(
            name=name,
            op="call_method",
            target=target,
            inputs=(input.name,),
            metadata=metadata,
        )
        self._operations.append(operation)
        return CompileTraceTensorProxy(self, name, metadata)

    def record_reshape(self, input, shape):
        self._ensure_open()
        self._require_owned_proxy(input)
        metadata = _reshape_output_metadata(input.metadata, shape)
        name = self._next_operation_name("reshape")
        self._operations.append(CompileTraceOperation(
            name=name, op="call_method", target="reshape", inputs=(input.name,),
            metadata=metadata, shape=shape,
        ))
        return CompileTraceTensorProxy(self, name, metadata)

    def record_transpose(self, input, dim0, dim1):
        self._ensure_open()
        self._require_owned_proxy(input)
        axes = (dim0, dim1)
        metadata = _transpose_output_metadata(input.metadata, axes)
        name = self._next_operation_name("transpose")
        self._operations.append(CompileTraceOperation(
            name=name, op="call_method", target="transpose", inputs=(input.name,),
            metadata=metadata, axes=axes,
        ))
        return CompileTraceTensorProxy(self, name, metadata)

    def record_reduction(self, input, dim, keepdim):
        self._ensure_open()
        self._require_owned_proxy(input)
        metadata = _reduction_output_metadata(input.metadata, dim, keepdim)
        name = self._next_operation_name("sum")
        self._operations.append(CompileTraceOperation(
            name=name, op="call_reduction", target="sum", inputs=(input.name,),
            metadata=metadata, reduction=(1, keepdim),
        ))
        return CompileTraceTensorProxy(self, name, metadata)

    def record_scalar(self, target, input, scalar):
        self._ensure_open()
        if target not in _SUPPORTED_SCALAR_TARGETS:
            _unsupported_operation(f"Tensor.{target}")
        self._require_owned_proxy(input)
        metadata = _scalar_output_metadata(input.metadata)
        scalar = _normalize_mul_scalar(scalar)
        name = self._next_operation_name(target)
        self._operations.append(CompileTraceOperation(
            name=name, op="call_method", target=target, inputs=(input.name,),
            metadata=metadata, scalar=scalar,
        ))
        return CompileTraceTensorProxy(self, name, metadata)

    def record_binary(self, target, left, right, operation_name):
        self._ensure_open()
        if target not in _SUPPORTED_BINARY_TARGETS:
            _unsupported_operation(operation_name)
        self._require_owned_proxy(
            left,
            operation_name=operation_name,
            role="left operand",
        )
        self._require_owned_proxy(
            right,
            operation_name=operation_name,
            role="right operand",
        )
        name = self._next_operation_name(target)
        metadata = _binary_output_metadata(left.metadata, right.metadata, target=target)
        operation = CompileTraceOperation(
            name=name,
            op="call_method",
            target=target,
            inputs=(left.name, right.name),
            metadata=metadata,
        )
        self._operations.append(operation)
        return CompileTraceTensorProxy(self, name, metadata)

    def finish(self, output):
        self._ensure_open()
        output_spec, output_metadata = _output_spec_and_metadata(
            output,
            self,
            role="return value",
        )
        if len(self._inputs) not in (1, 2):
            raise CompileTraceUnsupportedError(
                "torch.compile trace currently supports one or two inputs"
            )
        self._closed = True
        return CompileTraceGraph(
            name=self._name,
            inputs=tuple(self._inputs),
            operations=tuple(self._operations),
            output=output_spec,
            output_metadata=output_metadata,
            captures=tuple(self._captures),
            dynamic=self._dynamic,
        )

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("compile trace recorder is already finished")

    def _has_value(self, name):
        return (
            any(input.name == name for input in self._inputs)
            or any(capture.name == name for capture in self._captures)
            or any(operation.name == name for operation in self._operations)
        )

    def _next_operation_name(self, target):
        index = len(self._operations)
        while True:
            name = f"{target}_{index}"
            if not self._has_value(name):
                return name
            index += 1

    def _require_owned_proxy(
        self,
        value,
        *,
        operation_name=None,
        role="value",
    ):
        if not _builtins.isinstance(value, CompileTraceTensorProxy):
            if operation_name is None:
                raise CompileTraceUnsupportedError(
                    "torch.compile trace only supports Tensor proxy values"
                )
            raise CompileTraceUnsupportedError(
                f"torch.compile trace {operation_name} only supports Tensor "
                f"operands, got {_type_name(value)} for {role}"
            )
        if value._recorder is not self:
            raise CompileTraceUnsupportedError(
                "torch.compile trace cannot mix Tensor operands from different "
                "recorders"
            )


class CompileTraceTorchModule:
    float32 = float32
    float = float32

    def __init__(self, recorder):
        self._recorder = recorder

    def tensor(
        self,
        data,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
        **kwargs,
    ):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise CompileTraceUnsupportedError(
                "torch.compile trace tensor() does not support keyword "
                f"arguments: {names}"
            )
        metadata = _metadata_from_data(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return self._recorder.input(
            shape=metadata.shape,
            stride=metadata.stride,
            dtype=metadata.dtype,
            device=metadata.device,
            requires_grad=metadata.requires_grad,
        )


def _trace_compile_graph(
    program,
    make_inputs,
    *,
    name=None,
    input_count,
    dynamic=False,
):
    if not _builtins.callable(program):
        raise TypeError("torch.compile trace program must be callable")
    if not _builtins.callable(make_inputs):
        raise TypeError("torch.compile trace input factory must be callable")

    recorder = CompileTraceRecorder(
        name or getattr(program, "__name__", "compile_trace"),
        dynamic=dynamic,
    )
    trace_module = CompileTraceTorchModule(recorder)
    inputs = make_inputs(trace_module)
    if not _builtins.isinstance(inputs, tuple):
        raise CompileTraceUnsupportedError(
            "torch.compile trace input factory must return a tuple of inputs"
        )
    if len(inputs) != input_count:
        raise CompileTraceUnsupportedError(
            "torch.compile trace expected "
            f"{input_count} positional Tensor inputs, got {len(inputs)}"
        )

    output = program(*inputs)
    return recorder.finish(output)


def trace_compile_graph(program, make_inputs, *, name=None, dynamic=False):
    return _trace_compile_graph(
        program,
        make_inputs,
        name=name,
        input_count=2,
        dynamic=dynamic,
    )


def trace_one_input_compile_graph(program, make_inputs, *, name=None, dynamic=False):
    return _trace_compile_graph(
        program,
        make_inputs,
        name=name,
        input_count=1,
        dynamic=dynamic,
    )


__all__ = [
    "CompileTraceDType",
    "CompileTraceDevice",
    "CompileTraceCapture",
    "CompileTraceGraph",
    "CompileTraceInput",
    "CompileTraceOperation",
    "CompileTraceOutputContainer",
    "CompileTraceRecorder",
    "CompileTraceTensorMetadata",
    "CompileTraceTensorProxy",
    "CompileTraceTorchModule",
    "CompileTraceUnsupportedError",
    "cpu",
    "execute_compile_trace_graph",
    "float",
    "float32",
    "trace_compile_graph",
    "trace_one_input_compile_graph",
]
