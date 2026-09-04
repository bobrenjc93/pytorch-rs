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

    def __post_init__(self):
        object.__setattr__(self, "device", _parse_device_metadata(self.device))


@dataclass(frozen=True, slots=True)
class CompileTraceInput:
    name: str
    index: int
    metadata: CompileTraceTensorMetadata


@dataclass(frozen=True, slots=True)
class CompileTraceOperation:
    name: str
    op: str
    target: str
    inputs: tuple[str, ...]
    metadata: CompileTraceTensorMetadata


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

    def forward(self, *inputs):
        return execute_compile_trace_graph(self, *inputs)


_SUPPORTED_UNARY_METHODS = (
    "Tensor.neg",
    "Tensor.negative",
    "Tensor.abs",
    "Tensor.absolute",
    "Tensor.relu",
    "Tensor.detach",
)
_SUPPORTED_VALUE_UNARY_TARGETS = frozenset(("neg", "abs", "relu"))
_SUPPORTED_ALIAS_UNARY_TARGETS = frozenset(("detach",))
_SUPPORTED_UNARY_TARGETS = (
    _SUPPORTED_VALUE_UNARY_TARGETS | _SUPPORTED_ALIAS_UNARY_TARGETS
)
_SUPPORTED_BINARY_METHODS = (
    "Tensor.__add__",
    "Tensor.add",
)
_SUPPORTED_BINARY_TARGETS = frozenset(("add",))
_SUPPORTED_OPERATION_TARGETS = _SUPPORTED_UNARY_TARGETS | _SUPPORTED_BINARY_TARGETS
_SUPPORTED_OPERATION_DESCRIPTION = ", ".join(
    (*_SUPPORTED_UNARY_METHODS, *_SUPPORTED_BINARY_METHODS)
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


def _elementwise_output_stride(shape, operand_layouts):
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
    next_stride = 1
    for position, axis in enumerate(permutation):
        stride[axis] = next_stride
        if position + 1 < rank:
            next_stride *= shape[axis]
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
    )


def _grad_enabled():
    return _native._compile_trace_grad_enabled()


def _unary_output_metadata(input_metadata, target, *, grad_enabled=None):
    if target in _SUPPORTED_ALIAS_UNARY_TARGETS:
        return CompileTraceTensorMetadata(
            shape=input_metadata.shape,
            stride=input_metadata.stride,
            dtype=input_metadata.dtype,
            device=input_metadata.device,
            requires_grad=False,
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
    )


def _binary_output_metadata(left_metadata, right_metadata, *, grad_enabled=None):
    if left_metadata.dtype is not right_metadata.dtype:
        raise CompileTraceUnsupportedError(
            "torch.compile trace Tensor.add only supports matching dtypes"
        )
    if left_metadata.device != right_metadata.device:
        raise CompileTraceUnsupportedError(
            "torch.compile trace Tensor.add only supports matching devices"
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


def _metadata_from_native_tensor(tensor):
    shape, stride, requires_grad = _native._compile_trace_tensor_metadata(tensor)
    return CompileTraceTensorMetadata(
        shape=_normalize_shape(shape),
        stride=_normalize_shape(stride),
        dtype=float32,
        device=cpu,
        requires_grad=_normalize_requires_grad(requires_grad),
    )


def _require_matching_metadata(
    actual,
    expected,
    *,
    value_name,
    check_requires_grad=True,
):
    if actual == expected and check_requires_grad:
        return

    mismatches = []
    fields = ["shape", "stride", "dtype", "device"]
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


def _output_spec_and_metadata(value, recorder, *, role):
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

    output_elements = []
    metadata_elements = []
    for index, element in enumerate(value):
        output_element, metadata_element = _output_spec_and_metadata(
            element,
            recorder,
            role=f"{role}[{index}]",
        )
        output_elements.append(output_element)
        metadata_elements.append(metadata_element)
    return (
        CompileTraceOutputContainer(kind, tuple(output_elements)),
        CompileTraceOutputContainer(kind, tuple(metadata_elements)),
    )


def _materialize_graph_output(output_spec, metadata_spec, values, *, value_name):
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
        _require_matching_metadata(
            _metadata_from_native_tensor(output),
            metadata_spec,
            value_name=output_spec,
            check_requires_grad=False,
        )
        return output

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

    elements = tuple(
        _materialize_graph_output(
            child_output,
            child_metadata,
            values,
            value_name=f"{value_name}[{index}]",
        )
        for index, (child_output, child_metadata) in enumerate(
            zip(output_spec.elements, metadata_spec.elements)
        )
    )
    if output_spec.kind == "tuple":
        return elements
    return list(elements)


def _execute_operation(operation, values):
    if operation.op != "call_method":
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution only supports recorded Tensor "
            f"call_method operations, got {operation.op!r}"
        )
    if operation.target not in _SUPPORTED_OPERATION_TARGETS:
        _unsupported_operation(f"Tensor.{operation.target}")

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


def _expected_operation_metadata(operation, metadata_values, *, grad_enabled):
    if operation.target in _SUPPORTED_UNARY_TARGETS:
        if len(operation.inputs) != 1:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution only supports unary operations "
                f"with one input, got {len(operation.inputs)} inputs for "
                f"{operation.name!r}"
            )
        (input_name,) = operation.inputs
        return _unary_output_metadata(
            metadata_values[input_name],
            operation.target,
            grad_enabled=grad_enabled,
        )

    if operation.target not in _SUPPORTED_BINARY_TARGETS:
        _unsupported_operation(f"Tensor.{operation.target}")
    if len(operation.inputs) != 2:
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution only supports binary operations "
            "with two inputs, "
            f"got {len(operation.inputs)} inputs for {operation.name!r}"
        )

    left_name, right_name = operation.inputs
    return _binary_output_metadata(
        metadata_values[left_name],
        metadata_values[right_name],
        grad_enabled=grad_enabled,
    )


def execute_compile_trace_graph(graph, *inputs):
    if not _builtins.isinstance(graph, CompileTraceGraph):
        raise TypeError(
            "torch.compile trace execution expected CompileTraceGraph, "
            f"got {_type_name(graph)}"
        )
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
    for graph_input, input in zip(graph.inputs, inputs):
        _require_native_tensor(input, graph_input.name)
        input_metadata = _metadata_from_native_tensor(input)
        _require_matching_metadata(
            input_metadata,
            graph_input.metadata,
            value_name=graph_input.name,
        )
        values[graph_input.name] = input
        metadata_values[graph_input.name] = input_metadata
    grad_enabled = _grad_enabled()
    for operation in graph.operations:
        if operation.name in values:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution encountered duplicate value "
                f"name {operation.name!r}"
            )
        for input_name in operation.inputs:
            if input_name not in values:
                raise CompileTraceUnsupportedError(
                    "torch.compile trace execution operation "
                    f"{operation.name!r} references unknown value {input_name!r}"
                )
        expected_metadata = _expected_operation_metadata(
            operation,
            metadata_values,
            grad_enabled=grad_enabled,
        )
        output = _execute_operation(operation, values)
        output_metadata = _metadata_from_native_tensor(output)
        _require_matching_metadata(
            output_metadata,
            expected_metadata,
            value_name=operation.name,
        )
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

    def detach(self):
        return self._recorder.record_unary("detach", self)

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

    def __iadd__(self, other):
        _unsupported_operation("Tensor.__iadd__")

    def __sub__(self, other):
        _unsupported_operation("Tensor.__sub__")

    def __rsub__(self, other):
        _unsupported_operation("Tensor.__rsub__")

    def __mul__(self, other):
        _unsupported_operation("Tensor.__mul__")

    def __rmul__(self, other):
        _unsupported_operation("Tensor.__rmul__")

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
    def __init__(self, name="compile_trace"):
        self._name = _builtins.str(name)
        self._inputs = []
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
        metadata = CompileTraceTensorMetadata(
            shape=shape,
            stride=stride,
            dtype=_normalize_dtype(dtype),
            device=_normalize_device(device),
            requires_grad=_normalize_requires_grad(requires_grad),
        )
        compile_input = CompileTraceInput(
            name=input_name,
            index=len(self._inputs),
            metadata=metadata,
        )
        self._inputs.append(compile_input)
        return CompileTraceTensorProxy(self, input_name, metadata)

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
        metadata = _binary_output_metadata(left.metadata, right.metadata)
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
        )

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("compile trace recorder is already finished")

    def _has_value(self, name):
        return any(input.name == name for input in self._inputs) or any(
            operation.name == name for operation in self._operations
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


def _trace_compile_graph(program, make_inputs, *, name=None, input_count):
    if not _builtins.callable(program):
        raise TypeError("torch.compile trace program must be callable")
    if not _builtins.callable(make_inputs):
        raise TypeError("torch.compile trace input factory must be callable")

    recorder = CompileTraceRecorder(
        name or getattr(program, "__name__", "compile_trace")
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


def trace_compile_graph(program, make_inputs, *, name=None):
    return _trace_compile_graph(program, make_inputs, name=name, input_count=2)


def trace_one_input_compile_graph(program, make_inputs, *, name=None):
    return _trace_compile_graph(program, make_inputs, name=name, input_count=1)


__all__ = [
    "CompileTraceDType",
    "CompileTraceDevice",
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
