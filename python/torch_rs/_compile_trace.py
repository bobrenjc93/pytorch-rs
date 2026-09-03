"""Private native-only tracing pieces for future ``torch.compile`` work."""

from __future__ import annotations

import builtins as _builtins
import dis as _dis
import operator as _operator
import types as _types
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


@dataclass(frozen=True, slots=True)
class CompileTraceTensorMetadata:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: CompileTraceDType
    device: str
    requires_grad: bool


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
class CompileTraceGraph:
    name: str
    inputs: tuple[CompileTraceInput, ...]
    operations: tuple[CompileTraceOperation, ...]
    output: str
    output_metadata: CompileTraceTensorMetadata

    def forward(self, input):
        return execute_compile_trace_graph(self, input)


_SUPPORTED_UNARY_METHODS = (
    "Tensor.neg",
    "Tensor.negative",
    "Tensor.abs",
    "Tensor.absolute",
)
_SUPPORTED_UNARY_TARGETS = frozenset(("neg", "abs"))
_SUPPORTED_BINARY_METHODS = (
    "Tensor.__add__",
    "Tensor.add",
)
_SUPPORTED_BINARY_TARGETS = frozenset(("add",))
_SUPPORTED_OPERATION_TARGETS = _SUPPORTED_UNARY_TARGETS | _SUPPORTED_BINARY_TARGETS
_SUPPORTED_OPERATION_DESCRIPTION = ", ".join(
    (*_SUPPORTED_UNARY_METHODS, *_SUPPORTED_BINARY_METHODS)
)
_UNARY_METHOD_TARGETS = {
    "neg": "neg",
    "negative": "neg",
    "__neg__": "neg",
    "abs": "abs",
    "absolute": "abs",
    "__abs__": "abs",
}
_BINARY_METHOD_TARGETS = {
    "add": "add",
    "__add__": "add",
    "__radd__": "add",
}
_IGNORED_BYTECODE_OPS = frozenset(
    (
        "CACHE",
        "EXTENDED_ARG",
        "NOP",
        "RESUME",
    )
)
_GLOBAL_BYTECODE_OPS = frozenset(
    (
        "DELETE_GLOBAL",
        "IMPORT_FROM",
        "IMPORT_NAME",
        "LOAD_BUILD_CLASS",
        "LOAD_GLOBAL",
        "LOAD_NAME",
        "STORE_GLOBAL",
        "STORE_NAME",
    )
)
_MUTATING_BYTECODE_OPS = frozenset(
    (
        "DELETE_ATTR",
        "DELETE_DEREF",
        "DELETE_FAST",
        "DELETE_SUBSCR",
        "STORE_ATTR",
        "STORE_DEREF",
        "STORE_SUBSCR",
    )
)
_CONTROL_FLOW_BYTECODE_OPS = frozenset(
    (
        "FOR_ITER",
        "GET_ITER",
        "JUMP",
        "JUMP_BACKWARD",
        "JUMP_BACKWARD_NO_INTERRUPT",
        "JUMP_FORWARD",
        "POP_JUMP_BACKWARD_IF_FALSE",
        "POP_JUMP_BACKWARD_IF_NONE",
        "POP_JUMP_BACKWARD_IF_NOT_NONE",
        "POP_JUMP_BACKWARD_IF_TRUE",
        "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_FORWARD_IF_NONE",
        "POP_JUMP_FORWARD_IF_NOT_NONE",
        "POP_JUMP_FORWARD_IF_TRUE",
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_TRUE",
        "SETUP_FINALLY",
        "SETUP_WITH",
    )
)
_CODE_FLAG_VARARGS = 0x04
_CODE_FLAG_VARKEYWORDS = 0x08


def _unsupported_operation(operation):
    raise CompileTraceUnsupportedError(
        "torch.compile trace does not support "
        f"{operation}; only {_SUPPORTED_OPERATION_DESCRIPTION} are implemented"
    )


def _unsupported_bytecode(program, instruction, reason):
    program_name = getattr(
        program,
        "__qualname__",
        getattr(program, "__name__", program),
    )
    raise CompileTraceUnsupportedError(
        "torch.compile trace bytecode lowering does not support "
        f"{reason} in {program_name!r}: {instruction.opname}"
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
    if device is None:
        return "cpu"
    if device == "cpu" or _builtins.str(device) == "cpu":
        return "cpu"
    raise CompileTraceUnsupportedError(
        "torch.compile trace tensor() only supports CPU inputs"
    )


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


def _unary_output_metadata(input_metadata, *, grad_enabled=None):
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
        device="cpu",
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


def execute_compile_trace_graph(graph, input):
    if not _builtins.isinstance(graph, CompileTraceGraph):
        raise TypeError(
            "torch.compile trace execution expected CompileTraceGraph, "
            f"got {_type_name(graph)}"
        )
    if len(graph.inputs) != 1:
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution currently supports exactly one input"
        )

    graph_input = graph.inputs[0]
    _require_native_tensor(input, graph_input.name)
    input_metadata = _metadata_from_native_tensor(input)
    _require_matching_metadata(
        input_metadata,
        graph_input.metadata,
        value_name=graph_input.name,
    )

    values = {graph_input.name: input}
    metadata_values = {graph_input.name: input_metadata}
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

    try:
        output = values[graph.output]
    except KeyError:
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution graph output references unknown "
            f"value {graph.output!r}"
        ) from None
    _require_matching_metadata(
        _metadata_from_native_tensor(output),
        graph.output_metadata,
        value_name=graph.output,
        check_requires_grad=False,
    )
    return output


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

    def relu(self):
        _unsupported_operation("Tensor.relu")

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
        metadata = _unary_output_metadata(input.metadata)
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
        self._require_owned_proxy(output)
        if len(self._inputs) != 1:
            raise CompileTraceUnsupportedError(
                "torch.compile trace currently supports exactly one input"
            )
        self._closed = True
        return CompileTraceGraph(
            name=self._name,
            inputs=tuple(self._inputs),
            operations=tuple(self._operations),
            output=output.name,
            output_metadata=output.metadata,
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


@dataclass(frozen=True, slots=True)
class _BytecodeMethod:
    receiver: CompileTraceTensorProxy
    name: str


@dataclass(frozen=True, slots=True)
class _BytecodeConstant:
    value: object


def _validate_bytecode_lowering_program(program):
    if _builtins.type(program) is not _types.FunctionType:
        raise CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering currently supports exact "
            "Python functions only"
        )

    code = program.__code__
    if (
        code.co_argcount != 1
        or code.co_kwonlyargcount != 0
        or code.co_flags & (_CODE_FLAG_VARARGS | _CODE_FLAG_VARKEYWORDS)
    ):
        raise CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering currently supports exact "
            "Python functions with one positional Tensor argument"
        )
    if program.__closure__ is not None or code.co_freevars or code.co_cellvars:
        raise CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering does not support closures"
        )
    return code


def _pop_bytecode_value(stack, program, instruction):
    try:
        return stack.pop()
    except IndexError:
        _unsupported_bytecode(program, instruction, "stack underflow")


def _local_names_from_instruction(program, instruction, count):
    argval = instruction.argval
    if _builtins.isinstance(argval, tuple):
        names = argval
    elif _builtins.isinstance(argval, list):
        names = tuple(argval)
    elif count == 1 and _builtins.isinstance(argval, _builtins.str):
        names = (argval,)
    else:
        argrepr = _builtins.str(instruction.argrepr).strip()
        if argrepr.startswith("(") and argrepr.endswith(")"):
            argrepr = argrepr[1:-1]
        names = tuple(name.strip() for name in argrepr.split(",") if name.strip())

    if len(names) != count or not all(
        _builtins.isinstance(name, _builtins.str) and name for name in names
    ):
        _unsupported_bytecode(
            program,
            instruction,
            f"local operand decoding for {count} locals",
        )
    return names


def _load_bytecode_local(locals, stack, program, instruction, name):
    try:
        stack.append(locals[name])
    except KeyError:
        _unsupported_bytecode(
            program,
            instruction,
            f"unbound local {name!r}",
        )


def _store_bytecode_local(locals, stack, program, instruction, name):
    value = _require_bytecode_tensor(
        _pop_bytecode_value(stack, program, instruction),
        program,
        instruction,
        f"stored local {name!r}",
    )
    locals[name] = value


def _require_bytecode_tensor(value, program, instruction, role):
    if not _builtins.isinstance(value, CompileTraceTensorProxy):
        _unsupported_bytecode(program, instruction, f"non-Tensor {role}")
    return value


def _load_attr_pushes_method(instruction):
    if instruction.opname == "LOAD_METHOD":
        return True
    return instruction.opname == "LOAD_ATTR" and instruction.arg is not None and (
        instruction.arg & 1
    )


def _is_two_local_load_opcode(opname):
    return (
        opname in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW")
        or (
            opname.startswith("LOAD_FAST")
            and "_LOAD_FAST" in opname
            and "AND_CLEAR" not in opname
        )
    )


def _record_bytecode_method_call(recorder, method, args, program, instruction):
    target = _UNARY_METHOD_TARGETS.get(method.name)
    if target is not None:
        if args:
            _unsupported_bytecode(
                program,
                instruction,
                f"Tensor.{method.name} arguments",
            )
        return recorder.record_unary(target, method.receiver)

    target = _BINARY_METHOD_TARGETS.get(method.name)
    if target is not None:
        if len(args) != 1:
            _unsupported_bytecode(
                program,
                instruction,
                f"Tensor.{method.name} argument count {len(args)}",
            )
        other = _require_bytecode_tensor(
            args[0],
            program,
            instruction,
            f"Tensor.{method.name} operand",
        )
        if method.name == "__radd__":
            return recorder.record_binary(
                target,
                other,
                method.receiver,
                "Tensor.__radd__",
            )
        return recorder.record_binary(
            target,
            method.receiver,
            other,
            f"Tensor.{method.name}",
        )

    _unsupported_operation(f"Tensor.{method.name}")


def _record_bytecode_binary_add(recorder, stack, program, instruction):
    right = _require_bytecode_tensor(
        _pop_bytecode_value(stack, program, instruction),
        program,
        instruction,
        "right operand",
    )
    left = _require_bytecode_tensor(
        _pop_bytecode_value(stack, program, instruction),
        program,
        instruction,
        "left operand",
    )
    stack.append(recorder.record_binary("add", left, right, "Tensor.__add__"))


def _record_bytecode_unary(recorder, stack, program, instruction, target):
    input = _require_bytecode_tensor(
        _pop_bytecode_value(stack, program, instruction),
        program,
        instruction,
        "operand",
    )
    stack.append(recorder.record_unary(target, input))


def _lower_bytecode_instruction(
    recorder,
    locals,
    stack,
    program,
    instruction,
):
    opname = instruction.opname
    if opname in _IGNORED_BYTECODE_OPS:
        return None
    if opname in _GLOBAL_BYTECODE_OPS:
        _unsupported_bytecode(program, instruction, "global or import access")
    if opname in _MUTATING_BYTECODE_OPS:
        _unsupported_bytecode(program, instruction, "mutation")
    if opname in _CONTROL_FLOW_BYTECODE_OPS or "JUMP" in opname:
        _unsupported_bytecode(program, instruction, "control flow")
    if opname in ("KW_NAMES", "CALL_FUNCTION_KW", "CALL_METHOD_KW"):
        _unsupported_bytecode(program, instruction, "keyword arguments")

    if opname in ("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"):
        (name,) = _local_names_from_instruction(program, instruction, 1)
        _load_bytecode_local(locals, stack, program, instruction, name)
        return None

    if _is_two_local_load_opcode(opname):
        first_name, second_name = _local_names_from_instruction(
            program,
            instruction,
            2,
        )
        _load_bytecode_local(locals, stack, program, instruction, first_name)
        _load_bytecode_local(locals, stack, program, instruction, second_name)
        return None

    if opname == "STORE_FAST":
        (name,) = _local_names_from_instruction(program, instruction, 1)
        _store_bytecode_local(locals, stack, program, instruction, name)
        return None

    if opname == "STORE_FAST_LOAD_FAST":
        store_name, load_name = _local_names_from_instruction(
            program,
            instruction,
            2,
        )
        _store_bytecode_local(locals, stack, program, instruction, store_name)
        _load_bytecode_local(locals, stack, program, instruction, load_name)
        return None

    if opname == "STORE_FAST_STORE_FAST":
        first_name, second_name = _local_names_from_instruction(
            program,
            instruction,
            2,
        )
        _store_bytecode_local(locals, stack, program, instruction, first_name)
        _store_bytecode_local(locals, stack, program, instruction, second_name)
        return None

    if opname in ("LOAD_METHOD", "LOAD_ATTR"):
        if not _load_attr_pushes_method(instruction):
            _unsupported_bytecode(program, instruction, "attribute access")
        receiver = _require_bytecode_tensor(
            _pop_bytecode_value(stack, program, instruction),
            program,
            instruction,
            f"Tensor.{instruction.argval} receiver",
        )
        stack.append(_BytecodeMethod(receiver, _builtins.str(instruction.argval)))
        return None

    if opname in ("CALL", "CALL_METHOD", "CALL_FUNCTION"):
        argument_count = instruction.arg or 0
        args = [
            _pop_bytecode_value(stack, program, instruction)
            for _ in range(argument_count)
        ]
        args.reverse()
        callable_value = _pop_bytecode_value(stack, program, instruction)
        if not _builtins.isinstance(callable_value, _BytecodeMethod):
            _unsupported_bytecode(program, instruction, "function calls")
        stack.append(
            _record_bytecode_method_call(
                recorder,
                callable_value,
                tuple(args),
                program,
                instruction,
            )
        )
        return None

    if opname == "PRECALL":
        return None

    if opname == "LOAD_CONST":
        stack.append(_BytecodeConstant(instruction.argval))
        return None

    if opname in ("BINARY_ADD", "INPLACE_ADD"):
        if opname == "INPLACE_ADD":
            _unsupported_bytecode(program, instruction, "mutation")
        _record_bytecode_binary_add(recorder, stack, program, instruction)
        return None

    if opname == "BINARY_OP":
        if instruction.argrepr == "+":
            _record_bytecode_binary_add(recorder, stack, program, instruction)
            return None
        if instruction.argrepr == "+=":
            _unsupported_bytecode(program, instruction, "mutation")
        _unsupported_bytecode(
            program,
            instruction,
            f"binary operator {instruction.argrepr!r}",
        )

    if opname == "UNARY_NEGATIVE":
        _record_bytecode_unary(recorder, stack, program, instruction, "neg")
        return None

    if opname == "RETURN_VALUE":
        output = _require_bytecode_tensor(
            _pop_bytecode_value(stack, program, instruction),
            program,
            instruction,
            "return value",
        )
        if stack:
            _unsupported_bytecode(program, instruction, "residual stack values")
        return output

    if opname == "RETURN_CONST":
        _unsupported_bytecode(program, instruction, "non-Tensor return value")

    _unsupported_bytecode(program, instruction, "unsupported bytecode")


def lower_one_input_compile_graph(program, input_metadata, *, name=None):
    """Lower a narrow straight-line Python function into a native trace graph."""
    code = _validate_bytecode_lowering_program(program)
    if not _builtins.isinstance(input_metadata, CompileTraceTensorMetadata):
        raise TypeError(
            "torch.compile trace bytecode lowering expected "
            "CompileTraceTensorMetadata"
        )

    recorder = CompileTraceRecorder(
        name or getattr(program, "__name__", "compile_trace")
    )
    input_name = code.co_varnames[0]
    input_proxy = recorder.input(
        name=input_name,
        shape=input_metadata.shape,
        stride=input_metadata.stride,
        dtype=input_metadata.dtype,
        device=input_metadata.device,
        requires_grad=input_metadata.requires_grad,
    )
    locals = {input_name: input_proxy}
    stack = []

    for instruction in _dis.get_instructions(program):
        output = _lower_bytecode_instruction(
            recorder,
            locals,
            stack,
            program,
            instruction,
        )
        if output is not None:
            return recorder.finish(output)

    raise CompileTraceUnsupportedError(
        "torch.compile trace bytecode lowering did not find a Tensor return"
    )


def trace_one_input_compile_graph(program, make_inputs, *, name=None):
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
    if len(inputs) != 1:
        raise CompileTraceUnsupportedError(
            "torch.compile trace currently supports exactly one input"
        )

    output = program(*inputs)
    return recorder.finish(output)


__all__ = [
    "CompileTraceDType",
    "CompileTraceGraph",
    "CompileTraceInput",
    "CompileTraceOperation",
    "CompileTraceRecorder",
    "CompileTraceTensorMetadata",
    "CompileTraceTensorProxy",
    "CompileTraceTorchModule",
    "CompileTraceUnsupportedError",
    "execute_compile_trace_graph",
    "float",
    "float32",
    "lower_one_input_compile_graph",
    "trace_one_input_compile_graph",
]
