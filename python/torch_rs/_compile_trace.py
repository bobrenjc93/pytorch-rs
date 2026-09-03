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
_SUPPORTED_UNARY_DESCRIPTION = ", ".join(_SUPPORTED_UNARY_METHODS)


def _unsupported_operation(operation):
    raise CompileTraceUnsupportedError(
        "torch.compile trace does not support "
        f"{operation}; only {_SUPPORTED_UNARY_DESCRIPTION} are implemented"
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
    return CompileTraceTensorMetadata(
        shape=_normalize_shape(tensor.shape),
        stride=_normalize_shape(tensor.stride()),
        dtype=_normalize_dtype(tensor.dtype),
        device=_normalize_device(tensor.device),
        requires_grad=_normalize_requires_grad(tensor.requires_grad),
    )


def _require_matching_metadata(actual, expected, *, value_name):
    if actual == expected:
        return

    mismatches = []
    for field in ("shape", "stride", "dtype", "device", "requires_grad"):
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            mismatches.append(
                f"{field} expected {expected_value!r}, got {actual_value!r}"
            )
    mismatch_details = "; ".join(mismatches)
    raise ValueError(
        "torch.compile trace execution metadata mismatch for "
        f"{value_name!r}: {mismatch_details}"
    )


def _execute_operation(operation, values):
    if operation.op != "call_method":
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution only supports recorded Tensor.neg "
            f"and Tensor.abs call_method operations, got {operation.op!r}"
        )
    if operation.target not in _SUPPORTED_UNARY_TARGETS:
        _unsupported_operation(f"Tensor.{operation.target}")
    if len(operation.inputs) != 1:
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution only supports unary operations, "
            f"got {len(operation.inputs)} inputs for {operation.name!r}"
        )

    (input_name,) = operation.inputs
    try:
        input = values[input_name]
    except KeyError:
        raise CompileTraceUnsupportedError(
            "torch.compile trace execution operation "
            f"{operation.name!r} references unknown value {input_name!r}"
        ) from None

    if operation.target == "neg":
        return _native.Tensor.neg(input)
    if operation.target == "abs":
        return _native.Tensor.abs(input)
    _unsupported_operation(f"Tensor.{operation.target}")


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
    _require_matching_metadata(
        _metadata_from_native_tensor(input),
        graph_input.metadata,
        value_name=graph_input.name,
    )

    values = {graph_input.name: input}
    for operation in graph.operations:
        if operation.name in values:
            raise CompileTraceUnsupportedError(
                "torch.compile trace execution encountered duplicate value "
                f"name {operation.name!r}"
            )
        output = _execute_operation(operation, values)
        _require_matching_metadata(
            _metadata_from_native_tensor(output),
            operation.metadata,
            value_name=operation.name,
        )
        values[operation.name] = output

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
        _unsupported_operation("Tensor.__add__")

    def __radd__(self, other):
        _unsupported_operation("Tensor.__radd__")

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
        operation = CompileTraceOperation(
            name=name,
            op="call_method",
            target=target,
            inputs=(input.name,),
            metadata=input.metadata,
        )
        self._operations.append(operation)
        return CompileTraceTensorProxy(self, name, input.metadata)

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

    def _require_owned_proxy(self, value):
        if not _builtins.isinstance(value, CompileTraceTensorProxy):
            raise CompileTraceUnsupportedError(
                "torch.compile trace only supports Tensor proxy values"
            )
        if value._recorder is not self:
            raise CompileTraceUnsupportedError(
                "torch.compile trace cannot mix values from different recorders"
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
    "trace_one_input_compile_graph",
]
