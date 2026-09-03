"""Private helpers for the small supported ``torch.compile`` graph subset."""

import dis as _dis
import inspect as _inspect
import sys as _sys
import types as _types


_MISSING = object()


class _UnsupportedGraphlet(Exception):
    pass


class _ReluTraceProxy:
    __slots__ = ("_node",)

    def __init__(self, node):
        self._node = node

    def relu(self):
        if self._node != "input":
            raise _UnsupportedGraphlet
        return _ReluTraceProxy("relu")


def try_compile_supported_graphlet(model, unsupported_message):
    """Return a compiled callable for the exact supported graphlet, if any."""
    graphlet = _supported_relu_graphlet(model)
    if graphlet is None:
        return None
    relu_kind, signature, parameter_name = graphlet
    if relu_kind is None:
        return None

    captured_top_level_relu = (
        _package_module().relu if relu_kind == "top_level" else None
    )
    if not _trace_returns_unary_relu(model, relu_kind, captured_top_level_relu):
        return None

    def compiled_model(*args, **kwargs):
        if (
            captured_top_level_relu is not None
            and _current_package_relu() is not captured_top_level_relu
        ):
            raise NotImplementedError(unsupported_message)
        tensor = _bind_runtime_relu_argument(
            signature,
            parameter_name,
            args,
            kwargs,
            unsupported_message,
        )
        return tensor.relu()

    return compiled_model


def _supported_relu_graphlet(model):
    if type(model) is not _types.FunctionType:
        return None

    signature, parameter_name = _single_positional_parameter(model)
    if signature is None:
        return None

    relu_kind = _bytecode_relu_kind(model, parameter_name)
    if relu_kind is None:
        return None

    return relu_kind, signature, parameter_name


def _single_positional_parameter(model):
    try:
        signature = _inspect.signature(model)
    except (TypeError, ValueError):
        return None, None

    parameters = tuple(signature.parameters.values())
    if len(parameters) != 1:
        return None, None

    parameter = parameters[0]
    if parameter.kind not in (
        _inspect.Parameter.POSITIONAL_ONLY,
        _inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return None, None
    if parameter.default is not _inspect.Parameter.empty:
        return None, None
    return signature, parameter.name


def _bytecode_relu_kind(model, parameter_name):
    instructions = [
        instruction
        for instruction in _dis.get_instructions(model)
        if instruction.opname
        not in {
            "CACHE",
            "COPY_FREE_VARS",
            "EXTENDED_ARG",
            "NOP",
            "PRECALL",
            "PUSH_NULL",
            "RESUME",
        }
    ]
    if len(instructions) not in (4, 5):
        return None

    if _matches_method_relu_bytecode(instructions, parameter_name):
        return "method"
    if _matches_top_level_relu_bytecode(model, instructions, parameter_name):
        return "top_level"
    return None


def _matches_method_relu_bytecode(instructions, parameter_name):
    if len(instructions) != 4:
        return False
    load_input, load_relu, call, return_value = instructions
    return (
        load_input.opname == "LOAD_FAST"
        and load_input.argval == parameter_name
        and load_relu.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and load_relu.argval == "relu"
        and call.opname in {"CALL", "CALL_METHOD"}
        and call.arg == 0
        and return_value.opname == "RETURN_VALUE"
    )


def _matches_top_level_relu_bytecode(model, instructions, parameter_name):
    if len(instructions) != 5:
        return False
    load_module, load_relu, load_input, call, return_value = instructions
    if (
        load_module.opname != "LOAD_GLOBAL"
        or not _global_is_canonical_package_relu(model, load_module.argval)
        or load_relu.opname not in {"LOAD_ATTR", "LOAD_METHOD"}
        or load_relu.argval != "relu"
        or load_input.opname != "LOAD_FAST"
        or load_input.argval != parameter_name
        or call.opname not in {"CALL", "CALL_METHOD"}
        or call.arg != 1
        or return_value.opname != "RETURN_VALUE"
    ):
        return False
    return True


def _global_is_canonical_package_relu(model, name):
    module = model.__globals__.get(name)
    package = _package_module()
    return (
        module is package
        and _current_package_relu(package) is _canonical_package_relu(package)
        and _current_package_relu(package) is not _MISSING
    )


def _trace_returns_unary_relu(model, relu_kind, captured_top_level_relu):
    package = _package_module()
    original_relu = None

    def trace_relu(input, *args, **kwargs):
        if type(input) is _ReluTraceProxy and not args and not kwargs:
            return input.relu()
        return original_relu(input, *args, **kwargs)

    proxy = _ReluTraceProxy("input")
    if relu_kind == "top_level":
        if _current_package_relu(package) is not captured_top_level_relu:
            return False
        original_relu = _current_package_relu(package)
        package.relu = trace_relu
    try:
        result = model(proxy)
    except _UnsupportedGraphlet:
        return False
    finally:
        if original_relu is not None:
            package.relu = original_relu

    return type(result) is _ReluTraceProxy and result._node == "relu"


def _bind_runtime_relu_argument(
    signature,
    parameter_name,
    args,
    kwargs,
    unsupported_message,
):
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as error:
        raise NotImplementedError(unsupported_message) from error

    tensor = bound.arguments.get(parameter_name, _MISSING)
    if tensor is _MISSING:
        raise NotImplementedError(unsupported_message)
    package = _package_module()
    if type(tensor) is not package.Tensor:
        raise NotImplementedError(unsupported_message)
    if tensor.dtype is not package.float32:
        raise NotImplementedError(unsupported_message)
    if tensor.device != package.device("cpu"):
        raise NotImplementedError(unsupported_message)
    if tensor.requires_grad:
        raise NotImplementedError(unsupported_message)
    return tensor


def _current_package_relu(package=None):
    if package is None:
        package = _package_module()
    return getattr(package, "relu", _MISSING)


def _canonical_package_relu(package=None):
    if package is None:
        package = _package_module()
    native = getattr(package, "_C", None)
    return getattr(native, "relu", _MISSING)


def _package_module():
    return _sys.modules[__name__.partition(".")[0]]
