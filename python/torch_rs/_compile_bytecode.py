"""Private CPython bytecode frontend for the narrow ``torch.compile`` path."""

from __future__ import annotations

import builtins as _builtins
import dis as _dis
import types as _types
from dataclasses import dataclass

from . import _compile_trace as _trace


_CODE_FLAG_VARARGS = 0x04
_CODE_FLAG_VARKEYWORDS = 0x08


@dataclass(frozen=True, slots=True)
class _BytecodeMethod:
    receiver: _trace.CompileTraceTensorProxy
    name: str


@dataclass(frozen=True, slots=True)
class _BytecodeConstant:
    value: object


@dataclass(frozen=True, slots=True)
class _MethodTarget:
    kind: str
    target: str
    argument_count: int
    operation_name: str
    reverse: bool = False


@dataclass(frozen=True, slots=True)
class _OpcodeForm:
    kind: str
    names: frozenset[str] = frozenset()
    predicate: object = None
    reason: str | None = None

    def matches(self, instruction):
        return instruction.opname in self.names or (
            self.predicate is not None and self.predicate(instruction)
        )


_METHOD_TARGETS = {
    "neg": _MethodTarget("unary", "neg", 0, "Tensor.neg"),
    "negative": _MethodTarget("unary", "neg", 0, "Tensor.negative"),
    "__neg__": _MethodTarget("unary", "neg", 0, "Tensor.__neg__"),
    "abs": _MethodTarget("unary", "abs", 0, "Tensor.abs"),
    "absolute": _MethodTarget("unary", "abs", 0, "Tensor.absolute"),
    "__abs__": _MethodTarget("unary", "abs", 0, "Tensor.__abs__"),
    "add": _MethodTarget("binary", "add", 1, "Tensor.add"),
    "__add__": _MethodTarget("binary", "add", 1, "Tensor.__add__"),
    "__radd__": _MethodTarget(
        "binary",
        "add",
        1,
        "Tensor.__radd__",
        reverse=True,
    ),
}
_BYTECODE_METHOD_NAMES = frozenset(_METHOD_TARGETS)


def _is_two_local_load(instruction):
    opname = instruction.opname
    return (
        opname in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW")
        or (
            opname.startswith("LOAD_FAST")
            and "_LOAD_FAST" in opname
            and "AND_CLEAR" not in opname
        )
    )


def _contains_jump(instruction):
    return "JUMP" in instruction.opname


_OPCODE_FORMS = (
    _OpcodeForm(
        "ignored",
        frozenset(("CACHE", "EXTENDED_ARG", "NOP", "RESUME")),
    ),
    _OpcodeForm(
        "unsupported",
        frozenset(
            (
                "BEFORE_ASYNC_WITH",
                "BEFORE_WITH",
                "CHECK_EG_MATCH",
                "CHECK_EXC_MATCH",
                "END_ASYNC_FOR",
                "POP_EXCEPT",
                "PUSH_EXC_INFO",
                "RERAISE",
                "SETUP_ASYNC_WITH",
                "SETUP_EXCEPT",
                "SETUP_FINALLY",
                "SETUP_WITH",
                "WITH_EXCEPT_START",
            )
        ),
        reason="exception handling",
    ),
    _OpcodeForm(
        "unsupported",
        frozenset(
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
        ),
        reason="global or import access",
    ),
    _OpcodeForm(
        "unsupported",
        frozenset(
            (
                "DELETE_ATTR",
                "DELETE_DEREF",
                "DELETE_FAST",
                "DELETE_SUBSCR",
                "STORE_ATTR",
                "STORE_DEREF",
                "STORE_SUBSCR",
            )
        ),
        reason="mutation",
    ),
    _OpcodeForm(
        "unsupported",
        frozenset(
            (
                "FOR_ITER",
                "GET_ITER",
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
                "TO_BOOL",
            )
        ),
        _contains_jump,
        "control flow",
    ),
    _OpcodeForm(
        "unsupported",
        frozenset(("CALL_FUNCTION_KW", "CALL_KW", "CALL_METHOD_KW", "KW_NAMES")),
        reason="keyword arguments",
    ),
    _OpcodeForm(
        "local_load",
        frozenset(("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW")),
    ),
    _OpcodeForm("local_load_pair", predicate=_is_two_local_load),
    _OpcodeForm("store", frozenset(("STORE_FAST",))),
    _OpcodeForm("store_load", frozenset(("STORE_FAST_LOAD_FAST",))),
    _OpcodeForm("store_store", frozenset(("STORE_FAST_STORE_FAST",))),
    _OpcodeForm("load_method", frozenset(("LOAD_ATTR", "LOAD_METHOD"))),
    _OpcodeForm("call", frozenset(("CALL", "CALL_FUNCTION", "CALL_METHOD"))),
    _OpcodeForm("precall", frozenset(("PRECALL",))),
    _OpcodeForm("load_const", frozenset(("LOAD_CONST",))),
    _OpcodeForm("binary", frozenset(("BINARY_ADD", "BINARY_OP", "INPLACE_ADD"))),
    _OpcodeForm("unary_neg", frozenset(("UNARY_NEGATIVE",))),
    _OpcodeForm("return", frozenset(("RETURN_VALUE", "RETURN_CONST"))),
)


def _unsupported_bytecode(program, instruction, reason):
    program_name = getattr(
        program,
        "__qualname__",
        getattr(program, "__name__", program),
    )
    raise _trace.CompileTraceUnsupportedError(
        "torch.compile trace bytecode lowering does not support "
        f"{reason} in {program_name!r}: {instruction.opname}"
    )


def _instruction_form(program, instruction):
    for form in _OPCODE_FORMS:
        if not form.matches(instruction):
            continue
        if form.reason is not None:
            _unsupported_bytecode(program, instruction, form.reason)
        return form.kind
    _unsupported_bytecode(program, instruction, "unsupported bytecode")


def _validate_program(program, input_count):
    if _builtins.type(program) is not _types.FunctionType:
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering currently supports exact "
            "Python functions only"
        )
    if input_count not in (1, 2):
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering currently supports one or "
            "two positional Tensor arguments"
        )

    code = program.__code__
    if (
        code.co_argcount != input_count
        or code.co_kwonlyargcount != 0
        or code.co_flags & (_CODE_FLAG_VARARGS | _CODE_FLAG_VARKEYWORDS)
    ):
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering currently supports exact "
            "Python functions with one or two positional Tensor arguments "
            "matching the runtime input count"
        )
    if program.__closure__ is not None or code.co_freevars or code.co_cellvars:
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering does not support closures"
        )
    if getattr(code, "co_exceptiontable", b""):
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering does not support exception "
            "handling"
        )
    return code


def _pop(stack, program, instruction):
    try:
        return stack.pop()
    except IndexError:
        _unsupported_bytecode(program, instruction, "stack underflow")


def _local_names(program, instruction, count):
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


def _load_local(locals, stack, program, instruction, name):
    try:
        stack.append(locals[name])
    except KeyError:
        _unsupported_bytecode(program, instruction, f"unbound local {name!r}")


def _require_tensor(value, program, instruction, role):
    if not _builtins.isinstance(value, _trace.CompileTraceTensorProxy):
        _unsupported_bytecode(program, instruction, f"non-Tensor {role}")
    return value


def _store_local(locals, stack, program, instruction, name):
    locals[name] = _require_tensor(
        _pop(stack, program, instruction),
        program,
        instruction,
        f"stored local {name!r}",
    )


def _load_attr_pushes_method(instruction):
    if instruction.opname == "LOAD_METHOD":
        return True
    if instruction.opname != "LOAD_ATTR":
        return False
    if instruction.arg is not None and instruction.arg & 1:
        return True
    return instruction.argval in _BYTECODE_METHOD_NAMES


def _handle_load_method(recorder, locals, stack, program, instruction):
    if not _load_attr_pushes_method(instruction):
        _unsupported_bytecode(program, instruction, "attribute access")
    receiver = _require_tensor(
        _pop(stack, program, instruction),
        program,
        instruction,
        f"Tensor.{instruction.argval} receiver",
    )
    stack.append(_BytecodeMethod(receiver, _builtins.str(instruction.argval)))


def _record_method_call(recorder, method, args, program, instruction):
    method_target = _METHOD_TARGETS.get(method.name)
    if method_target is None:
        _trace._unsupported_operation(f"Tensor.{method.name}")
    if len(args) != method_target.argument_count:
        _unsupported_bytecode(
            program,
            instruction,
            f"{method_target.operation_name} argument count {len(args)}",
        )
    if method_target.kind == "unary":
        return recorder.record_unary(method_target.target, method.receiver)

    other = _require_tensor(
        args[0],
        program,
        instruction,
        f"{method_target.operation_name} operand",
    )
    if method_target.reverse:
        return recorder.record_binary(
            method_target.target,
            other,
            method.receiver,
            method_target.operation_name,
        )
    return recorder.record_binary(
        method_target.target,
        method.receiver,
        other,
        method_target.operation_name,
    )


def _handle_call(recorder, locals, stack, program, instruction):
    argument_count = instruction.arg or 0
    args = [_pop(stack, program, instruction) for _ in range(argument_count)]
    args.reverse()
    callable_value = _pop(stack, program, instruction)
    if not _builtins.isinstance(callable_value, _BytecodeMethod):
        _unsupported_bytecode(program, instruction, "function calls")
    stack.append(
        _record_method_call(
            recorder,
            callable_value,
            tuple(args),
            program,
            instruction,
        )
    )


def _record_binary_add(recorder, stack, program, instruction):
    right = _require_tensor(
        _pop(stack, program, instruction),
        program,
        instruction,
        "right operand",
    )
    left = _require_tensor(
        _pop(stack, program, instruction),
        program,
        instruction,
        "left operand",
    )
    stack.append(recorder.record_binary("add", left, right, "Tensor.__add__"))


def _binary_operator_symbol(instruction):
    if instruction.opname == "BINARY_ADD":
        return "+"
    if instruction.opname == "INPLACE_ADD":
        return "+="
    return instruction.argrepr


def _handle_binary(recorder, locals, stack, program, instruction):
    symbol = _binary_operator_symbol(instruction)
    if symbol == "+":
        _record_binary_add(recorder, stack, program, instruction)
        return
    if symbol == "+=":
        _unsupported_bytecode(program, instruction, "mutation")
    _unsupported_bytecode(program, instruction, f"binary operator {symbol!r}")


def _handle_unary_neg(recorder, locals, stack, program, instruction):
    input = _require_tensor(
        _pop(stack, program, instruction),
        program,
        instruction,
        "operand",
    )
    stack.append(recorder.record_unary("neg", input))


def _handle_return(recorder, locals, stack, program, instruction):
    if instruction.opname == "RETURN_CONST":
        _unsupported_bytecode(program, instruction, "non-Tensor return value")
    output = _require_tensor(
        _pop(stack, program, instruction),
        program,
        instruction,
        "return value",
    )
    if stack:
        _unsupported_bytecode(program, instruction, "residual stack values")
    return output


def _handle_local_load(recorder, locals, stack, program, instruction):
    (name,) = _local_names(program, instruction, 1)
    _load_local(locals, stack, program, instruction, name)


def _handle_local_load_pair(recorder, locals, stack, program, instruction):
    first_name, second_name = _local_names(program, instruction, 2)
    _load_local(locals, stack, program, instruction, first_name)
    _load_local(locals, stack, program, instruction, second_name)


def _handle_store(recorder, locals, stack, program, instruction):
    (name,) = _local_names(program, instruction, 1)
    _store_local(locals, stack, program, instruction, name)


def _handle_store_load(recorder, locals, stack, program, instruction):
    store_name, load_name = _local_names(program, instruction, 2)
    _store_local(locals, stack, program, instruction, store_name)
    _load_local(locals, stack, program, instruction, load_name)


def _handle_store_store(recorder, locals, stack, program, instruction):
    first_name, second_name = _local_names(program, instruction, 2)
    _store_local(locals, stack, program, instruction, first_name)
    _store_local(locals, stack, program, instruction, second_name)


def _handle_load_const(recorder, locals, stack, program, instruction):
    stack.append(_BytecodeConstant(instruction.argval))


def _handle_noop(recorder, locals, stack, program, instruction):
    return None


_OPCODE_HANDLERS = {
    "ignored": _handle_noop,
    "local_load": _handle_local_load,
    "local_load_pair": _handle_local_load_pair,
    "store": _handle_store,
    "store_load": _handle_store_load,
    "store_store": _handle_store_store,
    "load_method": _handle_load_method,
    "call": _handle_call,
    "precall": _handle_noop,
    "load_const": _handle_load_const,
    "binary": _handle_binary,
    "unary_neg": _handle_unary_neg,
    "return": _handle_return,
}


def _lower_instruction(recorder, locals, stack, program, instruction):
    kind = _instruction_form(program, instruction)
    return _OPCODE_HANDLERS[kind](recorder, locals, stack, program, instruction)


def _normalize_input_metadata(input_metadata):
    if _builtins.isinstance(input_metadata, _trace.CompileTraceTensorMetadata):
        input_metadata = (input_metadata,)
    else:
        try:
            input_metadata = tuple(input_metadata)
        except TypeError:
            raise TypeError(
                "torch.compile trace bytecode lowering expected "
                "CompileTraceTensorMetadata input metadata"
            ) from None
    if len(input_metadata) not in (1, 2):
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering currently supports one or "
            "two positional Tensor arguments"
        )
    for metadata in input_metadata:
        if not _builtins.isinstance(
            metadata,
            _trace.CompileTraceTensorMetadata,
        ):
            raise TypeError(
                "torch.compile trace bytecode lowering expected "
                "CompileTraceTensorMetadata input metadata"
            )
    return input_metadata


def lower_compile_graph(program, input_metadata, *, name=None):
    """Lower a narrow straight-line Python function into a native trace graph."""
    input_metadata = _normalize_input_metadata(input_metadata)
    code = _validate_program(program, len(input_metadata))

    recorder = _trace.CompileTraceRecorder(
        name or getattr(program, "__name__", "compile_trace")
    )
    locals = {}
    for index, metadata in enumerate(input_metadata):
        input_name = code.co_varnames[index]
        locals[input_name] = recorder.input(
            name=input_name,
            shape=metadata.shape,
            stride=metadata.stride,
            dtype=metadata.dtype,
            device=metadata.device,
            requires_grad=metadata.requires_grad,
        )
    stack = []

    for instruction in _dis.get_instructions(program):
        output = _lower_instruction(recorder, locals, stack, program, instruction)
        if output is not None:
            return recorder.finish(output)

    raise _trace.CompileTraceUnsupportedError(
        "torch.compile trace bytecode lowering did not find a Tensor return"
    )


def lower_one_input_compile_graph(program, input_metadata, *, name=None):
    graph = lower_compile_graph(program, (input_metadata,), name=name)
    if len(graph.inputs) != 1:
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering expected exactly one input"
        )
    return graph


def lower_two_input_compile_graph(
    program,
    left_metadata,
    right_metadata,
    *,
    name=None,
):
    graph = lower_compile_graph(
        program,
        (left_metadata, right_metadata),
        name=name,
    )
    if len(graph.inputs) != 2:
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering expected exactly two inputs"
        )
    return graph


__all__ = [
    "lower_compile_graph",
    "lower_one_input_compile_graph",
    "lower_two_input_compile_graph",
]
