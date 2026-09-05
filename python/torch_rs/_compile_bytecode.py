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
class _BytecodeFunction:
    function: object
    name: str
    code: object


@dataclass(frozen=True, slots=True)
class _HelperCacheDependency:
    global_name: str
    function: object
    code: object


@dataclass(frozen=True, slots=True)
class _GlobalLoadDependency:
    name: str
    instruction: object


@dataclass(frozen=True, slots=True)
class _CompileProgramDescriptor:
    code: object
    global_loads: tuple[_GlobalLoadDependency, ...]


@dataclass(frozen=True, slots=True)
class _CompileCacheRequest:
    key: object
    descriptor: _CompileProgramDescriptor
    input_metadatas: tuple
    helper_dependencies: tuple[_HelperCacheDependency, ...]


@dataclass(slots=True)
class _LoweringState:
    root_program: object
    helper_dependencies: tuple[_HelperCacheDependency, ...] = ()
    helper_call_count: int = 0


@dataclass(frozen=True, slots=True)
class _RequiresGradBranchLayout:
    input_name: str
    predicate_start: int
    jump_index: int
    false_start: int
    true_body: tuple
    false_body: tuple
    tail: tuple = ()


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
    "relu": _MethodTarget("unary", "relu", 0, "Tensor.relu"),
    "square": _MethodTarget("unary", "square", 0, "Tensor.square"),
    "detach": _MethodTarget("unary", "detach", 0, "Tensor.detach"),
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
_IGNORED_OPCODE_NAMES = frozenset(("CACHE", "EXTENDED_ARG", "NOP", "RESUME"))
_REQUIRES_GRAD_BRANCH_JUMPS = frozenset(
    ("POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_IF_FALSE")
)
_UNCONDITIONAL_FORWARD_JUMPS = frozenset(("JUMP_FORWARD",))


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


def _is_return_instruction(instruction):
    return instruction.opname in ("RETURN_VALUE", "RETURN_CONST")


def _is_ignored_instruction(instruction):
    return instruction.opname in _IGNORED_OPCODE_NAMES


_OPCODE_FORMS = (
    _OpcodeForm(
        "ignored",
        _IGNORED_OPCODE_NAMES,
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
    _OpcodeForm("load_global", frozenset(("LOAD_GLOBAL",))),
    _OpcodeForm("load_method", frozenset(("LOAD_ATTR", "LOAD_METHOD"))),
    _OpcodeForm("call", frozenset(("CALL", "CALL_FUNCTION", "CALL_METHOD"))),
    _OpcodeForm("precall", frozenset(("PRECALL",))),
    _OpcodeForm("push_null", frozenset(("PUSH_NULL",))),
    _OpcodeForm("load_const", frozenset(("LOAD_CONST", "LOAD_SMALL_INT"))),
    _OpcodeForm("build_tuple", frozenset(("BUILD_TUPLE",))),
    _OpcodeForm("build_list", frozenset(("BUILD_LIST",))),
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


def _validate_function_code(program, code):
    if (
        code.co_argcount not in (1, 2)
        or code.co_kwonlyargcount != 0
        or code.co_flags & (_CODE_FLAG_VARARGS | _CODE_FLAG_VARKEYWORDS)
    ):
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering currently supports exact "
            "Python functions with one or two positional Tensor arguments"
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


def _validate_program(program):
    if _builtins.type(program) is not _types.FunctionType:
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering currently supports exact "
            "Python functions only"
        )

    code = program.__code__
    return _validate_function_code(program, code)


def _validate_helper_function(root_program, helper, program, instruction, code=None):
    if _builtins.type(helper) is not _types.FunctionType:
        _unsupported_bytecode(program, instruction, "global or import access")
    if helper.__defaults__ is not None or helper.__kwdefaults__ is not None:
        _unsupported_bytecode(program, instruction, "helper function defaults")
    if code is None:
        code = helper.__code__
    if helper.__closure__ is not None or code.co_freevars or code.co_cellvars:
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering does not support closures"
        )
    if (
        helper.__globals__ is not root_program.__globals__
        or helper.__module__ != root_program.__module__
    ):
        _unsupported_bytecode(program, instruction, "global or import access")
    helper_name = helper.__name__
    if (
        not _builtins.isinstance(helper_name, _builtins.str)
        or root_program.__globals__.get(helper_name) is not helper
    ):
        _unsupported_bytecode(program, instruction, "global or import access")
    return _validate_function_code(helper, code)


def _global_name(program, instruction):
    name = instruction.argval
    if _builtins.isinstance(name, _builtins.str) and name:
        return name

    argrepr = _builtins.str(instruction.argrepr).strip()
    if argrepr.startswith("NULL + "):
        argrepr = argrepr[7:].strip()
    if argrepr:
        return argrepr
    _unsupported_bytecode(program, instruction, "global operand decoding")


def _resolve_live_global_helper(root_program, program, instruction, name):
    try:
        helper = program.__globals__[name]
    except KeyError:
        _unsupported_bytecode(program, instruction, "global or import access")
    helper_code = (
        helper.__code__ if _builtins.type(helper) is _types.FunctionType else None
    )
    helper_code = _validate_helper_function(
        root_program,
        helper,
        program,
        instruction,
        helper_code,
    )
    return _HelperCacheDependency(name, helper, helper_code)


def analyze_compile_program(program):
    code = _validate_program(program)
    global_loads = []
    for instruction in _analyzable_bytecode_instructions(program, code):
        kind = _instruction_form(program, instruction)
        if kind != "load_global":
            continue
        name = _global_name(program, instruction)
        _resolve_live_global_helper(program, program, instruction, name)
        global_loads.append(_GlobalLoadDependency(name, instruction))
    return _CompileProgramDescriptor(code, tuple(global_loads))


def _helper_cache_dependencies(program, descriptor):
    return tuple(
        _resolve_live_global_helper(
            program,
            program,
            global_load.instruction,
            global_load.name,
        )
        for global_load in descriptor.global_loads
    )


def _resolve_global_helper(state, program, instruction, name):
    if program is state.root_program:
        for dependency in state.helper_dependencies:
            if dependency.global_name == name:
                return dependency
        _unsupported_bytecode(program, instruction, "helper dependency snapshot")
    return _resolve_live_global_helper(state.root_program, program, instruction, name)


def _validate_input_metadatas(code, input_metadatas):
    if not _builtins.isinstance(input_metadatas, tuple):
        raise TypeError(
            "torch.compile trace bytecode lowering expected a tuple of "
            "CompileTraceTensorMetadata inputs"
        )
    if len(input_metadatas) != code.co_argcount:
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering expected metadata for "
            f"{code.co_argcount} positional Tensor arguments, got "
            f"{len(input_metadatas)}"
        )
    for input_metadata in input_metadatas:
        if not _builtins.isinstance(
            input_metadata,
            _trace.CompileTraceTensorMetadata,
        ):
            raise TypeError(
                "torch.compile trace bytecode lowering expected "
                "CompileTraceTensorMetadata"
            )


def prepare_compile_cache_request(program, input_metadatas, descriptor=None):
    """Return cache metadata and the exact helper snapshot used for lowering."""
    code = getattr(program, "__code__", None)
    if descriptor is None or descriptor.code is not code:
        descriptor = analyze_compile_program(program)
    else:
        _validate_function_code(program, descriptor.code)
    _validate_input_metadatas(descriptor.code, input_metadatas)
    helper_dependencies = _helper_cache_dependencies(program, descriptor)
    return _CompileCacheRequest(
        key=(descriptor.code, input_metadatas, helper_dependencies),
        descriptor=descriptor,
        input_metadatas=input_metadatas,
        helper_dependencies=helper_dependencies,
    )


def compile_cache_key(program, input_metadatas):
    """Return a cache key that includes validated helper function dependencies."""
    return prepare_compile_cache_request(program, input_metadatas).key


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


def _jump_offset(program, instruction):
    target = getattr(instruction, "argval", None)
    if _builtins.type(target) is not _builtins.int:
        _unsupported_bytecode(program, instruction, "control flow")
    return target


def _instruction_offsets(program, instructions):
    offsets = {}
    for index, instruction in enumerate(instructions):
        offset = getattr(instruction, "offset", None)
        if _builtins.type(offset) is not _builtins.int:
            _unsupported_bytecode(program, instruction, "control flow")
        if offset in offsets:
            _unsupported_bytecode(program, instruction, "control flow")
        offsets[offset] = index
    return offsets


def _instruction_index_for_offset(program, instructions, instruction):
    offsets = _instruction_offsets(program, instructions)
    target = _jump_offset(program, instruction)
    try:
        return offsets[target]
    except KeyError:
        _unsupported_bytecode(program, instruction, "control flow")


def _control_flow_indices(instructions):
    return tuple(
        index
        for index, instruction in enumerate(instructions)
        if _contains_jump(instruction) or instruction.opname == "FOR_ITER"
    )


def _last_non_ignored_index(instructions, start, stop):
    for index in range(stop - 1, start - 1, -1):
        if not _is_ignored_instruction(instructions[index]):
            return index
    return None


def _stored_local_names(program, instruction):
    kind = _instruction_form(program, instruction)
    if kind == "store":
        return _local_names(program, instruction, 1)
    if kind == "store_load":
        return _local_names(program, instruction, 2)[:1]
    if kind == "store_store":
        return _local_names(program, instruction, 2)
    return ()


def _branch_condition_input_name(program, code, instructions, jump_index):
    attr_index = jump_index - 1
    if attr_index >= 0 and instructions[attr_index].opname == "TO_BOOL":
        attr_index -= 1
    local_index = attr_index - 1
    if local_index < 0:
        _unsupported_bytecode(program, instructions[jump_index], "control flow")

    local_load = instructions[local_index]
    if _instruction_form(program, local_load) != "local_load":
        _unsupported_bytecode(program, instructions[jump_index], "control flow")
    (input_name,) = _local_names(program, local_load, 1)
    if input_name not in code.co_varnames[: code.co_argcount]:
        _unsupported_bytecode(program, local_load, "non-input requires_grad guard")

    load_attr = instructions[attr_index]
    if (
        load_attr.opname != "LOAD_ATTR"
        or _builtins.str(load_attr.argval) != "requires_grad"
        or _load_attr_pushes_method(load_attr)
    ):
        _unsupported_bytecode(program, load_attr, "control flow")
    for prefix_instruction in instructions[:local_index]:
        if input_name in _stored_local_names(program, prefix_instruction):
            _unsupported_bytecode(
                program,
                prefix_instruction,
                "non-input requires_grad guard",
            )
    return input_name, local_index


def _requires_grad_branch_layout(program, code, instructions):
    control_flow = _control_flow_indices(instructions)
    if not control_flow:
        return None

    branch_jumps = tuple(
        index
        for index in control_flow
        if instructions[index].opname in _REQUIRES_GRAD_BRANCH_JUMPS
    )
    if len(branch_jumps) != 1:
        _unsupported_bytecode(program, instructions[control_flow[0]], "control flow")
    (jump_index,) = branch_jumps

    other_jumps = tuple(index for index in control_flow if index != jump_index)
    if not all(
        instructions[index].opname in _UNCONDITIONAL_FORWARD_JUMPS
        for index in other_jumps
    ):
        _unsupported_bytecode(program, instructions[control_flow[0]], "control flow")
    if len(other_jumps) > 1:
        _unsupported_bytecode(program, instructions[other_jumps[1]], "control flow")

    input_name, predicate_start = _branch_condition_input_name(
        program,
        code,
        instructions,
        jump_index,
    )
    false_start = _instruction_index_for_offset(
        program,
        instructions,
        instructions[jump_index],
    )
    true_start = jump_index + 1
    if false_start <= true_start:
        _unsupported_bytecode(program, instructions[jump_index], "control flow")

    if other_jumps:
        (join_jump_index,) = other_jumps
        if not (true_start <= join_jump_index < false_start):
            _unsupported_bytecode(program, instructions[join_jump_index], "control flow")
        if _last_non_ignored_index(instructions, true_start, false_start) != (
            join_jump_index
        ):
            _unsupported_bytecode(program, instructions[join_jump_index], "control flow")
        join_start = _instruction_index_for_offset(
            program,
            instructions,
            instructions[join_jump_index],
        )
        if join_start <= false_start:
            _unsupported_bytecode(program, instructions[join_jump_index], "control flow")
        true_body = instructions[true_start:join_jump_index]
        false_body = instructions[false_start:join_start]
        tail = instructions[join_start:]
    else:
        last_true_instruction_index = _last_non_ignored_index(
            instructions,
            true_start,
            false_start,
        )
        if last_true_instruction_index is None or not _is_return_instruction(
            instructions[last_true_instruction_index]
        ):
            _unsupported_bytecode(program, instructions[jump_index], "control flow")
        true_body = instructions[true_start:false_start]
        false_body = instructions[false_start:]
        tail = ()

    return _RequiresGradBranchLayout(
        input_name=input_name,
        predicate_start=predicate_start,
        jump_index=jump_index,
        false_start=false_start,
        true_body=true_body,
        false_body=false_body,
        tail=tail,
    )


def _analyzable_bytecode_instructions(program, code):
    instructions = tuple(_dis.get_instructions(code))
    layout = _requires_grad_branch_layout(program, code, instructions)
    if layout is None:
        return instructions
    return (
        *instructions[: layout.predicate_start],
        *layout.true_body,
        *layout.false_body,
        *layout.tail,
    )


def _lowerable_bytecode_instructions(program, code, input_metadatas):
    instructions = tuple(_dis.get_instructions(code))
    layout = _requires_grad_branch_layout(program, code, instructions)
    if layout is None:
        return instructions

    input_names = code.co_varnames[: code.co_argcount]
    input_index = input_names.index(layout.input_name)
    selected_body = (
        layout.true_body
        if input_metadatas[input_index].requires_grad
        else layout.false_body
    )
    return (
        *instructions[: layout.predicate_start],
        *selected_body,
        *layout.tail,
    )


def _load_local(locals, stack, program, instruction, name):
    try:
        stack.append(locals[name])
    except KeyError:
        _unsupported_bytecode(program, instruction, f"unbound local {name!r}")


def _require_tensor(value, program, instruction, role):
    if not _builtins.isinstance(value, _trace.CompileTraceTensorProxy):
        _unsupported_bytecode(program, instruction, f"non-Tensor {role}")
    return value


def _require_output_value(value, program, instruction, role):
    if _builtins.isinstance(value, _trace.CompileTraceTensorProxy):
        return value
    if _builtins.type(value) in (_builtins.tuple, _builtins.list):
        for index, element in enumerate(value):
            _require_output_value(
                element,
                program,
                instruction,
                f"{role}[{index}]",
            )
        return value
    _unsupported_bytecode(program, instruction, f"non-Tensor {role}")


def _store_local(locals, stack, program, instruction, name):
    locals[name] = _require_output_value(
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


def _handle_load_global(
    recorder,
    locals,
    stack,
    program,
    instruction,
    state,
    active,
):
    del recorder, locals, active
    name = _global_name(program, instruction)
    helper_dependency = _resolve_global_helper(state, program, instruction, name)
    stack.append(
        _BytecodeFunction(
            helper_dependency.function,
            name,
            helper_dependency.code,
        )
    )


def _handle_load_method(
    recorder,
    locals,
    stack,
    program,
    instruction,
    state,
    active,
):
    del state, active
    if not _load_attr_pushes_method(instruction):
        _unsupported_bytecode(program, instruction, "attribute access")
    receiver = _require_tensor(
        _pop(stack, program, instruction),
        program,
        instruction,
        f"Tensor.{instruction.argval} receiver",
    )
    stack.append(_BytecodeMethod(receiver, _builtins.str(instruction.argval)))


def _lower_function_body(
    recorder,
    locals,
    stack,
    program,
    code,
    state,
    active,
    instructions=None,
):
    if instructions is None:
        instructions = _dis.get_instructions(code)
    for instruction in instructions:
        output = _lower_instruction(
            recorder,
            locals,
            stack,
            program,
            instruction,
            state,
            active,
        )
        if output is not None:
            return output

    raise _trace.CompileTraceUnsupportedError(
        "torch.compile trace bytecode lowering did not find a Tensor return"
    )


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


def _record_function_call(
    recorder,
    function,
    args,
    program,
    instruction,
    state,
    active,
):
    helper = function.function
    if helper in active:
        _unsupported_bytecode(program, instruction, "recursive helper function calls")
    if state.helper_call_count >= 1:
        _unsupported_bytecode(program, instruction, "helper function calls")
    helper_code = function.code
    if len(args) != helper_code.co_argcount:
        _unsupported_bytecode(
            program,
            instruction,
            f"helper function argument count {len(args)}",
        )
    helper_locals = {}
    for index, arg in enumerate(args):
        helper_locals[helper_code.co_varnames[index]] = _require_tensor(
            arg,
            program,
            instruction,
            f"helper argument {index}",
        )
    state.helper_call_count += 1
    return _lower_function_body(
        recorder,
        helper_locals,
        [],
        helper,
        helper_code,
        state,
        (*active, helper),
    )


def _handle_call(recorder, locals, stack, program, instruction, state, active):
    del locals
    argument_count = instruction.arg or 0
    args = [_pop(stack, program, instruction) for _ in range(argument_count)]
    args.reverse()
    callable_value = _pop(stack, program, instruction)
    if _builtins.isinstance(callable_value, _BytecodeMethod):
        stack.append(
            _record_method_call(
                recorder,
                callable_value,
                tuple(args),
                program,
                instruction,
            )
        )
        return
    if _builtins.isinstance(callable_value, _BytecodeFunction):
        stack.append(
            _record_function_call(
                recorder,
                callable_value,
                tuple(args),
                program,
                instruction,
                state,
                active,
            )
        )
        return
    _unsupported_bytecode(program, instruction, "function calls")


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


def _handle_binary(recorder, locals, stack, program, instruction, state, active):
    del locals, state, active
    symbol = _binary_operator_symbol(instruction)
    if symbol == "+":
        _record_binary_add(recorder, stack, program, instruction)
        return
    if symbol == "+=":
        _unsupported_bytecode(program, instruction, "mutation")
    _unsupported_bytecode(program, instruction, f"binary operator {symbol!r}")


def _handle_unary_neg(recorder, locals, stack, program, instruction, state, active):
    del locals, state, active
    input = _require_tensor(
        _pop(stack, program, instruction),
        program,
        instruction,
        "operand",
    )
    stack.append(recorder.record_unary("neg", input))


def _handle_return(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, state, active
    if instruction.opname == "RETURN_CONST":
        _unsupported_bytecode(program, instruction, "non-Tensor return value")
    output = _require_output_value(
        _pop(stack, program, instruction),
        program,
        instruction,
        "return value",
    )
    if stack:
        _unsupported_bytecode(program, instruction, "residual stack values")
    return output


def _handle_local_load(recorder, locals, stack, program, instruction, state, active):
    del recorder, state, active
    (name,) = _local_names(program, instruction, 1)
    _load_local(locals, stack, program, instruction, name)


def _handle_local_load_pair(
    recorder,
    locals,
    stack,
    program,
    instruction,
    state,
    active,
):
    del recorder, state, active
    first_name, second_name = _local_names(program, instruction, 2)
    _load_local(locals, stack, program, instruction, first_name)
    _load_local(locals, stack, program, instruction, second_name)


def _handle_store(recorder, locals, stack, program, instruction, state, active):
    del recorder, state, active
    (name,) = _local_names(program, instruction, 1)
    _store_local(locals, stack, program, instruction, name)


def _handle_store_load(
    recorder,
    locals,
    stack,
    program,
    instruction,
    state,
    active,
):
    del recorder, state, active
    store_name, load_name = _local_names(program, instruction, 2)
    _store_local(locals, stack, program, instruction, store_name)
    _load_local(locals, stack, program, instruction, load_name)


def _handle_store_store(
    recorder,
    locals,
    stack,
    program,
    instruction,
    state,
    active,
):
    del recorder, state, active
    first_name, second_name = _local_names(program, instruction, 2)
    _store_local(locals, stack, program, instruction, first_name)
    _store_local(locals, stack, program, instruction, second_name)


def _handle_load_const(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, program, state, active
    stack.append(_BytecodeConstant(instruction.argval))


def _handle_build_tuple(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, state, active
    argument_count = instruction.arg or 0
    values = [_pop(stack, program, instruction) for _ in range(argument_count)]
    values.reverse()
    output = tuple(values)
    _require_output_value(output, program, instruction, "tuple return value")
    stack.append(output)


def _handle_build_list(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, state, active
    argument_count = instruction.arg or 0
    values = [_pop(stack, program, instruction) for _ in range(argument_count)]
    values.reverse()
    output = list(values)
    _require_output_value(output, program, instruction, "list return value")
    stack.append(output)


def _handle_noop(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, stack, program, instruction, state, active
    return None


_OPCODE_HANDLERS = {
    "ignored": _handle_noop,
    "local_load": _handle_local_load,
    "local_load_pair": _handle_local_load_pair,
    "store": _handle_store,
    "store_load": _handle_store_load,
    "store_store": _handle_store_store,
    "load_global": _handle_load_global,
    "load_method": _handle_load_method,
    "call": _handle_call,
    "precall": _handle_noop,
    "push_null": _handle_noop,
    "load_const": _handle_load_const,
    "build_tuple": _handle_build_tuple,
    "build_list": _handle_build_list,
    "binary": _handle_binary,
    "unary_neg": _handle_unary_neg,
    "return": _handle_return,
}


def _lower_instruction(recorder, locals, stack, program, instruction, state, active):
    kind = _instruction_form(program, instruction)
    return _OPCODE_HANDLERS[kind](
        recorder,
        locals,
        stack,
        program,
        instruction,
        state,
        active,
    )


def lower_compile_graph(program, input_metadatas, *, name=None, compile_request=None):
    """Lower a narrow Python function into a native trace graph."""
    if compile_request is None:
        compile_request = prepare_compile_cache_request(program, input_metadatas)
    elif compile_request.input_metadatas != input_metadatas:
        raise TypeError(
            "torch.compile trace bytecode lowering compile request metadata "
            "does not match inputs"
        )
    code = compile_request.descriptor.code

    recorder = _trace.CompileTraceRecorder(
        name or getattr(program, "__name__", "compile_trace")
    )
    locals = {}
    for index, input_metadata in enumerate(input_metadatas):
        input_name = code.co_varnames[index]
        locals[input_name] = recorder.input(
            name=input_name,
            shape=input_metadata.shape,
            stride=input_metadata.stride,
            dtype=input_metadata.dtype,
            device=input_metadata.device,
            requires_grad=input_metadata.requires_grad,
        )
    stack = []
    state = _LoweringState(
        root_program=program,
        helper_dependencies=compile_request.helper_dependencies,
    )
    output = _lower_function_body(
        recorder,
        locals,
        stack,
        program,
        code,
        state,
        (program,),
        _lowerable_bytecode_instructions(program, code, input_metadatas),
    )
    return recorder.finish(output)


def lower_one_input_compile_graph(
    program,
    input_metadata,
    *,
    name=None,
    compile_request=None,
):
    return lower_compile_graph(
        program,
        (input_metadata,),
        name=name,
        compile_request=compile_request,
    )


__all__ = [
    "analyze_compile_program",
    "compile_cache_key",
    "prepare_compile_cache_request",
    "lower_compile_graph",
    "lower_one_input_compile_graph",
]
