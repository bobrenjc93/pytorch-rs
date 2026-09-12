"""Private CPython bytecode frontend for the narrow ``torch.compile`` path."""

from __future__ import annotations

import builtins as _builtins
import dis as _dis
import struct as _struct
import sys as _sys
import types as _types
from dataclasses import dataclass, field, replace

from . import _compile_trace as _trace
from ._compiler_state import native_function_owner as _NATIVE_FUNCTION_OWNER
from ._compiler_state import native_relu as _NATIVE_RELU
from ._compiler_state import native_squeeze as _NATIVE_SQUEEZE
from ._compiler_state import native_t as _NATIVE_T


_NATIVE_MODULE = _sys.modules[__package__]

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
class _BytecodeTuple:
    # Mixed constants/tensors are retained for shape call validation only.
    # They are not Tensor output pytrees or accepted helper arguments.
    elements: tuple


@dataclass(frozen=True, slots=True)
class _BytecodeKeywordNames:
    names: tuple[str, ...]


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
class _GlobalTensorCacheDependency:
    global_name: str
    tensor_identity: int
    metadata: _trace.CompileTraceTensorMetadata
    tensor: object = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True, slots=True)
class _BytecodeBuiltin:
    target: str
    arity: int


@dataclass(frozen=True, slots=True)
class _BytecodeModule:
    bindings: tuple


@dataclass(frozen=True, slots=True)
class _GlobalValueCacheDependency:
    global_name: str
    load_key: tuple
    guard: object
    value: object = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True, slots=True)
class _GlobalLoadDependency:
    name: str
    instruction: object
    module_attribute: str | None = None


@dataclass(frozen=True, slots=True)
class _CompileProgramDescriptor:
    code: object
    global_loads: tuple[_GlobalLoadDependency, ...]
    has_requires_grad_branch: bool


@dataclass(frozen=True, slots=True)
class _CompileCacheRequest:
    key: object
    descriptor: _CompileProgramDescriptor
    input_metadatas: tuple
    helper_dependencies: tuple[_HelperCacheDependency, ...]
    global_tensor_dependencies: tuple[_GlobalTensorCacheDependency, ...]
    global_value_dependencies: tuple[_GlobalValueCacheDependency, ...] = ()
    dynamic: bool = False


@dataclass(slots=True)
class _LoweringState:
    root_program: object
    helper_dependencies: tuple[_HelperCacheDependency, ...] = ()
    global_tensor_proxies: dict[str, _trace.CompileTraceTensorProxy] = field(
        default_factory=dict
    )
    global_values: dict[tuple, object] = field(default_factory=dict)
    helper_call_count: int = 0
    has_mixed_tuple: bool = False
    local_constant_error: _trace.CompileTraceUnsupportedError | None = None


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
    "view": _MethodTarget("view", "view", 1, "Tensor.view"),
    "reshape": _MethodTarget("view", "reshape", 1, "Tensor.reshape"),
    "transpose": _MethodTarget("view", "transpose", 2, "Tensor.transpose"),
    "squeeze": _MethodTarget("unary", "squeeze", 0, "Tensor.squeeze"),
    "t": _MethodTarget("unary", "t", 0, "Tensor.t"),
    "contiguous": _MethodTarget("unary", "contiguous", 0, "Tensor.contiguous"),
    "sum": _MethodTarget("reduction", "sum", 1, "Tensor.sum"),
    "neg": _MethodTarget("unary", "neg", 0, "Tensor.neg"),
    "negative": _MethodTarget("unary", "neg", 0, "Tensor.negative"),
    "__neg__": _MethodTarget("unary", "neg", 0, "Tensor.__neg__"),
    "abs": _MethodTarget("unary", "abs", 0, "Tensor.abs"),
    "absolute": _MethodTarget("unary", "abs", 0, "Tensor.absolute"),
    "__abs__": _MethodTarget("unary", "abs", 0, "Tensor.__abs__"),
    "relu": _MethodTarget("unary", "relu", 0, "Tensor.relu"),
    "square": _MethodTarget("unary", "square", 0, "Tensor.square"),
    "detach": _MethodTarget("unary", "detach", 0, "Tensor.detach"),
    "float": _MethodTarget("unary", "float", 0, "Tensor.float"),
    "mul": _MethodTarget("scalar", "mul_scalar", 1, "Tensor.mul"),
    "multiply": _MethodTarget("scalar", "mul_scalar", 1, "Tensor.multiply"),
    "__mul__": _MethodTarget("scalar", "mul_scalar", 1, "Tensor.__mul__"),
    "__rmul__": _MethodTarget("scalar", "mul_scalar", 1, "Tensor.__rmul__"),
    "matmul": _MethodTarget("binary", "matmul", 1, "Tensor.matmul"),
    "__matmul__": _MethodTarget("binary", "matmul", 1, "Tensor.__matmul__"),
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
_IGNORED_OPCODE_NAMES = frozenset(
    ("CACHE", "EXTENDED_ARG", "NOP", "NOT_TAKEN", "RESUME")
)
_EXCEPTION_HANDLING_OPCODE_NAMES = frozenset(
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
)
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
        _EXCEPTION_HANDLING_OPCODE_NAMES,
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
    _OpcodeForm("keyword_names", frozenset(("KW_NAMES",))),
    _OpcodeForm("call_kw", frozenset(("CALL_FUNCTION_KW", "CALL_KW", "CALL_METHOD_KW"))),
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
    _OpcodeForm("list_extend", frozenset(("LIST_EXTEND",))),
    _OpcodeForm("binary", frozenset(("BINARY_ADD", "BINARY_MATRIX_MULTIPLY", "BINARY_MULTIPLY", "BINARY_OP", "INPLACE_ADD"))),
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


def _is_exact_native_tensor(value):
    return _builtins.type(value) is _trace._native.Tensor


def _builtin_target(value):
    # Trusted startup bindings establish identity, not callable names or source
    # text. ReLU, squeeze and t have no member on the immutable arithmetic owner.
    if value is _NATIVE_RELU:
        return _BytecodeBuiltin("relu", 1)
    if value is _NATIVE_SQUEEZE:
        return _BytecodeBuiltin("squeeze", 1)
    if value is _NATIVE_T:
        return _BytecodeBuiltin("t", 1)
    owner = _NATIVE_FUNCTION_OWNER
    if value is owner.mul or value is owner.multiply:
        return _BytecodeBuiltin("mul_scalar", 2)
    if value is owner.matmul:
        return _BytecodeBuiltin("matmul", 2)
    if value is owner.add:
        return _BytecodeBuiltin("add", 2)
    if value is owner.neg or value is owner.negative:
        return _BytecodeBuiltin("neg", 1)
    return None


def _global_value_dependency(name, value, module_attribute=None):
    if _builtins.type(value) in (_builtins.bool, _builtins.int, _builtins.float):
        # Use the original type and binary64 bits, preserving signed zero and
        # stable NaN guards without executing equality or conversion callbacks.
        guard = (
            _builtins.type(value),
            _struct.pack("!d", value) if _builtins.type(value) is _builtins.float else value,
        )
        return _GlobalValueCacheDependency(name, (), guard, _BytecodeConstant(value))
    builtin = _builtin_target(value)
    if builtin is not None:
        return _GlobalValueCacheDependency(name, (), value, builtin)
    if _builtins.type(value) is _types.ModuleType and value is _NATIVE_MODULE:
        # Retain the legacy module guard. Newly supported fields are guarded
        # only at loads that actually use them, so unrelated mutations neither
        # reject old programs nor spend their recompile budget.
        attributes = ("mul", "multiply", "matmul")
        if module_attribute in ("add", "neg", "negative", "relu", "squeeze", "t"):
            attributes += (module_attribute,)
        bindings = tuple((attr, vars(value).get(attr)) for attr in attributes)
        if all(_builtin_target(fn) is not None for _, fn in bindings):
            lowered = tuple((attr, _builtin_target(fn)) for attr, fn in bindings)
            return _GlobalValueCacheDependency(name, (), (value, bindings), _BytecodeModule(lowered))
    return None


def _global_load_key(program, instruction):
    # One global module can load several fields, including from a helper.
    # Keep each snapshot rather than overwriting them by global name.
    return (id(program.__globals__), program.__code__, instruction.offset)


def _resolve_live_global_dependency(
    root_program, program, instruction, name, module_attribute=None,
):
    try:
        value = program.__globals__[name]
    except KeyError:
        _unsupported_bytecode(program, instruction, "global or import access")

    if _is_exact_native_tensor(value):
        metadata = _trace._metadata_from_native_tensor(value)
        return _GlobalTensorCacheDependency(
            global_name=name,
            tensor_identity=_builtins.id(value),
            metadata=metadata,
            tensor=value,
        )

    dependency = _global_value_dependency(name, value, module_attribute)
    if dependency is not None:
        return replace(dependency, load_key=_global_load_key(program, instruction))

    helper = value
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
    instructions = tuple(_dis.get_instructions(code))
    layout = _requires_grad_branch_layout(program, code, instructions)
    if layout is None:
        analyzable_instructions = instructions
    else:
        analyzable_instructions = (
            *instructions[: layout.predicate_start],
            *layout.true_body,
            *layout.false_body,
            *layout.tail,
        )
    global_loads = _global_load_dependencies_from_instructions(
        program,
        analyzable_instructions,
        resolve=layout is None,
    )
    return _CompileProgramDescriptor(code, global_loads, layout is not None)


def _global_load_dependencies_from_instructions(
    program, instructions, *, resolve=False, validate_opcodes=True,
):
    global_loads = []
    for index, instruction in enumerate(instructions):
        if validate_opcodes:
            kind = _instruction_form(program, instruction)
            if kind != "load_global":
                continue
        elif instruction.opname != "LOAD_GLOBAL":
            continue
        name = _global_name(program, instruction)
        following = instructions[index + 1] if index + 1 < len(instructions) else None
        attribute = (following.argval if following is not None
                     and following.opname in ("LOAD_ATTR", "LOAD_METHOD") else None)
        if resolve:
            _resolve_live_global_dependency(program, program, instruction, name, attribute)
        global_loads.append(_GlobalLoadDependency(name, instruction, attribute))
    return tuple(global_loads)


def _global_cache_dependencies_for_loads(
    root_program,
    program,
    global_loads,
    active_programs,
):
    helper_dependencies = []
    global_tensor_dependencies = []
    global_value_dependencies = []
    for global_load in global_loads:
        dependency = _resolve_live_global_dependency(
            root_program,
            program,
            global_load.instruction,
            global_load.name,
            global_load.module_attribute,
        )
        if _builtins.isinstance(dependency, _GlobalTensorCacheDependency):
            global_tensor_dependencies.append(dependency)
            continue

        if _builtins.isinstance(dependency, _GlobalValueCacheDependency):
            global_value_dependencies.append(dependency)
            continue

        helper_dependencies.append(dependency)
        helper = dependency.function
        if helper in active_programs:
            _unsupported_bytecode(
                program,
                global_load.instruction,
                "recursive helper function calls",
            )
        if program is not root_program:
            _unsupported_bytecode(
                program,
                global_load.instruction,
                "helper function calls",
            )
        # Preserve helper error precedence: resolve its globals before the
        # lowering pass validates other opcodes (which may contain effects).
        helper_loads = _global_load_dependencies_from_instructions(
            helper, tuple(_analyzable_bytecode_instructions(helper, dependency.code)),
            validate_opcodes=False,
        )
        nested_helpers, nested_tensors, nested_values = _global_cache_dependencies_for_loads(
            root_program,
            helper,
            helper_loads,
            (*active_programs, helper),
        )
        helper_dependencies.extend(nested_helpers)
        global_tensor_dependencies.extend(nested_tensors)
        global_value_dependencies.extend(nested_values)
    return (
        tuple(helper_dependencies), tuple(global_tensor_dependencies),
        tuple(global_value_dependencies),
    )


def _global_cache_dependencies(program, descriptor, input_metadatas):
    if not descriptor.global_loads:
        return (), (), ()
    # A descriptor is bound to one immutable code object. Straight-line loads
    # cannot change until that code is replaced; their live values still must
    # be resolved below on every call. Branches retain input-dependent loads.
    global_loads = descriptor.global_loads
    if descriptor.has_requires_grad_branch:
        global_loads = _global_load_dependencies_from_instructions(
            program,
            _lowerable_bytecode_instructions(
                program,
                descriptor.code,
                input_metadatas,
            ),
        )
    return _global_cache_dependencies_for_loads(
        program,
        program,
        global_loads,
        (program,),
    )


def _resolve_global_dependency(state, program, instruction, name):
    if (
        program is state.root_program
        or getattr(program, "__globals__", None)
        is getattr(state.root_program, "__globals__", None)
    ):
        for dependency in state.helper_dependencies:
            if dependency.global_name == name:
                return dependency
        load_key = _global_load_key(program, instruction)
        if load_key in state.global_values:
            return state.global_values[load_key]
        if name in state.global_tensor_proxies:
            return state.global_tensor_proxies[name]
        _unsupported_bytecode(program, instruction, "global dependency snapshot")
    return _resolve_live_global_dependency(state.root_program, program, instruction, name)


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


def _dynamic_metadata_key(input_metadatas):
    return tuple(
        (
            "dynamic_rank",
            len(input_metadata.shape),
            input_metadata.stride,
            input_metadata.dtype,
            input_metadata.device,
            input_metadata.requires_grad,
            input_metadata.storage_offset,
        )
        for input_metadata in input_metadatas
    )


def prepare_compile_cache_request(
    program,
    input_metadatas,
    descriptor=None,
    *,
    dynamic=False,
):
    """Return cache metadata and exact dependency snapshots used for lowering."""
    if _builtins.type(dynamic) is not _builtins.bool:
        raise TypeError("torch.compile trace cache dynamic flag must be bool")
    code = getattr(program, "__code__", None)
    if descriptor is None or descriptor.code is not code:
        descriptor = analyze_compile_program(program)
    else:
        _validate_function_code(program, descriptor.code)
    _validate_input_metadatas(descriptor.code, input_metadatas)
    dependencies = _global_cache_dependencies(program, descriptor, input_metadatas)
    helper_dependencies, global_tensor_dependencies, global_value_dependencies = dependencies
    if global_value_dependencies and all(m.device.type == "cpu" for m in input_metadatas):
        # Keep the established CPU global-access boundary. Scalar capture is
        # deliberately admitted only for the bounded CUDA grammar.
        _unsupported_bytecode(
            program, descriptor.global_loads[0].instruction, "global or import access"
        )
    all_metadatas = (*input_metadatas, *(d.metadata for d in global_tensor_dependencies))
    if any(m.device.type == "cuda" for m in all_metadatas):
        for metadata in all_metadatas:
            _trace._validate_cuda_metadata(metadata, require_contiguous=False)
            if metadata.device != all_metadatas[0].device:
                raise _trace.CompileTraceUnsupportedError(
                    "torch.compile trace CUDA inputs and captures require matching devices"
                )
    metadata_key = (
        _dynamic_metadata_key(input_metadatas) if dynamic else input_metadatas
    )
    return _CompileCacheRequest(
        key=(
            descriptor.code,
            metadata_key,
            helper_dependencies,
            global_tensor_dependencies,
            global_value_dependencies,
        ),
        descriptor=descriptor,
        input_metadatas=input_metadatas,
        helper_dependencies=helper_dependencies,
        global_tensor_dependencies=global_tensor_dependencies,
        global_value_dependencies=global_value_dependencies,
        dynamic=dynamic,
    )


def compile_cache_key(program, input_metadatas, *, dynamic=False):
    """Return a cache key with validated helper and global tensor dependencies."""
    return prepare_compile_cache_request(
        program,
        input_metadatas,
        dynamic=dynamic,
    ).key


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
    for instruction in instructions:
        if instruction.opname in _EXCEPTION_HANDLING_OPCODE_NAMES:
            _unsupported_bytecode(program, instruction, "exception handling")

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
    if isinstance(value, _BytecodeTuple):
        return _require_output_value(value.elements, program, instruction, "tuple return value")
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


def _store_local(locals, stack, program, instruction, name, state):
    value = _pop(stack, program, instruction)
    if isinstance(value, _BytecodeTuple):
        locals[name] = value
        return
    if _builtins.isinstance(value, _BytecodeConstant):
        if type(value.value) not in (int, tuple):
            try:
                _trace._normalize_mul_scalar(value.value)
            except _trace.CompileTraceUnsupportedError as error:
                # Preserve the value for public shape argument validation.
                # Still reject the graph if the local is unused or overwritten.
                if state.local_constant_error is None:
                    state.local_constant_error = error
        locals[name] = value
        return
    locals[name] = _require_output_value(
        value,
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
    dependency = _resolve_global_dependency(state, program, instruction, name)
    if _builtins.isinstance(dependency, (
        _trace.CompileTraceTensorProxy, _BytecodeConstant, _BytecodeBuiltin, _BytecodeModule
    )):
        stack.append(dependency)
        return
    stack.append(
        _BytecodeFunction(
            dependency.function,
            name,
            dependency.code,
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
    if stack and _builtins.isinstance(stack[-1], _BytecodeModule):
        module = stack.pop()
        for name, builtin in module.bindings:
            if instruction.argval == name:
                stack.append(builtin)
                return
        _unsupported_bytecode(program, instruction, "module attribute access")
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
        if instruction.opname == "KW_NAMES":
            # 3.11/3.12 hide KW_NAMES.argval; 3.10 and 3.13/3.14 instead
            # push a LOAD_CONST tuple consumed by their keyword call opcode.
            instruction = instruction._replace(argval=code.co_consts[instruction.arg])
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


def _reshape_argument(value):
    if isinstance(value, _BytecodeConstant):
        return value.value
    if isinstance(value, _BytecodeTuple):
        return tuple(_reshape_argument(element) for element in value.elements)
    if type(value) is tuple:
        return tuple(_reshape_argument(element) for element in value)
    if type(value) is list:
        return [_reshape_argument(element) for element in value]
    return _trace._RESHAPE_UNKNOWN_DIMENSION


def _record_method_call(recorder, method, args, program, instruction, names=()):
    if method.name in ("reshape", "view"):
        positional = len(args) - len(names)
        values = tuple(_reshape_argument(value) for value in args)
        positional_args = values[:positional]
        kwargs = dict(zip(names, values[positional:]))
        binder = _trace._bind_view_shape if method.name == "view" else _trace._bind_reshape_shape
        shape = binder(positional_args, kwargs, method.receiver.metadata)
        return recorder._record_shape(method.receiver, shape, method.name)
    if method.name == "transpose":
        positional = len(args) - len(names)
        if positional > 2:
            raise TypeError("transpose() requires dim0 and dim1")
        options = dict(zip(("dim0", "dim1"), args[:positional]))
        for name, value in zip(names, args[positional:]):
            if name not in ("dim0", "dim1") or name in options:
                raise TypeError(f"transpose() invalid or duplicate argument {name!r}")
            options[name] = value
        if len(options) != 2:
            raise TypeError("transpose() missing dim0 or dim1")
        if any(not isinstance(value, _BytecodeConstant) for value in options.values()):
            _unsupported_bytecode(program, instruction, "Tensor.transpose requires constant axes")
        return recorder.record_transpose(method.receiver, options["dim0"].value,
                                         options["dim1"].value)
    if method.name == "sum":
        positional = len(args) - len(names)
        if positional > 2:
            _unsupported_bytecode(program, instruction, "Tensor.sum argument count")
        options = dict(zip(("dim", "keepdim"), args[:positional]))
        for name, value in zip(names, args[positional:]):
            if name not in ("dim", "keepdim") or name in options:
                _unsupported_bytecode(program, instruction, "Tensor.sum keyword arguments")
            options[name] = value
        if "dim" not in options:
            _unsupported_bytecode(program, instruction, "Tensor.sum requires constant dim=1")
        options.setdefault("keepdim", _BytecodeConstant(False))
        if any(not isinstance(v, _BytecodeConstant) for v in options.values()):
            _unsupported_bytecode(program, instruction, "Tensor.sum requires constant options")
        return recorder.record_reduction(method.receiver, options["dim"].value,
                                         options["keepdim"].value)
    if names:
        _unsupported_bytecode(program, instruction, "keyword arguments")
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

    if method_target.kind == "scalar":
        return _record_scalar_multiply(recorder, method.receiver, args[0], program, instruction)

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
    names = ()
    if stack and isinstance(stack[-1], _BytecodeKeywordNames):
        names = stack.pop().names
    if len(names) > argument_count:
        _unsupported_bytecode(program, instruction, "keyword argument count")
    args = [_pop(stack, program, instruction) for _ in range(argument_count)]
    args.reverse()
    callable_value = _pop(stack, program, instruction)
    if names and not isinstance(callable_value, _BytecodeMethod):
        _unsupported_bytecode(program, instruction, "keyword arguments")
    if _builtins.isinstance(callable_value, _BytecodeBuiltin):
        target = callable_value.target
        if len(args) != callable_value.arity:
            _unsupported_bytecode(program, instruction, f"native {target} argument count")
        if target in ("add", "matmul"):
            left, right = (_require_tensor(arg, program, instruction, f"{target} operand") for arg in args)
            stack.append(recorder.record_binary(target, left, right, f"torch.{target}"))
        elif target in ("neg", "relu", "squeeze", "t"):
            operand = _require_tensor(args[0], program, instruction, f"{target} operand")
            stack.append(recorder.record_unary(target, operand))
        elif target == "mul_scalar":
            stack.append(_record_scalar_multiply(recorder, *args, program, instruction))
        else:
            _unsupported_bytecode(program, instruction, "native function target")
        return
    if _builtins.isinstance(callable_value, _BytecodeMethod):
        stack.append(
            _record_method_call(
                recorder,
                callable_value,
                tuple(args),
                program,
                instruction,
                names,
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


def _keyword_names(value, program, instruction):
    if (type(value) is not tuple or not value
            or any(type(name) is not str for name in value)
            or len(set(value)) != len(value)):
        _unsupported_bytecode(program, instruction, "keyword names")
    return _BytecodeKeywordNames(value)


def _handle_keyword_names(recorder, locals, stack, program, instruction, state, active):
    stack.append(_keyword_names(instruction.argval, program, instruction))


def _handle_call_kw(recorder, locals, stack, program, instruction, state, active):
    value = _pop(stack, program, instruction)
    if not isinstance(value, _BytecodeConstant):
        _unsupported_bytecode(program, instruction, "non-constant keyword names")
    stack.append(_keyword_names(value.value, program, instruction))
    return _handle_call(recorder, locals, stack, program, instruction, state, active)


def _record_binary_add(recorder, stack, program, instruction, target="add"):
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
    stack.append(recorder.record_binary(target, left, right, f"Tensor.{target}"))


def _record_scalar_multiply(recorder, left, right, program, instruction):
    if _builtins.isinstance(left, _BytecodeConstant):
        left, right = right, left
    tensor = _require_tensor(left, program, instruction, "scalar multiply operand")
    if not _builtins.isinstance(right, _BytecodeConstant):
        _unsupported_bytecode(program, instruction, "non-constant scalar or Tensor multiplication")
    return recorder.record_scalar("mul_scalar", tensor, right.value)


def _binary_operator_symbol(instruction):
    if instruction.opname == "BINARY_MATRIX_MULTIPLY":
        return "@"
    if instruction.opname == "BINARY_MULTIPLY":
        return "*"
    if instruction.opname == "BINARY_ADD":
        return "+"
    if instruction.opname == "INPLACE_ADD":
        return "+="
    return instruction.argrepr


def _defer_binary(stack, program, instruction, state, symbol):
    # Consume the operands without evaluating them, including user operators.
    _pop(stack, program, instruction)
    _pop(stack, program, instruction)
    state.local_constant_error = state.local_constant_error or _trace.CompileTraceUnsupportedError(
        f"torch.compile binary operator {symbol!r} is only retained for argument validation"
    )
    stack.append(_BytecodeConstant(_trace._RESHAPE_UNKNOWN_DIMENSION))


def _handle_binary(recorder, locals, stack, program, instruction, state, active):
    del locals, active
    symbol = _binary_operator_symbol(instruction)
    # Bounded exact integer arithmetic is retained only to diagnose public
    # argument errors later in the call. Even an unused/overwritten result
    # rejects the graph; computed dimensions never become captured constants.
    if (symbol in ("+", "-", "*") and len(stack) >= 2
            and all(isinstance(v, _BytecodeConstant) and type(v.value) is int
                    and -(2**63) <= v.value < 2**63 for v in stack[-2:])):
        right, left = stack.pop().value, stack.pop().value
        value = left + right if symbol == "+" else left - right if symbol == "-" else left * right
        state.local_constant_error = state.local_constant_error or _trace.CompileTraceUnsupportedError(
            "torch.compile computed integer values are only retained for argument validation"
        )
        stack.append(_BytecodeConstant(value))
        return
    if (symbol in ("+", "*", "@") and len(stack) >= 2
            and (all(isinstance(v, _BytecodeConstant) for v in stack[-2:])
                 or any(isinstance(v, _BytecodeConstant)
                        and v.value is _trace._RESHAPE_UNKNOWN_DIMENSION for v in stack[-2:]))):
        # A surrounding expression must not turn an already opaque dimension
        # into an immediate rejection (e.g. (k // 1) + 1 or (k // 1) * x).
        _defer_binary(stack, program, instruction, state, symbol)
        return
    if symbol == "*":
        right = _pop(stack, program, instruction)
        left = _pop(stack, program, instruction)
        stack.append(_record_scalar_multiply(recorder, left, right, program, instruction))
        return
    if symbol in ("+", "@"):
        _record_binary_add(recorder, stack, program, instruction, "matmul" if symbol == "@" else "add")
        return
    if symbol == "+=":
        _unsupported_bytecode(program, instruction, "mutation")
    if symbol in ("-", "/", "//", "%", "**", "<<", ">>", "&", "|", "^"):
        # Do not evaluate unsupported expressions, even on apparently constant
        # operands. Preserve their stack position so a later call can diagnose
        # independently known binding/range/shape errors. The deferred error
        # still rejects unused, overwritten and returned results.
        _defer_binary(stack, program, instruction, state, symbol)
        return
    _unsupported_bytecode(program, instruction, f"binary operator {symbol!r}")


def _handle_unary_neg(recorder, locals, stack, program, instruction, state, active):
    del locals, active
    operand = _pop(stack, program, instruction)
    if not isinstance(operand, _trace.CompileTraceTensorProxy):
        # As with bounded binary arithmetic, retain exact integer values only
        # for public error validation. Never invoke a user __neg__ conversion,
        # and keep the rejection even if the computed result is discarded.
        value = _trace._RESHAPE_UNKNOWN_DIMENSION
        if (isinstance(operand, _BytecodeConstant) and type(operand.value) is int
                and -(2**63) <= operand.value < 2**63):
            value = -operand.value
        state.local_constant_error = state.local_constant_error or _trace.CompileTraceUnsupportedError(
            "torch.compile computed unary values are only retained for argument validation"
        )
        stack.append(_BytecodeConstant(value))
        return
    input = _require_tensor(
        operand,
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
    del recorder, active
    (name,) = _local_names(program, instruction, 1)
    _store_local(locals, stack, program, instruction, name, state)


def _handle_store_load(
    recorder,
    locals,
    stack,
    program,
    instruction,
    state,
    active,
):
    del recorder, active
    store_name, load_name = _local_names(program, instruction, 2)
    _store_local(locals, stack, program, instruction, store_name, state)
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
    del recorder, active
    first_name, second_name = _local_names(program, instruction, 2)
    _store_local(locals, stack, program, instruction, first_name, state)
    _store_local(locals, stack, program, instruction, second_name, state)


def _handle_load_const(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, program, state, active
    stack.append(_BytecodeConstant(instruction.argval))


def _handle_build_tuple(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, active
    argument_count = instruction.arg or 0
    values = [_pop(stack, program, instruction) for _ in range(argument_count)]
    values.reverse()
    if values and all(isinstance(value, _BytecodeConstant) for value in values):
        stack.append(_BytecodeConstant(tuple(value.value for value in values)))
        return
    if any(isinstance(value, (_BytecodeConstant, _BytecodeTuple)) for value in values):
        state.has_mixed_tuple = True
        stack.append(_BytecodeTuple(tuple(values)))
        return
    output = tuple(values)
    _require_output_value(output, program, instruction, "tuple return value")
    stack.append(output)


def _handle_build_list(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, active
    argument_count = instruction.arg or 0
    values = [_pop(stack, program, instruction) for _ in range(argument_count)]
    values.reverse()
    output = list(values)
    try:
        _require_output_value(output, program, instruction, "list return value")
    except _trace.CompileTraceUnsupportedError as error:
        # Preserve a literal/local container for binding/type validation. It
        # remains forbidden as a shape, output, helper input or unused local.
        state.local_constant_error = state.local_constant_error or error
        stack.append(_BytecodeConstant([_reshape_argument(v) for v in values]))
        return
    stack.append(output)


def _handle_list_extend(recorder, locals, stack, program, instruction, state, active):
    del recorder, locals, active
    values = _pop(stack, program, instruction)
    depth = instruction.arg or 0
    if not 0 < depth <= len(stack):
        _unsupported_bytecode(program, instruction, "list extension stack")
    target = _reshape_argument(stack[-depth])
    if type(target) is not list:
        _unsupported_bytecode(program, instruction, "list extension target")
    # CPython emits BUILD_LIST 0 / LOAD_CONST / LIST_EXTEND for longer
    # literals. Only inspect exact internal tuples/lists; never iterate a user
    # object. This is validation-only, including empty and Tensor containers.
    values = _reshape_argument(values)
    items = values if type(values) in (tuple, list) else (_trace._RESHAPE_UNKNOWN_DIMENSION,)
    stack[-depth] = _BytecodeConstant(target + list(items))
    state.local_constant_error = state.local_constant_error or _trace.CompileTraceUnsupportedError(
        "torch.compile list extension is only retained for argument validation"
    )


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
    "call_kw": _handle_call_kw,
    "keyword_names": _handle_keyword_names,
    "precall": _handle_noop,
    "push_null": _handle_noop,
    "load_const": _handle_load_const,
    "build_tuple": _handle_build_tuple,
    "build_list": _handle_build_list,
    "list_extend": _handle_list_extend,
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
        name or getattr(program, "__name__", "compile_trace"),
        dynamic=compile_request.dynamic,
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
            storage_offset=input_metadata.storage_offset,
        )
    global_tensor_proxies = {}
    for dependency in compile_request.global_tensor_dependencies:
        if dependency.global_name in global_tensor_proxies:
            continue
        global_tensor_proxies[dependency.global_name] = recorder.capture(
            name=f"global:{dependency.global_name}",
            value=dependency.tensor,
            metadata=dependency.metadata,
        )
    stack = []
    state = _LoweringState(
        root_program=program,
        helper_dependencies=compile_request.helper_dependencies,
        global_tensor_proxies=global_tensor_proxies,
        global_values={d.load_key: d.value for d in compile_request.global_value_dependencies},
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
    if state.local_constant_error is not None:
        raise state.local_constant_error
    if state.has_mixed_tuple:
        raise _trace.CompileTraceUnsupportedError(
            "torch.compile mixed constant/Tensor tuples are only retained for shape argument validation"
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
