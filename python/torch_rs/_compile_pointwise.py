"""Static bytecode to typed float32 SSA for the native default CUDA JIT.

The frontend never invokes the function, Tensor methods, or a Python operator
on user objects. Warm calls resolve binding/metadata guards and enter one native
kernel. It deliberately has no graph breaks or eager fallback.
"""
from dataclasses import dataclass
import dis
import math
import struct
import sys
import types

from . import torch_rs as _native
from . import _compiler_state as _state

_ROOT = sys.modules[__package__]
_TENSOR_TYPE = _ROOT.Tensor
_UNARY = {"neg": "neg", "negative": "neg", "__neg__": "neg",
          "relu": "relu", "sin": "sin", "cos": "cos"}
_BINARY = {"add": "add", "__add__": "add", "__radd__": "add",
           "sub": "sub", "subtract": "sub", "__sub__": "sub", "__rsub__": "sub",
           "mul": "mul", "multiply": "mul", "__mul__": "mul", "__rmul__": "mul"}
_IGNORED = {"RESUME", "CACHE", "EXTENDED_ARG", "NOP", "NOT_TAKEN", "PUSH_NULL", "PRECALL", "COPY_FREE_VARS"}
_ALLOWED = _IGNORED | {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "LOAD_FAST_LOAD_FAST",
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW", "STORE_FAST", "STORE_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST",
    "LOAD_CONST", "LOAD_SMALL_INT", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_ATTR", "LOAD_METHOD", "BINARY_OP",
    "BINARY_ADD", "BINARY_SUBTRACT", "BINARY_MULTIPLY", "UNARY_NEGATIVE", "CALL", "CALL_FUNCTION",
    "CALL_METHOD", "RETURN_VALUE", "COPY", "DUP_TOP", "SWAP"}
_METHODS = tuple(_UNARY) + tuple(_BINARY) + ("__getattribute__",)
_MISSING = object()
# Imported during package initialization, before public bindings can be patched.
_FUNCTIONS = tuple((name, _ROOT.__dict__.get(name)) for name in
                   ("neg", "negative", "relu", "sin", "cos", "add", "sub", "subtract", "mul", "multiply"))
_METHOD_GUARDS = tuple((cls, name, cls.__dict__.get(name, _MISSING))
                      for cls in (_ROOT.Tensor, _ROOT.Tensor.__base__) for name in _METHODS)


def unsupported(reason):
    raise NotImplementedError("torch.compile native CUDA pointwise: " + reason)


def is_scalar(value):
    kind = type(value)
    # Equality against a type can execute a user-defined metaclass callback.
    return kind is bool or kind is int or kind is float


def scalar_bits(value):
    if not is_scalar(value):
        unsupported("constants must be exact bool/int/float values")
    if type(value) is int and not -(1 << 63) <= value < (1 << 64):
        unsupported("integer scalar is outside native scalar range")
    # Inductor propagates Python floating values before materializing float32.
    # Preserve the original binary64 value, including signed zero and values
    # outside the float32 range, across the private IR bridge.
    return struct.unpack("=Q", struct.pack("=d", value))[0]


@dataclass(frozen=True)
class Value:
    index: int
    tensor: bool = True
    dtype: str = "float32"


@dataclass(frozen=True)
class RuntimeScalar:
    index: int
    negative: bool = False


@dataclass(frozen=True)
class Call:
    op: str
    receiver: Value | None = None
    reverse: bool = False


@dataclass(frozen=True)
class Graph:
    inputs: int
    nodes: tuple
    output: int


@dataclass(frozen=True)
class Program:
    code: object
    instructions: tuple
    dependencies: tuple


def validate_signature_containers(model):
    # Function attributes permit container subclasses. Check their exact types
    # before emptiness so admission and warm guards never invoke user hooks.
    defaults, kwdefaults = model.__defaults__, model.__kwdefaults__
    if defaults is not None and (type(defaults) is not tuple or len(defaults)):
        unsupported("function defaults must be absent or an empty exact tuple")
    if kwdefaults is not None and (type(kwdefaults) is not dict or len(kwdefaults)):
        unsupported("keyword defaults must be absent or an empty exact dict")
    if model.__closure__ is not None and type(model.__closure__) is not tuple:
        unsupported("closure must be an exact tuple")


def analyze(model, arity):
    if type(model) is not types.FunctionType or arity not in (1, 2):
        unsupported("expected an exact Python function and one or two inputs")
    validate_signature_containers(model)
    code = model.__code__
    if (code.co_argcount != arity or code.co_kwonlyargcount or code.co_flags & (0x04 | 0x08 | 0x20 | 0x80 | 0x200)
            or code.co_cellvars
            or getattr(code, "co_exceptiontable", b"")):
        unsupported("requires a straight-line function with one or two positional tensor inputs")
    # dis formats co_consts with repr, before yielding even a LOAD_CONST.
    # Admit the entire pool first; None and exact strings also hold compiler
    # metadata (implicit returns and docstrings), but cannot be scalar operands.
    for constant in code.co_consts:
        if constant is not None and type(constant) is not str:
            scalar_bits(constant)
    instructions = tuple(dis.get_instructions(code))
    if len(instructions) > 16384:
        unsupported("function exceeds pointwise instruction limit")
    for instruction in instructions:
        if instruction.opname not in _ALLOWED:
            unsupported(f"unsupported bytecode {instruction.opname}; control flow, mutation and non-pointwise graphs are unsupported")
    dependencies = tuple(dict.fromkeys((i.opname, i.argval) for i in instructions
                                      if i.opname in ("LOAD_GLOBAL", "LOAD_DEREF")))
    return Program(code, instructions, dependencies)


def binding(value):
    if is_scalar(value):
        # Static float guards equate signed zeros, retaining the first graph's
        # sign. Keep the actual value for new graphs and runtime parameters;
        # literal scalar bits and distinct nonzero float values remain intact.
        key = struct.pack("=d", 0.0 if value == 0.0 else value) if type(value) is float else value
        return (type(value), key), value
    if value is _ROOT:
        # Attribute identities are checked when resolving this guard on every call.
        for name, expected in _FUNCTIONS:
            if _ROOT.__dict__.get(name) is not expected:
                unsupported("patched native function binding: " + name)
        return ("module", id(value)), value
    for name, expected in _FUNCTIONS:
        if value is expected and expected is not None:
            return ("function", name, id(value)), Call((_UNARY | _BINARY)[name])
    unsupported("only native operator bindings and scalar constants may be captured")


def resolve(model, program):
    # FunctionType permits a dict subclass as globals. Never invoke its lookup
    # hooks, including on a warm call or before rejecting a later graph node.
    globals_ = model.__globals__
    if type(globals_) is not dict:
        unsupported("function globals must be an exact dict")
    # Even exact dict lookup can call a colliding key's equality hook. Iteration
    # does not hash or compare keys; validate all keys before any lookup.
    if any(type(key) is not str for key in globals_):
        unsupported("function globals keys must be exact strings")
    values, keys = {}, []
    closure = dict(zip(program.code.co_freevars, model.__closure__ or ()))
    for kind, name in program.dependencies:
        if kind == "LOAD_GLOBAL":
            if name not in globals_:
                unsupported("unbound global: " + name)
            value = globals_[name]
        else:
            try:
                value = closure[name].cell_contents
            except (KeyError, ValueError):
                unsupported("empty closure binding: " + name)
        key, value = binding(value)
        keys.append(key)
        values[kind, name] = value
    return tuple(keys), values


def runtime_bindings(program, bindings, values, graphs, *, promote=True):
    """Promote changed captured floats using successful, reset-owned history."""
    previous = [key[1] for key in graphs if key[0] is program.code]
    keys, resolved, scalars = list(bindings), dict(values), []
    for position, dependency in enumerate(program.dependencies):
        value = values[dependency]
        if type(value) is not float:
            continue
        observations = [keys_[position] for keys_ in previous]
        dynamic = any(key[0] == "runtime_float" for key in observations)
        if promote and not dynamic and math.isfinite(value):
            # Exact builtins only: numeric comparison matches the reference's
            # promotion history, including equal signed zeros and int/float
            # transitions. Booleans do not participate in scalar dynamism.
            dynamic = any(
                (kind is float and struct.unpack("=d", prior)[0] != value)
                or (kind is int and prior != value)
                for kind, prior, *unused in observations
            )
        if dynamic:
            if len(scalars) >= 64:
                unsupported("at most 64 runtime scalar bindings are supported")
            resolved[dependency] = RuntimeScalar(len(scalars))
            scalars.append(value)
            keys[position] = ("runtime_float",)
    return tuple(keys), resolved, tuple(scalars)


def lower(program, values, arity, input_ids=None):
    if input_ids is None:
        input_ids = tuple(range(arity))
    nodes = [("input", i, 0, 0) for i in input_ids]
    runtime = sorted(value.index for value in values.values() if type(value) is RuntimeScalar)
    nodes.extend(("scalar", index, 0, 0) for index in runtime)
    locals_ = dict(zip(program.code.co_varnames, (Value(i) for i in range(arity))))
    stack = []

    def value(obj):
        if type(obj) is RuntimeScalar:
            if not obj.negative:
                return Value(arity + obj.index, False)
            nodes.append(("scalar", obj.index, 1, 0))
            return Value(len(nodes) - 1, False)
        if isinstance(obj, Value):
            return obj
        bits = scalar_bits(obj)
        boolean = type(obj) is bool
        kind = "boolean" if boolean else "integer" if type(obj) is int else "constant"
        nodes.append((kind, 0, 0, int(obj) if boolean else bits))
        return Value(len(nodes) - 1, False, "bool" if boolean else "float32")

    def emit(op, operands):
        args = [value(arg) for arg in operands]
        if len(args) == 1 and not args[0].tensor or not any(arg.tensor for arg in args):
            unsupported("operators require tensor expressions")
        nodes.append((op, args[0].index, args[1].index if len(args) == 2 else 0, 0))
        if len(nodes) > 4096:
            unsupported("graph exceeds 4096-node limit")
        return Value(len(nodes) - 1)

    def load(name):
        if name not in locals_:
            unsupported("unbound local: " + name)
        stack.append(locals_[name])

    for instruction in program.instructions:
        op, arg = instruction.opname, instruction.argval
        if op in _IGNORED:
            continue
        if op in ("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"):
            load(arg)
        elif op in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"):
            for name in arg:
                load(name)
        elif op == "STORE_FAST":
            locals_[arg] = stack.pop()
        elif op == "STORE_FAST_LOAD_FAST":
            locals_[arg[0]] = stack.pop()
            load(arg[1])
        elif op == "STORE_FAST_STORE_FAST":
            for name in arg:
                locals_[name] = stack.pop()
        elif op in ("LOAD_CONST", "LOAD_SMALL_INT"):
            scalar_bits(arg)  # Validate before retaining; never invoke user conversion.
            stack.append(arg)
        elif op in ("LOAD_GLOBAL", "LOAD_DEREF"):
            stack.append(values[op, arg])
        elif op in ("LOAD_ATTR", "LOAD_METHOD"):
            owner = stack.pop()
            if owner is _ROOT:
                if arg not in dict(_FUNCTIONS):
                    unsupported("unsupported native function: " + arg)
                stack.append(binding(_ROOT.__dict__.get(arg))[1])
            elif isinstance(owner, Value) and owner.tensor and arg in (_UNARY | _BINARY):
                stack.append(Call((_UNARY | _BINARY)[arg], owner, arg == "__rsub__"))
            else:
                unsupported("unsupported attribute: " + str(arg))
        elif op == "UNARY_NEGATIVE":
            operand = stack.pop()
            if type(operand) is RuntimeScalar:
                stack.append(RuntimeScalar(operand.index, not operand.negative))
            elif isinstance(operand, Value):
                stack.append(emit("neg", [operand]))
            else:
                scalar_bits(operand)
                stack.append(-operand)
        elif op in ("BINARY_OP", "BINARY_ADD", "BINARY_SUBTRACT", "BINARY_MULTIPLY"):
            symbol = instruction.argrepr if op == "BINARY_OP" else {"BINARY_ADD": "+", "BINARY_SUBTRACT": "-", "BINARY_MULTIPLY": "*"}[op]
            if symbol not in ("+", "-", "*"):
                unsupported("unsupported binary operator: " + symbol)
            right, left = stack.pop(), stack.pop()
            stack.append(emit({"+": "add", "-": "sub", "*": "mul"}[symbol], [left, right]))
        elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD"):
            operands = [stack.pop() for _ in range(instruction.arg)][::-1]
            target = stack.pop()
            if not isinstance(target, Call):
                unsupported("only native pointwise operators may be called")
            if target.receiver is not None:
                operands.insert(0, target.receiver)
            if len(operands) != (1 if target.op in set(_UNARY.values()) else 2):
                unsupported("operator argument count mismatch")
            if target.reverse:
                operands.reverse()
            stack.append(emit(target.op, operands))
        elif op == "COPY":
            stack.append(stack[-instruction.arg])
        elif op == "DUP_TOP":
            stack.append(stack[-1])
        elif op == "SWAP":
            stack[-1], stack[-instruction.arg] = stack[-instruction.arg], stack[-1]
        elif op == "RETURN_VALUE":
            result = stack.pop()
            if stack or not isinstance(result, Value) or not result.tensor or result.index < arity:
                unsupported("return one computed pointwise tensor")
            return Graph(arity, tuple(nodes), result.index)
    unsupported("missing tensor return")


def implementation(model, recompile_limit):
    # Shared reset registry/lock discipline, but no eager graph evaluator.
    cache = _state.new_native_eager_compile_cache()
    program = None

    def compiled(*args, **kwargs):
        nonlocal program
        if kwargs or len(args) not in (1, 2):
            unsupported("expected one or two positional tensor arguments")
        if type(_ROOT) is not types.ModuleType:
            unsupported("patched native package type")
        if any(type(arg) is not _TENSOR_TYPE for arg in args):
            raise NotImplementedError(_ROOT._COMPILE_UNSUPPORTED_MESSAGE)
        if _ROOT.overrides._get_current_function_mode() is not None:
            unsupported("active __torch_function__ mode")
        for cls, name, expected in _METHOD_GUARDS:
            if cls.__dict__.get(name, _MISSING) is not expected:
                unsupported("patched Tensor operation binding: " + name)
        # The native bridge checks all metadata and storage bounds again on launch.
        metadata = tuple(_native._compile_trace_tensor_metadata(arg) for arg in args)
        if any(m[4] == "cpu" for m in metadata):
            raise NotImplementedError(_ROOT._COMPILE_UNSUPPORTED_MESSAGE)
        _native._pointwise_validate_inputs(args)
        # Contiguous input offsets are launch-time addresses, not compiler
        # specialization guards. Inductor reuses a static scalar specialization
        # across them. Bounds are still checked above and by native execution.
        guard_metadata = tuple(m[:5] for m in metadata)
        with cache.lock:
            if program is None or program.code is not model.__code__:
                program = analyze(model, len(args))
            validate_signature_containers(model)
            if program.code.co_argcount != len(args):
                unsupported("function signature changed")
            static_bindings, static_values = resolve(model, program)
            # Persistent runtime promotion takes precedence over old static
            # entries. Discover new promotion only after the full guard misses:
            # a nonfinite specialization must not invalidate a prior static hit.
            bindings, values, scalars = runtime_bindings(
                program, static_bindings, static_values, cache.graphs, promote=False)
            # Object identity, not equal values or shared storage, determines
            # whether two parameters denote the same expression. Retain only
            # the relationship so fresh tensors can reuse graphs and code.
            input_ids = (0, 0) if len(args) == 2 and args[0] is args[1] else tuple(range(len(args)))
            key = (program.code, bindings, guard_metadata, input_ids)
            executor = cache.graphs.get(key)
            if executor is None:
                bindings, values, scalars = runtime_bindings(
                    program, static_bindings, static_values, cache.graphs)
                key = (program.code, bindings, guard_metadata, input_ids)
                executor = cache.graphs.get(key)
            if executor is None:
                if len(cache.graphs) >= recompile_limit:
                    unsupported(f"hit recompile_limit={recompile_limit}")
                graph = lower(program, values, len(args), input_ids)
                # Code specializes expressions and device, not shapes, values or pointers.
                # Keep this map inside reset-owned cache entries, so reset releases modules.
                code_key = (graph, metadata[0][4])
                executor = next((entry for entry in cache.graphs.values()
                                 if entry[0] == code_key), None)
                if executor is None:
                    executor = (code_key, _native._pointwise_compile(args, graph.nodes, graph.output))
                result = executor[1].run(args, scalars)
                cache.graphs[key] = executor
            else:
                result = executor[1].run(args, scalars)
            return result

    compiled._torch_rs_pointwise_cache = cache
    return compiled
