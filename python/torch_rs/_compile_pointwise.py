"""Static bytecode to typed float32 SSA for the native default CUDA JIT.

The frontend never invokes the function, Tensor methods, or a Python operator
on user objects. Warm calls resolve binding/metadata guards and enter one native
kernel. It deliberately has no graph breaks or eager fallback.
"""
from dataclasses import dataclass, field
import dis
from itertools import islice
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
    "CALL_METHOD", "RETURN_VALUE", "RETURN_CONST", "COPY", "DUP_TOP", "SWAP"}
_METHODS = tuple(_UNARY) + tuple(_BINARY) + ("__getattribute__",)
_MISSING = object()
_RANGE = range
_LOOP_OPS = {"GET_ITER", "FOR_ITER", "JUMP_ABSOLUTE", "JUMP_BACKWARD",
             "END_FOR", "POP_TOP", "POP_ITER"}
# Imported during package initialization, before public bindings can be patched.
_FUNCTIONS = tuple((name, _ROOT.__dict__.get(name)) for name in
                   ("neg", "negative", "relu", "sin", "cos", "add", "sub", "subtract", "mul", "multiply"))
_METHOD_GUARDS = tuple((cls, name, cls.__dict__.get(name, _MISSING))
                      for cls in (_ROOT.Tensor, _ROOT.Tensor.__base__) for name in _METHODS)


def unsupported(reason):
    raise NotImplementedError("torch.compile(): native CUDA pointwise: " + reason)


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


@dataclass(frozen=True, eq=False)
class Helper:
    """Validated code identity, never the function or its mutable containers."""
    code: types.CodeType

    def __eq__(self, other):
        return type(other) is Helper and self.code is other.code

    def __hash__(self):
        return id(self.code)


@dataclass(frozen=True)
class Graph:
    inputs: int
    nodes: tuple
    output: int
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        # Executable lookup uses structural equality across specializations;
        # hashing immutable IR need not walk every node again on each launch.
        object.__setattr__(self, "_hash", hash((self.inputs, self.nodes, self.output)))

    def __hash__(self):
        return self._hash


@dataclass(frozen=True)
class BindingSource:
    # Parameter positions belong to the public signature, never to the native
    # tensor tuple or scalar ABI. Captures retain their bytecode origin.
    kind: str
    name: str
    position: int | None = None


@dataclass(frozen=True)
class Program:
    code: object
    instructions: tuple
    dependencies: tuple
    range_sources: tuple = ()
    loop_overhead: int = 0
    positions: object = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "positions", types.MappingProxyType(
            {source: index for index, source in enumerate(self.dependencies)}))


@dataclass(frozen=True)
class BoundValue:
    """A lazy source read; assigning an unused local does not create a guard."""
    source: BindingSource
    value: object


def validate_data(obj):
    """Check a data boundary without realizing or guarding a lazy source."""
    item = obj.value if type(obj) is BoundValue else obj
    if type(item) is not Value and type(item) is not RuntimeScalar:
        scalar_bits(item)
    return obj


@dataclass(frozen=True)
class TensorGuard:
    """Reference shape/stride predicates, independent of native address maps."""
    sizes: tuple
    strides: tuple
    properties: tuple

    def matches(self, metadata, all_metadata):
        shape, strides = metadata[:2]
        if len(shape) != len(self.sizes) or metadata[2:5] != self.properties:
            return False
        if any((size < 2 if expected is None else size != expected)
               for size, expected in zip(shape, self.sizes)):
            return False
        for stride, expected in zip(strides, self.strides):
            if expected is None:
                if stride < 2:
                    return False
            elif type(expected) is int:
                if stride != expected:
                    return False
            elif expected[0] == "product":
                axis = expected[1]
                if stride != shape[axis] * strides[axis]:
                    return False
            elif stride != all_metadata[expected[1]][1][expected[2]]:
                return False
        return True


@dataclass(frozen=True)
class ShapeGuards:
    """The admitted pointwise graph's tensor and cross-tensor shape contract."""
    tensors: tuple
    equal_axes: tuple
    index_bounds: tuple

    def matches(self, metadata):
        if not all(guard.matches(metadata[source], metadata) for source, guard in self.tensors):
            return False
        if any(metadata[a][0][i] != metadata[b][0][j] for a, i, b, j in self.equal_axes):
            return False
        return all(0 <= _broadcast_elements([metadata[s][0] for s in group]) <= 2147483647
                   for group in self.index_bounds)


@dataclass
class Specialization:
    """Frozen logical semantics with lowerings for concrete tensor operand ABIs."""
    # Scalars/operators are frozen here; tensor Values are rebound to the current
    # filtered ABI only when a concrete lowering is missing. No tensor is retained.
    values: dict
    observed: tuple
    observations: dict
    lowerings: dict
    # Immutable projections retain source semantics without repeating dependency
    # searches for each guard candidate. Tensor indices always come from the call.
    binding_checks: tuple = ()
    tensor_positions: tuple = ()
    # Sources passed through data boundaries need admission even if ignored by
    # the helper. These positions impose no scalar-value or tensor-shape guard.
    data_positions: tuple = ()


def _broadcast_elements(shapes):
    result = ()
    for shape in shapes:
        rank = max(len(result), len(shape))
        left, right = (1,) * (rank-len(result)) + result, (1,) * (rank-len(shape)) + shape
        if any(a != b and a != 1 and b != 1 for a, b in zip(left, right)):
            return -1
        result = tuple(b if a == 1 else a for a, b in zip(left, right))
    return math.prod(result)


def _stride_observation(metadata):
    # torch/_dynamo/pgo.py: FrameStateSizeEntry stride observations.
    shape, strides = metadata[:2]
    candidates, result = {}, [None] * len(shape)
    for stride, negative_axis in sorted((s, -i) for i, s in enumerate(strides)):
        axis = -negative_axis
        result[axis] = candidates.get(stride, stride)
        candidates.setdefault(stride * shape[axis], ("product", axis))
    return tuple(result)


def _tensor_guard(metadata, previous, duck_strides, source):
    """Generalize successful source history while retaining zero/one guards."""
    # torch/_dynamo/variables/builder.py: _automatic_dynamic and
    # torch/fx/experimental/symbolic_shapes.py: _compute_symbolic_stride.
    # This models only the admitted contiguous float32 pointwise surface.
    shape, strides = metadata[:2]
    tensors = [m for m in previous if type(m) is tuple]
    dynamic_rank = "scalar" in previous or any(len(m[0]) != len(shape) for m in tensors)
    dynamic = tuple(dynamic_rank or any(m[0][i] != size for m in tensors)
                    for i, size in enumerate(shape))
    sizes = tuple(None if changed and size > 1 else size for size, changed in zip(shape, dynamic))
    # PGO only promotes independent strides while size history is fully static.
    observed = _stride_observation(metadata)
    dynamic_strides = tuple(not any(dynamic) and any(_stride_observation(m)[i] != atom for m in tensors)
                            for i, atom in enumerate(observed))
    candidates, guards = {}, [None] * len(shape)
    for stride, negative_axis in sorted((s, -i) for i, s in enumerate(strides)):
        axis = -negative_axis
        contiguous = axis+1 < len(shape) and stride == shape[axis+1] * strides[axis+1]
        if stride in (0, 1) and not contiguous:
            guard = stride
        elif dynamic_strides[axis]:
            guard = stride if stride in (0, 1) else None
        elif stride in candidates:
            guard = ("product", candidates[stride])
        elif None not in sizes:
            guard = stride
        else:
            guard = duck_strides.get(stride)
            duck_strides.setdefault(stride, ("equal", source, axis))
        guards[axis] = guard
        candidates[stride * shape[axis]] = axis
    return TensorGuard(sizes, tuple(guards), metadata[2:5])


def _logical_keys(program, bindings, values, observed, tensors):
    """Guard realized sources and aliases without exposing native operand indices."""
    keys, first, aliases = list(bindings), {}, []
    for source in observed:
        value = values[source]
        if type(value) is Value:
            tensor = tensors[value.index]
            owner = next((s for s, index in first.items() if tensors[index] is tensor), None)
            position = program.positions[source]
            if owner is None:
                first[source] = value.index
                keys[position] = ("tensor", None)
                aliases.append(len(first)-1)
            else:
                keys[position] = ("alias", owner)
                aliases.append(tuple(first).index(owner))
    for position, source in enumerate(program.dependencies):
        if source not in observed:
            keys[position] = ("ignored",)
    return tuple(keys), first, tuple(aliases)


def _shape_guards(program, graph, first, metadata, history):
    """Build logical guards; actual full-input admission stays with native code."""
    observations, guards, duck_strides = {}, [], {}
    for source, index in first.items():
        current = metadata[index][:5]
        previous = [entry.observations[source] for key, entry in history.items()
                    if key[0] is program.code and source in entry.observations]
        guards.append((source, _tensor_guard(current, previous, duck_strides, source)))
        observations[source] = current
    # Graph topology supplies broadcast equalities and live iteration/buffer
    # bounds. Rust remains the owner of actual-shape admission at compile/run.
    dependencies, combined = [], False
    for op, a, b, _ in graph.nodes:
        if op == "input":
            deps = {a}
        elif op in ("add", "sub", "mul"):
            deps = dependencies[a] | dependencies[b]
            combined |= len(deps) == 2
        elif op in ("neg", "relu", "sin", "cos"):
            deps = dependencies[a]
        else:
            deps = set()
        dependencies.append(deps)
    equal_axes = []
    if combined and len(first) == 2:
        (left, li), (right, ri) = first.items()
        ls, rs = metadata[li][0], metadata[ri][0]
        for offset in range(1, min(len(ls), len(rs))+1):
            if ls[-offset] > 1 and rs[-offset] > 1:
                equal_axes.append((left, len(ls)-offset, right, len(rs)-offset))
    live = tuple(s for s, i in first.items() if graph.nodes[i][1] in dependencies[graph.output])
    groups = tuple((s,) for s in live) + ((live,) if len(live) == 2 else ())
    # torch/_inductor/codegen/simd.py: can_use_32bit_indexing installs an
    # upper-bound conjunction only if the whole kernel is 32-bit eligible.
    # A 64-bit specialization has no inverse bound and can accept small shapes.
    bounds = groups if all(_broadcast_elements([observations[s][0] for s in group]) <= 2147483647
                           for group in groups) else ()
    return ShapeGuards(tuple(guards), tuple(equal_axes), bounds), observations


def _select_specialization(program, bindings, values, tensors, metadata, graphs):
    """Select newest matching semantics before considering new scalar promotion.

    A generalized entry can supersede an older exact shape even when its native
    broadcast executable is absent. Conversely, a rank miss can expose an older
    static scalar entry beneath a newer runtime entry.
    """
    resolved = tuple(values.values())  # resolve preserves dependency order.
    by_source = None
    for key, entry in reversed(graphs.items()):
        if key[0] is not program.code:
            continue
        scalars = []
        for position, expected in entry.binding_checks:
            current = bindings[position]
            if expected[0] == "runtime_float" and type(resolved[position]) is float:
                scalars.append(resolved[position])
            elif expected[0] in ("tensor", "alias"):
                if current[0] != "tensor":
                    break
            elif expected != current:
                break
        else:
            # Source realization order defines aliases, not public slot order.
            # Tensor kind checks above precede every current operand lookup.
            first, aliases = [], []
            for position in entry.tensor_positions:
                tensor = tensors[resolved[position].index]
                owner = next((i for i, previous in enumerate(first) if previous is tensor), None)
                if owner is None:
                    owner = len(first)
                    first.append(tensor)
                aliases.append(owner)
            if tuple(aliases) != key[3]:
                continue
            if by_source is None:
                by_source = {s: metadata[v.index][:5] for s, v in values.items() if type(v) is Value}
            if key[2].matches(by_source):
                return key, entry, tuple(scalars)
    return None


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


def validate_code(code, arity, *, helper=False):
    if (arity < 1 or code.co_argcount != arity or code.co_kwonlyargcount
            or code.co_flags & (0x04 | 0x08 | 0x20 | 0x80 | 0x200)
            or code.co_cellvars or (helper and code.co_freevars)
            or getattr(code, "co_exceptiontable", b"")):
        unsupported("requires a straight-line function with positional inputs and no defaults")
    # dis formats co_consts with repr, before yielding even a LOAD_CONST.
    # Validate before retaining helper code too: unused constants stay alive in it.
    # None and exact strings are metadata only, never data operands or returns.
    for constant in code.co_consts:
        if constant is not None and type(constant) is not str:
            scalar_bits(constant)


def instructions_for(code, *, helper=False):
    # Preserve the visible-instruction limit across versions with differing
    # inline CACHE layouts, without first allocating an unbounded tuple.
    instructions = tuple(islice(dis.get_instructions(code), 16385))
    if len(instructions) > 16384:
        unsupported("function exceeds pointwise instruction limit")
    for instruction in instructions:
        if (instruction.opname not in (_ALLOWED if helper else _ALLOWED | _LOOP_OPS)
                or (helper and instruction.opname in ("LOAD_GLOBAL", "LOAD_DEREF"))):
            unsupported(f"unsupported bytecode {instruction.opname}; control flow, mutation and non-pointwise graphs are unsupported")
    return instructions


def validate_namespaces(model):
    # Check every key before any lookup: a colliding non-string key can execute
    # equality even in an exact dict. Use the function's actual builtins table,
    # which CPython freezes at function creation, not globals['__builtins__'].
    for namespace, label in ((model.__globals__, "globals"),
                             (model.__builtins__, "builtins")):
        if type(namespace) is not dict:
            unsupported("function " + label + " must be an exact dict")
        if any(type(key) is not str for key in namespace):
            unsupported("function " + label + " keys must be exact strings")


def validate_ranges(model, sources):
    for source in sources:
        name = source.name
        if source.kind == "LOAD_DEREF":
            cell = model.__closure__[model.__code__.co_freevars.index(name)]
            try:
                value = cell.cell_contents
            except ValueError:
                value = _MISSING
        else:
            value = (model.__globals__[name] if name in model.__globals__
                     else model.__builtins__.get(name, _MISSING))
        if value is not _RANGE:
            unsupported("literal loop requires the actual built-in range: " + name)


def validate_loop_stack(body):
    # The iterator lives below the body in CPython. Our straight-line frame
    # omits it, so no body instruction may read/swap/copy below its own stack.
    depth = 1  # FOR_ITER's index, consumed by STORE_FAST (possibly fused).
    for instruction in body:
        op, arg = instruction.opname, instruction.arg
        required, delta = 0, 0
        if op in _IGNORED:
            continue
        if op.startswith("LOAD_FAST"):
            delta = 2 if op in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW") else 1
        elif op in ("LOAD_CONST", "LOAD_SMALL_INT", "LOAD_GLOBAL", "LOAD_DEREF"):
            delta = 1
        elif op in ("STORE_FAST", "STORE_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST"):
            required = 2 if op == "STORE_FAST_STORE_FAST" else 1
            delta = 0 if op == "STORE_FAST_LOAD_FAST" else -required
        elif op in ("LOAD_ATTR", "LOAD_METHOD", "UNARY_NEGATIVE"):
            required = 1
        elif op in ("BINARY_OP", "BINARY_ADD", "BINARY_SUBTRACT", "BINARY_MULTIPLY"):
            if op == "BINARY_OP" and instruction.argrepr not in ("+", "-", "*"):
                unsupported("unsupported loop binary operator")
            required, delta = 2, -1
        elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD"):
            required, delta = arg + 1, -arg
        elif op in ("COPY", "DUP_TOP"):
            required, delta = (arg if op == "COPY" else 1), 1
        elif op == "SWAP":
            required = arg
        else:
            unsupported("unsupported loop control flow: " + op)
        if required < 0 or depth < required:
            unsupported("invalid loop body stack")
        depth += delta
    if depth:
        unsupported("invalid loop body stack")


def normalize_loops(model, instructions):
    """Validate all structured root loops, then boundedly expand wordcode.

    CPython 3.10/3.11 FOR_ITER removes the exhausted iterator; 3.12 END_FOR
    removes iterator and sentinel; 3.13 uses END_FOR/POP_TOP and 3.14 uses
    END_FOR/POP_ITER. Back edges are absolute in 3.10, relative thereafter.
    Only these complete regions disappear; other edges always reject.
    """
    compact = tuple(i for i in instructions if i.opname not in _IGNORED)
    offsets = {i.offset: n for n, i in enumerate(compact)}
    # Labels can precede a no-op (e.g. a following `pass`) or EXTENDED_ARG
    # prefixes on large loops. Resolve only these transparent instructions,
    # not arbitrary ignored call/stack setup, to their next retained opcode.
    target = None
    for instruction in reversed(instructions):
        if instruction.opname not in _IGNORED:
            target = offsets[instruction.offset]
        elif instruction.opname in ("NOP", "EXTENDED_ARG", "NOT_TAKEN"):
            if target is not None:
                offsets[instruction.offset] = target
        else:
            target = None
    original_positions = {i.offset: n for n, i in enumerate(instructions)}
    regions, sources, covered = [], [], set()
    expanded_size = len(instructions)
    for position, instruction in enumerate(compact):
        if instruction.opname != "GET_ITER":
            continue
        call = compact[position - 1] if position else None
        if call is None or call.opname not in ("CALL", "CALL_FUNCTION") or call.arg not in (1, 2, 3):
            unsupported("loop requires a direct literal range call")
        start = position - call.arg - 2
        if start < 0 or compact[start].opname not in ("LOAD_GLOBAL", "LOAD_DEREF"):
            unsupported("loop requires a direct range binding")
        literals = compact[start + 1:position - 1]
        if any(i.opname not in ("LOAD_CONST", "LOAD_SMALL_INT") or type(i.argval) is not int
               for i in literals):
            unsupported("range bounds must be literal exact integers")
        args = tuple(i.argval for i in literals)
        if len(args) == 3 and args[2] == 0:
            unsupported("range step must be nonzero")
        if position + 1 >= len(compact) or compact[position + 1].opname != "FOR_ITER":
            unsupported("invalid loop iterator control flow")
        head = compact[position + 1]
        end = offsets.get(head.argval, -1)
        if end <= position + 3 or end >= len(compact):
            unsupported("invalid loop exit")
        back = compact[end - 1]
        if (back.opname not in ("JUMP_ABSOLUTE", "JUMP_BACKWARD")
                or offsets.get(back.argval) != position + 1):
            unsupported("invalid loop back edge")
        body = compact[position + 2:end - 1]
        if body[0].opname not in ("STORE_FAST", "STORE_FAST_LOAD_FAST"):
            unsupported("loop index must be a local")
        validate_loop_stack(body)
        stop = end
        if sys.version_info >= (3, 12):
            if compact[stop].opname != "END_FOR":
                unsupported("invalid loop cleanup")
            stop += 1
            if sys.version_info >= (3, 13):
                cleanup = "POP_ITER" if sys.version_info >= (3, 14) else "POP_TOP"
                if stop >= len(compact) or compact[stop].opname != cleanup:
                    unsupported("invalid loop cleanup")
                stop += 1
        if regions and start < regions[-1][1]:
            unsupported("nested or overlapping loops are unsupported")
        sources.append(BindingSource(compact[start].opname, compact[start].argval))
        # Exact integer arithmetic computes arbitrarily large trip counts
        # without len(range), iteration, or allocating a repeated body.
        first, limit, step = (0, args[0], 1) if len(args) == 1 else (*args, 1) if len(args) == 2 else args
        trips = max(0, (limit - first + step - (1 if step > 0 else -1)) // step)
        # Ignored instructions inside the body (including PRECALL/NOP) repeat
        # too. Removing them from normalization must not relax the shared limit.
        body_size = original_positions[back.offset] - original_positions[head.offset] - 1
        # Zero-trip bodies still receive one admission pass, isolated from the
        # executing frame. Charge that pass and its two internal scope markers.
        expanded_size += max(1, trips) * (body_size + 1) + (2 if not trips else 0)
        if expanded_size > 16384:
            unsupported("expanded function exceeds pointwise instruction limit")
        regions.append((start, stop, body, first, step, trips))
        covered.update(_RANGE(start, stop))
    for position, instruction in enumerate(compact):
        if instruction.opname in _LOOP_OPS and position not in covered:
            unsupported("unsupported loop control flow: " + instruction.opname)
    validate_ranges(model, sources)
    if not regions:
        return instructions, (), 0
    result, cursor = [], 0
    for start, stop, body, first, step, trips in regions:
        result.extend(compact[cursor:start])
        if not trips:
            result.append(body[0]._replace(opname="_LOOP_CHECK_START"))
        for index in _RANGE(max(1, trips)):
            # Reuse the dis instruction record; lowering only consumes opname,
            # argval and argrepr. Original code remains the semantic owner.
            result.append(body[0]._replace(opname="LOAD_CONST", argval=first + index * step))
            result.extend(body)
        if not trips:
            result.append(body[0]._replace(opname="_LOOP_CHECK_END"))
        cursor = stop
    result.extend(compact[cursor:])
    # Charge original setup/cleanup too, sharing lower()'s helper-call budget.
    return tuple(result), tuple(dict.fromkeys(sources)), expanded_size - len(result)


def freeze_helper(model):
    validate_signature_containers(model)
    if model.__closure__ is not None and len(model.__closure__):
        unsupported("helpers must not have closures")
    attributes = model.__dict__
    if type(attributes) is not dict:
        unsupported("helper attributes must be an exact dict")
    if any(type(key) is not str for key in attributes):
        unsupported("helper attribute keys must be exact strings")
    for directive in ("_torchdynamo_inline", "_dynamo_marked_constant", "_torchdynamo_disable"):
        if directive in attributes:
            unsupported("helper compiler directives are unsupported")
    code = model.__code__
    validate_code(code, code.co_argcount, helper=True)
    return Helper(code)


def analyze(model, arity):
    if type(model) is not types.FunctionType or arity < 1:
        unsupported("expected an exact Python function and positional inputs")
    validate_signature_containers(model)
    code = model.__code__
    validate_code(code, arity)
    validate_namespaces(model)
    instructions, range_sources, loop_overhead = normalize_loops(model, instructions_for(code))
    # Only initial parameter values read by the bytecode have scalar guards.
    # Unused/overwritten tensors still remain in the complete native input tuple.
    initial, read = set(code.co_varnames[:arity]), set()
    for instruction in instructions:
        op, arg = instruction.opname, instruction.argval
        if op == "_LOOP_CHECK_START":
            saved_initial = initial.copy()
        elif op == "_LOOP_CHECK_END":
            initial = saved_initial
        if op in ("STORE_FAST", "STORE_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST"):
            stored = (arg,) if op == "STORE_FAST" else arg[:1] if op == "STORE_FAST_LOAD_FAST" else arg
            initial.difference_update(stored)
        loaded = ((arg,) if op in ("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW")
                  else arg if op in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW")
                  else arg[1:] if op == "STORE_FAST_LOAD_FAST" else ())
        read.update(name for name in loaded if name in initial)
    parameters = tuple(BindingSource("parameter", name, position)
                       for position, name in enumerate(code.co_varnames[:arity]) if name in read)
    captures = tuple(dict.fromkeys(BindingSource(i.opname, i.argval) for i in instructions
                                  if i.opname in ("LOAD_GLOBAL", "LOAD_DEREF")))
    dependencies = parameters + captures
    return Program(code, instructions, dependencies, range_sources, loop_overhead)


def binding(value):
    if is_scalar(value):
        # Reference static float guards equate signed zeros and guard NaNs by
        # exact float type plus isnan, independent of sign/payload. Canonicalize
        # only guard identity; frozen values, IR bits and runtime inputs retain
        # the original scalar. Packed NaNs still participate in finite promotion.
        key = value
        if type(value) is float:
            key = struct.pack("=d", math.nan if math.isnan(value) else 0.0 if value == 0.0 else value)
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
    if type(value) is types.FunctionType:
        helper = freeze_helper(value)
        return ("helper", helper), helper
    unsupported("only native operators, exact Python helpers and scalar constants may be captured")


def bind_arguments(args):
    """Filter once, retaining every tensor (including unused and repeated ones)."""
    tensors, parameters = [], []
    for arg in args:
        kind = type(arg)
        if kind is _TENSOR_TYPE:
            parameters.append(Value(len(tensors)))
            tensors.append(arg)
        elif kind is float or kind is bool:
            parameters.append(arg)
        else:
            unsupported("default backend requires exact native CUDA float32 Tensor inputs "
                        "and exact float/bool positional scalars; positional integers and "
                        "objects are unsupported; see docs/compile-pointwise-jit.md")
    if len(tensors) not in (1, 2):
        unsupported("expected one or two positional tensor arguments")
    return tuple(tensors), tuple(parameters)


def resolve(model, program, parameters=None):
    # FunctionType permits a dict subclass as globals. Never invoke its lookup
    # hooks, including on a warm call or before rejecting a later graph node.
    validate_signature_containers(model)
    validate_code(model.__code__, program.code.co_argcount)
    validate_namespaces(model)
    validate_ranges(model, program.range_sources)
    globals_ = model.__globals__
    values, keys = {}, []
    closure = dict(zip(program.code.co_freevars, model.__closure__ or ()))
    if parameters is None:
        # Private hardware-free lowering callers model a tensor-only signature.
        parameters = tuple(Value(i) for i in range(program.code.co_argcount))
    for source in program.dependencies:
        kind, name = source.kind, source.name
        if kind == "parameter":
            value = parameters[source.position]
            if type(value) is Value:
                keys.append(("tensor", value.index))
                values[source] = value
                continue
        elif kind == "LOAD_GLOBAL":
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
        values[source] = value
    return tuple(keys), values


def runtime_bindings(program, bindings, values, graphs, *, observed=None):
    """Resolve a new specialization after guard misses, from successful history."""
    previous = [key[1] for key in graphs if key[0] is program.code]
    keys, resolved, scalars = list(bindings), dict(values), []
    for position, dependency in enumerate(program.dependencies):
        value = values[dependency]
        if (observed is not None and dependency not in observed) or type(value) is not float:
            continue
        # A new reference trace specializes nonfinite values even after prior
        # promotion. An existing runtime guard can still accept them on a hit.
        if not math.isfinite(value):
            continue
        observations = [keys_[position] for keys_ in previous]
        dynamic = any(key[0] in ("runtime_float", "tensor") for key in observations)
        if not dynamic:
            # Exact builtins only: numeric comparison matches the reference's
            # promotion history, including equal signed zeros and int/float
            # transitions. Booleans do not participate in scalar dynamism.
            dynamic = any(
                (key[0] is float and struct.unpack("=d", key[1])[0] != value)
                or (key[0] is int and key[1] != value)
                for key in observations
            )
        if dynamic:
            if len(scalars) >= 64:
                unsupported("at most 64 runtime scalar bindings are supported")
            resolved[dependency] = RuntimeScalar(len(scalars))
            scalars.append(value)
            keys[position] = ("runtime_float",)
    return tuple(keys), resolved, tuple(scalars)


def lower(program, values, arity, input_ids=None, *, observed=None, data_sources=None):
    if input_ids is None:
        input_ids = tuple(range(arity))
    nodes = [("input", i, 0, 0) for i in input_ids]
    runtime = sorted(value.index for value in values.values() if type(value) is RuntimeScalar)
    nodes.extend(("scalar", index, 0, 0) for index in runtime)
    remaining = 16384 - program.loop_overhead
    checked_nodes = 0
    helper_instructions = {}  # Per lowering only; warm hits never parse helpers.

    def data(obj):
        # Validate without realizing a lazy root source. Ignored parameters must
        # not acquire guards, but cannot carry callable/module/higher-order data.
        validate_data(obj)
        if data_sources is not None and type(obj) is BoundValue:
            data_sources.add(obj.source)
        return obj

    def realize(obj):
        if type(obj) is BoundValue:
            if observed is not None and obj.source not in observed:
                observed.append(obj.source)
            return obj.value
        return obj

    def value(obj):
        obj = realize(obj)
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
        if len(nodes) + checked_nodes > 4096:
            unsupported("graph exceeds 4096-node limit")
        return Value(len(nodes) - 1)

    def frame(instructions, locals_):
        nonlocal remaining, checked_nodes
        remaining -= len(instructions)
        if remaining < 0:
            unsupported("expanded function exceeds pointwise instruction limit")
        stack = []

        def load(name):
            if name not in locals_:
                unsupported("unbound local: " + name)
            stack.append(locals_[name])

        for instruction in instructions:
            op, arg = instruction.opname, instruction.argval
            if op in _IGNORED:
                continue
            if op == "_LOOP_CHECK_START":
                # Reuse typed operator/helper/data admission without executing
                # Python or changing the real frame/IR. Realized sources keep
                # the existing warm semantic and ignored-data guards.
                saved_locals, node_start = locals_, len(nodes)
                locals_ = locals_.copy()
            elif op == "_LOOP_CHECK_END":
                checked_nodes += len(nodes) - node_start
                del nodes[node_start:]
                locals_ = saved_locals
            elif op in ("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"):
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
                source = BindingSource(op, arg)
                stack.append(BoundValue(source, values[source]))
            elif op in ("LOAD_ATTR", "LOAD_METHOD"):
                owner = realize(stack.pop())
                if owner is _ROOT:
                    if arg not in dict(_FUNCTIONS):
                        unsupported("unsupported native function: " + arg)
                    stack.append(binding(_ROOT.__dict__.get(arg))[1])
                elif isinstance(owner, Value) and owner.tensor and arg in (_UNARY | _BINARY):
                    stack.append(Call((_UNARY | _BINARY)[arg], owner, arg == "__rsub__"))
                else:
                    unsupported("unsupported attribute: " + str(arg))
            elif op == "UNARY_NEGATIVE":
                operand = realize(stack.pop())
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
                target = realize(stack.pop())
                if type(target) is Helper:
                    if len(operands) != target.code.co_argcount:
                        unsupported("helper argument count mismatch")
                    parameters = [data(operand) for operand in operands]
                    if target not in helper_instructions:
                        helper_instructions[target] = instructions_for(target.code, helper=True)
                    stack.append(frame(helper_instructions[target],
                                       dict(zip(target.code.co_varnames, parameters))))
                    continue
                if type(target) is not Call:
                    unsupported("only native pointwise operators and direct helpers may be called")
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
            elif op in ("RETURN_VALUE", "RETURN_CONST"):
                result = data(stack.pop() if op == "RETURN_VALUE" else arg)
                if stack:
                    unsupported("return one data value")
                return result
        unsupported("missing data return")

    locals_ = {source.name: BoundValue(source, values[source]) for source in program.dependencies
               if source.kind == "parameter"}
    result = realize(frame(program.instructions, locals_))
    if type(result) is not Value or not result.tensor or result.index < arity:
        unsupported("return one computed pointwise tensor")
    return Graph(arity, tuple(nodes), result.index)


def implementation(model, recompile_limit):
    # Shared reset registry/lock discipline, but no eager graph evaluator.
    cache = _state.new_native_eager_compile_cache()
    program = None

    def compiled(*args, **kwargs):
        nonlocal program
        if kwargs:
            unsupported("expected positional arguments without keywords")
        if type(_ROOT) is not types.ModuleType:
            unsupported("patched native package type")
        tensors, parameters = bind_arguments(args)
        if _ROOT.overrides._get_current_function_mode() is not None:
            unsupported("active __torch_function__ mode")
        for cls, name, expected in _METHOD_GUARDS:
            if cls.__dict__.get(name, _MISSING) is not expected:
                unsupported("patched Tensor operation binding: " + name)
        # The native bridge checks all metadata and storage bounds again on launch.
        metadata = tuple(_native._compile_trace_tensor_metadata(arg) for arg in tensors)
        if any(m[4] == "cpu" for m in metadata):
            unsupported("default backend does not compile CPU tensors; use backend='eager' "
                        "for the documented CPU capture subset; see docs/compile-pointwise-jit.md")
        _native._pointwise_validate_inputs(tensors)
        with cache.lock:
            if program is None or program.code is not model.__code__:
                program = analyze(model, len(args))
            validate_signature_containers(model)
            if program.code.co_argcount != len(args):
                unsupported("function signature changed")
            static_bindings, static_values = resolve(model, program, parameters)
            input_ids = (0, 0) if len(tensors) == 2 and tensors[0] is tensors[1] else tuple(range(len(tensors)))
            abi = (len(tensors), input_ids, tuple((s, v.index) for s, v in static_values.items()
                                               if type(v) is Value))
            selected = _select_specialization(program, static_bindings, static_values,
                                              tensors, metadata, cache.graphs)
            if selected is None:
                if len(cache.graphs) >= recompile_limit:
                    unsupported(f"hit recompile_limit={recompile_limit}")
                # The same lowering records which sources actually materialize,
                # including dead operations but excluding unused local bindings.
                observed = []
                data_sources = set()
                graph = lower(program, static_values, len(tensors), input_ids,
                              observed=observed, data_sources=data_sources)
                bindings, values, scalars = runtime_bindings(
                    program, static_bindings, static_values, cache.graphs, observed=observed)
                if scalars:
                    graph = lower(program, values, len(tensors), input_ids)
                bindings, first, aliases = _logical_keys(program, bindings, values, observed, tensors)
                guards, observations = _shape_guards(program, graph, first, metadata, cache.graphs)
                observations.update((s, "scalar") for s in observed
                                    if type(static_values[s]) is float or type(static_values[s]) is int)
                key = (program.code, bindings, guards, aliases)
                entry = Specialization(
                    values, tuple(observed), observations, {},
                    tuple((position, expected) for position, expected in enumerate(bindings)
                          if expected[0] != "ignored"),
                    tuple(program.positions[source] for source in observed if type(values[source]) is Value),
                    tuple(position for position, source in enumerate(program.dependencies)
                          if source in data_sources))
            else:
                key, entry, scalars = selected
                # A cached graph omits ignored sources, but helper argument
                # admission still applies to their current values on every hit.
                for position in entry.data_positions:
                    validate_data(static_values[program.dependencies[position]])
                graph = entry.lowerings.get(abi)
                if graph is None:
                    values = dict(entry.values)
                    for source in program.dependencies:
                        if type(values[source]) is Value or source not in entry.observed:
                            values[source] = static_values[source]
                    graph = lower(program, values, len(tensors), input_ids)
            # Logical guards choose scalar semantics. Concrete executables still
            # specialize the full native ABI and exact broadcast address formula.
            # Offsets/addresses and all original-IR admission are checked at run.
            shapes = tuple(m[0] for m in metadata)
            indexing_key = None if all(s == shapes[0] for s in shapes) else shapes
            code_key = (graph, metadata[0][4], indexing_key)
            executor = cache.executors.get(code_key)
            if executor is None:
                executor = _native._pointwise_compile(tensors, graph.nodes, graph.output)
            result = executor.run(tensors, scalars)
            # Publish both levels only after success. Executable and lowering LRU
            # eviction bounds retained modules without consuming logical slots.
            for mapping, item_key, item in ((entry.lowerings, abi, graph),
                                            (cache.executors, code_key, executor),
                                            (cache.graphs, key, entry)):
                # Recency belongs to each map independently: a shared executor
                # can already be newest while its logical entry/lowering is not.
                if (mapping and next(reversed(mapping)) == item_key
                        and next(reversed(mapping.values())) is item):
                    continue
                mapping.pop(item_key, None)
                mapping[item_key] = item
                while len(mapping) > recompile_limit:
                    del mapping[next(iter(mapping))]
            return result

    compiled._torch_rs_pointwise_cache = cache
    return compiled
