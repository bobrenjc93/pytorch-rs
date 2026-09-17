"""Static bytecode to typed float32 SSA for the native default CUDA JIT.

The frontend never invokes the function, Tensor methods, or a Python operator
on user objects. Warm calls resolve binding/metadata guards and enter one native
kernel. It deliberately has no graph breaks or eager fallback.
"""
from dataclasses import dataclass, field
import dis
from itertools import islice
import math
import operator
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
_ROTATIONS = {"ROT_TWO": 2, "ROT_THREE": 3}  # CPython 3.10 fixed tuple assignments.
_ALLOWED = _IGNORED | _ROTATIONS.keys() | {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "LOAD_FAST_LOAD_FAST",
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW", "STORE_FAST", "STORE_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST",
    "LOAD_CONST", "LOAD_SMALL_INT", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_ATTR", "LOAD_METHOD", "BINARY_OP",
    "BINARY_ADD", "BINARY_SUBTRACT", "BINARY_MULTIPLY", "UNARY_NEGATIVE", "CALL", "CALL_FUNCTION",
    "CALL_METHOD", "RETURN_VALUE", "RETURN_CONST", "COPY", "DUP_TOP", "SWAP",
    "BUILD_TUPLE", "BUILD_LIST", "BUILD_MAP", "BUILD_CONST_KEY_MAP",
    "BINARY_SUBSCR", "UNPACK_SEQUENCE"}
_METHODS = tuple(_UNARY) + tuple(_BINARY) + ("__getattribute__", "shape")
_MISSING = object()
_RANGE = range
# Retained numerical data per wrapper, not a process/module/allocator-pool or
# transient allocation budget. Larger admitted plans execute without retention.
_PREPARED_CACHE_BYTES = 32 * 1024 * 1024
_LOOP_OPS = {"GET_ITER", "FOR_ITER", "JUMP_ABSOLUTE", "JUMP_BACKWARD",
             "END_FOR", "POP_TOP", "POP_ITER"}
_CONDITIONALS = {"POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
                 "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_FORWARD_IF_TRUE"}
_BRANCH_OPS = _CONDITIONALS | {"JUMP_FORWARD", "COMPARE_OP", "BINARY_SUBSCR", "TO_BOOL"}
_COMPARE = {"<": operator.lt, "<=": operator.le, "==": operator.eq,
            "!=": operator.ne, ">=": operator.ge, ">": operator.gt}
_REVERSE_COMPARE = {"<": ">", "<=": ">=", "==": "==", "!=": "!=", ">=": "<=", ">": "<"}
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
    outputs: tuple
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        # Executable lookup uses structural equality across specializations;
        # hashing immutable IR need not walk every node again on each launch.
        object.__setattr__(self, "_hash", hash((self.inputs, self.nodes, self.outputs)))

    def __hash__(self):
        return self._hash


@dataclass(frozen=True)
class BindingSource:
    # Parameter positions belong to the public signature, never to the native
    # tensor tuple or scalar ABI. Captures retain their bytecode origin.
    kind: str
    name: str
    position: int | None = None
    path: tuple = ()
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash((self.kind, self.name, self.position, self.path)))

    def __hash__(self):
        return self._hash

    def child(self, item):
        return BindingSource(self.kind, self.name, self.position, self.path + (item,))


@dataclass(frozen=True)
class Program:
    code: object
    instructions: tuple
    dependencies: tuple
    range_sources: tuple = ()
    loop_overhead: int = 0


@dataclass(frozen=True)
class BoundValue:
    """A lazy source read; assigning an unused local does not create a guard."""
    source: BindingSource
    value: object
    predicate_origin: bool = True


@dataclass(frozen=True)
class Literal:
    """Admitted literal metadata; only root integers can form shape predicates."""
    value: object
    predicate_origin: bool = True


@dataclass(frozen=True)
class ShapeValue:
    source: BindingSource
    axis: int | None = None
    predicate_origin: bool = True


@dataclass(frozen=True)
class InputTree:
    """Invocation-local admitted snapshot, never retained by a specialization."""
    kind: str
    items: tuple
    keys: tuple = ()


@dataclass(frozen=True, eq=False)
class Container:
    """A constructor identity in the frame, never a mutable Python result."""
    kind: str
    items: tuple
    keys: tuple = ()
    depth: int = 1
    identity: object = field(default_factory=object)


@dataclass(frozen=True)
class ResultSpec:
    """Immutable topology with no runtime Tensor owners or past dimensions."""
    # A topologically ordered DAG. Entries contain only source identities,
    # literal metadata, output slots and earlier entry indices.
    entries: tuple
    root: int
    # First observable occurrence of each computed root, projected onto the
    # canonical Graph slots. Numerical planning consumes this immutable order;
    # container kinds, keys, aliases and metadata never enter the executable key.
    output_order: tuple = ()

    def reconstruct(self, outputs, values, tensors, metadata):
        """Build fresh containers from this call's owners before cache publication."""
        objects = []
        for kind, payload in self.entries:
            if kind == "output":
                obj = outputs[payload]
            elif kind == "input":
                obj = tensors[values[payload].index]
            elif kind == "shape":
                source, axis = payload
                obj = metadata[values[source].index][0][axis]
            elif kind == "literal":
                obj = payload
            elif kind == "tuple":
                obj = tuple(objects[index] for index in payload)
            elif kind == "list":
                obj = [objects[index] for index in payload]
            else:
                obj = {key: objects[index] for key, index in payload}
            objects.append(obj)
        return objects[self.root]


@dataclass(frozen=True)
class Lowering:
    """Pair native computation with Python topology at the lowering cache owner."""
    graph: Graph
    result: ResultSpec


@dataclass(frozen=True)
class ShapePredicate:
    source: BindingSource
    axis: int
    comparison: str
    threshold: int
    outcome: bool

    def matches(self, metadata):
        shape = metadata[self.source][0]
        return (-len(shape) <= self.axis < len(shape)
                and _COMPARE[self.comparison](shape[self.axis], self.threshold) == self.outcome)


@dataclass(frozen=True)
class Branch:
    fallthrough: tuple
    jumped: tuple
    jump_when: bool


def walk_instructions(instructions):
    for instruction in instructions:
        if type(instruction) is Branch:
            yield from walk_instructions(instruction.fallthrough)
            yield from walk_instructions(instruction.jumped)
        else:
            yield instruction


def without_origin(obj, memo=None):
    """Remove helper-return predicate capability while preserving the result DAG.

    The shared memo retains constructor aliases and original container identity;
    literal metadata remains valid even when it cannot form a shape predicate.
    """
    if memo is None:
        memo = {}
    if type(obj) is Literal:
        return Literal(obj.value, False)
    if type(obj) is BoundValue:
        return BoundValue(obj.source, obj.value, False)
    if type(obj) is ShapeValue:
        return ShapeValue(obj.source, obj.axis, False)
    if type(obj) is Container:
        if id(obj) not in memo:
            memo[id(obj)] = Container(obj.kind, tuple(without_origin(item, memo) for item in obj.items),
                                      obj.keys, obj.depth, obj.identity)
        return memo[id(obj)]
    return obj


def validate_data(obj):
    """Check a data boundary without realizing or guarding a lazy source."""
    item = obj.value if type(obj) is BoundValue else obj
    if type(item) is InputTree:
        return obj
    if type(item) is Container:
        # Constructors have already validated every edge. Do not expand DAGs.
        return obj
    if type(item) is Literal:
        if item.value is None or type(item.value) is str:
            return obj
        item = item.value
    if type(item) is ShapeValue and item.axis is not None:
        return obj
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
    predicates: tuple = ()

    def matches(self, metadata):
        if not all(predicate.matches(metadata) for predicate in self.predicates):
            return False
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
    tensor_sources: tuple = ()
    # Sources passed through data boundaries need admission even if ignored by
    # the helper. These sources impose no scalar-value or tensor-shape guard.
    data_sources: tuple = ()
    # Non-guarding iteration hint from this successful specialization. Dynamic
    # shape hits retain its numerical partition without changing executable keys.
    numerical_hint: int = 1


def _broadcast_elements(shapes):
    shapes = iter(shapes)
    result = next(shapes, ())
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
    keys, first, aliases = {}, {}, []
    for source in observed:
        value = values[source]
        keys[source] = bindings[source]
        if type(value) is Value:
            tensor = tensors[value.index]
            owner = next((s for s, index in first.items() if tensors[index] is tensor), None)
            if owner is None:
                first[source] = value.index
                keys[source] = ("tensor", None)
                aliases.append(len(first)-1)
            else:
                keys[source] = ("alias", owner)
                aliases.append(tuple(first).index(owner))
    # Binding check order also defines the frozen runtime scalar slot order.
    return tuple((s, keys[s]) for s in bindings if s in keys), first, tuple(aliases)


def _shape_guards(program, graph, first, metadata, history, predicates=()):
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
    live = tuple(s for s, i in first.items() if graph.nodes[i][1] in set().union(*(dependencies[root] for root in graph.outputs)))
    groups = tuple((s,) for s in live) + ((live,) if len(live) == 2 else ())
    # torch/_inductor/codegen/simd.py: can_use_32bit_indexing installs an
    # upper-bound conjunction only if the whole kernel is 32-bit eligible.
    # A 64-bit specialization has no inverse bound and can accept small shapes.
    bounds = groups if all(_broadcast_elements([observations[s][0] for s in group]) <= 2147483647
                           for group in groups) else ()
    numerical_hint = _broadcast_elements([observations[s][0] for s in live])
    return (ShapeGuards(tuple(guards), tuple(equal_axes), bounds, tuple(predicates)),
            observations, numerical_hint)


def _select_specialization(program, bindings, values, tensors, metadata, graphs):
    """Select newest matching semantics before considering new scalar promotion.

    A generalized entry can supersede an older exact shape even when its native
    broadcast executable is absent. Conversely, a rank miss can expose an older
    static scalar entry beneath a newer runtime entry.
    """
    by_source = None
    for key, entry in reversed(graphs.items()):
        if key[0] is not program.code:
            continue
        scalars = []
        for source, expected in entry.binding_checks:
            current = bindings.get(source)
            if current is None:
                break
            if expected[0] == "runtime_float" and type(values[source]) is float:
                scalars.append(values[source])
            elif expected[0] in ("tensor", "alias"):
                if current[0] != "tensor":
                    break
            elif expected != current:
                break
        else:
            # Source realization order defines aliases, not public slot order.
            # Tensor kind checks above precede every current operand lookup.
            first, aliases = [], []
            for source in entry.tensor_sources:
                tensor = tensors[values[source].index]
                owner = next((i for i, previous in enumerate(first) if previous is tensor), None)
                if owner is None:
                    owner = len(first)
                    first.append(tensor)
                aliases.append(owner)
            if any(source not in values for source in entry.data_sources):
                continue
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
    for constant in code.co_consts:
        if type(constant) is tuple:
            # CPython <=3.13 stores BUILD_CONST_KEY_MAP keys in co_consts.
            # Check before disassembly can format even an unused object.
            if len(constant) > 4096 or any(type(key) is not str for key in constant):
                unsupported("constant tuples must contain bounded exact string keys")
        elif constant is not None and type(constant) is not str:
            scalar_bits(constant)


def instructions_for(code, *, helper=False):
    # Preserve the visible-instruction limit across versions with differing
    # inline CACHE layouts, without first allocating an unbounded tuple.
    instructions = tuple(islice(dis.get_instructions(code), 16385))
    if len(instructions) > 16384:
        unsupported("function exceeds pointwise instruction limit")
    for instruction in instructions:
        if (instruction.opname not in (_ALLOWED if helper else _ALLOWED | _LOOP_OPS | _BRANCH_OPS)
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
        if not _native._pointwise_namespace_keys_exact(namespace):
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
        elif op == "BINARY_SUBSCR" or (op == "BINARY_OP" and instruction.argrepr == "[]"):
            required, delta = 2, -1
        elif op == "UNPACK_SEQUENCE":
            if not 0 <= arg <= 4096:
                unsupported("unpack exceeds 4096 input reference limit")
            required, delta = 1, arg - 1
        elif op in ("BINARY_OP", "BINARY_ADD", "BINARY_SUBTRACT", "BINARY_MULTIPLY"):
            if op == "BINARY_OP" and instruction.argrepr not in ("+", "-", "*"):
                unsupported("unsupported loop binary operator")
            required, delta = 2, -1
        elif op in ("BUILD_TUPLE", "BUILD_LIST", "BUILD_MAP", "BUILD_CONST_KEY_MAP"):
            required = arg * 2 if op == "BUILD_MAP" else arg + (op == "BUILD_CONST_KEY_MAP")
            delta = 1 - required
        elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD"):
            required, delta = arg + 1, -arg
        elif op in ("COPY", "DUP_TOP"):
            required, delta = (arg if op == "COPY" else 1), 1
        elif op == "SWAP":
            required = arg
        elif op in _ROTATIONS:
            required = _ROTATIONS[op]
        else:
            unsupported("unsupported loop control flow: " + op)
        if required < 0 or depth < required:
            unsupported("invalid loop body stack")
        depth += delta
    if depth:
        unsupported("invalid loop body stack")


def normalize_control_flow(model, instructions):
    """Resolve root branch and loop regions on original offsets, then expand.

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
    # Branch regions share the original target map with loops. No expanded
    # instruction offset is ever interpreted as a control-flow target.
    branches = []
    for position, instruction in enumerate(compact):
        if instruction.opname not in _CONDITIONALS or position in covered:
            continue
        target = offsets.get(instruction.argval, -1)
        if target <= position or target >= len(compact):
            unsupported("branch requires a structured forward target")
        stop = target
        arm = compact[position + 1:target]
        other = ()
        if arm and arm[-1].opname in ("JUMP_FORWARD", "JUMP_ABSOLUTE"):
            stop = offsets.get(arm[-1].argval, -1)
            if stop < target or stop >= len(compact):
                unsupported("branch requires a forward join")
            arm, other = arm[:-1], compact[target:stop]
        for part in (arm, other):
            for index, item in enumerate(part):
                if item.opname in _LOOP_OPS | _CONDITIONALS | {"JUMP_FORWARD"}:
                    unsupported("nested branches and loops in branches are unsupported")
                if item.opname in ("RETURN_VALUE", "RETURN_CONST") and index != len(part)-1:
                    unsupported("unstructured branch return")
        if any(index in covered for index in _RANGE(position, stop)):
            unsupported("overlapping control-flow regions")
        branches.append((position, stop, Branch(tuple(arm), tuple(other),
                                               instruction.opname.endswith("IF_TRUE"))))
        covered.update(_RANGE(position, stop))
    for position, instruction in enumerate(compact):
        if instruction.opname in _LOOP_OPS | _CONDITIONALS | {"JUMP_FORWARD"} and position not in covered:
            unsupported("unsupported control flow: " + instruction.opname)
    validate_ranges(model, sources)
    if not regions and not branches:
        return instructions, (), 0
    replacements = list(branches)
    for start, stop, body, first, step, trips in regions:
        expanded = []
        if not trips:
            expanded.append(body[0]._replace(opname="_LOOP_CHECK_START"))
        for index in _RANGE(max(1, trips)):
            # Synthetic indices remain numerical constants, never root literals.
            expanded.append(body[0]._replace(opname="_LOOP_INDEX", argval=first + index * step))
            expanded.extend(body)
        if not trips:
            expanded.append(body[0]._replace(opname="_LOOP_CHECK_END"))
        replacements.append((start, stop, tuple(expanded)))
    result, cursor = [], 0
    for start, stop, replacement in sorted(replacements, key=lambda region: region[0]):
        result.extend(compact[cursor:start])
        result.extend((replacement,) if type(replacement) is Branch else replacement)
        cursor = stop
    result.extend(compact[cursor:])
    # Count original instructions and repeated loop bodies, including removed
    # setup and transparent opcodes; helper invocations share this budget.
    size = sum(1 for _ in walk_instructions(result)) + len(branches)
    return tuple(result), tuple(dict.fromkeys(sources)), expanded_size - size


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
    instructions, range_sources, loop_overhead = normalize_control_flow(model, instructions_for(code))
    # A conservative lazy source catalogue. The frame is the sole owner of
    # overwrites/unbound reads; observed-source projection supplies value guards.
    parameters = tuple(BindingSource("parameter", name, position)
                       for position, name in enumerate(code.co_varnames[:arity]))
    captures = tuple(dict.fromkeys(BindingSource(i.opname, i.argval) for i in walk_instructions(instructions)
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
    """Snapshot exact trees; retain every Tensor occurrence only for this call.

    Flat calls retain their direct, unbounded scalar-root admission. The bounded
    walker checks types before iteration/hash/lookup, and snapshots dict pairs
    before validating keys so no mutable dictionary is subsequently reread.
    """
    tensors, parameters = [], []
    for arg in args:
        kind = type(arg)
        if kind is _TENSOR_TYPE:
            parameters.append(Value(len(tensors)))
            tensors.append(arg)
            if len(tensors) > 2:
                unsupported("expected one or two positional tensor occurrences")
        elif kind is float or kind is bool:
            parameters.append(arg)
        elif kind is tuple or kind is list or kind is dict:
            break
        else:
            unsupported("default backend requires exact native CUDA float32 Tensor inputs "
                        "and exact float/bool leaves in bounded exact tuple/list/dict trees; "
                        "see docs/compile-pointwise-jit.md")
    else:
        if len(tensors) not in (1, 2):
            unsupported("expected one or two positional tensor occurrences")
        return tuple(tensors), tuple(parameters)

    # A container switches this invocation to the complete bounded walker.
    # Discard the prefix so every root and Tensor occurrence is counted once.
    tensors = []

    def leaf(arg):
        kind = type(arg)
        if kind is _TENSOR_TYPE:
            value = Value(len(tensors))
            tensors.append(arg)
            if len(tensors) > 2:
                unsupported("expected one or two positional tensor occurrences")
            return value
        if kind is float or kind is bool:
            return arg
        unsupported("default backend requires exact native CUDA float32 Tensor inputs "
                    "and exact float/bool leaves in bounded exact tuple/list/dict trees; "
                    "see docs/compile-pointwise-jit.md")

    seen, edges = set(), len(args)
    if edges > 4096:
        unsupported("input tree exceeds 4096 reference edges")

    def snapshot(arg, depth):
        nonlocal edges
        kind = type(arg)
        if kind is not tuple and kind is not list and kind is not dict:
            return leaf(arg)
        if depth >= 64:
            unsupported("input tree exceeds container depth 64")
        if id(arg) in seen:
            unsupported("repeated input container identity or cycle")
        seen.add(id(arg))
        width = len(arg)
        edges += width
        if edges > 4096:
            unsupported("input tree exceeds 4096 reference edges")
        if kind is dict:
            pairs = tuple(islice(arg.items(), 4097))
            if any(type(key) is not str for key, _ in pairs):
                unsupported("input dict keys must be exact strings")
            keys = tuple(key for key, _ in pairs)
            children = tuple(value for _, value in pairs)
        else:
            keys, children = (), tuple(islice(arg, 4097))
        # A concurrent growth cannot bypass the budget at expansion.
        if len(children) != width:
            unsupported("input container changed during admission")
        return InputTree(kind.__name__, tuple(snapshot(child, depth + 1) for child in children), keys)

    parameters = tuple(snapshot(arg, 0) for arg in args)
    if len(tensors) not in (1, 2):
        unsupported("expected one or two positional tensor occurrences")
    return tuple(tensors), parameters


def resolve(model, program, parameters=None):
    # FunctionType permits a dict subclass as globals. Never invoke its lookup
    # hooks, including on a warm call or before rejecting a later graph node.
    validate_signature_containers(model)
    validate_code(model.__code__, program.code.co_argcount)
    return _resolve_bindings(model, program, parameters)


def _resolve_bindings(model, program, parameters):
    """Resolve mutable bindings after the caller validates the root/signature."""
    validate_namespaces(model)
    validate_ranges(model, program.range_sources)
    globals_ = model.__globals__
    values, keys = {}, {}
    closure = dict(zip(program.code.co_freevars, model.__closure__ or ()))
    if parameters is None:
        # Private hardware-free lowering callers model a tensor-only signature.
        parameters = tuple(Value(i) for i in range(program.code.co_argcount))
    def parameter(source, value):
        values[source] = value
        if type(value) is InputTree:
            keys[source] = (value.kind,) if value.kind == "dict" else (value.kind, len(value.items))
            for item, child in zip(value.keys if value.kind == "dict" else range(len(value.items)), value.items):
                parameter(source.child(item), child)
        elif type(value) is Value:
            keys[source] = ("tensor", value.index)
        else:
            keys[source], values[source] = binding(value)

    for source in program.dependencies:
        kind, name = source.kind, source.name
        if kind == "parameter":
            parameter(source, parameters[source.position])
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
        keys[source] = key
        values[source] = value
    return keys, values


def runtime_bindings(program, bindings, values, graphs, *, observed=None):
    """Resolve a new specialization after guard misses, from successful history."""
    previous = [dict(key[1]) for key in graphs if key[0] is program.code]
    keys, resolved, scalars = dict(bindings), dict(values), []
    for dependency in values:
        value = values[dependency]
        if (observed is not None and dependency not in observed) or type(value) is not float:
            continue
        # A new reference trace specializes nonfinite values even after prior
        # promotion. An existing runtime guard can still accept them on a hit.
        if not math.isfinite(value):
            continue
        observations = [keys_[dependency] for keys_ in previous if dependency in keys_]
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
            keys[dependency] = ("runtime_float",)
    return keys, resolved, tuple(scalars)


def lower(program, values, arity, input_ids=None, *, observed=None, data_sources=None,
          metadata=(), predicates=None):
    if input_ids is None:
        input_ids = tuple(range(arity))
    nodes = [("input", i, 0, 0) for i in input_ids]
    runtime = sorted(value.index for value in values.values() if type(value) is RuntimeScalar)
    nodes.extend(("scalar", index, 0, 0) for index in runtime)
    remaining = 16384 - program.loop_overhead
    checked_nodes = 0
    helper_instructions = {}  # Per lowering only; warm hits never parse helpers.
    pending_checks = []  # Early-return continuations; no recursion per condition.
    construction_edges = 0

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
        return obj.value if type(obj) is Literal else obj

    def container(obj):
        item = realize(obj)
        if type(item) is not InputTree and type(item) is not Container:
            unsupported("selection/unpacking requires a tuple/list/dict container")
        return item

    def child(obj, item, index):
        if type(item) is InputTree:
            source = obj.source.child(item.keys[index] if item.kind == "dict" else index)
            return data(BoundValue(source, values[source], obj.predicate_origin))
        return item.items[index]

    def select(obj, selector):
        if type(selector) is not Literal:
            unsupported("container selection requires a literal index/key")
        item = container(obj)
        key = selector.value
        if item.kind == "dict":
            if type(key) is not str or key not in item.keys:
                unsupported("dict selection requires an existing exact literal string key")
            index = item.keys.index(key)
        else:
            if type(key) is not int or not -len(item.items) <= key < len(item.items):
                unsupported("sequence selection requires an in-range literal integer")
            index = key % len(item.items)
        return child(obj, item, index)

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

    def isolated(instructions, locals_, *, charge=True):
        nonlocal observed, data_sources, predicates, checked_nodes
        saved = observed, data_sources, predicates
        observed, data_sources, predicates = None, None, None
        node_start = len(nodes)
        try:
            return frame(instructions, locals_, check_only=True, charge=charge)
        finally:
            checked_nodes += len(nodes) - node_start
            del nodes[node_start:]
            observed, data_sources, predicates = saved

    def result_spec(obj):
        """Separate result topology from sorted distinct computed SSA roots.

        Repeated container identities are memoized; repeated computed roots map
        to one output slot. Neither returned input owners nor dimensions are kept.
        """
        entries, memo, roots = [], {}, set()

        def visit(item):
            if type(item) is Container:
                if item.identity in memo:
                    return memo[item.identity]
                children = tuple(visit(child) for child in item.items)
                payload = tuple(zip(item.keys, children)) if item.kind == "dict" else children
                entry = (item.kind, payload)
            elif type(item) is BoundValue:
                resolved = realize(item)
                if type(resolved) is InputTree:
                    unsupported("original input container result passthrough is unsupported")
                if type(resolved) is not Value or not resolved.tensor or item.source.kind != "parameter":
                    unsupported("result metadata must be literal or an input shape axis")
                entry = ("input", item.source)
            elif type(item) is Value and item.tensor and (item.index >= arity or item.index == -1):
                roots.add(item.index)
                entry = ("output", item.index)
            elif type(item) is Literal:
                entry = ("literal", item.value)
            elif type(item) is ShapeValue and item.axis is not None:
                entry = ("shape", (item.source, item.axis))
            else:
                unsupported("return computed pointwise tensors with bounded literal metadata")
            index = len(entries)
            entries.append(entry)
            if len(entries) > 4096:
                unsupported("result exceeds 4096 output reference limit")
            if type(item) is Container:
                memo[item.identity] = index
            return index

        root = visit(obj)
        if not roots:
            unsupported("return at least one computed pointwise tensor")
        if len(roots) > 64:
            unsupported("result exceeds 64 computed output limit")
        outputs = tuple(sorted(roots))
        slots = {value: slot for slot, value in enumerate(outputs)}
        entries = tuple((kind, slots[payload] if kind == "output" else payload)
                        for kind, payload in entries)
        order = tuple(dict.fromkeys(payload for kind, payload in entries if kind == "output"))
        return outputs, ResultSpec(entries, root, order)

    def root_result(obj):
        result_spec(obj)  # Also admit every inactive return before cache publication.
        return obj

    def frame(instructions, locals_, *, check_only=False, helper=False, charge=True):
        nonlocal remaining, checked_nodes, construction_edges
        remaining -= len(instructions) if charge else 0
        if remaining < 0:
            unsupported("expanded function exceeds pointwise instruction limit")
        stack = []

        def load(name):
            if name not in locals_:
                if not check_only:
                    unsupported("unbound local: " + name)
                # No value/type exists for an unexecuted local read. Treat it
                # as pointwise data solely for body-language admission. This
                # placeholder and every node using it are discarded before
                # native IR validation; it is never an input or a real local.
                stack.append(Value(-1))
            else:
                stack.append(locals_[name])

        for position, instruction in enumerate(instructions):
            if type(instruction) is Branch:
                if len(stack) != 1 or type(stack[0]) is not ShapePredicate:
                    unsupported("branch requires an input shape comparison")
                predicate = stack.pop()
                if predicates is not None:
                    predicates.append(predicate)
                taken = predicate.outcome == instruction.jump_when
                active = instruction.jumped if taken else instruction.fallthrough
                inactive = instruction.fallthrough if taken else instruction.jumped
                inactive_locals = locals_.copy()
                inactive_result = isolated(inactive, inactive_locals)
                result = frame(active, locals_, check_only=check_only)
                if result is not _MISSING:
                    if inactive_result is _MISSING:
                        pending_checks.append((instructions[position + 1:], inactive_locals))
                    return result
                continue
            op, arg = instruction.opname, instruction.argval
            if op in _IGNORED:
                continue
            if op == "_LOOP_CHECK_START":
                # Reuse typed operator/helper/data admission without executing
                # Python or changing the real frame/IR. Realized sources keep
                # the existing warm semantic and ignored-data guards.
                saved_locals, node_start, saved_check_only = locals_, len(nodes), check_only
                locals_ = locals_.copy()
                check_only = True
            elif op == "_LOOP_CHECK_END":
                checked_nodes += len(nodes) - node_start
                del nodes[node_start:]
                locals_ = saved_locals
                check_only = saved_check_only
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
            elif op in ("LOAD_CONST", "LOAD_SMALL_INT", "_LOOP_INDEX"):
                if op == "_LOOP_INDEX":
                    scalar_bits(arg)
                    stack.append(arg)
                elif type(arg) is tuple:
                    stack.append(arg)  # Admitted key tuples only; never output data.
                else:
                    if arg is not None and type(arg) is not str:
                        scalar_bits(arg)
                    stack.append(Literal(arg, not helper and type(arg) is int))
            elif op in ("BUILD_TUPLE", "BUILD_LIST", "BUILD_MAP", "BUILD_CONST_KEY_MAP"):
                count = instruction.arg
                needed = count * 2 if op == "BUILD_MAP" else count + (op == "BUILD_CONST_KEY_MAP")
                if count < 0 or len(stack) < needed:
                    unsupported("invalid output constructor stack")
                construction_edges += needed
                if construction_edges > 4096:
                    unsupported("result exceeds 4096 output construction edge limit")
                operands = stack[-needed:] if needed else []
                if needed:
                    del stack[-needed:]
                keys = ()
                if op == "BUILD_CONST_KEY_MAP":
                    keys = operands.pop()
                    if type(keys) is not tuple or len(keys) != count or any(type(key) is not str for key in keys):
                        unsupported("dict keys must be exact literal strings")
                elif op == "BUILD_MAP":
                    key_values = operands[::2]
                    if any(type(key) is not Literal or type(key.value) is not str for key in key_values):
                        unsupported("dict keys must be exact literal strings")
                    keys, operands = tuple(key.value for key in key_values), operands[1::2]
                items = tuple(data(item) for item in operands)
                if op in ("BUILD_MAP", "BUILD_CONST_KEY_MAP"):
                    # Assignment preserves the first insertion position and last value.
                    mapping = dict(zip(keys, items))
                    keys, items = tuple(mapping), tuple(mapping.values())
                depth = 1 + max((item.depth for item in items if type(item) is Container), default=0)
                if depth > 64:
                    unsupported("result exceeds output depth 64")
                stack.append(Container({"BUILD_TUPLE": "tuple", "BUILD_LIST": "list"}.get(op, "dict"),
                                       items, keys, depth))
            elif op in ("LOAD_GLOBAL", "LOAD_DEREF"):
                source = BindingSource(op, arg)
                stack.append(BoundValue(source, values[source]))
            elif op in ("LOAD_ATTR", "LOAD_METHOD"):
                original = stack.pop()
                owner = realize(original)
                if arg == "shape":
                    if (helper or type(original) is not BoundValue
                            or original.source.kind != "parameter" or type(owner) is not Value):
                        unsupported("shape queries require an original input Tensor")
                    stack.append(ShapeValue(original.source, predicate_origin=original.predicate_origin))
                elif owner is _ROOT:
                    if arg not in dict(_FUNCTIONS):
                        unsupported("unsupported native function: " + arg)
                    stack.append(binding(_ROOT.__dict__.get(arg))[1])
                elif isinstance(owner, Value) and owner.tensor and arg in (_UNARY | _BINARY):
                    stack.append(Call((_UNARY | _BINARY)[arg], owner, arg == "__rsub__"))
                else:
                    unsupported("unsupported attribute: " + str(arg))
            elif op == "UNARY_NEGATIVE":
                original = stack.pop()
                if type(original) is Literal:
                    scalar_bits(original.value)
                    scalar_bits(-original.value)
                    stack.append(Literal(-original.value, original.predicate_origin))
                    continue
                operand = realize(original)
                if type(operand) is RuntimeScalar:
                    stack.append(RuntimeScalar(operand.index, not operand.negative))
                elif isinstance(operand, Value):
                    stack.append(emit("neg", [operand]))
                else:
                    scalar_bits(operand)
                    stack.append(-operand)
            elif op == "BINARY_SUBSCR" or (op == "BINARY_OP" and instruction.argrepr == "[]"):
                axis, shape = stack.pop(), stack.pop()
                if type(shape) is not ShapeValue:
                    stack.append(select(shape, axis))
                    continue
                if (type(shape) is not ShapeValue or shape.axis is not None or type(axis) is not Literal
                        or not axis.predicate_origin or type(axis.value) is not int):
                    unsupported("only input.shape[literal integer axis] is supported")
                source = values[shape.source]
                if not metadata or type(source) is not Value:
                    unsupported("shape axis requires input metadata")
                if not -len(metadata[source.index][0]) <= axis.value < len(metadata[source.index][0]):
                    unsupported("shape axis is out of range")
                stack.append(ShapeValue(shape.source, axis.value, shape.predicate_origin))
            elif op == "UNPACK_SEQUENCE":
                count = instruction.arg
                if not 0 <= count <= 4096 or not stack:
                    unsupported("unpack exceeds 4096 input reference limit or invalid stack")
                obj = stack.pop()
                item = container(obj)
                if item.kind == "dict" or len(item.items) != count:
                    unsupported("fixed unpack requires a matching tuple/list length")
                stack.extend(child(obj, item, index) for index in reversed(range(count)))
            elif op == "COMPARE_OP":
                right, left = stack.pop(), stack.pop()
                comparison = arg  # 3.13+ argrepr may be bool(>).
                if type(left) is Literal and type(right) is ShapeValue and comparison in _COMPARE:
                    left, right, comparison = right, left, _REVERSE_COMPARE[comparison]
                if (type(left) is not ShapeValue or left.axis is None or not left.predicate_origin
                        or type(right) is not Literal or type(right.value) is not int or not right.predicate_origin
                        or comparison not in _COMPARE):
                    unsupported("condition requires input shape and exact integer literal")
                source = values[left.source]
                if not metadata or type(source) is not Value:
                    unsupported("shape predicate requires input metadata")
                shape = metadata[source.index][0]
                if not -len(shape) <= left.axis < len(shape):
                    unsupported("shape predicate axis is out of range")
                stack.append(ShapePredicate(left.source, left.axis, comparison, right.value,
                                            _COMPARE[comparison](shape[left.axis], right.value)))
            elif op == "TO_BOOL":
                if not stack or type(stack[-1]) is not ShapePredicate:
                    unsupported("only shape comparison truthiness is supported")
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
                    origins = {}
                    parameters = [without_origin(data(operand), origins) for operand in operands]
                    if target not in helper_instructions:
                        helper_instructions[target] = instructions_for(target.code, helper=True)
                    stack.append(frame(helper_instructions[target],
                                       dict(zip(target.code.co_varnames, parameters)),
                                       check_only=check_only, helper=True))
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
            elif op in _ROTATIONS:
                count = _ROTATIONS[op]
                if len(stack) < count:
                    unsupported("invalid rotation stack")
                stack[-count:] = stack[-1:] + stack[-count:-1]
            elif op in ("RETURN_VALUE", "RETURN_CONST"):
                result = data(stack.pop() if op == "RETURN_VALUE" else Literal(arg, not helper))
                if stack:
                    unsupported("return one data value")
                return without_origin(result) if helper else root_result(result)
        if stack:
            unsupported("unbalanced structured region stack")
        if helper:
            unsupported("missing data return")
        return _MISSING

    locals_ = {source.name: BoundValue(source, values[source]) for source in program.dependencies
               if source.kind == "parameter"}
    result = root_result(frame(program.instructions, locals_))
    while pending_checks:
        continuation, inactive_locals = pending_checks.pop()
        isolated(continuation, inactive_locals, charge=False)
    outputs, specification = result_spec(result)
    return Lowering(Graph(arity, tuple(nodes), outputs), specification)


def _prepared_entry_bytes(key, prepared):
    """Conservative incremental retention charge, calculated only on a miss.

    Native accounting uses actual Vec capacities and device allocation bytes.
    Recursively charge every Python key referent, including repeated/shared
    references (even the already-owned Graph), plus the wrapper and a 512-byte
    allowance for the dict slot/value pair/accounting integer. This is a data
    retention budget, not an exact Python allocator-resident-memory claim.
    """
    def key_bytes(value):
        if type(value) is tuple:
            return sys.getsizeof(value) + sum(key_bytes(item) for item in value)
        if type(value) is Graph:
            return (sys.getsizeof(value) + sys.getsizeof(value.__dict__)
                    + key_bytes(value.inputs) + key_bytes(value.nodes)
                    + key_bytes(value.outputs) + key_bytes(value._hash))
        return sys.getsizeof(value)

    return prepared.retained_bytes + sys.getsizeof(prepared) + key_bytes(key) + 512


def _publish_preparation(cache, key, prepared, retained_bytes, recompile_limit):
    """Called under the existing lock only after successful reconstruction."""
    # A prepared Arc must not retain a module whose executor was evicted. The
    # executable transaction above is authoritative, not an independent module cache.
    for old_key in tuple(cache.prepared):
        if old_key[0] not in cache.executors:
            _, old_bytes = cache.prepared.pop(old_key)
            cache.prepared_bytes -= old_bytes
    if retained_bytes <= _PREPARED_CACHE_BYTES:
        newest = (cache.prepared and next(reversed(cache.prepared)) == key
                  and next(reversed(cache.prepared.values()))[0] is prepared)
        if not newest:
            previous = cache.prepared.pop(key, None)
            if previous is not None:
                cache.prepared_bytes -= previous[1]
            cache.prepared[key] = (prepared, retained_bytes)
            cache.prepared_bytes += retained_bytes
    while (len(cache.prepared) > recompile_limit
           or cache.prepared_bytes > _PREPARED_CACHE_BYTES):
        _, old_bytes = cache.prepared.pop(next(iter(cache.prepared)))
        cache.prepared_bytes -= old_bytes


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
        metadata = _native._pointwise_admit_inputs(tensors)
        with cache.lock:
            if program is None or program.code is not model.__code__:
                program = analyze(model, len(args))
            validate_signature_containers(model)
            if program.code.co_argcount != len(args):
                unsupported("function signature changed")
            # analyze() admitted this exact immutable code/constant pool. The
            # identity check above repeats admission on replacement; mutable
            # signature containers still receive their one check on every call.
            # Private resolve() remains fully validating for independent callers.
            static_bindings, static_values = _resolve_bindings(model, program, parameters)
            input_ids = (0, 0) if len(tensors) == 2 and tensors[0] is tensors[1] else tuple(range(len(tensors)))
            selected = _select_specialization(program, static_bindings, static_values,
                                              tensors, metadata, cache.graphs)
            if selected is None:
                if len(cache.graphs) >= recompile_limit:
                    unsupported(f"hit recompile_limit={recompile_limit}")
                # The same lowering records which sources actually materialize,
                # including dead operations but excluding unused local bindings.
                observed = []
                data_sources = set()
                predicates = []
                lowering = lower(program, static_values, len(tensors), input_ids,
                              observed=observed, data_sources=data_sources, metadata=metadata, predicates=predicates)
                bindings, values, scalars = runtime_bindings(
                    program, static_bindings, static_values, cache.graphs, observed=observed)
                if scalars:
                    lowering = lower(program, values, len(tensors), input_ids, metadata=metadata)
                graph = lowering.graph
                bindings, first, aliases = _logical_keys(program, bindings, values, observed, tensors)
                guards, observations, numerical_hint = _shape_guards(
                    program, graph, first, metadata, cache.graphs, predicates)
                observations.update((s, "scalar") for s in observed
                                    if type(static_values[s]) is float or type(static_values[s]) is int)
                key = (program.code, bindings, guards, aliases)
                entry = Specialization(
                    {s: values[s] for s in observed if type(values[s]) is not InputTree},
                    tuple(observed), observations, {}, bindings,
                    tuple(source for source in observed if type(values[source]) is Value),
                    tuple(data_sources), numerical_hint)
            else:
                key, entry, scalars = selected
                # Ignored helper data still receives current admission on hits.
                for source in entry.data_sources:
                    validate_data(static_values[source])
            abi = (len(tensors), input_ids,
                   tuple(static_values[source].index for source in entry.tensor_sources))
            if selected is not None:
                lowering = entry.lowerings.get(abi)
                if lowering is None:
                    values = dict(static_values)
                    values.update((s, v) for s, v in entry.values.items() if type(v) is not Value)
                    lowering = lower(program, values, len(tensors), input_ids, metadata=metadata)
            # Logical guards choose scalar semantics. Concrete executables still
            # specialize the full native ABI and exact broadcast address formula.
            # Preparation checks all original-IR admission. Exact native shapes
            # certify that result on reuse; offsets/storage are checked each run.
            graph = lowering.graph
            shapes = tuple(m[0] for m in metadata)
            indexing_key = None if all(s == shapes[0] for s in shapes) else shapes
            code_key = (graph, metadata[0][4], indexing_key)
            executor = cache.executors.get(code_key)
            if executor is None:
                executor = _native._pointwise_compile(tensors, graph.nodes, graph.outputs)
            prepared_key = (code_key, shapes, entry.numerical_hint, lowering.result.output_order)
            cached_preparation = cache.prepared.get(prepared_key)
            if cached_preparation is None:
                prepared = executor.prepare(tensors, entry.numerical_hint, lowering.result.output_order)
                # Metadata was read before taking this lock. Bind the published
                # key to the same native snapshot that passed graph admission;
                # never label an artifact with a stale Python shape signature.
                if prepared.input_shapes != shapes:
                    unsupported("input shapes changed during pointwise preparation")
                retained_bytes = _prepared_entry_bytes(prepared_key, prepared)
            else:
                prepared, retained_bytes = cached_preparation
            outputs = prepared.run(tensors, scalars)
            result = lowering.result.reconstruct(outputs, static_values, tensors, metadata)
            # Publish every cache level only after success. Executable/lowering
            # eviction bounds retention without consuming logical slots.
            for mapping, item_key, item in ((entry.lowerings, abi, lowering),
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
            _publish_preparation(cache, prepared_key, prepared, retained_bytes, recompile_limit)
            return result

    compiled._torch_rs_pointwise_cache = cache
    return compiled
