"""Static bytecode to typed float32 SSA for the native default CUDA JIT.

The frontend never invokes the function, Tensor methods, or a Python operator
on user objects. Warm numerical calls resolve binding/metadata guards and enter
one native kernel; ordered input-rooted alias recipes construct views/effects.
It deliberately has no graph breaks or eager fallback.
"""
from dataclasses import dataclass, field
from typing import NamedTuple
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
    "BINARY_SUBSCR", "UNPACK_SEQUENCE", "POP_TOP", "KW_NAMES", "CALL_KW", "CALL_FUNCTION_KW"}
_METHODS = tuple(_UNARY) + tuple(_BINARY) + ("__getattribute__", "shape")
_MISSING = object()
_RANGE = range
# Retained numerical data per wrapper, not a process/module/allocator-pool or
# transient allocation budget. Larger admitted plans execute without retention.
_PREPARED_CACHE_BYTES = 32 * 1024 * 1024
_LOOP_OPS = {"GET_ITER", "FOR_ITER", "JUMP_ABSOLUTE", "JUMP_BACKWARD",
             "END_FOR", "POP_ITER"}
_CONDITIONALS = {"POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
                 "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_FORWARD_IF_TRUE"}
_BRANCH_OPS = _CONDITIONALS | {"JUMP_FORWARD", "COMPARE_OP", "BINARY_SUBSCR", "TO_BOOL"}
_COMPARE = {"<": operator.lt, "<=": operator.le, "==": operator.eq,
            "!=": operator.ne, ">=": operator.ge, ">": operator.gt}
_REVERSE_COMPARE = {"<": ">", "<=": ">=", "==": "==", "!=": "!=", ">=": "<=", ">": "<"}
# Imported during package initialization, before public bindings can be patched.
_FUNCTIONS = tuple((name, _ROOT.__dict__.get(name)) for name in
                   ("neg", "negative", "relu", "sin", "cos", "add", "sub", "subtract", "mul", "multiply"))
_METHOD_GUARDS = tuple((cls, tuple((name, cls.__dict__.get(name, _MISSING)) for name in _METHODS))
                      for cls in (_ROOT.Tensor, _ROOT.Tensor.__base__))


# Guard only alias operations admitted by this lowering, including inactive bodies.
_ALIAS_GUARDS = {name: tuple((cls, cls.__dict__.get(name, _MISSING))
                            for cls in (_ROOT.Tensor, _ROOT.Tensor.__base__))
                 for name in ("transpose", "view", "add_")}


def validate_alias_bindings(names):
    for name in names:
        for cls, expected in _ALIAS_GUARDS[name]:
            if cls.__dict__.get(name, _MISSING) is not expected:
                unsupported("patched Tensor operation binding: " + name)


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
class View:
    # Construction slot, never numerical SSA or a geometry-based identity.
    index: int


@dataclass(frozen=True)
class RuntimeScalar:
    index: int
    negative: bool = False


@dataclass(frozen=True)
class Call:
    op: str
    receiver: Value | tuple | None = None
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
    negative: bool = False
    numeric_unary: bool = False


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


def container_signature(value):
    return (value.kind,) if value.kind == "dict" else (value.kind, len(value.items))


@dataclass(frozen=True, eq=False)
class Container:
    """A constructor identity in the frame, never a mutable Python result."""
    kind: str
    items: tuple
    keys: tuple = ()
    depth: int = 1
    identity: object = field(default_factory=object)
    # Only exact admitted co_consts integer tuples; never an input container.
    constant: tuple | None = None


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

    def reconstruct(self, outputs, values, tensors, metadata, views=()):
        """Build dynamic containers from current owners, retaining literal identity."""
        objects = []
        for kind, payload in self.entries:
            if kind == "output":
                obj = outputs[payload]
            elif kind == "view":
                obj = views[payload]
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
    graph: Graph | None
    result: ResultSpec
    # Retained admission includes inactive bodies; one ordered selected sequence
    # controls execution. Sources/payloads are immutable slots, never owners.
    operations: tuple = ()
    operation_bindings: tuple = ()
    active_operations: tuple = ()
    # Each concrete ABI may admit different inactive paths. None means child
    # presence only; tensor signatures contain (kind, slot, minimum rank).
    # These signatures retain no invocation-local owners or exact dimensions.
    structural_admission: tuple = ()

    def admits(self, resolved, metadata):
        for source, expected in self.structural_admission:
            current = resolved.get(source)
            if current is None:
                return False
            if expected is not None:
                if expected[0] == "tensor":
                    if current != expected[:2] or len(metadata[current[1]][0]) < expected[2]:
                        return False
                elif current != expected:
                    return False
        return True


def preflight_operations(lowering, values, metadata):
    """Validate every retained operation before executing the selected sequence."""
    layouts, nodes = [], []
    slots = {index: slot for slot, index in enumerate(lowering.active_operations)}
    for index, (operation, kind, source, payload) in enumerate(lowering.operations):
        if kind == "input":
            value = values.get(source)
            if type(value) is not Value:
                unsupported("alias operation requires a current original input Tensor")
            current = metadata[value.index]
            shape, strides, offset = current[0], current[1], current[5]
            operand = value.index
        else:
            shape, strides, offset = layouts[source]
            if index in slots:
                operand = len(metadata) + slots[source]
        if operation == "add_scalar_":
            args = []
            for spec in payload:
                if spec[0] == "source":
                    value = values[spec[1]]
                    if not is_scalar(value):
                        unsupported("add_ requires exact scalar values")
                    if spec[3] and type(value) is bool:
                        value = int(value)  # Exact bool negation produces an integer.
                    args.append(-value if spec[2] else value)
                else:
                    args.append(spec[1])
            payload = tuple(args)
        if operation == "transpose":
            layout = _native._compile_trace_cuda_transpose_metadata(shape, strides, offset, payload)
        else:
            layout = _native._compile_trace_cuda_alias_metadata(shape, strides, offset, operation, payload)
        layouts.append(layout)
        if index in slots:
            nodes.append((operation, (operand,), payload, layout[0], layout[1]))
    return nodes


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
        return BoundValue(obj.source, obj.value, False, obj.negative, obj.numeric_unary)
    if type(obj) is ShapeValue:
        return ShapeValue(obj.source, obj.axis, False)
    if type(obj) is Container:
        if id(obj) not in memo:
            memo[id(obj)] = Container(obj.kind, tuple(without_origin(item, memo) for item in obj.items),
                                      obj.keys, obj.depth, obj.identity, obj.constant)
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
    if type(item) is not Value and type(item) is not View and type(item) is not RuntimeScalar:
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
        for size, expected in zip(shape, self.sizes):
            if (size < 2 if expected is None else size != expected):
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
        # Warm matching needs ordered short-circuit checks, not temporary
        # generator frames for each predicate, tensor, equality and bound group.
        for predicate in self.predicates:
            if not predicate.matches(metadata):
                return False
        for source, guard in self.tensors:
            if not guard.matches(metadata[source], metadata):
                return False
        for a, i, b, j in self.equal_axes:
            if metadata[a][0][i] != metadata[b][0][j]:
                return False
        for group in self.index_bounds:
            if not 0 <= _broadcast_elements([metadata[s][0] for s in group]) <= 2147483647:
                return False
        return True


class SpecializationPayload(NamedTuple):
    """Shallow-frozen facts owned by one logical capture, shared by its entries."""
    # Scalars/operators are frozen here; tensor Values use the current filtered
    # ABI when creating or re-admitting a lowering. No tensor is retained.
    values: dict
    observed: tuple
    observations: dict
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


class Specialization(NamedTuple):
    """Published shell retaining facts and semantic lowering admission history."""
    payload: SpecializationPayload
    lowerings: dict


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
        previous = [entry.payload.observations[source] for key, entry in history.items()
                    if key[0] is program.code and source in entry.payload.observations]
        guards.append((source, _tensor_guard(current, previous, duck_strides, source)))
        observations[source] = current
    # Graph topology supplies broadcast equalities and live iteration/buffer
    # bounds. Rust remains the owner of actual-shape admission at compile/run.
    dependencies, combined = [], False
    for op, a, b, _ in (() if graph is None else graph.nodes):
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
    live_inputs = set().union(*(dependencies[root] for root in graph.outputs)) if graph is not None else set()
    live = tuple(s for s, i in first.items() if graph.nodes[i][1] in live_inputs) if graph is not None else ()
    groups = tuple((s,) for s in live) + ((live,) if len(live) == 2 else ())
    # torch/_inductor/codegen/simd.py: can_use_32bit_indexing installs an
    # upper-bound conjunction only if the whole kernel is 32-bit eligible.
    # A 64-bit specialization has no inverse bound and can accept small shapes.
    bounds = groups if all(_broadcast_elements([observations[s][0] for s in group]) <= 2147483647
                           for group in groups) else ()
    numerical_hint = _broadcast_elements([observations[s][0] for s in live])
    return (ShapeGuards(tuple(guards), tuple(equal_axes), bounds, tuple(predicates)),
            observations, numerical_hint)


def _select_specialization(program, resolved, tensors, metadata, graphs):
    """Select newest matching semantics before considering new scalar promotion.

    A generalized entry can supersede an older exact shape even when its native
    broadcast executable is absent. Conversely, a rank miss can expose an older
    static scalar entry beneath a newer runtime entry.
    """
    values = resolved.values
    for key, entry in reversed(graphs.items()):
        if key[0] is not program.code:
            continue
        payload = entry.payload
        scalars = []
        for source, expected in payload.binding_checks:
            current = resolved.get(source)
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
            for source in payload.tensor_sources:
                tensor = tensors[values[source].index]
                owner = next((i for i, previous in enumerate(first) if previous is tensor), None)
                if owner is None:
                    owner = len(first)
                    first.append(tensor)
                aliases.append(owner)
            if any(resolved.get(source) is None for source in payload.data_sources):
                continue
            if tuple(aliases) != key[3]:
                continue
            # Each candidate can project additional sources. Shape guards use
            # its realized tensor sources, whose kinds were checked above.
            by_source = {s: metadata[values[s].index][:5] for s in payload.tensor_sources}
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
            if (len(constant) > 4096 or not (all(type(key) is str for key in constant)
                                              or all(type(key) is int for key in constant))):
                unsupported("constant tuples require bounded exact string keys or integer dimensions")
            for item in constant:
                if type(item) is int:
                    scalar_bits(item)
        elif constant is not None and type(constant) is not str:
            scalar_bits(constant)


def instructions_for(code, *, helper=False):
    # Preserve the visible-instruction limit across versions with differing
    # inline CACHE layouts, without first allocating an unbounded tuple.
    instructions = tuple(islice(dis.get_instructions(code), 16385))
    if len(instructions) > 16384:
        unsupported("function exceeds pointwise instruction limit")
    instructions = tuple(i._replace(argval=code.co_consts[i.arg]) if i.opname == "KW_NAMES" else i
                         for i in instructions)
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
        if op in _IGNORED or op == "KW_NAMES":
            continue
        if op == "POP_TOP":
            required, delta = 1, -1
        elif op.startswith("LOAD_FAST"):
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
        elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_KW", "CALL_FUNCTION_KW"):
            names = op in ("CALL_KW", "CALL_FUNCTION_KW")
            required, delta = arg + 1 + names, -arg - names
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

    seen, edges = set(), len(args)
    if edges > 4096:
        unsupported("input tree exceeds 4096 reference edges")

    def snapshot(arg, depth):
        nonlocal edges
        kind = type(arg)
        if kind is _TENSOR_TYPE:
            value = Value(len(tensors))
            tensors.append(arg)
            if len(tensors) > 2:
                unsupported("expected one or two positional tensor occurrences")
            return value
        if kind is float or kind is bool:
            return arg
        if kind is not tuple and kind is not list and kind is not dict:
            unsupported("default backend requires exact native CUDA float32 Tensor inputs "
                        "and exact float/bool leaves in bounded exact tuple/list/dict trees; "
                        "see docs/compile-pointwise-jit.md")
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
            if len(pairs) != width:
                unsupported("input container changed during admission")
            keys = tuple(key for key, _ in pairs)
            # The pairs already own a consistent snapshot. Do not project a
            # second temporary tuple of values before admitting its children.
            return InputTree("dict", tuple(snapshot(value, depth + 1) for _, value in pairs), keys)
        # An exact tuple is already immutable; only lists need a child snapshot.
        children = arg if kind is tuple else tuple(islice(arg, 4097))
        # A concurrent growth cannot bypass the budget at expansion.
        if len(children) != width:
            unsupported("input container changed during admission")
        return InputTree(kind.__name__, tuple(snapshot(child, depth + 1) for child in children))

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


class _BindingResolution:
    """Invocation-local projection of admitted snapshots onto immutable sources.

    Captures remain eager. Warm guards supply their existing source identities;
    lowering enumerates the complete snapshot in the original binding order.
    Neither this resolver nor its snapshots belong to a retained cache entry.
    """
    def __init__(self, program):
        self.program = program
        self.keys, self.values = {}, {}

    def bind(self, source, value):
        self.values[source] = value
        if type(value) is InputTree:
            self.keys[source] = container_signature(value)
        elif type(value) is Value:
            self.keys[source] = ("tensor", value.index)
        else:
            self.keys[source], self.values[source] = binding(value)

    def expand(self, source, value):
        self.bind(source, value)
        if type(value) is InputTree:
            for item, child in zip(value.keys if value.kind == "dict" else range(len(value.items)), value.items):
                self.expand(source.child(item), child)

    def dependencies(self, model, parameters, parameter):
        validate_namespaces(model)
        validate_ranges(model, self.program.range_sources)
        globals_ = model.__globals__
        closure = dict(zip(self.program.code.co_freevars, model.__closure__ or ()))
        if parameters is None:
            # Private hardware-free lowering callers model a tensor-only signature.
            parameters = tuple(Value(i) for i in range(self.program.code.co_argcount))
        for source in self.program.dependencies:
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
            self.keys[source], self.values[source] = binding(value)

    def get(self, source):
        key = self.keys.get(source)
        if key is not None:
            return key
        if source.kind != "parameter" or not source.path:
            return None
        root = self.program.dependencies[source.position]
        if root.name != source.name:
            return None
        value = self.values[root]
        # Only read the admitted snapshot, never the caller's mutable containers.
        # A stale path is a guard miss, not an input-admission failure.
        for item in source.path:
            if type(value) is not InputTree:
                return None
            if value.kind == "dict":
                if type(item) is not str:
                    return None
                try:
                    index = value.keys.index(item)
                except ValueError:
                    return None
            else:
                if type(item) is not int or not 0 <= item < len(value.items):
                    return None
                index = item
            value = value.items[index]
        self.bind(source, value)
        return self.keys[source]

    def complete(self):
        # Guard lookup order must not become scalar-promotion/ABI order. Rebuild
        # the full maps in dependency/DFS order, keeping already-frozen captures.
        keys, values = self.keys, self.values
        self.keys, self.values = {}, {}
        for source in self.program.dependencies:
            if source.kind == "parameter":
                self.expand(source, values[source])
            else:
                self.keys[source], self.values[source] = keys[source], values[source]
        return self.keys, self.values


def _resolve_bindings(model, program, parameters):
    """Resolve all mutable bindings for private lowering callers."""
    resolved = _BindingResolution(program)
    resolved.dependencies(model, parameters, resolved.expand)
    return resolved.keys, resolved.values


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
    # Per lowering; matching retained-lowering reuse does not parse helpers.
    helper_instructions = {}
    pending_checks = []  # Early-return continuations; no recursion per condition.
    construction_edges = 0
    operations, active_operations = [], []
    operation_bindings = set()
    structural_admission = {}
    has_effect = has_numerical = False
    returns = []
    literal_tuples = {}  # Constant-pool identity, local to this lowering.

    def integer_tuple(items):
        nonlocal construction_edges
        identity = id(items)
        if identity not in literal_tuples:
            construction_edges += len(items)
            if construction_edges > 4096:
                unsupported("result exceeds 4096 output construction edge limit")
            literal_tuples[identity] = Container("tuple", tuple(Literal(x) for x in items),
                                                 constant=items)
        return literal_tuples[identity]

    def data(obj):
        # Validate without realizing a lazy root source. Ignored parameters must
        # not acquire guards, but cannot carry callable/module/higher-order data.
        validate_data(obj)
        if data_sources is not None and type(obj) is BoundValue:
            data_sources.add(obj.source)
        return obj

    def realize(obj):
        if type(obj) is BoundValue:
            if type(obj.value) is InputTree:
                structural_admission[obj.source] = container_signature(obj.value)
            if observed is not None and obj.source not in observed:
                observed.append(obj.source)
            return obj.value
        return obj.value if type(obj) is Literal else obj

    def admit_tensor(source, value, minimum_rank=0):
        previous = structural_admission.get(source)
        if previous is not None:
            minimum_rank = max(minimum_rank, previous[2])
        structural_admission[source] = ("tensor", value.index, minimum_rank)

    def container(obj):
        item = realize(obj)
        if type(item) is not InputTree and type(item) is not Container:
            unsupported("selection/unpacking requires a tuple/list/dict container")
        return item

    def child(obj, item, index):
        if type(item) is InputTree:
            source = obj.source.child(item.keys[index] if item.kind == "dict" else index)
            structural_admission.setdefault(source, None)
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
        nonlocal has_numerical
        has_numerical = True
        args = [value(arg) for arg in operands]
        if len(args) == 1 and not args[0].tensor or not any(arg.tensor for arg in args):
            unsupported("operators require tensor expressions")
        nodes.append((op, args[0].index, args[1].index if len(args) == 2 else 0, 0))
        if len(nodes) + len(operations) + checked_nodes > 4096:
            unsupported("graph exceeds 4096-node limit")
        return Value(len(nodes) - 1)

    def effect_scalar(obj):
        value = realize(obj)
        if type(obj) is BoundValue:
            if not is_scalar(value) and type(value) is not RuntimeScalar:
                unsupported("add_ requires exact bool/int/float scalars")
            return ("source", obj.source, obj.negative, obj.numeric_unary)
        if type(value) is RuntimeScalar:
            source = next(source for source, v in values.items()
                          if type(v) is RuntimeScalar and v.index == value.index)
            return ("source", source, value.negative, False)
        if not is_scalar(value):
            unsupported("add_ requires exact bool/int/float scalars")
        return ("constant", value)

    def isolated(instructions, locals_, *, charge=True, snapshot=False):
        nonlocal observed, data_sources, predicates, checked_nodes
        saved = observed, data_sources, predicates
        observed, data_sources, predicates = None, None, None
        node_start = len(nodes)
        try:
            result = frame(instructions, locals_, check_only=True, charge=charge)
            return (result, tuple(nodes)) if snapshot else result
        finally:
            # View slots remain globally unique for immutable warm preflight.
            # Only numerical instructions are discarded after body admission.
            checked_nodes += len(nodes) - node_start
            del nodes[node_start:]
            observed, data_sources, predicates = saved

    def result_spec(obj, view_slots=None, *, require_root=True, numerical=None):
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
                entry = ("literal", item.constant) if item.constant is not None else (item.kind, payload)
            elif type(item) is BoundValue:
                resolved = realize(item)
                if type(resolved) is InputTree:
                    unsupported("original input container result passthrough is unsupported")
                if type(resolved) is not Value or not resolved.tensor or item.source.kind != "parameter":
                    unsupported("result metadata must be literal or an input shape axis")
                # Inactive aliases require current tensor admission on reuse,
                # independently of the active path's logical shape guards.
                admit_tensor(item.source, resolved)
                entry = ("input", item.source)
            elif type(item) is Value and item.tensor and (item.index >= arity or item.index == -1):
                roots.add(item.index)
                entry = ("output", item.index)
            elif type(item) is View:
                entry = ("view", item.index if view_slots is None else view_slots[item.index])
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
        if not roots and require_root:
            if numerical is None:
                numerical = any(node[0] in set(_UNARY.values()) | set(_BINARY.values()) for node in nodes)
            if numerical or (not has_effect and not any(kind == "view" for kind, _ in entries)):
                unsupported("return at least one computed pointwise tensor or a purely input-rooted view")
        if len(roots) > 64:
            unsupported("result exceeds 64 computed output limit")
        outputs = tuple(sorted(roots))
        slots = {value: slot for slot, value in enumerate(outputs)}
        entries = tuple((kind, slots[payload] if kind == "output" else payload)
                        for kind, payload in entries)
        order = tuple(dict.fromkeys(payload for kind, payload in entries if kind == "output"))
        return outputs, ResultSpec(entries, root, order)

    def root_result(obj):
        result_spec(obj, require_root=False)
        # Preserve the path-local numerical admission when checking returns
        # after all bodies have established whether the program has effects.
        numerical = any(node[0] in set(_UNARY.values()) | set(_BINARY.values()) for node in nodes)
        returns.append((obj, numerical))
        return obj

    def frame(instructions, locals_, *, check_only=False, helper=False, charge=True):
        nonlocal remaining, checked_nodes, construction_edges, has_effect
        remaining -= len(instructions) if charge else 0
        if remaining < 0:
            unsupported("expanded function exceeds pointwise instruction limit")
        stack = []
        keywords = ()

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
                inactive_result, inactive_nodes = isolated(inactive, inactive_locals, snapshot=True)
                result = frame(active, locals_, check_only=check_only)
                if result is not _MISSING:
                    if inactive_result is _MISSING:
                        pending_checks.append((instructions[position + 1:], inactive_locals, inactive_nodes))
                    return result
                continue
            op, arg = instruction.opname, instruction.argval
            if op in _IGNORED:
                continue
            if op == "KW_NAMES":
                keywords = arg
            elif op == "POP_TOP":
                discarded = stack.pop()
                if not (type(discarded) is View or (type(discarded) is BoundValue
                        and type(discarded.value) is Value)):
                    unsupported("only alias/effect results may be discarded")
            elif op == "_LOOP_CHECK_START":
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
                    stack.append(integer_tuple(arg)
                                 if all(type(x) is int for x in arg) else arg)
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
                    admit_tensor(original.source, owner)
                    stack.append(ShapeValue(original.source, predicate_origin=original.predicate_origin))
                elif owner is _ROOT:
                    if arg not in dict(_FUNCTIONS):
                        unsupported("unsupported native function: " + arg)
                    stack.append(binding(_ROOT.__dict__.get(arg))[1])
                elif arg in _ALIAS_GUARDS:
                    validate_alias_bindings((arg,))
                    operation_bindings.add(arg)
                    if type(owner) is View:
                        source = ("view", owner.index)
                    elif (type(original) is BoundValue and original.source.kind == "parameter"
                          and type(owner) is Value and 0 <= owner.index < arity):
                        source = ("input", original.source)
                    else:
                        unsupported("alias operations require an original input or input-rooted view")
                    stack.append(Call(arg, source))
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
                    # Keep the source when bool negation produces an exact int.
                    # Helpers/constructors must not turn runtime input leaves
                    # into constants eligible for metadata-only operations.
                    stack.append(BoundValue(original.source, -operand, False, not original.negative, True)
                                 if type(original) is BoundValue else -operand)
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
                admit_tensor(shape.source, source, axis.value + 1 if axis.value >= 0 else -axis.value)
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
            elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_KW", "CALL_FUNCTION_KW"):
                if op in ("CALL_KW", "CALL_FUNCTION_KW"):
                    keywords = stack.pop()
                if (type(keywords) is not tuple or any(type(name) is not str for name in keywords)
                        or len(keywords) > instruction.arg):
                    unsupported("keyword names must be exact literal strings")
                operands = [stack.pop() for _ in range(instruction.arg)][::-1]
                target = realize(stack.pop())
                if keywords and (type(target) is not Call or target.op != "add_"):
                    unsupported("keywords are supported only for scalar add_")
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
                if target.op in _ALIAS_GUARDS:
                    if target.op == "transpose":
                        payload = tuple(realize(operand) for operand in operands)
                        if (len(payload) != 2 or any(type(axis) is not int for axis in payload)
                                or any(type(operand) is BoundValue and operand.source.kind == "parameter"
                                       for operand in operands)):
                            unsupported("transpose requires two exact integer constants")
                    elif target.op == "view":
                        if len(operands) == 1 and type(operands[0]) is Container:
                            shape = operands[0]
                            if shape.kind != "tuple":
                                unsupported("view shape requires an exact integer tuple")
                            operands = shape.items
                        payload = tuple(realize(operand) for operand in operands)
                        if (not operands and instruction.arg == 0) or any(type(x) is not int for x in payload):
                            unsupported("view requires literal integer dimensions")
                        if any(type(x) is BoundValue and x.source.kind == "parameter" for x in operands):
                            unsupported("view dimensions must be constants")
                    else:
                        has_effect = True
                        positional = len(operands) - len(keywords)
                        if positional > 1 or len(set(keywords)) != len(keywords) or any(k not in ("other", "alpha") for k in keywords):
                            unsupported("add_ expects other and keyword-only alpha")
                        supplied = dict(zip(keywords, operands[positional:]))
                        if positional:
                            if "other" in supplied:
                                unsupported("add_ got duplicate other")
                            supplied["other"] = operands[0]
                        if "other" not in supplied:
                            unsupported("add_ requires other")
                        payload = tuple(effect_scalar(x) for x in (supplied["other"], supplied.get("alpha", Literal(1))))
                    keywords = ()
                    operation = "add_scalar_" if target.op == "add_" else target.op
                    operations.append((operation, *target.receiver, payload))
                    if len(nodes) + len(operations) + checked_nodes > 4096:
                        unsupported("graph exceeds 4096-node limit")
                    if not check_only:
                        active_operations.append(len(operations) - 1)
                    kind, source = target.receiver
                    result = (BoundValue(source, values[source]) if kind == "input" else View(source)) if target.op == "add_" else View(len(operations) - 1)
                    stack.append(result)
                    continue
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
                constant = (integer_tuple(arg)
                            if op == "RETURN_CONST" and type(arg) is tuple
                            and all(type(x) is int for x in arg) else Literal(arg, not helper))
                result = data(stack.pop() if op == "RETURN_VALUE" else constant)
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
        continuation, inactive_locals, inactive_nodes = pending_checks.pop()
        # An early-return continuation belongs to the inactive path's prefix,
        # never to the numerical operations or view slots of the returned arm.
        active_nodes = nodes[:]
        # Keep the shared node budget constant while swapping path prefixes:
        # active instructions remain charged, and already-checked inactive
        # prefix nodes must not be charged twice when restored here.
        budget_delta = len(active_nodes) - len(inactive_nodes)
        checked_nodes += budget_delta
        nodes[:] = inactive_nodes
        try:
            isolated(continuation, inactive_locals, charge=False)
        finally:
            nodes[:] = active_nodes
            checked_nodes -= budget_delta
    # Root returns already recorded observations in their own active/isolated
    # frames. Deferred validation must not promote inactive aliases into guards.
    saved_observed = observed
    observed = None
    try:
        for returned, numerical in returns:
            result_spec(returned, numerical=numerical)
    finally:
        observed = saved_observed
    if has_effect and has_numerical:
        unsupported("mutation-bearing programs cannot contain numerical Tensor operations")
    outputs, specification = result_spec(result, {index: slot for slot, index in enumerate(active_operations)})
    return Lowering(Graph(arity, tuple(nodes), outputs) if outputs else None,
                    specification, tuple(operations), tuple(sorted(operation_bindings)),
                    tuple(active_operations), tuple(structural_admission.items()))


def _prepared_entry_bytes(key, prepared, code_key):
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

    return (prepared.retained_bytes + sys.getsizeof(prepared)
            + key_bytes(key) + key_bytes(code_key) + 512)


def _stage_recent(mapping, key, value, limit):
    """Copy only changed logical maps; published maps remain immutable."""
    newest = None
    if mapping:
        newest = next(reversed(mapping))
        if newest == key and next(reversed(mapping.values())) is value:
            return mapping
    staged = mapping.copy()
    # Overwriting the identical newest key preserves its identity and order.
    if not mapping or newest is not key:
        staged.pop(key, None)
    staged[key] = value
    while len(staged) > limit:
        del staged[next(iter(staged))]
    return staged


def _publish_preparation(cache, key, prepared, retained_bytes, recompile_limit, code_key, executor, prepared_keys):
    """Called under the existing lock only after successful reconstruction."""
    # A prepared Arc must not retain a module whose executor was evicted. The
    # executable transaction above is authoritative, not an independent module cache.
    for old_key in prepared_keys:
        _, old_bytes, old_code, old_executor = cache.prepared[old_key]
        if cache.executors.get(old_code) is not old_executor:
            cache.prepared.pop(old_key)
            cache.prepared_bytes -= old_bytes
    if retained_bytes <= _PREPARED_CACHE_BYTES:
        newest = (cache.prepared and next(reversed(cache.prepared)) == key
                  and next(reversed(cache.prepared.values()))[0] is prepared)
        if not newest:
            previous = cache.prepared.pop(key, None)
            if previous is not None:
                cache.prepared_bytes -= previous[1]
            cache.prepared[key] = (prepared, retained_bytes, code_key, executor)
            cache.prepared_bytes += retained_bytes
    while (len(cache.prepared) > recompile_limit
           or cache.prepared_bytes > _PREPARED_CACHE_BYTES):
        _, old_bytes, _, _ = cache.prepared.pop(next(iter(cache.prepared)))
        cache.prepared_bytes -= old_bytes


def _receipt(result, prepared):
    # Allocate before publication, including warm selected-invocation receipts.
    return result, prepared


def implementation(model, recompile_limit):
    # Shared reset registry/lock discipline, but no eager graph evaluator.
    cache = _state.new_native_eager_compile_cache()
    program = None

    def execute(args, kwargs, want_receipt):
        nonlocal program
        if kwargs:
            unsupported("expected positional arguments without keywords")
        if type(_ROOT) is not types.ModuleType:
            unsupported("patched native package type")
        tensors, parameters = bind_arguments(args)
        if _ROOT.overrides._get_current_function_mode() is not None:
            unsupported("active __torch_function__ mode")
        mismatch = _native._pointwise_method_identity_mismatch(_METHOD_GUARDS, _MISSING)
        if mismatch is not None:
            unsupported("patched Tensor operation binding: " + mismatch)
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
            resolved = _BindingResolution(program)
            resolved.dependencies(model, parameters, resolved.bind)
            static_values = resolved.values
            input_ids = (0, 0) if len(tensors) == 2 and tensors[0] is tensors[1] else tuple(range(len(tensors)))
            selected = _select_specialization(program, resolved, tensors, metadata, cache.graphs)
            if selected is None:
                static_bindings, static_values = resolved.complete()
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
                payload = SpecializationPayload(
                    {s: values[s] for s in observed if type(values[s]) is not InputTree},
                    tuple(observed), observations, bindings,
                    tuple(source for source in observed if type(values[source]) is Value),
                    tuple(data_sources), numerical_hint)
                entry = Specialization(payload, {})
            else:
                key, entry, scalars = selected
                payload = entry.payload
                # Ignored helper data still receives current admission on hits.
                for source in payload.data_sources:
                    validate_data(static_values[source])
            abi = (len(tensors), input_ids,
                   tuple(static_values[source].index for source in payload.tensor_sources))
            if selected is not None:
                lowering = entry.lowerings.get(abi)
                if lowering is None or not lowering.admits(resolved, metadata):
                    _, static_values = resolved.complete()
                    values = dict(static_values)
                    values.update((s, v) for s, v in payload.values.items() if type(v) is not Value)
                    lowering = lower(program, values, len(tensors), input_ids, metadata=metadata)
            # Logical guards choose scalar semantics. Concrete executables still
            # specialize the full native ABI and exact broadcast address formula.
            # Preparation checks all original-IR admission. Exact native shapes
            # certify that result on reuse; offsets/storage are checked each run.
            if lowering.operation_bindings:
                validate_alias_bindings(lowering.operation_bindings)
            view_nodes = preflight_operations(lowering, static_values, metadata) if lowering.operations else ()
            outputs = ()
            prepared = None
            if lowering.graph is not None:
                graph = lowering.graph
                shapes = tuple(m[0] for m in metadata)
                indexing_key = None if all(s == shapes[0] for s in shapes) else shapes
                logical_code_key = (graph, metadata[0][4], indexing_key)
                prepared_key = (logical_code_key, shapes, payload.numerical_hint, lowering.result.output_order)
                cached_preparation = cache.prepared.get(prepared_key)
                if cached_preparation is not None:
                    prepared, retained_bytes, code_key, executor = cached_preparation
                    if cache.executors.get(code_key) is not executor:
                        cached_preparation = None
                if cached_preparation is None:
                    host = _native._pointwise_host_plan(
                        tensors, graph.nodes, graph.outputs,
                        payload.numerical_hint, lowering.result.output_order)
                    code_key = host.executable_identity
                    executor = cache.executors.get(code_key)
                    if executor is None:
                        executor = host.compile()
                    prepared = executor.bind(host)
                    # Drop construction-only graph, source, words and shape vectors.
                    del host
                    if prepared.input_shapes != shapes:
                        unsupported("input shapes changed during pointwise preparation")
                    # Frozen native owners preserve this relation for the lifetime
                    # of a frontend-admitted tuple. Hits still check the map owner.
                    if not prepared.belongs_to(executor):
                        unsupported("prepared executable owner mismatch")
                    retained_bytes = _prepared_entry_bytes(prepared_key, prepared, code_key)
                outputs = prepared.run(tensors, scalars)
            if lowering.active_operations:
                views = _native._compile_trace_cuda_graph(tensors, view_nodes)
                result = lowering.result.reconstruct(outputs, static_values, tensors, metadata, views)
            else:
                result = lowering.result.reconstruct(outputs, static_values, tensors, metadata)
            if want_receipt:
                result = _receipt(result, prepared)
            if lowering.graph is not None:
                # Stage the bounded owner scan before any retained-map mutation,
                # including warm same-key owner replacement. Cleanup runs after
                # publication so it sees the final executor owners and evictions.
                prepared_keys = tuple(cache.prepared)
            lowerings = _stage_recent(entry.lowerings, abi, lowering, recompile_limit)
            if lowerings is not entry.lowerings:
                entry = Specialization(payload, lowerings)
            graphs = _stage_recent(cache.graphs, key, entry, recompile_limit)
            if lowering.graph is not None:
                try:
                    mapping = cache.executors
                    # Executor recency is independent of logical recency.
                    if not (mapping and next(reversed(mapping)) == code_key
                            and next(reversed(mapping.values())) is executor):
                        mapping.pop(code_key, None)
                        mapping[code_key] = executor
                    while len(mapping) > recompile_limit:
                        del mapping[next(iter(mapping))]
                    _publish_preparation(cache, prepared_key, prepared, retained_bytes, recompile_limit, code_key, executor, prepared_keys)
                except BaseException:
                    cache._clear_derived_locked()
                    raise
            # The sole logical commit follows every fallible publication step.
            cache.graphs = graphs
            return result

    def compiled(*args, **kwargs):
        return execute(args, kwargs, False)

    def receipt(*args, **kwargs):
        return execute(args, kwargs, True)

    compiled._torch_rs_pointwise_receipt = receipt
    compiled._torch_rs_pointwise_cache = cache
    return compiled
