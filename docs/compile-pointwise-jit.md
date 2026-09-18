# Native default CUDA pointwise compilation

`torch_rs.compile(fn)` with untouched public defaults now lowers a bounded
pointwise language to a generated fused CUDA kernel and supports terminal
input-rooted transpose views. Pure-view results need no numerical kernel.
Backend resolution still
chooses `inductor`; this native implementation does not import or execute
PyTorch. It is not general Inductor, CPU compiler, or training equivalence.

Start with the example and [supported programs](#supported-programs). For setup
and repeated calls, see [runtime requirements](#runtime-requirements) and
[cache behavior](#cache-behavior). The [pipeline](#compiler-pipeline) and
[numerical contract](compile-pointwise-numerics.md) are maintainer references.

## Example

```python
import torch_rs as torch

def pointwise(x, y):
    wave = torch.sin(x)
    return (wave * wave + y.cos() - 0.375).relu()

x = torch.tensor([0.2, -1.3], dtype=torch.float32).to("cuda:0")
y = torch.tensor([1.1, 0.7], dtype=torch.float32).to("cuda:0")
result = torch.compile(pointwise)(x, y)
```

## Supported programs

The function accepts one or two exact native CUDA float32 Tensor inputs on the
same device with contiguous storage, plus exact built-in `float` and `bool`
arguments in any positional slots, including leaves of the bounded input trees
below. For equal-shaped tensors:

```python
def f(scale, x, enabled, y):
    return x * scale + y * enabled
```

Positional integers, numeric subclasses, keyword arguments, default expansion and scalar-only
programs are unsupported. Equal-shape inputs support
local intermediate variables, reused expressions, binary
add/subtract/multiply (tensor/tensor or tensor/scalar in either order), and
unary negation/ReLU/sin/cos. Operator syntax, positional Tensor methods, and
positional native top-level functions are accepted. Exact bool/int/float
constants may be literal, module-global, or closure values; native operator
and package bindings may also be captured. Function globals must be an exact
`dict` with exact string keys; custom globals mappings and keys are rejected
before any lookup hooks can execute,
including on repeated calls. Scalar admission uses type identity, and the complete
constant pool is validated before disassembly can format any constants. Rejected
objects cannot execute metaclass equality or representation callbacks. Signature
defaults must be absent or empty exact containers, and closures must use an exact tuple; container subclasses are
rejected before truthiness or iteration hooks can execute. Integers must fit the native scalar range
`[-2**63, 2**64-1]`; float32 overflow rounds to signed infinity. Graphs
are bounded to 4096 nodes and 16384 bytecode instructions. Return a computed Tensor
or a bounded nested result as described below.
Scalars and empty tensor shapes and contiguous views with storage offsets are
supported. Computed outputs have fresh storage and canonical contiguous strides;
returned input aliases preserve their original storage and strides. Inputs are
unchanged. Input-only returns remain outside this subset.

Terminal `Tensor.transpose(dim0, dim1)` results are also admitted when rooted
in an original input Tensor leaf or an earlier such transpose, at rank 0, 1 or 2.
Axes must be two positional exact integer constants (literal, local, global or
closure). Runtime input leaves remain ineligible axes through unary negation,
local assignment, helper forwarding and container selection, even when negation
turns a boolean into an exact integer. Nested results may mix these views with original aliases, computed
outputs and the existing literal/current-input-shape metadata. Repeated
references to a construction share one Python wrapper; distinct constructions,
including equal-axis and inverse transposes, create distinct wrappers sharing
the original storage. A function returning only metadata/views must return at
least one view and perform no numerical Tensor operation. Mixed numerical
functions still require a computed output root.

Computed-source transposes, arithmetic or shape queries consuming a view,
`t`, `view`, `reshape`, `contiguous`, top-level transpose, keyword/runtime axes,
strided inputs and training remain outside the default subset. All original
inputs, including unused leaves, retain the contiguous CUDA float32 admission.

View recipes retain only source/construction slots and axes. Every current view
is preflighted through the native planner before numerical compilation or
execution. Warm calls also preflight retained inactive and zero-trip recipes
against current inputs, without rescanning inactive helper bodies. Mixed
calls run the unchanged pointwise pipeline, then the existing native alias
bridge, then result reconstruction. Pure-view calls create no numerical graph,
kernel, executor or preparation entry. The bridge independently replans actual
inputs, and caches publish only after successful wrapping and reconstruction.
Transpose binding guards apply only to programs that admit transpose; warm
same-arm reuse preserves the existing inactive-helper behavior. These changes
extend metadata compilation coverage and make no performance-parity claim.
The input views share storage with their original owners. After the compiled
call, public [CUDA scalar `Tensor.add_`](cuda-add-inplace.md) can mutate a dense
returned view; a computed output remains independent. Mutation inside the
compiled function is unsupported. For the broader explicit-eager view language,
see [CUDA transpose capture](compile-cuda-t.md).
Focused validation is recorded in [input transpose validation](compile-input-transpose-validation.md).

Unequal input shapes additionally require either the tensor-leaf multiply-add
described below, or at most one arithmetic stage, including live sin/cos. Input and scalar nodes start at depth
zero; add/subtract/multiply add one to the maximum operand depth, tensor negation
adds one, and ReLU/sin/cos preserve depth. Scalar sign metadata adds no tensor operation.
This rule applies to the original typed expression before simplification, even
when one input is unused or singleton-only reshaping yields linear addresses.
Broadcast dimensions must match or one must be singleton, including scalar
tensors, leading/interior singleton dimensions and empty outputs. For example,
`(x.relu() - y.sin()).cos()`, `x.sin()` and `x * scale` are supported;
`x*y + 0.5`, `(x*y).sin()+x` and `x*y + (x+1.0)*(x+1.0)` are rejected
for unequal Tensor `x,y`; the scalar-leaf `x*y + 0.5` differs from the accepted
tensor-leaf `x*y+y` exception below.
Equal-shape support retains the full pointwise language above.

The sole two-stage broadcast exception is `a*b+c` or `c+a*b`, where all three
leaves are tensor inputs. Input IDs may repeat within the one/two-tensor limit
(for example, `x*y+x` or `y+x*x`); an unused argument still receives validation.
The original returned IR must have exactly this structure. Scalar leaves,
subtraction, negation, ReLU/sin/cos, a second product, extra arithmetic and
identity wrappers do not qualify, even if later simplification removes them.
This exception additionally requires no live sin/cos in any returned root.
For example, `p=x*y; return (p+x,p.sin())` shares a product between arithmetic
and trig consumers and is excluded in both output orders. This graph-wide guard
preserves the bounded single-FMA exception. Dead trig does not restrict live
admission. The exception reuses the existing lowering and checked addresses.

One-stage trig uses the existing numerical planner and selected native CUDA
executable, including its scalar-kind and shape-history behavior.
Finite default-Inductor comparisons do not establish general Inductor coverage
or performance parity. Generic executor PTX and reconstructed scalar plans are
not independent device traces; empty outputs execute no trig.

Strided inputs, other dtypes, gradients (even inside no-grad),
mutation, control flow outside the bounded root branches and literal loops below, module calls, keyword operator
arguments, reductions, matrix operations, and device/dtype conversions are
explicitly rejected. No original body or Python operator is run during
admission or warm execution. Unsupported configurations keep their existing
contracts; `disable=True`, configured/custom backend resolution, and the
explicit `backend="eager"` capture implementation remain separate.

### Bounded positional input trees

Exact `tuple`, `list` and `dict` inputs may nest existing exact native Tensor,
`float` and `bool` leaves. Dict keys must be exact strings. Select literal integer
sequence indices (including negative indices), literal string keys, or unpack a
fixed tuple/list length. Local aliases, constructed containers and data-only
helpers compose with the same frame lowerer:

```python
def scaled(payload):
    x, gain = payload["pair"]
    return {"scaled": x * gain, "input": x}

result = torch.compile(scaled)({"pair": [x, 0.5], "unused": False})
```

Every input edge is admitted before execution, including unread leaves. A call
containing containers permits at most 64 nested container levels and 4096
reference edges, counting positional roots and contained values (including empty
containers). Flat calls retain their previous scalar-argument admission without
this edge bound. The native ABI still requires one or two Tensor **occurrences**,
including unused or repeated Tensor leaves; it does not deduplicate input arity.
The existing 64 runtime-float limit and all native numerical limits still apply.

Caller containers must form a tree: cycles and repeated container identities
anywhere in the arguments, including shared tuples, are rejected. Repeated Tensor
owners and distinct storage-sharing Tensor views remain supported within native
storage limits. Returning an original input container anywhere in the public
result is rejected; selected input Tensor leaves may still return by identity
alongside a computed Tensor. Helpers may pass input containers back for subsequent
selection inside the compiled function.

Observation remains lazy. Selecting/unpacking a sequence checks its exact type
and length; dict selection checks its type and the selected path, without guarding
unrelated keys or insertion order. Scalar history follows the public parameter
and normalized item path: list and tuple indices share source identity under
separate structural guards. On a logical hit, changed Tensor traversal order
rebinds current operands while preserving the selected specialization's frozen
scalars and runtime scalar slots. Caches retain required source projections, not
caller containers, Tensor owners, input descriptors or unused keys. Admission
snapshots are invocation-local; concurrent caller mutation is not an atomic
whole-tree transaction.

Container subclasses, custom mappings, positional int/None/string leaves,
runtime or captured selectors, slices, dict iteration/unpacking, starred forms,
mutation, and captured/default containers remain unsupported. Constant-pool and
shape-predicate provenance rules are unchanged: a helper cannot grant an input
shape predicate new authority. This is a bounded frontend extension, not general
pytree, Dynamo, Inductor or accelerator parity.

The [structured-input validation and performance repair](diagnostics/compile-pointwise-structured-inputs/performance-repair/README.md)
links the focused contracts, paired GPU histories and frontend diagnostics.
The [clean-commit repair capture](diagnostics/compile-pointwise-structured-inputs/postcommit-c648878/README.md)
records focused correctness and build/import checks. Canonical qualification
remains pending; the diagnostic timings are not performance or coverage scores.

### Bounded nested results

Small exact tuple/list/dict constructors can combine computed Tensor leaves,
input-rooted transpose views, original input aliases, exact literal `None`/bool/int/float/string metadata and
current `input.shape[literal_integer_axis]` values:

```python
def structured(x, y):
    product = x * y
    shared = [product, x, x.shape[-1]]
    return {"shared": shared, "again": shared, "sum": product + x, "tag": "native"}
```

Numerical functions require at least one computed Tensor, with at most 64 distinct
original SSA roots; purely input-rooted view functions follow the rule above. Every computed root must have the same actual shape; input aliases may
retain another shape. Dict keys must be exact literal strings; insertion order
and duplicate-key replacement follow Python. Repeated Tensor or container leaves
preserve identity within a call. Separate equal computations get separate output
storage, even when kernel arithmetic is shared. Computed outputs and dynamic
containers are fresh across calls, so retaining earlier results is safe.

Constructors work in direct helpers and expanded root literal loops and branches.
They share a 4096 construction/reference-edge budget and depth limit 64, including
overwrites and expanded iterations. Shared container DAGs are accounted and
rebuilt without exponential expansion. The admitted constructor bytecodes are
`BUILD_TUPLE`, `BUILD_LIST`, `BUILD_MAP` and `BUILD_CONST_KEY_MAP`; bounded exact
string key tuples are checked before disassembly. Compiler-optimized constant
containers (including `()`) and large literals requiring other opcodes are
unsupported. Literal selection and fixed tuple/list unpacking are supported
for these symbolic containers too. Mutation, starred forms,
comprehensions, constructor calls, whole `torch.Size` returns and shape arithmetic
remain unsupported.
Forms optimized to identical admitted bytecode are indistinguishable.

Positional/captured scalar metadata returns are unsupported; arithmetic scalar
specialization is unchanged. Shape metadata must come from an original input in
the root, with an in-range literal axis. Helper boundaries cannot create root
shape-predicate provenance. All original nodes and inputs retain admission checks,
and every computed root must satisfy the existing numerical domain. Returned
intermediates do not impose eager rounding: consumers can still use fused FMA.

One native graph, output-vector ABI and CUDA launch serve all computed leaves.
The immutable Python result specification carries topology separately, so changes
to keys or container order do not fragment native executable caching. Current
input aliases and dimensions are reconstructed after native completion and before
any cache publication or LRU update. Inputs and all output allocations remain
owned through launch, synchronization and Python conversion, including failures.
Single-Tensor functions use this same path and still return a Tensor.

The result specification also projects the first observable occurrence of each
computed root into native numerical planning. Realization and fusion determine
logical regions, including rounded intermediate imports and exports. A native
scalar instruction plan executes those regions within one generated CUDA kernel;
changing return order may supply a different Program and executable. An immutable
preparation retains the validated metadata for matching warm calls, plus a
completed read-only instruction upload for VM execution. Computed outputs and
VM scratch remain invocation-owned through synchronization and failure;
neither input tensors nor previous outputs are retained by preparations.
VM scratch is capped at 64 MiB by limiting active workers and using a grid-stride
loop. Direct executables keep the same launch geometry and completion boundary,
using local registers without instruction upload or scratch. Correctness tests
do not establish a performance improvement.

The [structured-output evidence index](diagnostics/compile-pointwise-structured-outputs/README.md)
links the clean `d0f965a2` correctness capture, bounded guard timing diagnostic,
historical failures and retention limits. Captures remain scoped to their recorded
source revisions.

### Bounded root shape branches

Root `if`/`else` statements may compare an input Tensor's
`shape[literal_integer_axis]` with an exact integer literal using
`<`, `<=`, `==`, `!=`, `>=` or `>`, in either operand order:

```python
def shaped(x):
    if x.shape[-1] > 9:
        result = x.sin()
    else:
        result = x.cos()
    return result + 0.5
```

Input and literal local aliases, negative axes, early returns, assignments at
joins and sequential conditions are supported. The compiler reads native input
metadata without running the function or a public descriptor. Source-bound
branch outcome guards survive dimension generalization and tensor ABI rebinding;
crossing a threshold selects or lowers the appropriate graph. Public `shape`
descriptors on both Tensor classes remain identity-guarded.

Both arms pass bounded language/type admission on each new lowering. Inactive
locals, numerical IR and source observations do not enter the selected graph.
Warm calls check capture/signature bindings and active helper code, without
reparsing inactive helper bodies. A later branch crossing or new ABI lowering
admits those bodies again; invalid bodies fail without publishing cache changes.

Straight-line helpers can occur in arms, and root literal loops can occur outside
branch regions. Nested branches, branches in loops/helpers, loops in arms,
tensor truthiness, shape arithmetic, arbitrary attributes/subscripts, and
runtime/captured thresholds are unsupported. Captures, helper returns, computed
tensors and synthetic loop indices cannot acquire input/literal predicate origin
through aliases or unary negation. Admission follows CPython 3.10–3.14 bytecode;
source spellings optimized to identical instructions are indistinguishable.
Both arms and all helper/loop expansion share the existing instruction/node limits.
The conservative lazy parameter catalogue avoids separate liveness analysis; large
signatures with many unused parameters still incur per-call binding work.

[Clean `be808651` validation](diagnostics/compile-pointwise-shape-branches/postcommit-be808651/README.md)
records the committed branch tests, H100 default-Inductor comparisons and portable
CPython 3.10–3.14 checks, with source/wheel/runtime provenance and retained raw logs.

### Bounded root literal loops

Root functions may use sequential, non-nested `for` loops with a direct global,
closure or built-in lookup of the actual built-in `range`, including aliases:

```python
def recurrence(x, scale):
    for i in range(3, -2, -2):
        x = (x * scale + i).relu()
    return x
```

One to three literal exact integer arguments and a nonzero step are required.
Positive/negative steps and zero/one/many trips are supported. Bounds retain the
existing integer scalar range. Runtime/captured bounds, iterator expressions,
nested/helper-local loops, conditional/early-exit edges and mutation are rejected.
Admission follows CPython 3.10–3.14 bytecode semantics; source forms optimized to
identical bytecode are indistinguishable. Loop bodies use the same pointwise
operations and direct helpers as straight-line programs; index use adds no new
scalar arithmetic or indexing operations.

Normalization validates complete loop regions and stack cleanup, bounds expansion,
then expands before lazy source binding and frame lowering. Index assignments and
carry-over locals retain frame semantics. Zero trips preserve previous locals and
initial parameters; an index never assigned remains unbound. A zero-trip loop
does not admit an identity-only root return. Its body still passes the existing
typed operator, helper and data admission in an isolated local frame; its
temporary assignments and IR are discarded, while helper code/binding and data
guards remain active on warm calls. Unbound local reads in this skipped frame
(including its helper calls) use temporary data placeholders, not executed-local
lookups; no placeholder or skipped assignment escapes into the executing frame
or native IR. A genuinely executed unbound read still rejects. Reductions and
helper-local loops therefore reject even in a zero-trip body. This admission pass
shares the instruction/node budgets with executed bodies and helper calls.
Original instructions, every repeated
body/index assignment and every helper invocation share the 16384-instruction
budget; the 4096-node limit and original-IR numerical boundary are unchanged.

The original root code remains the semantic cache owner. Every used range lookup,
including zero-trip loops, is checked by identity on every call before lowering or
execution. Globals and the function's actual builtins table must be exact dicts
with exact string keys, checked before lookup or disassembly. Signature containers
are also revalidated on warm hits. Admission of the unchanged immutable root code
and its complete constant pool is reused by code identity; replacement repeats
full admission before disassembly. Private `resolve()` callers still receive
full code/constant validation. Restoring a
valid range binding can reuse existing entries; invalid bindings publish no cache
changes. Normalization adds no persistent cache or execution backend.

See the [loop evidence index](diagnostics/compile-pointwise-loops/README.md) for
the latest measured implementation, historical captures, original failures and
retention limitations. Those source-bound measurements do not qualify later
documentation commits.

### Direct Python helpers

A root global or closure binding may be an exact Python function with one or
more positional parameters. Calls must match its positional arity. Helpers may
use parameters, admitted scalar literals, local assignments, native Tensor
methods and the arithmetic above. Repeated calls, multiple helpers and calls
composed in the root all emit operations into the same graph:

```python
def wave(x):
    return x.sin() * 0.5

def pointwise(x):
    return wave(x) + wave(x + 0.25)
```

Helpers cannot read globals or closures, look up other helpers, branch, mutate,
handle exceptions, yield or await. Keyword-only/variadic parameters, nonempty
defaults and closures, and compiler directive attributes (`_torchdynamo_inline`,
`_dynamo_marked_constant`, `_torchdynamo_disable`) are rejected. Container types,
attribute keys and the entire constant pool are validated without callbacks,
including unused constants and warm calls. Strings and `None` are literal result metadata only.

Arguments and returns may also contain admitted small constructors and literal metadata; functions,
native call objects and modules cannot pass through helpers even as ignored
arguments. Identity and scalar-literal returns may feed later tensor operations;
numerical functions must still return at least one computed tensor. Scalar binary arithmetic remains
unsupported. Both `RETURN_VALUE` and Python 3.12 `RETURN_CONST` use this data-only
boundary. Passing an ignored input through a helper creates no scalar value guard,
but its data admission is rechecked on warm cache hits, including global and
closure rebinding. All tensor inputs still undergo native validation.

Each helper binding freezes only its code identity in the existing logical
specialization. Rebinding to another function with the same code reuses that
guard; structurally equal but distinct code does not. A new concrete tensor ABI
lowers the retained code, even if the original function has since changed.
Defaults, closures, constants and directive presence are revalidated on every
call. Helpers share root source realization, runtime scalar slots, SSA nodes and
budgets: every call charges its full instruction count toward the 16384 expanded
instruction limit, with the same 4096-node limit. Parsing is local to lowering;
ordinary warm hits do not disassemble helpers. Neither Python body executes.
Original-IR numerical admission, executor sharing, failure-atomic publication,
LRU bounds and reset ownership remain unchanged.

See the [helper diagnostics](diagnostics/compile-pointwise-helpers/README.md) for
source-bound checks and their limits.

## Runtime requirements

NVRTC is discovered by ordinary shared-library names (`libnvrtc.so.13`,
`libnvrtc.so.12`, `libnvrtc.so`) or an explicit `TORCH_RS_NVRTC` override.
The installed toolkit supplies NVRTC's libdevice implementation. Missing or
incompatible tooling produces a diagnostic failure, never eager fallback.
The target compute capability comes from the actual guarded CUDA device.
The private kernel object exposes generated source/PTX, compiler version,
options and device for regression evidence.

## Compiler pipeline

`_compile_pointwise.py` resolves bounded root branches and literal loops on
original bytecode regions before loop expansion, then lowers the selected path and
constructs typed SSA nodes with float32 tensor values and scalar kinds.
`pointwise_ir.rs` independently validates node topology. `pointwise_regions.rs`
plans realization, locality ordering and fusion over the admitted graph.
`pointwise_lowering.rs` canonicalizes each region into declarative scalar
instructions with explicit rounding and FMA decisions. `pointwise_program.rs`
allocates registers and validates instruction dataflow. `pointwise_codegen.rs`
mechanically emits those exact words when there are at most 256 instructions,
128 registers and 65,536 complete UTF-8 source bytes, including addresses and ABI.
Empty or over-cap plans use the existing VM. Both domains retain the existing
precise intrinsics, compiler options, launch geometry and completion path. Direct
code embeds its instructions and needs neither a device instruction buffer nor
invocation scratch. VM preparations still upload instructions and VM calls retain
bounded scratch. Compiler failures propagate; they never trigger VM fallback.
Plan disassembly is separate from actual selected kernel source/PTX.
`pointwise_indexing.rs` checks every expression's broadcast shape and size before
numerical rewriting, including dead expressions. The same Rust admission check
enforces the unequal-shape original-IR boundary during compilation and direct
cached-kernel execution, before NVRTC, output allocation or launch. Only live
returned nodes determine numerical capability; dead expressions still receive
full graph and shape validation. It checks a common shape across all computed roots
and derives each live input's address from row-major coordinates; unused inputs do not
expand the result. Singleton axes contribute no address increment. The same
generated kernel fuses all supported pointwise operations.
`cuda/jit.rs` compiles it with NVRTC and loads the resulting PTX through the
existing native driver. No fixed workload, shape, name, or corpus recognizer
is involved.

The [numerical contract](compile-pointwise-numerics.md) explains expression
rewriting, FMA selection, constant precision, signed zeros and libdevice rounding.
The unequal-shape boundary excludes competing products. The earlier
broad candidate had finite cancellation and IEEE failures, including fresh large
shapes and persistent shape transitions; default Inductor also varies contraction
choices across timing-selected configurations. The [historical review
record](diagnostics/compile-pointwise-broadcast/review-autotune-blocker.md)
preserves these failures. They are excluded expressions, not numerical repairs.

## Cache behavior

### Tensor metadata and identity

Each wrapper caches validated graphs and compiled modules, never tensor data,
results or input pointers. Shape/stride/dtype/device/gradient, live
scalar/function bindings, and repeated-input object relationships are guarded.
Contiguous storage offsets are read from the current inputs at launch and
bounds-checked on every call; they do not require separate specializations.
Passing the same Tensor for both parameters shares its input expression;
distinct tensors, even equal-valued tensors or views sharing storage, keep
separate expressions. Only this identity relationship is cached, so fresh
tensors reuse the same specialization.

### Scalar bindings

Every positional parameter and captured global/closure cell has an explicit lazy
source identity; the frame determines which values are observed.
Public parameter positions are distinct from filtered tensor indices and runtime
scalar operand indices. All tensors, including unused ones, enter the same
validation, alias, shape, code-generation and launch path. Scalar parameters that are unused or overwritten before their first read have
no value guards; exact-type admission still checks every
argument. Boolean bindings stay static. Integer literals and captures retain their
existing support; positional integers are rejected before user hooks can run.

Used positional and captured scalars are initially constant
specializations. Static float guards equate positive and negative
zero: a cache hit retains the sign captured by that graph, while a new graph
uses the current value. Literal zeros and promoted runtime parameters retain
their actual sign. Static NaN guards accept all exact-float NaN signs and payloads
as one specialization, matching the reference's `isnan` guard. The frozen scalar
value and IR bits remain those of the selected graph; runtime arguments keep their
actual bits. A changed finite float becomes a runtime float32
kernel parameter, matching the reference's warm-call materialization boundary.
Logical guards are checked from most recently selected to oldest before any new
promotion. An existing runtime specialization accepts earlier floats and nonfinite
values when its other guards match; a rank miss can instead select an older
static specialization. Only a complete guard miss consults successful source
history for new promotion. New traces specialize nonfinite values even after
runtime promotion. Reset clears this history. Integer and Boolean bindings retain
their scalar kinds.
At most 64 runtime scalar parameters are supported in total across captures and
positional arguments, by the existing single promotion pass. Their current values are
passed by value at launch and are never retained in graph or code cache keys.
The binding regressions keep both wrappers alive across changes without
resetting the reference. Persistent default-Inductor tests independently check
positional, global and closure histories, repeated bindings, slot changes,
nonfinite values, overflow, signed zeros and Boolean transitions. These tests
characterize binding policy and logical specialization selection across shape
histories, including signed-zero revisits.
See the [positional binding evidence](diagnostics/compile-pointwise-positional/README.md).

### Recompilation and reset

Each wrapper keeps logical specializations and concrete native executors under
one reset owner. A source's changed dimensions generalize after a guard miss;
zero and singleton dimensions remain static. Rank, stride relations, broadcast
equalities and applicable 32-bit upper bounds constrain reuse. Unused tensors
create no logical shape guards but still participate in all native validation.
A generalized specialization retains its frozen constants when an older shape
returns, including the sign of zero.

A checked native host plan performs original-graph admission and builds one
validated Program before compiler discovery or upload. Native executable identity
contains versioned exact bytes for the structural original Graph, complete ABI,
device/context, checked address formulas and VM/direct domain. Direct identity
also includes every instruction word and the register count; VM identity omits
the uploaded Program. Shapes and raw numerical hints are not executable identity.
Equal Programs and addresses share an executable even after preparation eviction;
different resulting Programs may require a new compilation. A retained identical
executable can rebuild a preparation without discovering or loading NVRTC. A logical
hit may compile a new concrete executor without consuming a logical slot or
updating promotion history. Preparation checks original-IR numerical admission on
actual shapes, including unused tensors and singleton-only linear maps. A warm
preparation requires the identical native input shapes (including rank); that
checked signature certifies reuse of the graph's shape analysis. Every run still
rechecks current types, device, dtype, gradients, contiguity, element/storage
bounds and exact shapes, and uses current offsets and runtime scalar values.

`recompile_limit` defaults to eight logical specializations. The executable LRU
and each specialization's ABI-lowering LRU are independently bounded by the same
limit. Immutable preparations form a separate data LRU keyed by logical graph/device/indexing,
exact input shapes, the retained numerical hint and computed-root output order.
Container-only changes do not fragment this data cache. Different exact shapes
may prepare or evict data within one generalized logical specialization without
consuming another logical slot. Equivalent executable identity avoids another module.

The preparation LRU is bounded by the same entry count and a 32 MiB per-wrapper
retained-data budget. Construction-only host instruction vectors are dropped
after binding; VM upload completes before their release. Native charges include
actual retained device allocation bytes (including best-fit excess capacity) and
owned signature/layout storage. Direct preparations retain instruction/register
counts without device instruction storage. Python charges conservatively count every key referent,
including shared/repeated references, its graph and the value-carried actual
executor key, plus the wrapper and a
512-byte per-entry bookkeeping allowance. This is not a bound on total process
memory, CUDA allocator pools, modules, outputs, scratch or transient preparation.
An oversized preparation executes by the same mechanism without being retained.
Kernel/module/source/PTX and the exact native executable identity belong to the
count-bounded executor LRU. Construction-only host plans are dropped after bind.
Eviction prunes preparations by their recorded actual executor identity and owner.
Each newly bound preparation certifies its native Arc ownership before execution,
accounting or publication. Frozen native owners preserve that relation in the
frontend-admitted cache tuple. Every selected hit checks that the recorded
executor is still the exact current map owner; it performs no native ownership
query, planning, emission, compilation, upload or retention reaccounting. The
existing bounded scan prunes stale map owners only after success. Executor-map
replacement and clear remain supported; manually fabricating an inconsistent
tuple in the private preparation dictionary is outside this contract.

Failed admission, preparation, compilation, execution, output conversion or result
reconstruction publishes no entry, history or LRU change. All cache publication
happens after successful result reconstruction and optional selected-invocation
receipt allocation. The private `_torch_rs_pointwise_receipt` calls the same
implementation body and returns `(result, prepared)` with the exact successfully
used owner, or `(result, None)` for graph-free input views. Source/PTX is not
independent execution tracing or physical GPU UUID attestation. `torch.compiler.reset()` clears
all three levels under the same lock; the next call recompiles. Explicitly held
private prepared objects have ordinary independent ownership, like held private
kernel objects, and may outlive wrapper reset.

## Storage and device ownership

A native bridge revalidates input layouts, ranges and device before allocation
or launch. Preparation and execution both check the kernel's recorded driver
context under the existing device guard; preparation checks before any device
allocation, including direct preparations without instruction storage. A successful
VM upload has its own completion boundary, which is not repeated on a cache hit.
Nonempty direct calls still launch and complete through the same owner as VM calls.
Storage owners remain borrowed through legacy-stream completion,
including errors. One launch produces fresh computed outputs; empty outputs need
no launch, instruction upload, scratch or input pointer. An empty computed output
does not require every input to be empty; unused inputs still receive validation.
The private legacy kernel `.run` and `.prepare` remain VM-only; ordinary-default
executables instead bind a typed checked host plan. No arbitrary CUDA source or
caller-provided instruction stream is accepted by the Python bridge. Modules are shared with immutable preparations, keyed by device
and checked against the active driver context. The existing device guard
restores the caller's device on compilation, execution and module destruction.
Wrapper locks serialize cache publication and reset.

## Validation and campaign evidence

The [guard traversal author diagnostic](diagnostics/default-compile-guard-scan-20260918.md)
records bounded public-call measurements and validation attempts. It is separate
from canonical performance evaluation.

The [positional binding evidence index](diagnostics/compile-pointwise-positional/README.md)
links the clean `7dd1a811` fixed measurements against main `a281503f`, 37 passing
focused regressions and three dispatched-module captures. The separate 16-leg
v2 warm-dispatch comparison passes all 56 paired histories. The index preserves
earlier clean snapshots, the failed v1 comparison, signed-zero and NaN-guard
failures, and development-only repair checks. These bounded regressions and
fixed-corpus measurements do not establish general Inductor parity.

[Independent regression tests](../tests/test_compile_pointwise_jit.py) cover
hardware-free admission/codegen, generated expression trees against default
PyTorch Inductor and eager semantics, changed values/shapes/constants, offsets,
IEEE values, failure/retry, concurrent calls, reset and lifetimes. Two-device
restoration runs separately with `CUDA_VISIBLE_DEVICES=0,1`; portable runs skip
hardware-only cases explicitly. The [validation record](diagnostics/compile-pointwise-jit/README.md)
records the local commands, failures and generated-kernel evidence.
The [broadcast regressions](../tests/test_compile_pointwise_broadcast.py) add
independently generated ranks, singleton patterns and expression graphs, both
operand orders, empty/scalar/offset inputs and cache/lifetime checks.
The [bounded primitive tests](../tests/test_compile_pointwise_broadcast_primitives.py)
keep both frameworks' wrappers alive across IEEE inputs and shape changes;
[admission regressions](../tests/test_compile_pointwise_broadcast_priority.py)
check original-IR rejection through compilation and direct kernel execution.
Their [validation record](diagnostics/compile-pointwise-broadcast/README.md)
retains historical full compiler sweeps and broad-candidate failures alongside
new clean measurements of the narrowed candidate and main. The old broadcast
scores do not describe the narrowed domain; the current record identifies each
measured source and build separately.

The unchanged [public-default compiler gates](torch-compile-default-evaluator.md)
remain the scoring authority with all 112 coverage and 56 CUDA performance
cells. These focused tests do not change their denominator. Unsupported
categories remain zero. The [post-commit evidence](diagnostics/compile-pointwise-jit/README.md)
records fresh clean-commit coverage, CUDA-performance and generated-code captures
for `aab2fd7`, including the libdevice product-order repair, alongside the unchanged source-bound baseline
and original failures.
[Repair validation](diagnostics/compile-pointwise-jit/review-libdevice-order.md) records
the development checks separately from these clean campaign measurements.

The [tensor-leaf multiply-add record](diagnostics/compile-pointwise-tensor-madd/README.md)
documents the additional original-IR exception, default-Inductor comparisons,
direct cached-kernel revalidation and dispatched CUDA/PTX captures.

[Clean post-commit evidence](diagnostics/compile-pointwise-tensor-madd/postcommit-bf908578/README.md)
records candidate `bf908578` and main `77aa16fc`, both fixed-corpus measurement
orders, exact multiply-add comparisons and the raw-report retention manifest.
