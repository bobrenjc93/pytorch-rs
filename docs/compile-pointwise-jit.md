# Native default CUDA pointwise compilation

`torch_rs.compile(fn)` with untouched public defaults now lowers a bounded
pointwise language to a generated fused CUDA kernel. Backend resolution still
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

The positional-scalar extension and shared specialization fix have
[clean-commit measurements](diagnostics/compile-pointwise-positional/postcommit-7a74596b/README.md)
and passing persistent default-Inductor regressions for the formerly failing
signed-zero shape histories. The
[validation record](diagnostics/compile-pointwise-positional/README.md) preserves
those failures alongside revision checks. Burner's independent review and exact-head
qualification remain pending. This bounded surface does not establish
general Inductor parity.

The function accepts one or two exact native CUDA float32 Tensor inputs on the
same device with contiguous storage, plus exact built-in `float` and `bool`
arguments in any positional slots. For equal-shaped tensors:

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
are bounded to 4096 nodes and 16384 bytecode instructions. Return one computed Tensor.
Scalars and empty tensor shapes and contiguous views with storage offsets are
supported. Outputs have fresh storage and canonical contiguous strides;
inputs are unchanged. Returning an input unchanged is outside this subset.

Unequal input shapes additionally require the returned expression to have at most
one arithmetic stage and no live sin/cos. Input and scalar nodes start at depth
zero; add/subtract/multiply add one to the maximum operand depth, tensor negation
adds one, and ReLU preserves depth. Scalar sign metadata adds no tensor operation.
This rule applies to the original typed expression before simplification, even
when one input is unused or singleton-only reshaping yields linear addresses.
Broadcast dimensions must match or one must be singleton, including scalar
tensors, leading/interior singleton dimensions and empty outputs. For example,
`(x.relu() - y.relu()).relu()` and `x * scale` are supported; `x * scale + scale`,
`x*y + (x+1.0)*(x+1.0)` and live sin/cos are rejected with unequal input shapes.
Equal-shape support retains the full pointwise language above.

Strided inputs, other dtypes, gradients (even inside no-grad),
mutation, control flow, containers, helper calls, module calls, keyword operator
arguments, reductions, matrix operations, and device/dtype conversions are
explicitly rejected. No original body or Python operator is run during
admission or warm execution. Unsupported configurations keep their existing
contracts; `disable=True`, configured/custom backend resolution, and the
explicit `backend="eager"` capture implementation remain separate.

## Runtime requirements

NVRTC is discovered by ordinary shared-library names (`libnvrtc.so.13`,
`libnvrtc.so.12`, `libnvrtc.so`) or an explicit `TORCH_RS_NVRTC` override.
The installed toolkit supplies NVRTC's libdevice implementation. Missing or
incompatible tooling produces a diagnostic failure, never eager fallback.
The target compute capability comes from the actual guarded CUDA device.
The private kernel object exposes generated source/PTX, compiler version,
options and device for regression evidence.

## Compiler pipeline

`_compile_pointwise.py` statically admits CPython straight-line bytecode and
constructs typed SSA nodes with float32 tensor values and scalar kinds.
`pointwise_ir.rs` independently validates node topology; `pointwise_lowering.rs` canonicalizes expressions and emits CUDA
C from operator rules, retaining intermediates.
`pointwise_indexing.rs` checks every expression's broadcast shape and size before
numerical rewriting, including dead expressions. The same Rust admission check
enforces the unequal-shape arithmetic-depth boundary during compilation and direct
cached-kernel execution, before NVRTC, output allocation or launch. Only live
returned nodes determine numerical capability; dead expressions still receive
full graph and shape validation. It derives the returned shape
and each live input's address from row-major coordinates; unused inputs do not
expand the result. Singleton axes contribute no address increment. The same
generated kernel fuses all supported pointwise operations.
`cuda/jit.rs` compiles it with NVRTC and loads the resulting PTX through the
existing native driver. No fixed expression, shape, name, or corpus recognizer
is involved.

The [numerical contract](compile-pointwise-numerics.md) explains expression
rewriting, FMA selection, constant precision, signed zeros and libdevice rounding.
The unequal-shape boundary excludes competing arithmetic contractions. The earlier
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

Each used parameter, global and closure cell has an explicit source identity.
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
their actual sign. A changed finite float becomes a runtime float32
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

Native executors specialize the full filtered tensor ABI, device and exact
broadcast address formulas. Equal-shaped inputs share linear-load code. A logical
hit may compile a new concrete executor without consuming a logical slot or
updating promotion history. Every launch rechecks original-IR numerical admission
on actual shapes, including unused tensors and singleton-only linear maps.

`recompile_limit` defaults to eight logical specializations. The executable LRU
and each specialization's ABI-lowering LRU are independently bounded by the same
limit. Failed admission, compilation or execution publishes no entry, history or
LRU change. `torch.compiler.reset()` clears both cache levels; the next call
recompiles.

## Storage and device ownership

A native bridge revalidates input layouts, ranges and device before allocation
or launch. Storage owners remain borrowed through legacy-stream completion,
including errors. One fused launch produces a fresh output; empty outputs need
no launch or input pointer. Modules are owned by cache entries, keyed by device
and checked against the active driver context. The existing device guard
restores the caller's device on compilation, execution and module destruction.
Wrapper locks serialize cache publication and reset.

## Validation and campaign evidence

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
