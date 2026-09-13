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

The function must have one or two positional exact native CUDA float32 Tensor
inputs, on the same device, with identical shapes and contiguous storage. The
language supports local intermediate variables, reused expressions, binary
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

Broadcasting, strided inputs, other dtypes, gradients (even inside no-grad),
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
`cuda/jit.rs` compiles it with NVRTC and loads the resulting PTX through the
existing native driver. No fixed expression, shape, name, or corpus recognizer
is involved.

The [numerical contract](compile-pointwise-numerics.md) explains expression
rewriting, FMA selection, constant precision, signed zeros and libdevice rounding.

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

### Captured scalars

Captured scalars are initially constant
specializations. Static captured-float guards equate positive and negative
zero: a cache hit retains the sign captured by that graph, while a new graph
uses the current value. Literal zeros and promoted runtime parameters retain
their actual sign. A changed finite captured float becomes a runtime float32
kernel parameter, matching the reference's warm-call materialization boundary.
Promotion is per binding, persists when earlier float values return, and is
cleared by reset. Integer and Boolean bindings retain their scalar kinds.
Existing runtime promotions are applied before the complete graph-cache guard
is checked; new promotion is discovered only on a cache miss. Returning to a
cached finite specialization after an infinity/NaN-only interlude retains that
specialization, while shape misses still consult the full binding history.
At most 64 runtime scalar parameters are supported. Their current values are
passed by value at launch and are never retained in graph or code cache keys.
The binding regressions keep both wrappers alive across changes without
resetting the reference.

### Recompilation and reset

A changed shape creates a graph cache
entry but reuses code for the same expression and device. Failed admission,
compilation or execution does not consume a cache slot. `recompile_limit`
retains the existing default of eight metadata/binding specializations.
`torch.compiler.reset()` clears these caches; the next call recompiles.

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

The unchanged [public-default compiler gates](torch-compile-default-evaluator.md)
remain the scoring authority with all 112 coverage and 56 CUDA performance
cells. These focused tests do not change their denominator. Unsupported
categories remain zero. The [post-commit evidence](diagnostics/compile-pointwise-jit/README.md)
records fresh clean-commit coverage, CUDA-performance and generated-code captures
for `aab2fd7`, including the libdevice product-order repair, alongside the unchanged source-bound baseline
and original failures.
[Repair validation](diagnostics/compile-pointwise-jit/review-libdevice-order.md) records
the development checks separately from these clean campaign measurements.
