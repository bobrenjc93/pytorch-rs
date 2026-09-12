# Native default CUDA pointwise compilation

`torch_rs.compile(fn)` with untouched public defaults now lowers a bounded
pointwise language to a generated fused CUDA kernel. Backend resolution still
chooses `inductor`; this native implementation does not import or execute
PyTorch. It is not general Inductor, CPU compiler, or training equivalence.

```python
import torch_rs as torch

def pointwise(x, y):
    wave = torch.sin(x)
    return (wave * wave + y.cos() - 0.375).relu()

x = torch.tensor([0.2, -1.3], dtype=torch.float32).to("cuda:0")
y = torch.tensor([1.1, 0.7], dtype=torch.float32).to("cuda:0")
result = torch.compile(pointwise)(x, y)
```

The function must have one or two positional exact native CUDA float32 Tensor
inputs, on the same device, with identical shapes and contiguous storage. The
language supports local intermediate variables, reused expressions, binary
add/subtract/multiply (tensor/tensor or tensor/scalar in either order), and
unary negation/ReLU/sin/cos. Operator syntax, positional Tensor methods, and
positional native top-level functions are accepted. Exact bool/int/float
constants may be literal, module-global, or closure values; native operator
and package bindings may also be captured. Function globals must be an exact
`dict`; custom globals mappings are rejected before any lookup hooks can execute,
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

## Lowering, arithmetic, and ownership

`_compile_pointwise.py` statically admits CPython straight-line bytecode and
constructs typed SSA nodes with float32 tensor values and scalar kinds.
`pointwise_ir.rs` independently validates node topology; `pointwise_lowering.rs` canonicalizes expressions and emits CUDA
C from operator rules, retaining intermediates.
`cuda/jit.rs` compiles it with NVRTC and loads the resulting PTX through the
existing native driver. No fixed expression, shape, name, or corpus recognizer
is involved. Accurate libdevice `sinf`/`cosf` and `--ftz=false` are used;
fast math is not enabled. Arithmetic explicitly allows FMA contraction
(`--fmad=true`) to match default Inductor, including cancellation and overflow
cases where CUDA eager's separately rounded operations differ. SSA variables
retain expression dependencies and reuse through native compiler optimization.
Numerical lowering shares identical ordered expressions before selecting
explicit `fmaf` operations for each consumer. Self-subtraction established before
sign rewriting uses the shared rounded value: finite values produce positive zero, while infinities and NaNs
produce NaN. Products may contract at multiple consumers while retaining their
rounded value for other uses. Operand order is retained, including for
commutative operators, to match the reference's expression deduplication.
Subtraction introduced by sign normalization retains its contraction eligibility
even when the normalized operands coincide. Sign-flipped products retain their
factors: direct products take contraction priority, but an otherwise unpaired
signed product can still contract rather than prematurely overflowing.
Sign normalization precedes contraction: `-(a-b)` becomes `(b-a)+0`, products
with negative coefficients in sums become subtraction, and unit multipliers
and signed doubling follow the reference's normalization. Negated products use
`fmaf(-a, b, +0)`.
Uncontracted operations use explicit round-to-nearest CUDA intrinsics so NVRTC
cannot choose a different contraction after strength reduction.
The complete IR is validated before unused expressions are removed; an unused
local cannot change rounding of the returned expression. Invalid unused
operations remain rejected.
Boolean, integer and floating scalars retain distinct IR kinds until their
operator-specific rules and float32 promotion have run. Boolean `False` and
integer zero multiplication produce a known positive-zero tensor, including
for non-finite inputs; floating zero multiplication retains IEEE NaN and zero
signs. The resulting constant tensor is distinct from a scalar operand, so
further tensor multiplication still has float semantics. Only zeros created by
this early integer/Boolean rewrite qualify for tensor addition identities or a
right-hand subtraction identity. Addition identities run before subtraction
identities; a zero exposed by subtraction cannot retroactively remove an addition.
Zeros computed by subsequent arithmetic retain
the arithmetic operation at runtime consumers: `(x*0)+x` preserves an input
negative zero, while `(x*0+0.0)+x` produces positive zero.
Scalar literals and add/subtract/multiply/negate of known constant tensors retain
Python binary64 precision and zero signs until float32 materialization. For
example, `((x*0)+16777216.0)+1.0-16777216.0` folds to one. A constant is rounded
to float32 at each runtime consumer or output; other constant consumers of the
same intermediate retain its binary64 value. ReLU/sin/cos stop this propagation
and consume float32 values. Folded negation flips the constant's sign instead of
lowering to positive-zero subtraction or FMA. Runtime arithmetic retains its
existing contraction rules. IEEE bits preserve negative zero and non-finite
values across the private IR bridge.
Integer constants follow default Inductor's Python binary64-to-float32
normalization; large integers at rounding boundaries can differ by one float32
ULP from eager's direct integer conversion.
The zero-sign contract follows default PyTorch 2.13 Inductor: standalone negation
uses `0 - x`, so negating positive zero produces positive zero; ReLU preserves
negative zero. Contracted negation of a nonzero positive product that underflows
produces negative zero; an exactly zero floating product produces positive zero.
Negating a known zero tensor from Boolean or integer multiplication instead folds
to negative zero. CUDA eager also differs for standalone positive-zero negation
and negative-zero ReLU. Tests assert those differences and check the covered
eager/reference values. Finite libdevice results retain
gradual underflow; some reference compositions flush subnormals. Finite values
are compared with numerical tolerances, and zero signs are checked wherever
both results are zero.

NVRTC is discovered by ordinary shared-library names (`libnvrtc.so.13`,
`libnvrtc.so.12`, `libnvrtc.so`) or an explicit `TORCH_RS_NVRTC` override.
The installed toolkit supplies NVRTC's libdevice implementation. Missing or
incompatible tooling produces a diagnostic failure, never eager fallback.
The target compute capability comes from the actual guarded CUDA device.
The private kernel object exposes generated source/PTX, compiler version,
options and device for regression evidence.

Each wrapper caches validated graphs and compiled modules, never tensor data,
results or input pointers. Shape/stride/offset/dtype/device/gradient, live
scalar/function bindings, and repeated-input object relationships are guarded.
Passing the same Tensor for both parameters shares its input expression;
distinct tensors, even equal-valued tensors or views sharing storage, keep
separate expressions. Only this identity relationship is cached, so fresh
tensors reuse the same specialization. Captured scalars remain statically
specialized by their full value. Inductor may instead promote changed floating
bindings to runtime scalar tensors and change their rounding boundaries; the
binding regression compares each native specialization with a fresh constant-
specialized reference, not Inductor's adaptive warm-call strategy.
A changed shape creates a graph cache
entry but reuses code for the same expression and device. Failed admission,
compilation or execution does not consume a cache slot. `recompile_limit`
retains the existing default of eight metadata/binding specializations.
`torch.compiler.reset()` clears these caches; the next call recompiles.

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
for `40a57b36`, including the constant-folding precision and zero-origin repair,
alongside the unchanged source-bound baseline and original failures.
[Repair validation](diagnostics/compile-pointwise-jit/review-folding.md) records
the development checks separately from these clean campaign measurements.
