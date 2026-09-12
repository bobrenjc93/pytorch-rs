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
and package bindings may also be captured. Integers must fit the native scalar
range `[-2**63, 2**64-1]`; float32 overflow rounds to signed infinity. Graphs
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
constructs typed float32 SSA nodes. `pointwise_ir.rs` independently validates
node topology and emits CUDA C from operator rules, retaining intermediates.
`cuda/jit.rs` compiles it with NVRTC and loads the resulting PTX through the
existing native driver. No fixed expression, shape, name, or corpus recognizer
is involved. Accurate libdevice `sinf`/`cosf` and `--ftz=false` are used;
fast math is not enabled. Arithmetic explicitly allows FMA contraction
(`--fmad=true`) to match default Inductor, including cancellation and overflow
cases where CUDA eager's separately rounded operations differ. SSA variables
retain expression dependencies and reuse through native compiler optimization.
A local single-use multiply/add-subtract rule emits explicit `fmaf` so toolkit
strength reduction cannot lose the intended contraction. Float constants are rounded to float32 once and
encoded by their IEEE bits, including negative zero and non-finite values.
Integer constants follow default Inductor's Python binary64-to-float32
normalization; large integers at rounding boundaries can differ by one float32
ULP from eager's direct integer conversion.
The zero-sign contract follows default PyTorch 2.13 Inductor: negation uses
`0 - x`, so negating positive zero produces positive zero; ReLU preserves
negative zero. CUDA eager differs in these two cases (negative and positive
zero respectively). Tests assert those differences explicitly as well as
checking all other eager/reference values. Finite libdevice results retain
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
results or input pointers. Shape/stride/offset/dtype/device/gradient and live
scalar/function bindings are guarded. A changed shape creates a graph cache
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
records fresh clean-commit coverage and CUDA-performance captures, alongside
the unchanged source-bound baseline and original development failures.
