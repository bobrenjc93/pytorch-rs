# Compiled CUDA reshape

The bounded eager compiler captures `Tensor.reshape`, exactly two-positional-argument
`torch.reshape(x, constant_shape)`, and genuine direct-import aliases on native CUDA
float32 rank-0/1/2 tensors without gradients, with rank-0/1/2 outputs:

```python
import torch_rs as torch

def flatten_transpose(x):
    return torch.reshape(x.t(), [-1])

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]]).to("cuda:0")
f = torch.compile(flatten_transpose, backend="eager", fullgraph=True)
y = f(x)
assert y.cpu().tolist() == [1., 4., 2., 5., 3., 6.]
assert y.data_ptr() != x.data_ptr()
```

Method shapes accept positional exact integer constants (`reshape(2, -1)`) or one
exact flat tuple (`reshape((2, -1))`, also `shape=(2, -1)`). `reshape(())`
produces a scalar from one element. One `-1` may infer a dimension. Literal and
local constants are frozen; live module-global integers have type/value guards.
Invalid binding, invalid types and signed-64-bit dimension overflow raise
`TypeError`; negative dimensions other than `-1`, multiple inference dimensions,
element-count mismatch, ambiguous zero-element inference and layout overflow
raise `RuntimeError` before any graph operation executes.

PyTorch checks the first ordinary shape element before keyword binding, then
unpacks the remaining dimensions in order. In particular, a first boolean is
invalid, while some later booleans are accepted by reference PyTorch. Those
later boolean forms are deliberately unsupported here, as are integer subclasses,
index conversions, method lists, tuple subclasses, arbitrary containers, symbolic or
computed dimensions and scalar function arguments. The compiler never invokes
user conversion methods. Tensor-only input binding retains its existing errors.
Known dimension type/overflow errors and invalid call structure are checked even
when another value is nonconstant. Local nonnumeric constants retain the same
reshape binding errors as inline literals. Unsupported locals and mixed
literal/local tuples are retained only for validation; they cannot enter a
compiled graph or become output pytrees, even when unused or overwritten.

Top-level calls require exactly two positional arguments and an exact tuple/list
of constant integer dimensions, such as `torch.reshape(x, (2, -1))` or a genuine
`from torch_rs import reshape` alias called with `[2, -1]`. Keywords, variadic
shapes, shape metadata expressions and arbitrary shape globals remain excluded.
The frontend recognizes the immutable native variable-function owner's identity,
retained at package startup. It guards only used module fields and imported
bindings, so unrelated reshape replacement/deletion leaves existing graphs valid.
Retained genuine aliases survive public/native export mutation; fake callables
reject without invoking their bodies, equality, descriptors or conversions.
Nonempty integer lists must feed a supported top-level reshape; this does not
admit unused integer lists, integer-list outputs or method list shapes.

A view-compatible reshape creates a distinct Python object sharing storage,
with the native view planner's strides and the input offset. A copy-required
positive-stride rank-1/rank-2 input packs logical values directly on its CUDA
device, then uses canonical output strides and offset zero. Both forms preserve
float32 bits, including signed zero and NaN payloads. Views observe shared
mutations and retain storage after input deletion; copies are independent.
Repeated output references preserve identity, and separate reshape calls create
separate wrappers even for unchanged shapes, scalars, singletons and empties.
Metadata allocation is not a tensor-storage copy.

The explicit shape IR payload is separate from scalar values, reduction options
and transpose axes. Frontend metadata inference and whole-graph execution share
the checked native eager reshape resolver and view-stride planner. Execution
reuses eager CUDA packing without CPU staging, Python-body/method replay or
reference imports. Reshape composes with `t()`, `transpose()`, `contiguous()` and
packed negation/scalar/add/matmul/row-sum nodes. A strided reshape view still
needs packing before arithmetic.

Existing one/two-input, same-module helper/global capture, output-pytree,
static/dynamic and no-break fullgraph policies apply. Dynamic cache hits may
switch between views and packs as sizes change; shape requests are resolved
again while rank/stride/device/offset guards remain exact. Both planners validate
the whole graph before execution, including early/late nodes, cached declarations,
output field types and repeated output/metadata pairs. Actual inputs and unused
captures must share a CUDA device. CPU capture, higher ranks, new dtypes,
gradients, `reshape_as`, dtype-changing views, new backends and fusion are outside this increment.

[Compiled `Tensor.view`](compile-cuda-view.md) accepts the same bounded method shape
surface with `size=` keyword binding, but always requires alias-compatible strides.

## Validation

Use the [locked contributor setup](../CONTRIBUTING.md), then run:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_reshape tests.test_compile_cuda_reshape
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_reshape.ModuleReshapeDeviceTests tests.test_compile_cuda_reshape.ReshapeDeviceTests
CUDA_VISIBLE_DEVICES=0 cargo test --locked --features python-bindings --lib cuda_graph
CUDA_VISIBLE_DEVICES=0 cargo test --locked --test cuda_contiguous
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest -v tests.test_compile_cuda_module_reshape tests.test_compile_cuda_reshape
```

The [top-level development evidence](diagnostics/compile-cuda-module-reshape/README.md)
retains the exact-main 144-cell baseline: 96 native rejections, of which 94 are
strict-reference eligible. Two empty add-then-reshape cells retain their surveyed
Inductor stride discrepancy and earn zero strict-parity credit. The new frontend
closes all 96 rejections without changing that classification. No performance
or general compiler/Inductor/training parity is claimed.

Hardware-only cases skip clearly when the required devices are unavailable.
The [latest clean-commit capture](diagnostics/compile-cuda-reshape/postcommit-93441a9f/README.md)
measures `93441a9fc13e8124a507a34810f2c3d97f7e69f1` with a fresh local environment
and release build, including all three reviews' argument-validation regressions.
H100 differentials, compiler and CPU/layout regressions, two-device restoration,
CUDA-hidden portability, Rust checks, Clippy and the example above passed;
hardware skips remain recorded.

The [original development capture](diagnostics/compile-cuda-reshape/README.md)
preserves the fresh PR1977 baseline gap and failed attempts. The
[binding review](diagnostics/compile-cuda-reshape/binding-review/README.md),
[partial-argument review](diagnostics/compile-cuda-reshape/validation-review/README.md) and
[local-constant review](diagnostics/compile-cuda-reshape/local-constant-review/README.md)
retain their development failures and checks. Earlier clean captures at
[`1ad5fd62`](diagnostics/compile-cuda-reshape/postcommit-1ad5fd62/README.md),
[`7533cbbc`](diagnostics/compile-cuda-reshape/postcommit-7533cbbc/README.md),
[`3d44687d`](diagnostics/compile-cuda-reshape/postcommit-3d44687d/README.md) and
[`73b74ec0`](diagnostics/compile-cuda-reshape/postcommit-73b74ec0/README.md) remain
pinned to their original code and build identities.

These graphlets are non-scoring diagnostics. Frozen38, performance workloads,
evaluator definitions, observer/hardware contracts and the separate unadopted
PR1970/PR1971 campaigns are unchanged.
