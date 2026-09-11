# Compiled CUDA view

The eager compiler captures alias-only `Tensor.view` on exact native CUDA
float32 tensors without gradients, with input and output ranks 0, 1 or 2:

```python
import torch_rs as torch

def flatten_relu(x, w):
    y = (x @ w).relu()
    flat = y.view(size=(-1,))
    return y, flat, flat

x = torch.tensor([[1., -2., 3.], [4., 5., -6.]]).to("cuda:0")
w = torch.tensor([[1.], [2.], [1.]]).to("cuda:0")
f = torch.compile(flatten_relu, backend="eager", fullgraph=True)
y, flat, repeated = f(x, w)
assert flat.cpu().tolist() == [0., 8.]
assert flat is repeated and flat is not y
assert flat.data_ptr() == y.data_ptr()
```

Shapes accept exact integer positional constants (`view(3, -1)`) or one exact
flat tuple (`view((3, -1))`, `view(size=(3, -1))`). An empty tuple produces a
scalar from one element. One `-1` may infer a dimension. Literal/local constants
are frozen; live module-global integers retain type/value guards, including
inside the existing same-module helper. The existing one/two Tensor input,
output-pytree, static/dynamic and no-break fullgraph policies apply.

View never packs or copies storage. A transposed matrix's `view(-1)` fails with
the eager stride-compatibility error; use `reshape(-1)` or
`contiguous().view(-1)` when a copy is intended. Compatible views retain storage,
offsets, planner strides and raw float32 bits. This includes unchanged shapes,
empty tensors, scalars and noncanonical singleton strides. Each call creates a
distinct Python wrapper; repeated references to one result retain identity.
Mutations are shared and outputs keep storage alive after input deletion.

View's overloaded binder differs from `reshape(shape=...)`: it accepts `size=`,
rejects `shape=`, and rejects invalid keyword combinations before unpacking
integer dimensions. Invalid bindings/types and signed-64-bit dimension overflow
raise `TypeError`; invalid inferred sizes, element counts and incompatible
layouts raise `RuntimeError`. Known argument errors remain checked alongside
nonconstant dimensions. Literal/local lists and bounded exact integer arithmetic
are retained only for argument validation; even unused or overwritten values
still reject capture. No user index conversion is invoked. Lists, tuple/int
subclasses, later boolean dimensions, symbolic/computed dimensions, scalar call
inputs and dtype overloads (even same-dtype) remain unsupported capture. Eager
view's broader supported argument forms are unchanged.

Frontend metadata inference and the explicit native whole-graph planner share
the eager checked shape resolver and alias-compatible stride planner. Both
validate every node and output before execution, including late failures and
cached declarations. Dynamic cache hits recheck compatibility as sizes change;
a failed call executes no graph operations and leaves a compatible cached graph
usable. Exact rank/stride/device/offset guards, unused-input/capture checks,
method/helper/global guards and repeated output/metadata checks remain active.

View composes with reshape, `t`, transpose, contiguous, arithmetic, matmul,
row sums and ReLU. Strided arithmetic still requires explicit packing. CPU view
capture, new dtypes, gradients, `view_as`, top-level variants, other backends and
fusion are excluded. Execution uses native operations without the original
Python body, per-node Python replay or reference-PyTorch forwarding.

## Validation

After the [locked contributor setup](../CONTRIBUTING.md), run:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_view
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_view.ViewDeviceTests
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest -v tests.test_compile_cuda_view
CUDA_VISIBLE_DEVICES=0 cargo test --locked --features python-bindings --lib cuda_graph
```

Metadata/planner tests run without hardware; hardware-only cases skip clearly.
The [development capture](diagnostics/compile-cuda-view/README.md) records the
fresh main baseline, release build, seeded H100 differentials and regressions,
including failed attempts. Clean-commit capture belongs to Burner's later
delivery commit. These are non-scoring diagnostics; scoring corpora, performance
workloads, evaluator contracts and the separate PR1970/PR1971 campaigns are unchanged.
