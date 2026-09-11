# Compiled CUDA squeeze

The bounded native eager compiler captures exactly `Tensor.squeeze()` with no
arguments on exact CUDA float32 tensors of rank 0, 1 or 2 without gradients:

```python
import torch_rs as torch

def remove_singletons(x):
    y = x.squeeze()
    return y, y, x.squeeze()

x = torch.tensor([[1., 2., 3.]]).to("cuda:0")
f = torch.compile(remove_singletons, backend="eager", fullgraph=True)
a, b, c = f(x)
assert a.cpu().tolist() == [1., 2., 3.]
assert a is b and a is not c and a is not x
assert a.data_ptr() == c.data_ptr() == x.data_ptr()
```

Every squeeze creates a fresh Tensor wrapper sharing storage, including unchanged
matrices, vectors and scalars. Only dimensions of size one and their stride entries
are removed. Surviving strides, storage offset, device, dtype and data bits remain
unchanged, including empty layouts, signed zero and NaN payloads. Views observe
shared mutations and retain storage after their inputs are deleted. Empty pointer
equality alone does not prove shared ownership; native tests check storage identity.

The existing one/two-input policies apply: `backend="eager"`, `fullgraph=True`
with `dynamic=None`, `False` or `True`, and no-break `fullgraph=False` with
`dynamic=None`. Dynamic captures guard input rank, strides and offset while
replanning singleton removal as sizes change. Output rank can change within the
rank-0/1/2 bound. Repeated references to a node retain identity; distinct squeeze
nodes and separate invocations return distinct objects.

Negation, ReLU and addition replan their output rank when squeezed dimensions
disappear or reappear. Cached declarations are checked against the originally
captured inputs separately from runtime planning. Surviving noncontiguous strides
still require `contiguous()` before arithmetic.

The native graph planner shares eager squeeze's layout calculation. Execution
calls the checked native view primitive without a copy or kernel. Both planners
validate the whole graph, including cached declarations and nested output metadata,
before any native operation. Supported execution never calls the Python body,
user hooks or per-node Python methods. Method identity and unused input/capture
device guards remain active.

Squeeze composes with [view](compile-cuda-view.md),
[reshape](compile-cuda-reshape.md), [transpose](compile-cuda-t.md),
[contiguous](compile-cuda-contiguous.md), and supported arithmetic, matmul,
row sums and ReLU. Strided arithmetic still requires packing first.

Dimension-bearing calls, including `dim=`, tuples, lists and variadic axes,
all other keywords, top-level `torch.squeeze`, `squeeze_`, `unsqueeze` and
`flatten` remain unsupported for capture. These raise
`CompileTraceUnsupportedError`; existing native eager squeeze overloads are
unchanged. CPU capture of squeeze, higher ranks, other dtypes, gradients,
new backends, fusion, Python fallback and installed-PyTorch replay are excluded.
This is not general `torch.compile`, Inductor, training or performance parity.

## Validation

Use the [locked contributor setup](../CONTRIBUTING.md), then run:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_squeeze tests.test_squeeze tests.test_squeeze_reference
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_squeeze.SqueezeDeviceTests
CUDA_VISIBLE_DEVICES=0 cargo test --locked --features python-bindings --lib cuda_graph
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest -v tests.test_compile_cuda_squeeze
```

The [clean-commit capture](diagnostics/compile-cuda-squeeze/postcommit-f8f8244/README.md)
validates `f8f8244c2cfa81e255c1814d1bd101c5d846b03d`, including dynamic arithmetic
consumers, with a fresh release wheel, H100 differentials, the complete compiler
sweep, two-device restoration and focused native/portable checks. The
[earlier capture](diagnostics/compile-cuda-squeeze/postcommit-e835f7f/README.md) and
[development evidence](diagnostics/compile-cuda-squeeze/README.md) retain their
original measurements, exact-main baseline and initial failed attempt. The
[dynamic-consumer review revision](diagnostics/compile-cuda-squeeze/review-dynamic-consumers/README.md)
records the reproduced cache-hit failure and development validation of its fix;
the new clean capture completes that revision's deferred measurement. Hardware
cases skip clearly when CUDA is unavailable. These are non-scoring diagnostics:
the frozen 38-case corpus, performance workloads, evaluator and hardware contracts
are unchanged. PR1970/PR1971 remain separate unadopted human-review campaigns;
independent review and Burner delivery gates remain required.
