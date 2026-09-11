# Compiled CUDA Tensor.t views

The bounded eager compiler captures parameterless `Tensor.t()` for native CUDA
float32 tensors of rank 0, 1 or 2 without gradients:

```python
import torch_rs as torch

def transpose_and_pack(x):
    return -x.t().contiguous()

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]]).to("cuda:0")
f = torch.compile(transpose_and_pack, backend="eager", fullgraph=True)
assert f(x).cpu().tolist() == [[-1., -4.], [-2., -5.], [-3., -6.]]
```

Each `t()` creates a new view object sharing the input's storage, dtype, device
and offset. Rank-2 shape and strides swap; scalars and vectors keep their
metadata but still return a distinct object. Repeated references to one result
preserve identity, distinct calls produce distinct objects, and `x.t().t()`
shares storage without returning `x` itself. Views retain storage after their
inputs go out of scope and observe shared mutations.

The node calls the existing checked native view primitive through the whole-graph
executor. It allocates view metadata only: no tensor-storage copy, CPU staging,
arithmetic kernel, Python-body replay, Tensor-method redispatch or reference
PyTorch import. `t().contiguous()` reuses native packing, while a transposed
input may become contiguous through `t()` alone. Negation, scalar multiplication,
addition, matmul and row sums still reject genuinely strided operands unless
packed first.

Existing one/two-input, global/helper, output-pytree, callable and cache guards
apply. Static captures guard shape/stride/offset; dynamic captures allow size
changes while retaining rank/stride/offset guards and replanning each operation.
Both planners validate the entire graph before execution. Metadata field types
are checked before equality, including early/late nodes, cached declarations
and output leaves. Repeated output containers retain identity while each distinct
output/metadata pairing is validated. All actual inputs and captures, including
unused captures, must be on the same CUDA device.

Arguments, rank >2, CPU capture, other dtypes, gradients, general
transpose/permute/reshape, `.T`/`.mT`, top-level `torch.t` and new backends remain
unsupported. No-break `fullgraph=False` retains its existing default-dynamic
policy. These graphlets are non-scoring diagnostics: the frozen 38-case corpus,
performance workloads, evaluator/observer/hardware contracts and unadopted
PR1970/PR1971 campaigns are unchanged.

## Validation

Use the [locked contributor setup](../CONTRIBUTING.md), then run:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_t
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_t.CompileTDeviceTests
CUDA_VISIBLE_DEVICES=0 cargo test --locked --features python-bindings --lib cuda_graph
```

Hardware-only tests skip clearly without the required devices. Rust test-only
accounting checks zero native operations for malformed early/late nodes; it is
absent from release builds and does not alter scoring observers.

The [initial clean-commit capture](diagnostics/compile-cuda-t/postcommit-e01f1d0f/README.md)
measured `e01f1d0f69dffa8018e1b333bb5ab642ce9fdcb9` with a fresh locked `.venv`
and release build. H100 t/packing differentials, compiler regressions, strict
metadata negatives, two-device restoration, focused Rust and CPU/docs checks
passed. Source, installed imports, native binary and clean status are verified
in its receipts. It predates the repeated-output metadata repair.

The [review-repair diagnostics](diagnostics/compile-cuda-t/review-output-metadata/README.md)
preserve the failing reproduction and fresh release-build validation of that
repair, including static/dynamic cache-hit rejection before native execution.
These are source-bound development measurements; a fresh clean-code capture
remains required after Burner commits the repair.

The [original development bundle](diagnostics/compile-cuda-t/README.md), including
the fresh PR1975 baseline reproduction and failed attempts, remains unchanged.
These diagnostics do not replace independent review or Burner merge gates.
