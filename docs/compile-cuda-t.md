# Compiled CUDA transpose views

The bounded eager compiler captures `Tensor.t()` and `Tensor.transpose(dim0, dim1)`
for native CUDA float32 tensors of rank 0, 1 or 2 without gradients:

```python
import torch_rs as torch

def transpose_and_pack(x):
    return -x.transpose(dim0=0, dim1=1).contiguous()

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]]).to("cuda:0")
f = torch.compile(transpose_and_pack, backend="eager", fullgraph=True)
assert f(x).cpu().tolist() == [[-1., -4.], [-2., -5.], [-3., -6.]]
```

Each `t()` or `transpose()` creates a new view object sharing the input's
storage, dtype, device and offset. Different rank-2 axes swap shape and strides;
scalars, vectors and same-axis transposes keep their metadata but still return
a distinct object. Repeated references to one result preserve identity,
distinct calls produce distinct objects, and double transpose
shares storage without returning `x` itself. Views retain storage after their
inputs go out of scope and observe shared mutations.

The node calls the existing checked native view primitive through the whole-graph
executor. It allocates view metadata only: no tensor-storage copy, CPU staging,
arithmetic kernel, Python-body replay, Tensor-method redispatch or reference
PyTorch import. `transpose(...).contiguous()` and `t().contiguous()` reuse native
packing, while a transposed input may become contiguous through either view
alone. Negation, scalar multiplication,
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

`transpose` binds two positional axes, `dim0`/`dim1` keywords in either order,
or a positional `dim0` with keyword `dim1`. Axes must be exact integer constants
(literals, frozen locals or guarded globals). Negative axes normalize against
rank; scalars accept `0` and `-1`. Same-axis calls still create views. Invalid
bindings and boolean/float axes raise `TypeError`, out-of-range axes raise
`IndexError`, and integers outside signed 64-bit range raise `ValueError`.
Index-like objects/integer subclasses, named dimensions and dynamic axis
expressions are deliberately unsupported. Static and dynamic policies do not
imply support for arbitrary shape-dependent Python expressions.

Arguments to `t()`, rank >2, CPU capture, other dtypes, gradients,
`swapdims`/`swapaxes`, general permute/reshape, `.T`/`.mT`, top-level
`torch.t`/`torch.transpose` and new backends remain unsupported. No-break
`fullgraph=False` retains its existing default-dynamic policy. These graphlets are non-scoring diagnostics: the frozen 38-case corpus,
performance workloads, evaluator/observer/hardware contracts and unadopted
PR1970/PR1971 campaigns are unchanged.

## Validation

Use the [locked contributor setup](../CONTRIBUTING.md), then run:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_t tests.test_compile_cuda_transpose
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_t.CompileTDeviceTests tests.test_compile_cuda_transpose.TransposeDeviceTests
CUDA_VISIBLE_DEVICES=0 cargo test --locked --features python-bindings --lib cuda_graph
```

Hardware-only tests skip clearly without the required devices. Rust test-only
accounting checks zero native operations for malformed early/late nodes; it is
absent from release builds and does not alter scoring observers.

The [clean-commit transpose capture](diagnostics/compile-cuda-transpose/postcommit-fef2ad82/README.md)
measured `fef2ad8258a09c6e3d2554d9477526500447fe27` with a fresh locked `.venv`
and release build. H100 transpose/composition, compiler, two-device, Rust and
CPU/docs checks passed with clean status throughout measurement. This completes
the capture deferred by the [development bundle](diagnostics/compile-cuda-transpose/README.md),
whose baseline and failed attempts remain unchanged. Independent review and
Burner merge gates remain required.

The [t-only clean-commit capture](diagnostics/compile-cuda-t/postcommit-2ce3dfbf/README.md)
measured `2ce3dfbf627c59a57d1e37131663a26c2ebfccad` with a fresh locked `.venv`
and release build. H100 t/packing differentials, repeated-output metadata
rejection, compiler regressions, two-device restoration, focused Rust and
CPU/docs checks passed. Its receipts verify source, installed imports, native
binary and clean status. That capture validates `t()` and its output-metadata
repair; current parameterized transpose validation is linked above.

The [review-repair diagnostics](diagnostics/compile-cuda-t/review-output-metadata/README.md)
preserve the failing reproduction and source-bound development validation.
The [initial clean capture](diagnostics/compile-cuda-t/postcommit-e01f1d0f/README.md)
remains pinned to `e01f1d0f`, before the repeated-output metadata repair.

The [original development bundle](diagnostics/compile-cuda-t/README.md), including
the fresh PR1975 baseline reproduction and failed attempts, remains unchanged.
These diagnostics do not replace independent review or Burner merge gates.
