# Marker-free CUDA addition graph capture

The generic bytecode compiler accepts exact native contiguous float32 CUDA
Tensors for one- and two-input addition graphs with
`torch.compile(fn, backend="eager", fullgraph=True)`. This is graph capture and
native operation execution, with one Rust/CUDA addition per graph operation.
It does not provide Inductor-style fusion, general default-backend CUDA
compilation, or a new CUDA performance score.

```python
import torch_rs as torch

def combine(left, right):
    intermediate = right + left
    return intermediate.add(left)

compiled = torch.compile(combine, backend="eager", fullgraph=True)
x = torch.tensor([1., 2., 3.]).to("cuda:0")
y = torch.tensor([4., 5., 6.]).to("cuda:0")
assert compiled(x, y).cpu().tolist() == [6., 9., 12.]
```

The existing operator `+` and positional `.add(tensor)` syntax supports
self-addition, chains, scalar tensors, empty tensors, and contiguous views with
nonzero offsets. Exact same-module helpers, live global tensor captures, and
tuple/list tensor outputs retain the existing bytecode restrictions. Names and
markers do not select execution paths. The original Python function is never
called by native compiled execution, and installed PyTorch is used only as a
test reference.

The existing no-break `backend="eager", fullgraph=False` option also accepts
this subset with `dynamic=None`; unsupported bytecode still raises without
eager fallback. `fullgraph=True` retains `dynamic=None/False/True`. Dynamic
variants guard rank and exact strides; a reused graph recomputes operation
shapes and requires equal CUDA addition operand shapes before executing any
operation. Dynamic shapes do not enable broadcasting.

CUDA graphs reject unary operations (including `float` and `detach`), scalar
number operands, broadcasting, noncontiguous layouts, gradients, mixed CPU/CUDA
inputs, and mixed CUDA ordinals. Keyword method arguments, other operations,
mutations, unsupported bytecode, and unsupported compiler options retain their
existing rejection behavior. Float64 CUDA tensors and CUDA tensors requiring
gradients cannot currently be constructed by the native substrate; the
compiler metadata boundary also explicitly rejects those properties.

## Metadata and cache contract

The Rust metadata hook reads dtype, device including ordinal, shape, strides,
requires-grad, and storage offset from the actual tensor, without Python
property dispatch. The compiler uses that native device metadata. CUDA input and
capture metadata guard the exact offset as well as shape/stride/dtype/device;
addition outputs have canonical contiguous strides and offset zero. CPU traces
retain their established offset-polymorphic behavior (`storage_offset=None`
in trace metadata), while the raw native metadata hook reports their real offset.

CPU, CUDA:0, and CUDA:1 specializations cannot share cache entries. Global
identity and metadata remain part of the key, including globals loaded by a
helper. Every call checks current input and capture devices and layouts.
Rebinding a global creates a specialization or hits the existing recompile
limit; an incompatible device transition is rejected. Rejected calls leave the
graph cache unchanged. A new graph is published only after successful native
execution, and the entire graph is validated before executing any operation.
Private metadata-only recorders can still describe CUDA unary graphs, but such
graphs cannot execute. Native unary hooks enforce CPU-only execution.

## Reproduction and evidence

Run from a clean checkout of the committed implementation with a freshly built
release wheel, PyTorch 2.13.0, and caches/temp directories inside the worktree.
Write all diagnostic outputs under ignored `target/` first; copy retained
reports into `docs/diagnostics/` only after all measurements have finished and
the checkout is still clean. The report's `base_commit` identifies the committed
source measured by the diagnostic.

```bash
mkdir -p target/tmp target/cache
export TMPDIR="$PWD/target/tmp" XDG_CACHE_HOME="$PWD/target/cache"
export CUDA_CACHE_PATH="$PWD/target/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$PWD/target/cache/inductor"
export TRITON_CACHE_DIR="$PWD/target/cache/triton"
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python .github/scripts/verify_native_extension.py
.venv/bin/python -m unittest tests.test_compile_cuda_boundary
.venv/bin/python scripts/diagnose_compile_cuda_add.py \
  --output target/compile-cuda-add-single.json
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_compile_cuda_boundary.CompileCudaDeviceTests
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/diagnose_compile_cuda_add.py \
  --output target/compile-cuda-add-multi.json
```

The diagnostic is separate from the frozen 38-case coverage corpus and all
scoring registries. It records reference eligibility by comparing stock
`torch.compile(..., backend="eager", fullgraph=...)` with stock eager execution
under both declared fullgraph settings. Each case uses two seeded input data
sets, materializes output, records metadata and value hashes, and detects calls
to the original function by the native compiler. Unsupported native behavior
is explicit, including rejection during input construction. Reference-ineligible
mixed-device operations are recorded separately. No timings or scores are
computed. The unit tests separately prove cache reuse, IEEE edge behavior,
layout/offset guards, global transitions, failures without cache publication,
PyTorch import independence, and restoration of current device on GPUs 0/1.

Checked-in reports in `docs/diagnostics/compile-cuda-add-*.json` bind evidence to
source hashes and the installed extension hash. They record Python, Rust,
PyTorch, driver, GPU, native runtime, and release build configuration. Native
addition uses embedded PTX 6.0 targeting sm_50, JIT-compiled by the NVIDIA driver;
CUDA 12.6 nvcc is available on this host but is not used by this native path.

See [validation history](compile-cuda-add-validation.md) for source validation
results, known baseline Python factory-order and boolean-buffer failures, and
composite evidence provenance. Those historical results are separate from the
usage and reproduction contract above.
