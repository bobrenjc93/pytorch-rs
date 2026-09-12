# Ordinary CUDA arithmetic and ReLU graph capture

The generic bytecode compiler accepts exact native contiguous float32 CUDA
Tensors for one- and two-input scalar-multiply/negation/ReLU/addition graphs, plus
[rank-2 matrix products](compile-cuda-matmul.md), with
`torch.compile(fn, backend="eager", fullgraph=True)`. This is graph capture and
native operation execution, with one Rust/CUDA kernel per recorded operation.
This is unfused bounded capture, not a general Inductor compiler or a
performance-parity claim.
It does not provide Inductor-style fusion, general default-backend CUDA
compilation, or a new CUDA performance score.

```python
import torch_rs as torch

def combine(left, right):
    intermediate = torch.add(torch.neg(right), left.mul(0.5))
    return 2 * intermediate

compiled = torch.compile(combine, backend="eager", fullgraph=True)
x = torch.tensor([1., 2., 3.]).to("cuda:0")
y = torch.tensor([4., 5., 6.]).to("cuda:0")
assert compiled(x, y).cpu().tolist() == [-7., -8., -9.]
```

`x * scalar`, `scalar * x`, positional `.mul(scalar)`/`.multiply(scalar)`
and positional `torch_rs.mul`/`torch_rs.multiply` calls (including imported
aliases) compose with negation, ReLU and addition. Scalars must be exact Python
`bool`, `int`, or `float` literals, local constants, or module globals, including
globals read by the existing same-module helper. Conversion uses the public
scalar multiplication parser, including integer overflow and float32 rounding.
Scalar function arguments, closures, numeric subclasses/NumPy scalars, complex
values, arithmetic on captured scalars, kwargs and `out` remain unsupported.
Top-level calls require an immutable native callable identity; unsupported replacements reject.

Unary `-`, zero-argument `.neg()`, `.negative()` and `.relu()` compose arbitrarily with
operator `+` and positional `.add(tensor)`. Positional `torch_rs.add(x, y)`,
`torch_rs.neg(x)` and `torch_rs.negative(x)` also capture, including module
aliases, direct imported aliases and calls inside supported same-module helpers.
These spellings reuse the existing add/neg nodes and native kernels. This syntax supports
equal-shape addition and exactly `(M, N)+(N,)` in either operand order,
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
shapes and validates equal shapes or exactly `(M, N)` and `(N,)` before
executing any operation. Dynamic shapes do not enable broader broadcasting.

[No-argument squeeze](compile-cuda-squeeze.md) also captures rank-0/1/2
shared-storage views; its strided outputs require packing before arithmetic.

CUDA graphs reject other unary operations (including `float` and `detach`), scalar
number operands outside multiplication, broader broadcasts (including `(M,N)+(1,N)`,
`(M,N)+(M,1)`, rank-three plus vector, or scalar expansion), noncontiguous layouts,
gradients, mixed CPU/CUDA inputs, and mixed CUDA ordinals. The bounded
[view](compile-cuda-view.md), [reshape](compile-cuda-reshape.md), transpose and
row-sum methods retain their documented keyword forms. Other keyword arguments and operations,
closures, mutations, unsupported bytecode,
and unsupported compiler options retain their existing rejection behavior.
Float64 CUDA tensors and CUDA tensors requiring gradients cannot currently be constructed by the native substrate; the
compiler metadata boundary also explicitly rejects those properties.
The compiler reuses [native CUDA scalar multiplication](cuda-mul-scalar-validation.md);
CPU multiplication capture and tensor-tensor multiplication remain unsupported.
CPU module/direct-import call capture remains unsupported, including these new
spellings; CPU operator and method capture retains its existing behavior.
Function calls accept only the stated positional Tensor arguments: no `alpha`,
`out`, keywords, scalar addition or extra/missing arguments.

ReLU uses [native CUDA compare/select](cuda-relu.md), preserving NaN payloads
and clamping negative zero to positive zero. Its method capture accepts any
contiguous rank already admitted by negation. Top-level `torch.relu` and
functional ReLU capture remain unsupported.

## Metadata and cache contract

The Rust metadata hook reads dtype, device including ordinal, shape, strides,
requires-grad, and storage offset from the actual tensor, without Python
property dispatch. The compiler uses that native device metadata. CUDA input and
capture metadata guard the exact offset as well as shape/stride/dtype/device;
negation, ReLU and equal-shape addition outputs own fresh CUDA storage with canonical
contiguous strides and offset zero. Matrix/vector addition uses the shared
elementwise stride planner, preserving singleton and empty output ordering;
scalar multiplication uses the scalar layout planner. All allocate fresh storage
with offset zero, including offset-view inputs. CPU traces retain their established offset-polymorphic behavior (`storage_offset=None`
in trace metadata), while the raw native metadata hook reports their real offset.

CPU, CUDA:0, and CUDA:1 specializations cannot share cache entries. Global
identity and metadata remain part of the key, including globals loaded by a
helper. Scalar globals also guard exact type and value (binary64 bits for floats,
including signed zero and NaNs); literals are guarded by the function code.
A specialization stores the validated float32 scalar value. Scalar and callable
bindings are reread on every call before cache lookup; helper dependencies use
the same snapshots. Changed values specialize subject to `recompile_limit`,
and unsupported replacements reject. Every call checks current input and capture
devices and layouts.
Module guards retain the legacy `mul`/`multiply`/`matmul` snapshots and guard
new `add`/`neg`/`negative` fields only at loads that use them. Unused new fields
do not invalidate warm graphs or spend `recompile_limit`, even when rebound
or deleted. Direct imported aliases guard their own binding and survive
unrelated public-attribute mutations. Replacing a used binding with another
canonical native operation dispatches that operation with its own arity;
unsupported callables, arbitrary modules, descriptors and missing used fields
reject without user callbacks. Each load keeps its own snapshot, including
multiple fields and helper loads in the same namespace.
The immutable callable owner is retained during package initialization. The lazy
frontend never resolves identities through the writable native owner export;
replacing or deleting that export before the first compile does not authorize
counterfeit callables or invoke their hooks.
Rebinding a global creates a specialization or hits the existing recompile
limit; an incompatible device transition is rejected. Rejected calls leave the
graph cache unchanged. A new graph is published only after successful native
execution, and the entire graph is validated before executing any operation.
Private metadata-only recorders can still describe unsupported CUDA unary graphs;
the executor rejects these before any kernel runs. Native unary hooks admit
negation and ReLU on CUDA and retain CPU-only guards for all other unary targets.
The private binary bridge independently delegates shape, layout, dtype, device
and autograd validation to `Tensor::add`; it cannot bypass the kernel boundary.
CUDA add nodes also validate declared result metadata before execution, while
dynamic outputs derive concrete metadata from current inputs.

## Public add/neg function-call validation

[Clean-commit module arithmetic evidence](diagnostics/compile-cuda-module-arithmetic/postcommit-35a52edc/README.md)
records validation at `35a52edc`, including the owner-initialization review fix, with the [baseline and development checks](diagnostics/compile-cuda-module-arithmetic/README.md) preserved. Reproduce the focused
suite with `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_arithmetic`;
use `CUDA_VISIBLE_DEVICES=0,1` for its `ModuleArithmeticDeviceTests` class and an
empty device mask for the hardware-free frontend checks. Tests cover all four
policies, cache hits, exact exception classes, helpers, generated compositions,
layout/broadcast boundaries, dynamic squeeze rank changes, lifetime, binding
mutations and output prevalidation. Single-node arithmetic uses the existing
native unary/binary hook; compositions execute through the native whole-graph
bridge without per-node Python replay. Neither path runs the program body.

The [owner-initialization review revision](diagnostics/compile-cuda-module-arithmetic/owner-startup-review/README.md)
adds fresh-process checks for substitutions made before the first frontend
import. The clean `35a52edc` capture includes those checks; its development
validation and the [earlier `e2378805` capture](diagnostics/compile-cuda-module-arithmetic/postcommit-e2378805/README.md) remain preserved at their original source identities.

## Matrix/vector validation

[Trailing-vector validation](compile-cuda-trailing-vector-validation.md) gives
the reproducible `trailing_vector_v1` diagnostic, H100 test commands, and evidence
requirements. Reference backend `eager` establishes bounded semantics only.
The frozen 38-case compiler corpus and four-workload CUDA benchmark are unchanged;
this capability does not imply a score gain.

## Scalar multiplication validation

[Scalar capture validation](compile-cuda-mul-scalar-validation.md) describes the
independent generated graphlets, concrete programs, cache/alias/fail-closed tests,
and `mul_neg_add_v1` non-scoring diagnostic. Reference tracing uses PyTorch 2.13
`backend="eager"`; it provides correctness coverage, not Inductor performance parity.
The frozen 38-case compile corpus, addition-only and neg/add diagnostics, private
four-workload benchmark, scores and historical reports remain unchanged.

## Negation validation

`tests/test_compile_cuda_neg.py` exercises generated shapes and graph lengths,
all three unary spellings, neg/add compositions, scalar/empty/offset inputs,
IEEE values, fresh output storage, cold and CPU-warmed caches, dynamic shape,
stride/offset/device guards, live global/helper captures, rejected closures,
nested outputs, repeated inputs, preserved inputs, streams, and two-device
ownership. Unsupported late operations fail before graph execution or cache
insertion. The original Python function is blocked during differential tests;
a subprocess additionally blocks installed-PyTorch imports.

Run `tests.test_compile_cuda_neg` on GPU 0 and
`tests.test_compile_cuda_neg.CompileCudaNegDeviceTests` on GPUs 0,1 alongside the
existing boundary tests below. See [candidate validation](compile-cuda-neg-validation.md)
for raw logs and fresh build evidence measured at clean integrated commit
`df1964bd297b6368bf8eb3b235394cde5f6e723b`. Source-PR reports remain pinned to
their original measured revisions in that guide.
The fixed scoring corpora and historical measurements are unchanged. The frozen
addition-only diagnostic retains two obsolete `reject_neg` expectations and
returns exit 1 on supported negation; those are historical expectation failures,
not harness passes. The historical `neg_add_v1` command below now returns exit 1 on GPU 0:
its four matrix/vector rejection expectations are obsolete. All other
expectations remain enforced; do not label that run a passing diagnostic.
Use [trailing-vector validation](compile-cuda-trailing-vector-validation.md) for
the versioned current capability diagnostic. It retains the addition matrix and unsupported guards, adds
all three negation spellings and composed graphs, and records every raw outcome.

## Reproduction and evidence

Run from a clean checkout of the committed implementation with a freshly built
release wheel, PyTorch 2.13.0, and caches/temp directories inside the worktree.
Write all diagnostic outputs under ignored `target/` first; copy retained
reports into `docs/diagnostics/` after measurements have finished. The report's
`base_commit` records HEAD; use `worktree_status` and `source_sha256` to identify
the actual measured source. A report with source edits is not exact-HEAD evidence.

```bash
mkdir -p target/tmp target/cache
export TMPDIR="$PWD/target/tmp" XDG_CACHE_HOME="$PWD/target/cache"
export CUDA_CACHE_PATH="$PWD/target/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$PWD/target/cache/inductor"
export TRITON_CACHE_DIR="$PWD/target/cache/triton"
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python .github/scripts/verify_native_extension.py
.venv/bin/python -m unittest tests.test_compile_cuda_boundary
.venv/bin/python scripts/diagnose_compile_cuda_neg_add.py \
  --case-set neg_add_v1 --output target/compile-cuda-neg-add-single.json
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_compile_cuda_boundary.CompileCudaDeviceTests
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/diagnose_compile_cuda_neg_add.py \
  --case-set neg_add_v1 --output target/compile-cuda-neg-add-multi.json
```

The diagnostic is separate from the frozen 38-case coverage corpus and all
scoring registries. It records reference eligibility by comparing stock
`torch.compile(..., backend="eager", fullgraph=...)` with stock eager execution
under both declared fullgraph settings. Each case uses two seeded input data
sets, materializes output, records metadata and value hashes, and detects calls
to the original function by the native compiler. Unsupported native behavior
is explicit, including rejection during input construction. Reference-ineligible
mixed-device operations are recorded separately. No timings or scores are
computed. Exit 0 requires every supported case to match both reference runs and
every unsupported case to reject. Unexpected reference failures, wrong outputs,
unexpected acceptance, and execution errors return nonzero, with raw results
retained. Only the declared nonscalar mixed-device cases may be reference-ineligible.
The canonical `.venv` import guard and installed-source hash
checks run before the cases; dirty-worktree status is recorded explicitly.
The unit tests separately prove cache reuse, IEEE edge behavior,
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
