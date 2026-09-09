# Native contiguous CUDA float32 scalar multiplication

`x * scalar`, `scalar * x`, `Tensor.mul`/`Tensor.multiply`, and
`torch_rs.mul`/`torch_rs.multiply` use the existing scalar conversion and override
bindings, then `Tensor::mul_scalar` and a native CUDA driver kernel. The kernel
uses round-to-nearest float32 multiplication without flush-to-zero and a 64-bit
grid-stride loop. It accepts arbitrary supported contiguous shapes and lengths,
rank-zero tensors, empty tensors, and contiguous nonzero-offset views. It does
not import PyTorch, compute values on the CPU, or use benchmark dispatch.

Output storage is fresh on the input's CUDA ordinal, with offset zero and the
shared scalar layout planner's strides. Singleton dimensions can have
noncanonical contiguous strides, matching PyTorch's scalar operation. Inputs
are preserved. Empty views never form or dereference an input pointer, including
views whose metadata offset exceeds the backing allocation. Bounds and byte
sizes are checked before launch. Device guards span allocation and initialization;
launch and completion failures propagate without publishing output, and disable
allocation reuse. The shared unary storage helper synchronizes the legacy stream
before releasing either allocation, including after launch errors.

Execution completes before return and restores the caller's current device.
Calling under an external PyTorch stream context still executes on the native
legacy stream; completed outputs can then be consumed on an independent stream.
Public stream selection and externally racing writes to native device pointers
remain unsupported.

Noncontiguous CUDA layouts, dtype conversion/complex promotion, autograd,
tensor-tensor multiplication, explicit `out` (including `out=None` under the
existing multiplication bindings), in-place variants, and scalar-multiply
compiler capture remain rejected. CUDA dropout's former CPU-only nonidentity
boundary is retained. Existing CPU arithmetic, scalar conversion errors,
overrides, layout and autograd behavior are unchanged. Eager multiplication
adds no compiler capture or cache-preflight support.

## Validation and provenance

The implementation tests in [test_cuda_mul_scalar.py](../tests/test_cuda_mul_scalar.py)
compare public CUDA outputs and metadata with stable PyTorch 2.13 on real H100
hardware. They cover generated mixed-sign inputs; positive, negative, zero and
nontrivial scalars; Python/NumPy scalar conversion and overflow errors; signed
zeros, subnormals, infinities and NaNs; irregular block/grid tails; a fully
materialized 17,000,003-element result exceeding the 64 MiB front cache; offset,
singleton and empty views; immutable inputs; output ownership; fresh-thread
context initialization; independent-stream reads; overrides; unsupported
boundaries; and compiler rejection before graph execution or cache insertion.
Non-NaN values compare bitwise; NaNs compare by classification rather than payload.
Two-device tests use only GPUs 0,1, query driver pointer memory type/device/managed
attributes, and check ownership and device restoration
through success, rejection and destruction. A request beyond device virtual
address capacity checks actual CUDA allocation failure and recovery without
filling physical GPU memory.

[Rust coverage](../tests/cuda_mul_scalar.rs) runs without Python and validates
native multiplication and CPU-only boundaries. Internal storage tests check
invalid/overflowing bounds, out-of-range empty offsets, allocation-size overflow,
and an injected launch failure after a real GPU allocation, followed by successful
execution with reuse disabled. The injection does not induce a hardware fault or
exhaust GPU memory. Hardware-only tests skip clearly without the required devices.

## Clean committed measurement

The post-commit [capture](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/) measures clean implementation commit
`e87d5db6d54c6f095867dba0720c3b8af73c4be6` on H100 GPU 0 against
PyTorch 2.13.0+cu130; explicit device checks use GPUs 0,1. All measurements
completed with an empty tracked-worktree status before evidence and documentation
were copied into the repository. No implementation or measurement harness changed.

The [fresh build receipt](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/build-record.json) records an initially absent
Cargo target, release configuration, compiler/runtime paths and source fingerprint
`7fa54535257b6e105d2330485dab6d09fc86b2a9ed263adc9c0ee383f4fcc930`.
The [installed-wheel receipt](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/installed-runtime.json) verifies the wheel's
native binary and every packaged Python source against the checkout. Math-evaluator
workers use the source-copy binary; the Python checks and diagnostics use the
installed wheel with `PYTHONPATH` empty. Both native hashes reproduce the author's
original build. [Command receipts](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/checks-record.json) and the
[final audit](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/provenance-audit.json) retain clean source identities,
commands, timestamps, device masks, cache paths, and artifact hashes.

| Clean-commit check | Result | Raw evidence |
| --- | --- | --- |
| Scalar differential, GPU 0 | 8 passed; 2 two-device skips | [log](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/differential-final.log) |
| Existing CUDA/compiler/scalar binding checks | 53 passed; 6 two-device skips | [log](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/focused.log) |
| Scalar ownership/allocation recovery, GPUs 0,1 | 2 passed | [log](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/two-device-final.log) |
| Rust bounds/launch failure cleanup, GPUs 0,1 | 1 passed | [log](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/rust-two-device.log) |
| Fixed CUDA math evaluator | 3/6 on each of the same three seeds; unsupported cases remain zero | [report](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/evaluation.json) |
| Fixed compile evaluator | 38/38 eligible cases passed | [report](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/compile-evaluation.json) |
| Existing neg/add diagnostic | 168 expectations met; 128 native passes, 40 unsupported | [report](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/neg-add-diagnostic.json) |
| Fixed CUDA performance, fresh / reused caches | 4/4 correct each; reference/native geometric means 1.2850× / 1.3368× | [fresh](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/cuda-performance-fresh.json), [reused](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/cuda-performance-reused.json) |
| Public-add diagnostic | 76 rows per cache condition; all bitwise checks passed | [report](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/cuda-add-regression.json) |

The performance recapture preserves the author's setup: new CUDA/Triton/Inductor
caches and a rebuilt private pointwise library in the first run, an existing
pointwise-reduce library in both runs, then reuse of all caches in the second.
The [cache receipt](diagnostics/cuda-mul-scalar/post-commit-e87d5db6/performance-cache-record.json) hashes the reduction
cache before measurement. Both runs retain the unchanged four shapes, five
warmups, 17 samples and three calls per sample. No timing run was selected or
retried. These private workload results establish no scalar-multiply compiler support.

The public-add recapture retains five warmups, nine samples, both implementation
orders and two fresh processes. Capped parity for isolated / 32-call / 64-call
work is 97.08 / 69.54 / 67.23 percent without the saturation prelude,
and 96.94 / 69.99 / 68.03 percent after saturation. All slow rows remain
in the raw report. These measurements establish neither general performance
non-regression nor scalar-multiplication performance parity.

## Original author validation and superseded measurements

The original provisional files below remain byte-for-byte intact. Their CUDA,
compiler and performance measurements are superseded by the clean-commit recapture
above and are not current-candidate performance evidence. The author’s full Rust
and Python validation and historical baseline reproductions were preserved without
rerunning unrelated full suites during this evidence-only step.

The [original evidence directory](diagnostics/cuda-mul-scalar/) records the author’s fresh
release build, source hashes, native binaries, test commands, baseline
reproductions and unchanged evaluator outputs. The source snapshot is based on
HEAD `41dcf4a5a015337a61f4507940cbeb20fd4ff006` plus the explicitly recorded
uncommitted implementation. Its production source fingerprint is
`7fa54535257b6e105d2330485dab6d09fc86b2a9ed263adc9c0ee383f4fcc930`.
The build receipt records a new empty Cargo target, not an incremental native
build. A subsequent release wheel build and install are recorded separately;
its stripped extension and Python sources were verified against the wheel and
checkout. Source-copy checks and installed-wheel checks have separate native
hashes in the provenance records.

| Check | Result | Raw evidence |
| --- | --- | --- |
| Rust default / Python bindings | 357 / 368 passed on GPU 0 | [default](diagnostics/cuda-mul-scalar/rust-default.log), [bindings](diagnostics/cuda-mul-scalar/rust-bindings.log) |
| Formatting and Clippy (both configurations) | passed | [format](diagnostics/cuda-mul-scalar/fmt.log), [default](diagnostics/cuda-mul-scalar/clippy-default.log), [bindings](diagnostics/cuda-mul-scalar/clippy-bindings.log) |
| Final scalar differential tests | 10 tests; 8 passed, 2 device-specific skips | [log](diagnostics/cuda-mul-scalar/differential-final.log) |
| Existing focused Python suite | 124 tests; passed, 8 device-specific skips | [log](diagnostics/cuda-mul-scalar/focused.log) |
| GPUs 0,1 | 8 Python checks and 1 Rust failure-path check passed; final scalar tests also passed direct pointer-ownership assertions | [Python](diagnostics/cuda-mul-scalar/two-device.log), [Rust](diagnostics/cuda-mul-scalar/rust-two-device.log), [final scalar](diagnostics/cuda-mul-scalar/two-device-final.log) |
| Full installed-wheel Python suite | 5,341 tests; 2 existing failures, 13 skips | [log](diagnostics/cuda-mul-scalar/python-full-installed.log) |
| Fixed CUDA math evaluator | 3/6 passed on all three seeds; all other cases retained at zero | [report](diagnostics/cuda-mul-scalar/evaluation.json) |
| Fixed compile evaluator | 38/38 eligible cases passed | [report](diagnostics/cuda-mul-scalar/compile-evaluation-installed.json) |
| Existing neg/add diagnostic | 168 expectations met: 128 native passes, 40 unsupported outcomes | [report](diagnostics/cuda-mul-scalar/neg-add-diagnostic-installed.json) |
| Fixed CUDA performance, new driver/compiler caches / reused caches | 4/4 correct in each run; reference/native geometric-mean ratios 1.5033× / 1.2264× | [fresh](diagnostics/cuda-mul-scalar/cuda-performance-fresh.json), [reused](diagnostics/cuda-mul-scalar/cuda-performance-reused.json) |
| Unchanged public-add diagnostic | 76 rows in each of two cache conditions; all bitwise checks passed | [report](diagnostics/cuda-mul-scalar/cuda-add-regression.json) |

The two private performance runs use the unchanged four-shape matrix, five
warmups, 17 samples and three calls per sample, with synchronization and output
materialization. Both report 100% capped scores within that private workload.
The first uses new CUDA/Triton/Inductor caches and rebuilds the private pointwise
library; the pointwise-reduce library is reused from the full suite in both runs.
Thus the first run is not a fully cold private-kernel build. The second reuses
all these caches. Both predeclared runs are retained, with no selection
or retry of timings. These results establish no new scalar-multiply compiler
support or general performance parity.

The public-add diagnostic retains its default seed, five warmups, nine samples
per implementation order, and two fresh processes. Its capped reference/native
parity percentages for isolated / 32-call / 64-call work were
96.27 / 69.66 / 67.52 with no saturation prelude, and
97.33 / 70.87 / 68.87 after saturation. The historical candidate report recorded
97.94 / 79.50 / 77.90 and 98.18 / 71.19 / 69.90 respectively. The lower sustained
ratios in this run's first condition are retained, not discarded or retried;
these measurements do not establish general performance non-regression or a
scalar-multiplication speed claim. Historical evidence remains unchanged.

Both full-suite failures are the existing noncanonical boolean-buffer cases in
`test_tensor_buffer_reference`: native conversion produces `[0., 1.]` for the
rejected comparison, while PyTorch produces `[1., 1.]`. Both reproduce in the
freshly compiled, unmodified baseline `41dcf4a5` archive inside this worktree at hash seeds
0 and 6. The [baseline receipt](diagnostics/cuda-mul-scalar/baseline-record.json)
verifies the archive's production blobs against that baseline and records the native
binary and commands; [seed 0](diagnostics/cuda-mul-scalar/baseline-failures-seed0.log)
and [seed 6](diagnostics/cuda-mul-scalar/baseline-failures-seed6.log) preserve
those failures. Factory-keyword tests passed in both baseline reproductions.
No unrelated implementation fixes were made.

The initial [source-copy full run](diagnostics/cuda-mul-scalar/python-full.log)
also had a diagnostic provenance failure because `PYTHONPATH=python` bypassed
the required installed wheel. The full installed-wheel rerun above resolved it.
The diagnostic's provenance guard remains unchanged.

Host: NVIDIA H100, compute capability 9.0, driver 580.82.07; CPython 3.12.14,
PyTorch 2.13.0+cu130, explicitly selected CUDA runtime 13.0. Rust/Cargo 1.92.0
built release `extension-module`/`abi3-py310`, thin LTO, one codegen unit.
Native eager kernels use embedded PTX 6.0 targeting `sm_50` through driver JIT;
they do not invoke nvcc. The unchanged private performance workload uses the
available nvcc 12.6.85. Exact paths, GPU inventory, runtime versions and caches
are retained in the receipts and raw reports. Historical measurements are unchanged.

## Reproduction

Use the worktree-local environment setup in the
[CUDA negation guide](cuda-neg-validation.md#reproduce-from-this-checkout), with
Rust/Cargo 1.92.0, CPython 3.12, stable PyTorch 2.13.0 and the explicit local CUDA
13 runtime. Keep build targets, dependency caches and temporary directories
inside this checkout. Use a new unused Cargo target for each fresh build.
The existing [capture procedure](diagnostics/composite-cuda-neg/reproduce.py)
can be imported with `OUT` set to a new directory under `target/`; it records
honest dirty/clean provenance and runs the unchanged six-case math evaluator.
Only copy reports into docs after all measurements finish.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python .venv/bin/python -m unittest -v \
  tests.test_cuda_mul_scalar tests.test_cuda_neg tests.test_cuda_add \
  tests.test_compile_cuda_boundary tests.test_compile_cuda_neg \
  tests.test_mul tests.test_multiply tests.test_top_level_mul tests.test_top_level_multiply
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python .venv/bin/python -m unittest -v \
  tests.test_cuda_mul_scalar.CudaMulScalarDeviceTests
CUDA_VISIBLE_DEVICES=0 cargo test --locked --all-targets --features python-bindings
CUDA_VISIBLE_DEVICES=0 cargo test --locked --all-targets
# The full suite's diagnostic checks require the installed wheel, not a source override.
.venv/bin/maturin build --release --locked --out target/mul-wheels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps target/mul-wheels/*.whl
CUDA_VISIBLE_DEVICES=0 PYTHONPATH= .venv/bin/python .github/scripts/verify_native_extension.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH= .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
CUDA_VISIBLE_DEVICES=0 PYTHONPATH= .venv/bin/python scripts/evaluate_torch_compile_coverage.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH= .venv/bin/python scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison --output target/mul-performance-fresh.json
```

The fixed math denominator stays six. Only negation, same-shape addition and
scalar multiplication can execute; trailing-vector addition, axis reduction and
matrix multiplication remain visible as unsupported and zero-credit. The fixed
compile corpus and private four-shape CUDA performance workload are unchanged;
reruns check regression risk and establish no new compiler or performance scope.
Historical reports and measurements remain untouched.
