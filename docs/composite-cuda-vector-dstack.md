# CUDA trailing-vector addition and depth stacking integration

This integrates the reviewed eager CUDA `(M,N)+(N,)` and CPU `dstack` scopes
on main `c8d56df6bbae2f3efcdfc4f9f124c063762f35c3`. README, FEATURES,
the canonical supported surface, and the architecture source map now describe
contiguous float32 CUDA matrix/vector addition on one device in either order.
Compiler addition still requires equal shapes. Other broadcasts, layouts,
dtypes, nonunit alpha, concrete out, and CUDA autograd remain excluded.
The source kernel, independent bounds checks, empty-input handling, device
restoration, completion-before-return, and lifecycle tests are preserved.

`dstack` still accepts only exact native CPU float32 scalars, vectors, and
matrices. Its rank-three normalized views now reach a generic native depth
concatenation fast path. This path borrows each input's storage and strides,
copies depth blocks, and validates entire strided rows for singleton depth.
It supports arbitrary row/column sizes, unequal depths, offset and transposed
views, and strided depth axes. Shared gradient storage keeps the generic
fallback. Public rank-three input rejection, dispatch, gradients, allocation
and shape-overflow checks, empty results, and fresh storage remain intact.

## Evidence status

The original [PR #1936 capture](cuda-add-trailing-vector.md#historical-source-validation-pr-1936)
remains pinned to its source revision, original paths, and binary hashes.
Existing historical benchmarks, failed records, frozen evaluation definitions,
feature weights, the 38-case compiler corpus, and the private four-workload CUDA
benchmark are unchanged. Source PR scores are repair leads, not measurements of
this composite against the current baseline.

New captures are **precommit diagnostics**, with actual dirty status and source
fingerprints. They do not establish clean-commit performance credit or a
repository-wide percentage. The integration task forbids committing, so fresh
clean-commit evidence, independent review, complete baseline recalculation,
confirmed no-regression gates, exact-head CI, and managed merge remain Burner
delivery gates after these changes are committed. No managed progress artifact
or generator is changed here.

## Reproduce after Burner commits the candidate

Use a real canonical `.venv` within this checkout; this integration selected
CPython 3.14.5 and the locked dev/reference dependencies. Python versions must
have separate self-contained checkouts. Keep caches and temporary files local.
The environment preflight needs a one-thread numerical-library budget to avoid
oversubscribing this 384-logical-CPU host during subprocess validation.

```bash
export CARGO_HOME="$PWD/target/cargo-home"
export UV_CACHE_DIR="$PWD/target/uv-cache"
export TMPDIR="$PWD/target/tmp"
export XDG_CACHE_HOME="$PWD/target/cache"
export CUDA_CACHE_PATH="$PWD/target/cache/cuda"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCH_RS_CUDART="$PWD/.venv/lib/python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13"
mkdir -p "$CARGO_HOME" "$UV_CACHE_DIR" "$TMPDIR" "$CUDA_CACHE_PATH"
```

Populate the local Cargo registry with `cargo fetch --locked` before the
offline build. The capture helper requires an unused output directory, records
actual commands and timestamps, and refuses dirty trees by default. It installs
the release wheel into `.venv` and supplies identical extension bytes to the
source package required by the unchanged CUDA math evaluator. `--allow-dirty`
is only for explicitly labeled precommit diagnostics, as in this integration.

```bash
.venv/bin/python scripts/capture_depth_concat_build.py --output target/clean-composite
.venv/bin/python -m unittest tests.test_readme_quickstart \
  tests.test_compile_cuda_neg_add_diagnostic \
  tests.test_top_level_stack_benchmark_artifact.TopLevelStackBenchmarkArtifactTests.test_generated_validator_smoke_artifact_validates
.venv/bin/python scripts/benchmark_depth_concat.py \
  --samples 21 --repeats 3 \
  --build-record target/clean-composite/build-record.json \
  --output target/clean-composite/depth.json
.venv/bin/python scripts/evaluate_cuda_math.py \
  --seed 5027103527016457416 --seed 3727445652986941402 --seed 6885146872121100961 \
  --build-record target/clean-composite/build-record.json \
  --output target/clean-composite/cuda-math.json
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_cuda_add_trailing_vector.CudaAddTrailingVectorDeviceTests
```

The depth benchmark uses 15 symmetric warmup blocks, 21 samples of three
calls per implementation in each of two reversed orders, fixed inputs,
one CPU thread and pinned affinity, and bitwise materialization after every
block. JSON retains all samples and per-order median/MAD, all cases, source,
harness and native binary hashes, interpreter identity, dependencies, and
build configuration. Inputs are constructed outside timing on both sides.
CUDA timing is not inferred from this CPU diagnostic; the six-case evaluator
is correctness-only and keeps unsupported reduction/matmul in its denominator.

## Build and runtime identity

The measured production fingerprint is
`5cf0651daeb2577d3a0816620cf54e92a7620bbace9f6782ff7818456bd8b967`;
the installed and source-package extension SHA-256 is
`5a2ee9083b4343da4c49544215f0966b8a99aca73c8bec32f23cc24cda9cb038`.
These identify dirty integrated source over merged HEAD
`444faba3f8f793c68a069d65355c46fe3f650417`, not a clean implementation commit.
The build uses Rust/cargo 1.92.0, release thin LTO, one codegen unit, and
`extension-module`; CPython is 3.14.5 (built with Clang 22.1.3), with locked
PyTorch 2.13.0+cu130 and NumPy 2.5.1. CPU timings pin CPU 0 on an AMD EPYC 9654.
CUDA checks use H100 GPU 0 (and 0,1 for the bounded ordinal test), compute
capability 9.0, driver 580.82.07, and the local CUDA runtime reporting 13000.
Installed nvcc is 12.6.85 and is unused: the driver JIT compiles embedded PTX.
Exact interpreter/compiler paths and hashes, dependencies, driver/runtime
library paths and hashes, source files, and build commands are in the receipts.

## Measured release timings (precommit)

The bottleneck reproduced in the merged source before repair. The following
values pool both execution orders; each entry is median ± MAD in microseconds.
Raw per-order results and samples remain in the linked JSON. Input creation is
excluded symmetrically and every measured block is materialized bitwise.

| Workload | Before: native | Final: native | Final: PyTorch | Confirmation: native | Confirmation: PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| scalar | 1.02 ± 0.03 | 0.94 ± 0.02 | 2.47 ± 0.06 | 0.96 ± 0.02 | 2.59 ± 0.08 |
| vector | 6.24 ± 0.06 | 1.95 ± 0.03 | 5.36 ± 0.12 | 2.07 ± 0.10 | 5.38 ± 0.04 |
| matrix 3x7 dense | 1.18 ± 0.02 | 0.95 ± 0.02 | 2.65 ± 0.06 | 0.97 ± 0.02 | 2.77 ± 0.10 |
| matrix 3x7 transpose | 1.20 ± 0.03 | 0.92 ± 0.01 | 3.83 ± 0.14 | 0.93 ± 0.03 | 3.76 ± 0.09 |
| matrix 3x7 offset | 1.20 ± 0.03 | 0.95 ± 0.02 | 2.66 ± 0.06 | 0.95 ± 0.03 | 2.82 ± 0.08 |
| matrix 3x7 strided | 1.22 ± 0.02 | 0.94 ± 0.02 | 3.67 ± 0.06 | 0.95 ± 0.02 | 3.73 ± 0.11 |
| matrix 257x263 dense | 870.72 ± 19.56 | 107.51 ± 3.96 | 249.30 ± 7.42 | 108.70 ± 5.29 | 240.39 ± 5.66 |
| matrix 257x263 transpose | 952.36 ± 20.78 | 221.89 ± 11.25 | 197.94 ± 16.92 | 197.13 ± 19.82 | 163.74 ± 13.74 |
| matrix 257x263 offset | 869.93 ± 13.63 | 110.92 ± 4.57 | 249.59 ± 5.81 | 107.02 ± 7.50 | 239.51 ± 6.74 |
| matrix 257x263 strided | 916.74 ± 9.18 | 117.44 ± 4.69 | 119.08 ± 6.46 | 116.88 ± 5.12 | 112.93 ± 6.96 |
| matrix 129x521 dense | 866.95 ± 12.96 | 109.62 ± 4.84 | 243.86 ± 4.56 | 105.80 ± 5.89 | 232.04 ± 5.23 |
| matrix 129x521 transpose | 1021.10 ± 21.42 | 188.42 ± 11.59 | 193.43 ± 14.90 | 158.23 ± 14.75 | 171.43 ± 13.01 |
| matrix 129x521 offset | 861.46 ± 7.80 | 106.88 ± 2.60 | 242.57 ± 5.40 | 108.16 ± 4.28 | 237.78 ± 7.39 |
| matrix 129x521 strided | 906.79 ± 10.01 | 113.93 ± 4.73 | 121.34 ± 5.57 | 112.45 ± 5.24 | 112.05 ± 6.33 |
| three matrices dense | 30.43 ± 1.34 | 5.25 ± 0.12 | 10.80 ± 0.18 | 5.32 ± 0.11 | 10.80 ± 0.16 |
| three matrices transpose | 31.48 ± 0.68 | 5.11 ± 0.07 | 8.19 ± 0.20 | 5.28 ± 0.09 | 8.19 ± 0.22 |
| three matrices offset | 29.13 ± 0.28 | 5.17 ± 0.11 | 10.80 ± 0.16 | 5.42 ± 0.11 | 10.94 ± 0.18 |
| three matrices strided | 30.56 ± 0.13 | 5.20 ± 0.10 | 8.09 ± 0.18 | 5.17 ± 0.09 | 7.88 ± 0.20 |
| cat axis 0 | 18.09 ± 0.12 | 18.00 ± 0.12 | 7.01 ± 0.28 | 17.96 ± 0.09 | 6.93 ± 0.19 |
| cat axis 1 | 18.63 ± 0.21 | 21.00 ± 2.05 | 6.84 ± 0.18 | 18.77 ± 0.12 | 6.85 ± 0.22 |
| empty | 0.86 ± 0.02 | 0.84 ± 0.02 | 2.20 ± 0.02 | 0.84 ± 0.03 | 2.25 ± 0.04 |

The 257×263 contiguous case improves from 870.72 to 107.51 µs; its transposed
case improves from 952.36 to 221.89 µs but remains slower than PyTorch’s
197.94 µs in this capture. The unchanged rank-two `cat(dim=1)` control measures
12.7% slower than before (21.00 versus 18.63 µs); this result is retained.
The [confirmation capture](diagnostics/composite-cuda-vector-dstack/precommit/depth-confirmation.json)
ran after the full suite, with no concurrent test process. The contiguous case
remains 108.70 µs versus PyTorch 240.39 µs. The 257×263 transposed case is
197.13 versus 163.74 µs, and its strided case is 116.88 versus 112.93 µs;
these remaining PyTorch advantages are not hidden. The rank-two `cat(dim=1)`
control is 18.77 ± 0.12 µs, 0.8% above its pre-repair median; the earlier 12.7%
slowdown did not repeat at that magnitude. Both captures remain in the record.
These are targeted diagnostics, not a repository-wide performance percentage.

## Retained artifacts

- [Pre-repair release timings](diagnostics/composite-cuda-vector-dstack/pre-repair/depth.json) bind the original merged source and original release wheel.
- [Intermediate timings](diagnostics/composite-cuda-vector-dstack/intermediate/depth.json) retain the first optimization before row-level bounds validation.
- [Failed intermediate math run](diagnostics/composite-cuda-vector-dstack/intermediate/cuda-math-failed.json) retains zero credit from the missing source-package extension. The evaluator was not changed.
- [Final precommit timings](diagnostics/composite-cuda-vector-dstack/precommit/depth.json), [math diagnostic](diagnostics/composite-cuda-vector-dstack/precommit/cuda-math.json), and [build receipt](diagnostics/composite-cuda-vector-dstack/precommit/build-record.json) bind the integrated implementation.
- [Identity audit](diagnostics/composite-cuda-vector-dstack/precommit/identities.json) verifies all 59 installed Python sources against this checkout, byte-identical installed/source extensions, and loaded CUDA library hashes.
- [Artifact manifest](diagnostics/composite-cuda-vector-dstack/manifest.json) records byte-for-byte copies, verified historical wheel contents, and source file hashes. Intermediate and final source patches reconstruct their measured code from merged HEAD `444faba`.

Raw failed smoke/setup logs are kept under `diagnostics/composite-cuda-vector-dstack/checks/`.
The first benchmark harness attempted unsupported public rank-three inputs;
the corrected benchmark retains the public boundary and exercises internal
rank-three shapes through Rust differential tests. An earlier harness startup
also tried an unavailable native interop-thread setter; the final harness checks
the existing one-thread native interop budget instead. Rust test startup needed
a local copy of `libpython3.14.so.1.0` plus the selected interpreter’s read-only
standard-library prefix (`PYTHONHOME`) and `PYTHONDONTWRITEBYTECODE=1`.
No test assertion or timeout was relaxed.

## Validation outcome

- The unchanged six-case real-CUDA math evaluator passes **4/6** at all three
  recorded seeds: equal-shape add, trailing-vector add, negation, and scalar
  multiplication. All 18 reference executions pass. Reduction and matmul retain
  zero credit. Candidate workers load no PyTorch modules.
- All **370 Rust tests** pass with Python bindings and real CUDA enabled;
  this includes generated native rank-three fast-path/fallback bit comparisons,
  existing autograd/layout/overflow coverage, and CUDA storage boundaries.
  Clippy with warnings denied and `cargo fmt --check` pass.
- The full Python 3.14 run executes **5,419 tests**, with **15 skips** and only
  two documentation failures. One assertion still required the superseded CUDA
  shape wording; the other required a colon in the documentation index. Both
  were repaired, and all **12 documentation tests pass** on the final text.
  All implementation tests in the full run passed; the entire suite was not
  repeated after these documentation-only fixes. The failed full log and
  successful focused rerun are both retained.
- Documentation-command smoke tests pass on GPU 0. The stack-validator smoke
  passes with one-thread numerical-library environment settings; earlier
  subprocess timeout logs are retained. No timeout or containment assertion
  was weakened, and `.venv` was never switched or symlinked.
- The bounded GPU 0,1 trailing-vector test passes device restoration and mixed
  ordinal rejection. The native-extension provenance check passes.
- Feature weights, locks, fixed evaluators/corpora/private benchmarks, and
  Burner-owned progress artifacts are unchanged. Clean-commit validation and
  managed review/scoring/CI/merge remain the explicit delivery gates above.
