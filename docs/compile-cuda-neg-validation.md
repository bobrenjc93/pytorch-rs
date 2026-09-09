# Native CUDA neg/add capture validation

The shared marker-free eager graph path accepts unary minus, `Tensor.neg()` and
`Tensor.negative()` over exact native contiguous float32 CUDA tensors, composed
with same-shape addition. This is unfused bounded capture, not a general
Inductor compiler or a performance-parity claim. See the [owning guide](compile-cuda-add.md).

## Integrated clean-commit evidence

The current capture measures clean implementation commit
`df1964bd297b6368bf8eb3b235394cde5f6e723b` in the composite worktree. It supersedes
the source-PR measurements below for validation of the integrated candidate.
The release wheel was rebuilt in a new empty Cargo target, using the existing
locked worktree dependencies. Installed Python sources and native binaries were
verified against the checkout and wheel. Every measurement checked the complete
tracked source snapshot and empty git status before and after execution. Reports
were staged under `target/post-commit-df1964bd/reports/` and copied byte-for-byte
only after the captures and provenance audit completed. This refresh changes
only evidence and its documentation.

The [build receipt](diagnostics/compile-cuda-neg/integrated-df1964bd/build-record.json),
[command receipts](diagnostics/compile-cuda-neg/integrated-df1964bd/checks-record.json),
[source hashes](diagnostics/compile-cuda-neg/integrated-df1964bd/source-files.json),
[provenance audit](diagnostics/compile-cuda-neg/integrated-df1964bd/provenance-audit.json),
and [artifact hashes](diagnostics/compile-cuda-neg/integrated-df1964bd/artifact-sha256.json)
bind the actual source/native hashes, commands, timestamps, environments, caches,
local import/build paths and raw results. All current worktree paths resolve
inside this composite. The original source-PR, author, baseline, kernel and
earlier composite artifacts remain unchanged.

This capture used CPython 3.12.14+meta, PyTorch 2.13.0+cu130, CUDA runtime 13.0,
NVIDIA H100 and driver 580.82.07. Rust/Cargo 1.92.0 built the release ABI3
extension with thin LTO and one codegen unit. Native neg/add uses driver-JIT PTX;
the unchanged private performance benchmark uses nvcc 12.6.85. This was a fresh
native build, not a dependency-installation timing.

| Check | Result | Raw evidence |
| --- | --- | --- |
| Focused integrated tests | 124 tests; passed, 6 device-specific skips | [log](diagnostics/compile-cuda-neg/integrated-df1964bd/focused.log) |
| Two-device checks, GPUs 0,1 | 6 passed | [log](diagnostics/compile-cuda-neg/integrated-df1964bd/two-device.log) |
| Fixed compile evaluator | 38/38 eligible cases passed | [report](diagnostics/compile-cuda-neg/integrated-df1964bd/compile-evaluation.json) |
| Fixed CUDA math evaluator | 2/6 cases passed on all three seeds | [report](diagnostics/compile-cuda-neg/integrated-df1964bd/cuda-math.json) |
| Current `neg_add_v1` diagnostic | 168 single-device and 12 two-device expectations met | [GPU 0](diagnostics/compile-cuda-neg/integrated-df1964bd/neg-add-single.json), [GPUs 0,1](diagnostics/compile-cuda-neg/integrated-df1964bd/neg-add-multi.json) |
| Fixed CUDA performance, fresh caches | 4/4 correct; 1.4673x geometric-mean ratio, 93.04% capped score | [report](diagnostics/compile-cuda-neg/integrated-df1964bd/cuda-performance-fresh.json) |
| Fixed CUDA performance, reused caches | 4/4 correct; 0.9027x geometric-mean ratio, 77.38% capped score | [report](diagnostics/compile-cuda-neg/integrated-df1964bd/cuda-performance.json) |

The compile denominator remains 38, and the CUDA math denominator remains six
with seeds 9173, 260909 and 903217. All four unsupported math cases retain zero
credit. The maintained diagnostic separately records 128 native passes and 40
unsupported outcomes on GPU 0, and eight passes and four unsupported outcomes
on GPUs 0,1. Unsupported outcomes are successful guard checks, not native
execution credit. The frozen addition-only diagnostic and its obsolete
`reject_neg` results remain preserved below.

Both performance runs retain the same four shapes, reference/options, five
warmups, 17 samples, three calls per sample, synchronization and materialization.
The first run used new CUDA/Triton/Inductor cache directories and rebuilt both
private kernel libraries; the second reused those caches. The
[cache receipt](diagnostics/compile-cuda-neg/integrated-df1964bd/performance-cache-record.json)
records the old caches preserved locally and the new paths. These were the two
predeclared runs, with no selection or retry of slow outcomes. Raw timings,
dispersion, memory pressure and zero-credit rules remain intact. The results do
not establish performance non-regression or general neg/add compilation parity.
Full suites and unrelated historical failure reproductions were not rerun in
this evidence step; independent review and merge gates still apply.

## Source-PR clean-commit evidence

The retained source-PR reports measure clean implementation commit
`c29e953e5cf3c74fe5fbcbd38177ecc0181dc08b` in its original compiler worktree. They replaced
the pre-commit compiler candidate measurements. They do not measure the integrated
column_stack/negation candidate and provide no current-composite performance
credit. Their original paths, hashes and raw results remain unchanged. An
integrated candidate requires its own clean-commit build and measurements;
local checks under ignored `target/` are not a substitute for that gate.
A new release build target and fresh native wheels were used for the source PR;
every build and measurement checked the complete
tracked source tree and an empty git status before and after execution. All
reports were first written under ignored `target/post-commit-c29e953/` and
copied here only after measurements finished. The source PR's subsequent
`f71ce298afe9c11193090d06271636c642c608dd` diff contains only evidence and its
documentation; that statement does not apply to the later composite integration.

The [build receipt](diagnostics/compile-cuda-neg/build-record.json),
[command receipts](diagnostics/compile-cuda-neg/checks-record.json),
[GPU command receipts](diagnostics/compile-cuda-neg/gpu-checks-record.json), and
[source/provenance audit](diagnostics/compile-cuda-neg/provenance-audit.log)
record the actual commit, source/build hashes, commands, timestamps, cache
state, local interpreter/import paths, runtime and device configuration.
[Artifact hashes](diagnostics/compile-cuda-neg/artifact-sha256.json) bind the
retained raw files. No measurements or provenance fields were hand-edited.

All build outputs, dependencies, caches and temporary files stayed inside the
original compiler worktree. The existing pinned local Python environment and
Cargo registry were reused; this is a fresh native build, not a dependency-installation timing.
CPython 3.12.14 and PyTorch 2.13.0+cu130 use the worktree-local CUDA 13 runtime.
The GPUs are NVIDIA H100, compute capability 9.0, driver 580.82.07. Rust/Cargo
1.92.0 build release extension-module/abi3-py310 with thin LTO and one codegen
unit. Native negation/addition use embedded PTX 6.0/sm_50 through driver JIT,
without nvcc. The unchanged private performance benchmark separately records
nvcc 12.6.85 targeting sm_90 and CUDA runtime 13.0.

## Source-PR measurements

| Check | Result | Raw evidence |
| --- | --- | --- |
| Focused GPU and compiler suite | 126 tests; passed, 6 device-specific skips | [log](diagnostics/compile-cuda-neg/focused-final.log) |
| Two-device suite, GPUs 0,1 | 6 passed | [log](diagnostics/compile-cuda-neg/two-device-final.log) |
| Unchanged compile evaluator | 38/38 existing cases passed | [report](diagnostics/compile-cuda-neg/compile-evaluation.json) |
| Unchanged CUDA math evaluator | 2/6 fixed cases passed on all 3 seeds | [report](diagnostics/compile-cuda-neg/cuda-math.json) |
| Unchanged CUDA performance benchmark | 4/4 shapes passed in each run | [fresh-cache report](diagnostics/compile-cuda-neg/cuda-performance-fresh.json), [cache-reusing report](diagnostics/compile-cuda-neg/cuda-performance.json) |
| Unchanged addition diagnostic | 58 passed, 24 unsupported on GPU 0; 4 passed, 2 unsupported on GPUs 0,1 | [GPU 0](diagnostics/compile-cuda-neg/addition-diagnostic.json), [GPUs 0,1](diagnostics/compile-cuda-neg/addition-diagnostic-multi.json) |

The fixed compile corpus still has 38 cases and does not measure the new CUDA
capture surface. CUDA math still has six cases with seeds 9173, 260909 and
903217. Broadcasting, scalar multiplication, axis reduction and matmul retain
zero credit. The focused tests separately exercise the new generated neg/add
programs, guards, output metadata, streams and two-device ownership.

Both performance runs retain the same four shapes, reference, five warmups,
17 samples, three calls per sample, synchronization and output materialization.
The fresh-cache run rebuilds the private pointwise library and uses new CUDA,
Triton and Inductor cache directories; the second run reuses those caches.
Recorded benchmark aggregates: fresh-cache: 1.2929x geometric-mean speed ratio, 100.00% capped score; cache-reusing: 1.2557x geometric-mean speed ratio, 100.00% capped score.
Raw samples, cold-call accounting, variance, memory pressure and slow results
remain in the reports. These timings measure the unchanged private workload,
not generic neg/add capture, and establish no new overall compiler, hardware,
training or performance parity.

The unchanged addition diagnostic's GPU-0 `reject_neg` expectation is obsolete:
its two negation outputs match the reference, but the script returns exit 1 for
those successful cases. The audit checks that these are its only expectation
mismatches. The multi-device diagnostic returns exit 0. Neither its expectation
flags nor any scoring definition was changed.

## Preserved author and baseline history

The earlier author full suites and failure reproductions below remain pinned
to their original measurements. They were not rerun or attributed to the
post-commit build. The [original author build receipt](diagnostics/compile-cuda-neg/author-build-record.json)
and [original check record](diagnostics/compile-cuda-neg/author-checks-record.json)
preserve the pre-commit production fingerprint and dirty-patch provenance.
The [production patch](diagnostics/compile-cuda-neg/production.patch) is an
original author artifact, not the source identity of the clean-commit measurements.

| Author check | Original result | Preserved evidence |
| --- | --- | --- |
| Rust default / Python bindings | 354 / 365 passed | [default](diagnostics/compile-cuda-neg/rust-default-tests.log), [bindings](diagnostics/compile-cuda-neg/rust-tests.log) |
| Formatting / Clippy | passed | [fmt](diagnostics/compile-cuda-neg/fmt.log), [Clippy](diagnostics/compile-cuda-neg/clippy.log) |
| Full Python suite | 5,311 tests; 5 failures, 11 skips | [log](diagnostics/compile-cuda-neg/python-full.log) |

The author full run found one obsolete compiled-negation rejection assertion
in `test_cuda_neg.py`, later corrected in the committed implementation. Its
layout and autograd rejections remain. The earlier
[focused run](diagnostics/compile-cuda-neg/focused.log) preserves the same
obsolete assertion failure; the [initial run](diagnostics/compile-cuda-neg/focused-initial.log)
preserves two documentation failures subsequently fixed. The source-PR focused
suite above verifies the final assertions. This evidence refresh does not
replace independent review or full-suite merge gates.

The other four author full-run failures concern factory keyword ordering and
noncanonical boolean buffers. The preserved clean-baseline wheel at
`e4cddf0ecb45c23c683953fa2f8c45904b170a80` reproduced both buffer failures
([log](diagnostics/compile-cuda-neg/baseline-buffer.log)) and factory-order
failures at `PYTHONHASHSEED=6`
([log](diagnostics/compile-cuda-neg/baseline-factory-seed6.log)). The author's
isolated candidate seed-6 factory rerun passed
([log](diagnostics/compile-cuda-neg/candidate-factory-seed6.log)); its buffer
rerun still failed both boolean cases
([log](diagnostics/compile-cuda-neg/candidate-baseline-failures-seed6.log)).
These intermittent ordering and buffer results were not refreshed. Baseline
[build](diagnostics/compile-cuda-neg/baseline-build-record.json) and
[environment](diagnostics/compile-cuda-neg/baseline-repro-provenance.json)
receipts and all historical logs are byte-for-byte unchanged.

## Reproduce the measurements

Use the worktree-local dependency and cache setup in the
[kernel validation guide](cuda-neg-validation.md#reproduce-from-this-checkout).
Start from a clean checkout of the committed code. Select a new, unused
`CARGO_TARGET_DIR` for the candidate build, and keep generated reports under
ignored `target/` until all measurements finish. With that configuration and
`CUDA_VISIBLE_DEVICES=0`:

```bash
.venv/bin/maturin build --release --locked --out target/compile-neg-wheels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/compile-neg-wheels/*.whl
cp "$CARGO_TARGET_DIR/release/libpytorch_rs.so" python/torch_rs/torch_rs.abi3.so
.venv/bin/python .github/scripts/verify_native_extension.py
.venv/bin/python -m unittest -v tests.test_compile_cuda_neg \
  tests.test_compile_cuda_boundary tests.test_cuda_neg tests.test_cuda_add \
  tests.test_cuda_native_views tests.test_top_level_compile \
  tests.test_readme_quickstart tests.test_cuda_math_evaluator \
  tests.test_hardware_heterogeneity_evaluation
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v \
  tests.test_compile_cuda_neg.CompileCudaNegDeviceTests \
  tests.test_compile_cuda_boundary.CompileCudaDeviceTests \
  tests.test_cuda_neg.CudaNegDeviceTests tests.test_cuda_add.CudaAddDeviceTests
bash scripts/evaluate_torch_compile_coverage.sh > target/compile-neg-coverage.json
# Use new CUDA_CACHE_PATH, TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR paths.
# Move any existing target/torch_rs_private_cuda_pointwise cache to a new local
# directory before the first run, so its report includes the nvcc command.
.venv/bin/python scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison --output target/compile-neg-performance-fresh.json
.venv/bin/python scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison --output target/compile-neg-performance.json
```

Before running the unchanged CUDA math evaluator, create a build receipt using
the [existing receipt procedure](hardware-heterogeneity-evaluator.md#reproduce-and-bind-a-native-build).
Then run:

```bash
.venv/bin/python scripts/evaluate_cuda_math.py \
  --seed 9173 --seed 260909 --seed 903217 \
  --build-record target/compile-neg-build-record.json \
  --output target/compile-neg-cuda-math.json
.venv/bin/python scripts/diagnose_compile_cuda_neg_add.py \
  --case-set neg_add_v1 --output target/compile-neg-add-diagnostic.json
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/diagnose_compile_cuda_neg_add.py \
  --case-set neg_add_v1 --output target/compile-neg-add-diagnostic-multi.json
```
