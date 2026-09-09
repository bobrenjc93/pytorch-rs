# Native CUDA addition diagnostics

This diagnostic covers public same-shape contiguous float32 CUDA addition. It
is feature evidence, not an evaluator, a general CUDA benchmark, or a change to
historical reports. The three public forms, scalar and empty tensors, nonzero
seeded data, generated shapes, offset inputs, chained calls, cache saturation,
and outputs above 64 MiB and 256 MiB are included.

Run from a clean worktree at the committed implementation revision, using a
locked Python 3.12 development/reference environment. Keep caches in
this worktree, including `UV_CACHE_DIR`, `CARGO_HOME`, `TMPDIR`, and
`CUDA_CACHE_PATH`. The build helper uses an empty target directory for each
source export and records actual setup, build and installation durations. Its
shared download caches are explicitly recorded as warm.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/build_cuda_add_diagnostic.py --name candidate --revision HEAD
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diagnose_cuda_add.py \
  --build-record target/cuda-add-diagnostic/candidate/build-record.json \
  --output docs/benchmark-data/cuda-add-candidate.json
```

Pass `--revision <commit>` to the build helper for a committed baseline. Each
name must be new. It exports source under `target/cuda-add-diagnostic/`, builds a
release wheel, force-installs it into `.venv`, and checks the installed native
extension against the wheel bytes. The diagnostic verifies source file hashes
and the installed extension hash before retaining evidence.

Retained baseline and candidate evidence must use clean committed source.
The candidate command above exports `HEAD` without an implementation overlay.
Omitting `--revision` remains useful for local scratch diagnostics under
`target/`, but those reports record an uncommitted overlay and are not suitable
as retained release evidence. After measurement, keep the follow-up diff limited
to reports and their documentation so the measured implementation and benchmark
harness remain unchanged. In Burner workflows, Burner owns both implementation
and subsequent evidence commits.

Each cache condition runs in a fresh process. The clean condition runs the
ordered matrix without a saturation prelude; it does not claim cold allocation
for every cell. Saturation holds and then drops the same 68 mixed-size zero
allocations for each implementation, exceeding the original cache's entry and
byte budgets. Each cell uses both implementation orders, five warmup blocks and
nine measured blocks per order. Every sample includes device completion and is
followed by bitwise output readback on both implementations outside timing.
Both implementations' inputs live in the same per-shape dictionary; temporary
loop bindings are cleared after each implementation so neither retains the
previous shape's inputs while the next shape allocates.
Isolated calls, 32-call and 64-call batches, and 32-call dependency chains are
reported separately, with raw samples, medians, dispersion and capped geometric
parity. A faster cell cannot offset a slower one in those capped aggregates.

The cache repair replaces old entries instead of refusing new sizes forever.
Best-fit reuse permits at most 25% excess capacity, with logical tensor bounds
kept separate from allocation capacity. The front cache is bounded globally by
32 allocations and 64 MiB. Where
supported, larger allocations and front-cache misses use a private CUDA pool
with legacy-stream allocation/release ordering. Each pool tracks live
allocation capacity and sets its release threshold to that capacity plus
256 MiB. This budgets unused backing independently of large live inputs.
Unused pool memory above the budget is reclaimed at synchronization, subject
to CUDA allocation granularity; pending frees may persist until the next wait.
The matrix includes a cell larger than the pool's unused-memory budget so its
remaining costs stay visible. Devices without pool support retain the
synchronous allocator and bounded front cache.
Aligned inputs use a general four-float vector kernel with scalar tail handling;
unaligned views keep the scalar kernel. Both use round-to-nearest float32
addition without flushing subnormals.

Repeated live arguments can use a cached driver graph, keyed by the
context-specific kernel function, input/output addresses and element count.
The cache retains at most 64 executable entries and 64 recent signatures.
Callers retain the executable and all tensor owners through completion;
entries own executable metadata, not tensor storage. This is an internal
eager-add optimization. The public CPU compiler still rejects CUDA inputs
and captures.

Addition still completes the legacy stream before returning, including on
launch failure, with all owners alive. This preserves independent-stream reads,
immediate input/output drops, aliases, cross-thread use and synchronous CPU
copies. It also leaves a per-call completion cost that PyTorch can amortize in
sustained batches. Performance claims must retain that limitation.

## Current candidate results

The [baseline raw report](benchmark-data/cuda-add-before.json) and
[candidate raw report](benchmark-data/cuda-add-candidate.json) use the same
byte-identical runner, fixed seed, timing boundaries and output checks. The
candidate build includes allocation reuse, graph replay, reduction threading
and compiler device checks from the committed composite. This matrix
measures CUDA addition performance; it does not measure CPU reduction speed.
Each report contains 76 cells per cache condition, with 18 raw samples per
implementation per cell. All bitwise output checks passed. These are feature
diagnostics, not evaluator scores or a claim of full PyTorch performance parity.

| Cache condition | Boundary | Before capped parity | Candidate capped parity |
| --- | --- | ---: | ---: |
| Clean | Isolated | 91.11% | 97.94% |
| Clean | 32-call batch | 58.37% | 70.57% |
| Clean | 64-call batch | 54.96% | 68.64% |
| Clean | 32-call chain | 56.95% | 69.76% |
| Saturated | Isolated | 78.08% | 98.20% |
| Saturated | 32-call batch | 37.82% | 75.77% |
| Saturated | 64-call batch | 36.94% | 73.97% |
| Saturated | 32-call chain | 37.23% | 75.01% |

Selected saturated-process operator cells below use 64-call batches. Values
are median microseconds per call; the raw reports retain every cell and sample.

| Elements | Native before | Native candidate | PyTorch before → candidate run |
| ---: | ---: | ---: | ---: |
| 3,079 | 12.76 | 8.96 | 4.70 → 6.24 |
| 1,048,603 | 164.20 | 9.44 | 5.44 → 5.43 |
| 4,194,359 | 205.40 | 33.00 | 27.75 → 26.83 |
| 17,000,003 | 380.57 | 106.54 | 97.87 → 97.64 |
| 33,554,467 | 646.19 | 204.86 | 185.93 → 185.77 |
| 67,108,867 | 1218.67 | 400.72 | 365.28 → 365.15 |

The severe allocation-cache cliffs are reduced in these measured cells,
including outputs above 64 MiB. Small-call sustained throughput still trails
PyTorch: the saturated 3,079-element cell takes 8.96 µs versus
6.24 µs. Completion remains synchronous per native call. A
67,108,867-element output exceeds the unused-pool budget; its isolated candidate
latency is 679.17 µs versus PyTorch’s 480.83 µs in the saturated
process. That remaining cost stays visible alongside improved batch latency.

Clean and saturated conditions run in separate processes. The higher saturated
aggregate does not establish a speedup caused by saturation: for 3,079 elements,
the clean process records 8.22 µs native and 4.65 µs PyTorch,
while the saturated process records 8.96 µs and 6.24 µs respectively.
The comparison with the original baseline also includes multiple implementation
repairs; it does not isolate graph replay's contribution.

| Build stage | Baseline seconds | Candidate seconds |
| --- | ---: | ---: |
| Locked dependency setup | 0.112 | 0.120 |
| Release wheel build | 44.327 | 45.724 |
| Wheel installation | 0.194 | 0.210 |

Both builds used empty per-export Cargo targets and warm worktree-local
Cargo/uv download caches. Timing used CPython 3.12.13, PyTorch 2.13.0+cu130,
one pinned CPU thread, H100, CUDA runtime 13000 and driver 580.82.07. Rust
1.92.0 used release optimization, thin LTO and one codegen unit. The existing
worktree-local CUDA driver JIT cache was not flushed; “clean” describes the
absence of an allocation-saturation prelude. Available nvcc was 12.6.85; the
addition kernel uses driver JIT, not nvcc. The reports record hardware, library,
compiler, import, source, wheel and native hashes.

The baseline is the unchanged clean export of
`cc0068c2acec798aa222edaf8c0fb62defbeb936`. The candidate is a clean export of
`2181d81ec70804016da8ed7f6f4b275edc832d89`, rebuilt with `--revision HEAD` from a clean
composite worktree. Its report records that commit in `measured_code_commit`,
`source_matches_commit: true`, and an empty `origin_status`. The exported file
hashes were checked against the committed Git archive. Source, build, wheel,
runner and import paths are rooted inside this composite worktree. The installed
native extension matches the release wheel, and the runner matches the committed
source. This follow-up changes only evidence and documentation; Burner owns its
commit. No implementation or benchmark-harness change intervenes between the
measured commit and this evidence refresh.

## Validation

The clean-export wheel matches all 60 package members of the wheel previously
used to validate the compiler fix, including the native extension, byte for byte.
That implementation passed full suites on managed CPython 3.12.12 and 3.14.5:
5,221 tests each, eight expected skips each. It also passed `cargo fmt --check`,
Clippy with `-D warnings` with and without `python-bindings`,
`cargo test --all-targets` (352 tests), its `--features python-bindings` variant
(363 tests), and `cargo test --doc`. Those full suites were not repeated for
this evidence-only refresh.

This refresh reran wheel/import verification, 60 focused CUDA, compiler
and README tests (five single-GPU skips), and all five two-GPU device-guard
tests. Both complete diagnostic matrices passed their bitwise output checks.
The audit also verified all 5,472 raw samples, medians and capped aggregates,
matching output hashes, and the unchanged runner and workload matrix.

Compute Sanitizer 2025.3.1 reported **zero errors** for the 50-test CUDA and
compiler suite (five single-GPU skips), including immediate drops,
aliases, cross-thread use, graph-cache eviction and CPU-copy correctness.
Memcheck kept memory and stream-order checking enabled; `--report-api-errors no`
excludes intentionally invalid device ordinals from its error count while the
tests continue to assert those errors. Commands used for the refresh include:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  .github/scripts/verify_native_extension.py
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest \
  tests.test_compile_cuda_boundary tests.test_cuda_add \
  tests.test_cuda_host_transfer tests.test_cuda_native_views \
  tests.test_cuda_zero_roundtrip tests.test_composite_cuda_mean_convert \
  tests.test_readme_quickstart
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_cuda_add.CudaAddDeviceTests \
  tests.test_cuda_host_transfer.CudaHostTransferDeviceGuardTests \
  tests.test_cuda_zero_roundtrip.CudaCurrentDeviceGuardTests
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck \
  --track-stream-ordered-races all --report-api-errors no \
  --target-processes all --error-exitcode 99 \
  .venv/bin/python -m unittest tests.test_compile_cuda_boundary \
  tests.test_cuda_add tests.test_cuda_host_transfer tests.test_cuda_native_views \
  tests.test_cuda_zero_roundtrip tests.test_composite_cuda_mean_convert
```

All commands used worktree-local caches and temporary directories. The full
3.12 correctness run additionally used an empty worktree-local
`PYTHONPYCACHEPREFIX` with `PYTHONDONTWRITEBYTECODE=1`; the performance diagnostic
retains the same CPython 3.12.13 interpreter on both implementations.
