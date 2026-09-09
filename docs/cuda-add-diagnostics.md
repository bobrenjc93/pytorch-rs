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

Addition still completes the legacy stream before returning, including on
launch failure, with all owners alive. This preserves independent-stream reads,
immediate input/output drops, aliases, cross-thread use and synchronous CPU
copies. It also leaves a per-call completion cost that PyTorch can amortize in
sustained batches. Performance claims must retain that limitation.

## Measured repair results

The [baseline raw report](benchmark-data/cuda-add-before.json) and
[candidate raw report](benchmark-data/cuda-add-candidate.json) use the same
corrected runner, fixed seed, timing boundaries and output checks. Each contains
76 cells per cache condition, with 18 raw samples per implementation per cell.
All bitwise output checks passed. These are feature diagnostics, not evaluator
scores or a claim of full PyTorch performance parity.

| Cache condition | Boundary | Before capped parity | Candidate capped parity |
| --- | --- | ---: | ---: |
| Clean | Isolated | 91.11% | 96.45% |
| Clean | 32-call batch | 58.37% | 68.62% |
| Clean | 64-call batch | 54.96% | 65.29% |
| Clean | 32-call chain | 56.95% | 67.17% |
| Saturated | Isolated | 78.08% | 96.93% |
| Saturated | 32-call batch | 37.82% | 69.10% |
| Saturated | 64-call batch | 36.94% | 67.58% |
| Saturated | 32-call chain | 37.23% | 68.06% |

Selected saturated-process operator cells below use 64-call batches. Values
are median microseconds per call; the raw reports retain every cell and sample.

| Elements | Native before | Native candidate | PyTorch before → candidate run |
| ---: | ---: | ---: | ---: |
| 3,079 | 12.76 | 8.63 | 4.70 → 4.65 |
| 1,048,603 | 164.20 | 9.92 | 5.44 → 5.69 |
| 4,194,359 | 205.40 | 33.21 | 27.75 → 26.96 |
| 17,000,003 | 380.57 | 106.48 | 97.87 → 97.56 |
| 33,554,467 | 646.19 | 204.02 | 185.93 → 185.63 |
| 67,108,867 | 1218.67 | 399.24 | 365.28 → 365.05 |

The severe allocation-cache cliffs are substantially reduced, including above
64 MiB and in chains. Small-call sustained throughput still trails PyTorch:
for example, the saturated 3,079-element cell takes 8.63 µs versus 4.65 µs.
Completion remains synchronous per native call. A 67,108,867-element output
exceeds the unused-pool budget; its isolated candidate latency is 693.80 µs
versus PyTorch’s 480.46 µs in the saturated process. That remaining allocation
cost is visible even though its batch latency improves substantially.

| Build stage | Baseline seconds | Candidate seconds |
| --- | ---: | ---: |
| Locked dependency setup | 0.112 | 0.104 |
| Release wheel build | 44.327 | 44.674 |
| Wheel installation | 0.194 | 0.176 |

Both builds used empty per-export Cargo targets and warm worktree-local
Cargo/uv download caches. Timing used CPython 3.12.13, PyTorch 2.13.0+cu130,
one pinned CPU thread, H100, CUDA runtime 13000 and driver 580.82.07. Rust
1.92.0 used release optimization, thin LTO and one codegen unit. The reports
record full hardware, library, compiler, import, source, wheel and native hashes.

The baseline is a clean export of `cc0068c2acec798aa222edaf8c0fb62defbeb936`.
The candidate is a clean export of `3c167233398803251121b8f1a9b7e0ab55ee54c5`,
rebuilt with `--revision HEAD` from a clean composite worktree. Its report records
that commit in `measured_code_commit`, `source_matches_commit: true`, and an empty
`origin_status`. The installed native extension matches the release wheel, and
the runner matches the committed source. Only regenerated evidence and this
report documentation change after the measured commit; implementation and
benchmark-harness files remain unchanged. Burner owns the subsequent delivery.

## Integration validation

For the committed-source provenance refresh, wheel/import verification passed,
as did 55 focused documentation/CUDA tests (four single-GPU skips) and all three
two-GPU device-guard tests. Both complete diagnostic matrices passed their
bitwise output checks. The rebuilt native extension and all Python package
files match the previously validated wheel byte for byte. The broader checks
below were performed during integration; this evidence-only refresh did not
repeat the full Python/Rust suites or memory-sanitizer runs.

The same verified release-wheel native extension passed the full Python suite
with managed CPython 3.12.12 and 3.14.5: 5,210 tests each, seven skips each.
Each environment occupied the real worktree `.venv` directory during its run,
and `.github/scripts/verify_native_extension.py` checked its imports. The final
rebuild reproduced the tested native extension and every Python module byte for
byte; only the wheel's CycloneDX build report and corresponding `RECORD` changed.
The raw diagnostic records the final wheel's own hash. Commands:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  .github/scripts/verify_native_extension.py
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  -m unittest discover -s tests -p 'test_*.py'
```

The successful 3.12 run additionally set `PYTHONPYCACHEPREFIX` to an empty
worktree-local directory and `PYTHONDONTWRITEBYTECODE=1`. The preceding full
run had three unchanged `nn.factory_kwargs` order-comparison failures. A probe
of identical source versus cached code demonstrated different set iteration
orders under the same hash seed; loading both packages from fresh source
eliminated those failures without changing their implementation or assertions.
Earlier noncanonical runs also exposed a validator's required `.venv` path
layout and CPython 3.12.13/GCC noncanonical-bool-buffer differences reproduced
with the baseline extension. The performance comparison retains that same
3.12.13 interpreter on both sides; it uses float32 inputs, not bool buffers.

`cargo fmt --check`, Clippy with `-D warnings` both with and without
`python-bindings`, `cargo test --all-targets` (350 tests), its
`--features python-bindings` variant (361 tests), and `cargo test --doc` passed.
The documented `python -m unittest tests.test_readme_quickstart` check passed
all ten tests. Normal two-device addition/transfer guards passed all three
tests with `CUDA_VISIBLE_DEVICES=0,1`.

Compute Sanitizer 2025.3.1 reported **zero errors** for:

- The Python CUDA addition, composite mean/conversion, host-transfer, view and
  roundtrip suites: 45 tests, four expected single-GPU skips.
- The three two-GPU device-guard tests, including cross-thread pool releases.
- All three standalone Rust CUDA-add tests, including actual concurrent Rust
  threads and owner drops.

Memcheck used `--track-stream-ordered-races all --error-exitcode 99` and, for
Python, `--target-processes all` and `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.
`--report-api-errors no` excludes intentionally invalid device ordinals from
the sanitizer error count; the tests still assert those errors. Memory and
stream-order checking remain enabled. For example:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
  compute-sanitizer --tool memcheck --track-stream-ordered-races all \
  --report-api-errors no --target-processes all --error-exitcode 99 \
  .venv/bin/python -m unittest tests.test_cuda_add \
  tests.test_composite_cuda_mean_convert tests.test_cuda_host_transfer \
  tests.test_cuda_native_views tests.test_cuda_zero_roundtrip
```

The primary runtime was CUDA 13.0 (`13000`), with H100 GPUs and driver
580.82.07. Standalone Rust addition (three tests) and the pool-budget test also
passed against CUDA 12.6 (`12060`). The pool test queries real CUDA accounting
after large live allocations and verifies that drained unused backing respects
the configured budget. Available nvcc was 12.6.85; the addition kernel uses
driver JIT, not nvcc.
