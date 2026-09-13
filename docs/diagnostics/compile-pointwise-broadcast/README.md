# Default CUDA pointwise broadcasting validation

The [broadcast FMA repair](review-broadcast-order.md) changes numerical lowering
after these captures. Candidate reports and generated-code captures below are
stale for that repair and must be refreshed after Burner commits it; they are
not evidence of the repaired candidate. Baseline measurements remain unchanged.

The unchanged public-default-compile-v2 gates measured clean implementation
`5adb2638f8b0d67adabc746ee38b2debabaa4f48` and clean current-main baseline
`166687a730a86235fa38decfa364703b60ebc595` on 2026-09-13 UTC. Burner saved the
implementation commit during worker recovery; the worker created no commits.
Commit `43f06fb` added only documentation and this evidence after measurement.
See the [compiler contract](../../compile-pointwise-jit.md).

| Measurement | Baseline | Broadcast implementation |
| --- | ---: | ---: |
| Weighted coverage | 6% (4/112) | 10% (8/112) |
| Weighted CUDA performance | 12% (4/56) | 20% (8/56) |
| Performance common-success geometric mean, reference/candidate | 1.8377081163578368 | 1.915793810988903 |

All four reports have `valid: true`, `diagnostic: false`. The new passing
cells belong to broadcasting; unsupported categories remain zero. These are
fixed-corpus measurements, not general Inductor parity. Common-success ratios
describe each implementation's successful subset, which differs between builds.
No evaluator, corpus, tolerance, weight, denominator, or historical evidence changed.

## Retained evidence

- [Baseline coverage](baseline-coverage.json.gz),
  [baseline performance](baseline-cuda-perf.json.gz),
  [candidate coverage](candidate-coverage.json.gz), and
  [candidate performance](candidate-cuda-perf.json.gz) retain every fixed cell,
  unsupported outcome, both CUDA orders, five warmups, 17 samples, cold/steady
  timings, synchronization and source/build/wheel provenance.
- [Gate receipts](gates-receipt.json) record exact commands, clean commits,
  timestamps, return codes and GPU snapshots before/after each run.
- [Validation bundle](validation.json.gz) maps filenames to their original text:
  all compiler logs, commands, environment, source/test manifests, build/Clippy
  logs, gate worker logs, original failures, and the exhaustive partition receipt
  (`compiler-summary.json`). Raw worker observations and wheels remain at the
  worktree-local paths recorded by the reports; their hashes were checked before
  packaging. Those large temporary artifacts are not duplicated here and may
  disappear during worktree cleanup. The reports and logs are durable.
- [Generated CUDA](kernel.cu), [PTX](kernel.ptx.gz),
  [provenance](provenance.json), and [source manifest](source-manifest.json.gz)
  come from the clean implementation's installed extension. The independent
  [capture program](capture.py) exercises changed values and shapes with ordinary
  default compilation, fresh outputs, two broadcast specializations and no
  installed-PyTorch import. [Checksums](manifest.json) cover this bundle.

## Checks and original failures

The complete compiler selection passed **896 tests in 84 disjoint module
processes**, with **23 explicit skips** under `CUDA_VISIBLE_DEVICES=0`. Recorded
test IDs verify exhaustive coverage without overlap. All **three** focused
pointwise device-restoration tests subsequently passed with devices `0,1`.
The new broadcast module passed nine tests with one two-device skip; with CUDA
hidden it passed two metadata/codegen tests and explicitly skipped all seven
hardware cases. Another 80 scalar IEEE comparisons passed against default
Inductor. Tests check numerical values, signed zeros, metadata, storage ownership,
changed bindings/shapes, liveness, warm execution, guards and module lifetimes.

All 29 focused Rust pointwise tests passed in both default and `python-bindings`
configurations. Default and bindings builds, all-target Clippy with warnings
denied, formatting, documentation checks and diff checks passed. Independent
Moduler design and focused implementation reviews found no unresolved issues.

Original logs retain two invalid test-fixture attempts (an unavailable native
integer dtype and an unsupported gradient-carrying CUDA transfer), initial
Clippy/type-inference corrections, and a Rust test loader failure corrected by
selecting the local Python 3.12 library. An existing matmul test initially failed
its hard-coded `.venv` provenance assertion in the evaluator environment; it
passed after installing the identical wheel in worktree-local `.venv`. Burner
interrupted the initial compiler sweep; the resumed sweep excluded completed
passing groups and retained the failed/interrupted logs. No failed measurement
was discarded or retried for a better score.

Hardware was H100 GPU 0, `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver
580.82.07. Python was 3.12.14 and PyTorch 2.13.0+cu130. Native JIT selected
NVRTC 13.0 and runtime 13000, targeting `compute_90` with explicit FMA and
`--ftz=false`. Queried nvcc 12.6 was not used for JIT generation. The tested,
captured and evaluated native extension has SHA256
`4c87977a0afb8180799d44882d64ba45a796ef98e803578c9363429dcf3b693a`.

## Reproduce

From each clean source revision, with all build and cache paths inside its checkout:

```bash
mkdir -p target/tmp target/cuda-cache
export TMPDIR="$PWD/target/tmp" CUDA_CACHE_PATH="$PWD/target/cuda-cache"
export XDG_CACHE_HOME="$PWD/target/cache"
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric coverage
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric cuda-perf
CUDA_VISIBLE_DEVICES=0 target/default-compile-eval/venv/bin/python \
  docs/diagnostics/compile-pointwise-broadcast/capture.py target/broadcast-capture
```

The baseline predates the broadcast capture script; run that last command on the
implementation. Full test commands and local environment settings are retained
in the validation bundle. Burner still owns final independent review,
exact-head evaluation, delivery and managed progress artifacts.
