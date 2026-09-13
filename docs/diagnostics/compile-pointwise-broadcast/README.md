# Default CUDA pointwise broadcasting validation

**Review status: blocked.** Subsequent H100 cancellation tests reproduce both
finite and IEEE broadcast failures, including persistent shape transitions.
Identical fresh default-Inductor runs also select numerically different kernels.
The [second review record](review-autotune-blocker.md) preserves these results
and the unresolved external numerical-contract dependency. The measurements
below remain valid for their fixed corpus; they do not resolve these findings.

The unchanged public-default-compile-v2 gates measured clean implementation
`c1f2d380abd012313741e629a5139c651be14cc6`, including the
[broadcast FMA repair](review-broadcast-order.md), on 2026-09-13 UTC. The clean
main baseline `166687a730a86235fa38decfa364703b60ebc595` retains its original
measurements from earlier that day. Only documentation and evidence changed
after the new captures; the worker created no commits.
See the [compiler contract](../../compile-pointwise-jit.md).

| Measurement | Baseline | Broadcast implementation |
| --- | ---: | ---: |
| Weighted coverage | 6% (4/112) | 10% (8/112) |
| Weighted CUDA performance | 12% (4/56) | 20% (8/56) |
| Performance common-success geometric mean, reference/candidate | 1.8377081163578368 | 1.8784593297982968 |

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
  timestamps, return codes and GPU snapshots for the original baseline and new
  candidate runs. [Post-commit validation](post-commit-validation.json.gz)
  preserves the new worker/build logs, capture commands and five passing focused
  broadcast-priority tests, all run from clean `c1f2d38`.
- The original [validation bundle](validation.json.gz) and subsequent
  [repair bundle](review-broadcast-order.json.gz) retain their original compiler
  logs, commands, environment, source/test manifests, build/Clippy logs, gate
  worker logs, failures and exhaustive partition receipts. The original bundle
  measures the initial implementation at `5adb263` or explicitly dirty development
  sources; the repair bundle explicitly records uncommitted repair validation.
  Neither substitutes for the clean candidate reports above.
  Raw worker observations and wheels remain at the
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

The repair's complete compiler selection passed **901 tests in 85 disjoint module
processes**, with **23 explicit skips** under `CUDA_VISIBLE_DEVICES=0`. Recorded
test IDs verify exhaustive coverage without overlap; source and test hashes still
match clean `c1f2d38`. The initial 896-test run remains in the original bundle.
All **three** focused
pointwise device-restoration tests subsequently passed with devices `0,1`.
The broadcast module passed nine tests with one two-device skip. Its original
hardware-absent run (two metadata tests and seven explicit skips) and 80 scalar
IEEE comparisons remain in the initial bundle. The repair's new priority module
passed all five tests again from the clean commit; its development run with CUDA
hidden passed both metadata tests and explicitly skipped all three hardware cases.
Tests check numerical values, signed zeros, metadata, storage ownership,
changed bindings/shapes, liveness, warm execution, guards and module lifetimes.

All 30 focused Rust pointwise tests passed in both default and `python-bindings`
configurations. Default and bindings builds, all-target Clippy with warnings
denied, formatting, documentation checks and diff checks passed. Independent
Moduler design and focused implementation reviews previously reported no
actionable issues within the documented concrete-shape contract. The second
review's measured failures supersede that conclusion.

The repair archive preserves the original broadcast FMA failure and three
exceptional-value failures caused by Inductor's automatic symbolic shape history.
The [numerical contract](../../compile-pointwise-numerics.md) records the limits
of that model. The recorded fresh-shape and finite-history tests passed, but
the [second review](review-autotune-blocker.md) demonstrates that their values
and reference resets missed finite cancellation failures. These measurements
do not establish unrestricted broadcast or symbolic-history parity.

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
`b1b32432462c6400e96e8c55827599af99e945d3efbd7348bcf78d006eb48251`.

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
