# Default CUDA pointwise broadcasting validation

The current refinement admits unequal input shapes only for a returned original
IR expression with arithmetic depth at most one and no live sin/cos. ReLU
preserves depth; add/subtract/multiply and tensor negation add a stage. Actual
shape equality applies even for unused inputs or linear address maps. Equal-shape
support is unchanged. See the [compiler contract](../../compile-pointwise-jit.md)
and [numerical boundary](../../compile-pointwise-numerics.md#unequal-shape-numerical-boundary).

New clean candidate and clean-main coverage/performance captures are deferred to
Burner's post-commit evidence phase. **The 10%/20% results below measure the old
broad candidate, not the narrowed implementation.** Excluded cells remain zero
under the unchanged gates. The new [bounded capture program](capture_bounded.py)
captures an accepted non-corpus expression and identifies the module dispatched
for each observation. No narrowed-domain measurement is claimed by the historical
files in this directory.

## Bounded-domain development validation

The [development bundle](bounded-validation.json.gz) records the uncommitted
refinement over `39b7983f42469900eacf1cecb60166ec5394df89`, with source/test hashes,
build logs, exact process receipts, reproduction commands and retained preliminary
probes. It is not a clean-commit evaluation. Both default and `python-bindings`
configurations pass 31 focused Rust tests, builds and all-target Clippy with
warnings denied. Formatting and 12 documentation tests pass.
The complete compiler selection passes **908 tests in 86 disjoint module
processes**, with **23 explicit skips** under `CUDA_VISIBLE_DEVICES=0`. The
receipt verifies exhaustive file coverage, unique test IDs, and final test-source
hashes; every process exits zero. This includes the existing equal-shape liveness
and arithmetic-contraction regressions.

Persistent default-native/default-Inductor wrappers exercise the accepted
primitives and ReLU compositions across changed shapes, bindings, scalar kinds,
IEEE values, offsets and empty outputs. Historical cancellation programs now
assert explicit rejection, including direct cached-kernel calls with unequal
shapes and identical linear address maps. Three focused two-device tests pass;
with CUDA hidden, six metadata tests pass and 15 hardware tests explicitly skip.
The independent admission review and a separately recorded hardware-free probe
of 1,232 accepted generated sources are retained in the bundle.

[Bounded generated CUDA](bounded-kernel.cu), [PTX](bounded-kernel.ptx.gz),
[provenance](bounded-provenance.json) and [source manifest](bounded-source-manifest.json.gz)
describe the actually dispatched final module for `(x.relu()-y.relu()).relu()`.
The capture observes a warm hit and a later shape specialization, matching each
to its cache guard and source/PTX hashes. H100 execution selected NVRTC 13.0,
runtime 13000 and `compute_90`; queried nvcc 12.6 was not the JIT compiler.
The release extension SHA256 is
`0aaf69050b98f1a7ad757ced29ba5a7a6409723d6d4d5883fcdfb7328038ac4b`.

Burner's post-commit phase must refresh these development captures and add new
bounded-candidate and clean-main coverage/performance reports using the unchanged
commands below. Preserve the historical reports and failures separately; excluded
cells receive zero in the fixed denominator.

## Historical broad candidate and review failures

**The broad candidate was rejected.** Subsequent H100 cancellation tests reproduce both
finite and IEEE broadcast failures, including persistent shape transitions.
Identical fresh default-Inductor runs also select numerically different kernels.
The [second review record](review-autotune-blocker.md) preserves these results
and the unresolved external numerical-contract dependency. The measurements
below remain valid for their fixed corpus; they do not resolve these findings.
The subsequent [development regression run](review-regressions.json.gz) removes per-shape
reference resets and adds cancellation-sensitive large-shape and persistent
IEEE cases. Its clean-commit rerun again fails eight subtests; the earlier
passing test records below describe the earlier test selection, not the
strengthened suite.

The unchanged public-default-compile-v2 gates measured the clean broad implementation
`a2ccf6fb0cf964b8dce446e308a83850fad76e62`, including the strengthened
regressions, on 2026-09-13 UTC. Native implementation sources remain unchanged
from the [broadcast FMA repair](review-broadcast-order.md) at `c1f2d38`. The clean
main baseline `166687a730a86235fa38decfa364703b60ebc595` retains its original
measurements from earlier that day. These reports retain their original source
identities; subsequent narrowing does not change or replace those measurements.
See the [compiler contract](../../compile-pointwise-jit.md).

| Measurement | Historical baseline | Historical broad candidate |
| --- | ---: | ---: |
| Weighted coverage | 6% (4/112) | 10% (8/112) |
| Weighted CUDA performance | 12% (4/56) | 20% (8/56) |
| Performance common-success geometric mean, reference/candidate | 1.8377081163578368 | 1.8767047192089992 |

All four reports have `valid: true`, `diagnostic: false`. The new passing
cells belong to broadcasting; unsupported categories remain zero. These are
fixed-corpus measurements, not general Inductor parity. Common-success ratios
describe each implementation's successful subset, which differs between builds.
No evaluator, corpus, tolerance, weight, denominator, or historical evidence changed.

## Retained historical evidence

- [Baseline coverage](baseline-coverage.json.gz),
  [baseline performance](baseline-cuda-perf.json.gz),
  [candidate coverage](candidate-coverage.json.gz), and
  [candidate performance](candidate-cuda-perf.json.gz) retain every fixed cell,
  unsupported outcome, both CUDA orders, five warmups, 17 samples, cold/steady
  timings, synchronization and source/build/wheel provenance.
- [Gate receipts](gates-receipt.json) record exact commands, clean commits,
  timestamps, return codes and GPU snapshots for the original baseline and new
  broad candidate runs. [Post-commit validation](post-commit-validation.json.gz)
  preserves the new worker/build logs and capture commands from clean `a2ccf6f`.
  The seven-test broadcast-priority suite reports eight failing subtests on H100.
  With CUDA hidden, two metadata tests pass and five hardware tests explicitly
  skip. Original `c1f2d38` reports, receipts and validation records are retained
  byte-for-byte in `previous_capture_files`; they do not supply current-candidate
  performance credit. Neither version supplies narrowed-candidate credit.
- The original [validation bundle](validation.json.gz) and subsequent
  [repair bundle](review-broadcast-order.json.gz) retain their original compiler
  logs, commands, environment, source/test manifests, build/Clippy logs, gate
  worker logs, failures and exhaustive partition receipts. The original bundle
  measures the initial implementation at `5adb263` or explicitly dirty development
  sources; the repair bundle explicitly records uncommitted repair validation.
  Neither substitutes for the clean broad-candidate reports above or future
  narrowed-candidate captures.
  Raw worker observations and wheels remain at the
  worktree-local paths recorded by the reports; their hashes were checked before
  packaging. Those large temporary artifacts are not duplicated here and may
  disappear during worktree cleanup. The reports and logs are durable.
- [Generated CUDA](kernel.cu), [PTX](kernel.ptx.gz),
  [provenance](provenance.json), and [source manifest](source-manifest.json.gz)
  come from the clean broad implementation's installed extension. The independent
  [capture program](capture.py) exercises changed values and shapes with ordinary
  default compilation, fresh outputs, two broadcast specializations and no
  installed-PyTorch import. Its multi-stage sin/cos expression is excluded by the
  refinement and the original capture script remains unchanged.
  [Checksums](manifest.json) cover this bundle.

## Checks and original failures

The earlier repair's complete compiler selection passed **901 tests in 85 disjoint module
processes**, with **23 explicit skips** under `CUDA_VISIBLE_DEVICES=0`. Recorded
test IDs verify exhaustive coverage without overlap at clean `c1f2d38`.
The changed priority suite is now captured separately at `a2ccf6f` and fails;
this evidence phase did not repeat the unrelated full selection. The initial
896-test run remains in the original bundle.
All **three** focused
pointwise device-restoration tests subsequently passed with devices `0,1`.
The broadcast module passed nine tests with one two-device skip. Its original
hardware-absent run (two metadata tests and seven explicit skips) and 80 scalar
IEEE comparisons remain in the initial bundle. The repair's new priority module
passed all five earlier tests again from `c1f2d38`; its development run with CUDA
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

## Reproduce the refined implementation

From each clean source revision, with all build and cache paths inside its checkout:

```bash
mkdir -p target/tmp target/cuda-cache
export TMPDIR="$PWD/target/tmp" CUDA_CACHE_PATH="$PWD/target/cuda-cache"
export XDG_CACHE_HOME="$PWD/target/cache"
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric coverage
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric cuda-perf
CUDA_VISIBLE_DEVICES=0 target/default-compile-eval/venv/bin/python \
  docs/diagnostics/compile-pointwise-broadcast/capture_bounded.py target/broadcast-bounded-capture
```

Run the last command on the narrowed implementation only. The original
`capture.py` command and its results remain historical broad-candidate evidence.
Full historical test commands and local environment settings are retained
in the validation bundle. Burner still owns final independent review,
exact-head evaluation, delivery and managed progress artifacts.
