# Clean composite evidence: immutable analysis and owned CPU tanh

Measured composite implementation: **`d94daecdd9c086ac23afe690a52df11cbaca4f8a`**.
Fresh actual-main baseline: **`b7936239ceb7d7713a15ecd8a2c82c4bddd6d214`**.
Evidence-only publication commit: **pending Burner delivery**. No source,
test, harness, dependency or scoring change is part of this publication.

This new, declared **COMPLETE** series contains four fresh baseline and four
fresh composite reports from clean checkouts inside this composite worktree.
It completes the capture deferred by the [integration notes](../../../composite-compile-tanh-integration.md).
Both sides used matching local CPython 3.12.14 interpreter bytes, installed
PyTorch 2.13.0+cu130 and locked dependencies, Rust 1.92.0, CUDA libraries and
unchanged diagnostic harness. Actual baseline paths are inside the nested
`target/composite-postcommit-d94daecd/baseline` checkout; candidate paths are
inside the enclosing composite. No old report was reused as the fresh baseline.

The [declaration](declaration.json) fixes primary **798431**, held-outs
**481723/926051**, and primary repeat **798431**, with baseline profiling and
all four baseline runs preceding candidate profiling and its four runs.
All 12 cells, both timing orders, individual calls, first calls, cache records
and failures are retained. Correctness passed in **48/48 baseline**
and **48/48 candidate** cells. [Accounting](comparison.json) verifies
**59,520 raw calls**, five-call block means, medians,
fixed aggregation, matching inputs and the unchanged timing policy.

## Paired latency and reference drift

Geometric means across every cell; native speedup is baseline native latency
/ candidate native latency. Reference drift is candidate PyTorch latency /
baseline PyTorch latency, so values above one mean the reference slowed.

| Run | Baseline native µs | Candidate native µs | Native speedup | Reference drift |
| --- | ---: | ---: | ---: | ---: |
| primary | 276.22 | 233.43 | 1.1833x | 0.9319x |
| held-481723 | 243.69 | 195.59 | 1.2459x | 0.8687x |
| held-926051 | 226.82 | 205.52 | 1.1036x | 1.0187x |
| primary-repeat | 273.26 | 247.23 | 1.1053x | 1.0716x |

The six composed cells, which exercise root global-load analysis reuse:

| Run | Baseline native µs | Candidate native µs | Native speedup | Reference drift |
| --- | ---: | ---: | ---: | ---: |
| primary | 408.00 | 302.45 | 1.3490x | 0.9111x |
| held-481723 | 354.58 | 258.03 | 1.3742x | 0.9000x |
| held-926051 | 348.18 | 270.14 | 1.2889x | 1.0037x |
| primary-repeat | 402.74 | 316.94 | 1.2707x | 1.0527x |

Capped diagnostic parity is reported separately; a higher cap can result from
reference drift and does not establish native acceleration.

| Run | Baseline cap | Candidate cap | Raw baseline | Raw candidate |
| --- | ---: | ---: | --- | --- |
| primary | 80.6881% | 89.3404% | [report](baseline/primary.json) | [report](candidate/primary.json) |
| held-481723 | 75.8686% | 84.2442% | [report](baseline/held-481723.json) | [report](candidate/held-481723.json) |
| held-926051 | 76.8113% | 86.5598% | [report](baseline/held-926051.json) | [report](candidate/held-926051.json) |
| primary-repeat | 78.0846% | 89.1147% | [report](baseline/primary-repeat.json) | [report](candidate/primary-repeat.json) |

Aggregate native latency improved in all four paired runs in this capture.
There are **12 native regressions among 48 comparable
paired cells**. All slow cells remain included. Per-cell values, timing-order
medians and first-call observations are in [comparison.json](comparison.json).
The unmodified matmul-only group regressed in the second held-out and repeat;
the six composed-cell aggregate improved in all four pairs.
The measured ratios describe this capture only; no isolated performance
nonregression, general compiler parity or performance milestone is claimed.

## Contention, caches and mechanism

The 20 capture commands have timestamped before/after GPU and process
inventories in
[commands.json](commands.json). The [admission observation](admission-observation.json)
records the paused Burner process; the task reports other Burner work completed.
No reservation API was available and no other user's process was interrupted.
All recorded boundary process inventories were empty. Empty snapshots do not
prove exclusive GPU use between them.

Fresh empty Cargo targets were used for both release builds. Each side used
separate initially empty diagnostic CUDA/Triton/Inductor caches. Profiling
performed 20 warmups and 200 calls per cell before that side's timing series;
caches then persisted across all four runs. First calls are fresh-wrapper and
Dynamo-reset observations, not cold disk-cache comparisons. Setup reused the
local locked download/registry caches, and copied the same CPython distribution
into the baseline checkout. No Python shortcut or global configuration changed.

The [24 raw profiles](profiles.json) confirm the mechanism on fresh actual main:
each composed baseline cell repeats root lowering/static global-load analysis
200 times; each candidate cell repeats it zero times, while **200 live global
resolutions remain on both sides**. Profiled durations are not timing evidence.
Immutable reuse does not cache live bindings or expand compiler support.

## Verification and reproducibility

[Verification](verification.json) checks complete committed source manifests,
installed Python files, wheel/native hashes, interpreter/reference identities,
loaded CUDA library hashes, private fixed-score kernels and command log hashes.
The [paired reference check](paired-reference-check.json) additionally compares
installed PyTorch and NumPy native-library bytes on both sides. Additional
read-only identity/accounting commands have their own
[verification receipts](verification-commands.json).
The existing private-kernel source checksum is BLAKE2b-128; SHA-256 is recorded
separately for the source and binary. Exact commands and timestamps are in the
[receipts](commands.json), with their orchestration in [command-sources](command-sources/).

- [Baseline build](baseline/build-capture/build-record.json) and
  [candidate build](candidate/build-capture/build-record.json): fresh locked
  offline release wheels, thin LTO, one codegen unit, extension-module/abi3-py310.
- [Baseline manifest](baseline/source-manifest.json) and
  [candidate manifest](candidate/source-manifest.json): every tracked file at
  the respective measured revision, including tests and harnesses.
- [Paired build check](paired-build-check.json): identical interpreter bytes,
  dependency versions, lockfile, Rust compiler and release configuration.
- [Candidate focused checks](candidate/focused-regressions.log): 18 tests run,
  17 passed and one device-mask skip; compiler/tanh composition, static/live
  guards and native CUDA whole-graph validation. The targeted persistent-view
  [regression](candidate/view-regression.log) also passed.
- [Two-GPU checks](candidate/two-gpu.log): composed graph and matmul device
  restoration/mixed-device rejection: two passed, with only devices 0 and 1 visible.
- [Frozen compiler corpus](candidate/frozen-compiler.log),
  [fixed CUDA math](candidate/fixed-math.json) and the separate unchanged
  [four-shape score](candidate/fixed-scoring.json) retain their own bounded
  definitions: 38/38 compiler cases and 18/18 math trials passed; all four
  fixed-score shapes were eligible. Their outcomes do not establish general
  parity or a score gain.

The timing harness and fixed math script kept physical `CUDA_VISIBLE_DEVICES=0`.
The host is NVIDIA H100, driver 580.82.07. Both sides used local CUDA 13
runtime/cuBLAS, recorded in every timing report. Native graph operations use
cuBLAS and driver-JIT PTX; nvcc 12.6.85 is recorded but unused by those operations.
The unchanged fixed-score private kernels use their recorded nvcc-built binaries.
The actual PyTorch/Triton compiler paths, hashes and build configuration remain
in the raw reports. All executable, installed-package, cache, build and output
paths are inside this composite worktree except read-only system toolchain and
driver identities.

Reproduce each side from its clean committed checkout with a local canonical
`.venv`, the recorded local environment, and a new output directory:

```bash
.venv/bin/python scripts/capture_depth_concat_build.py --output target/NEW/build-capture
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diagnose_compile_cuda_matmul.py \
  --build-record target/NEW/build-capture/build-record.json \
  --output target/NEW/primary.json --seed 798431
```

Use the complete sequence in [capture.py](command-sources/capture.py.txt), not
selected cells or replacement runs. Wheels and caches remain under `target`;
all reports, logs, receipts, raw profiles and provenance manifests are published.

The initial publication path audit included a build-log closing parenthesis
in a directory name and rejected it. Its [failed log](publication-check-failed.log)
and original script are retained; the corrected audit parses worktree names
and verifies the same unmodified reports.

## Historical evidence and remaining gates

All **255 files** in the two included source publications remain byte-for-byte
unchanged, as checked by the [historical manifest](historical-publication-manifest.json).
The [source development series](../static-analysis-b7936239-development/README.md)
and [source clean series](../postcommit-4cb0432/README.md) retain original source,
build and path identities, failed attempts, contention inventories and the
failed/corrected audit receipts. Their prior ratios (approximately 0.3346,
0.6599, 0.0863 and 0.9167) remain historical potentially-contended observations;
none supplies current-composite performance credit. No original result or
publication was relabeled or edited.

Native CUDA remains bounded `backend="eager"`. CPU tanh gradients are not CUDA
autograd and provide no hardware-score credit. The four-shape and frozen 38-case
corpora, weights, denominators and Burner-managed progress artifacts are unchanged.

Burner owns the evidence-only publication commit. The measured implementation
may precede that publication only while the intervening diff remains evidence
and documentation. Any later implementation/test/harness change invalidates the
affected captures. **Independent exact-head review, all ten current-definition
nonregression gates and exact-head combined CI remain required before merge.**
This evidence step does not approve the branch or authorize cleanup.
