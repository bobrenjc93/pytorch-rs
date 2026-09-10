# Immutable root compiler analysis reuse

This change reuses the existing global-load tuple only for descriptors whose
validated code has no `requires_grad` branch. It adds one immutable boolean to
that descriptor. Each call still resolves globals, helper bindings and code,
module/callable bindings, scalar values and tensor metadata. Conditional paths
keep their prior branch-sensitive load selection. Helper analysis, cache limits,
CPU boundaries, method guards and whole-graph validation are unchanged. There is
no global code cache and no new function/tensor retention.

The initial checkout and measured baseline are merged main
`b7936239ceb7d7713a15ecd8a2c82c4bddd6d214`. **There is no implementation or
publication commit yet.** Burner explicitly prohibited this agent from creating
commits. Candidate results are therefore development measurements of the
recorded source snapshot, not clean-commit or exact-head evidence. The `commit`
field in dirty build receipts names their base; it does not identify the
uncommitted implementation. A final candidate build used an empty target after
the test fixture correction. Production source and tests then stayed fixed.

## Profiling and measurement boundary

Fresh baseline profiling used all 12 native cells from the unchanged
`diagnose_compile_cuda_matmul.py`, 20 warmups and 200 profiled calls per cell.
Every composed cell performed 200 root disassemblies and static load scans.
Their combined cumulative cost was about 26–27% of instrumented elapsed time;
this is an overhead diagnosis, not an unprofiled speedup estimate. No-global
matmul already skipped that work. Candidate profiling is separate from timings. It confirms zero repeated root
disassemblies/static scans and 200 live resolutions per composed cell. See
[profile accounting](profiles.json) and the baseline/candidate raw `.prof` files.

The seeds and primary repeat were declared before measurements in
`declaration.json`: primary 798431, held-outs 481723 and 926051, and repeat
798431. Both sides use local CPython 3.12.14, the same locked PyTorch 2.13.0+cu130,
and the unchanged complete 12-cell diagnostic. Each report preserves both
orders, 31 blocks of five calls per implementation per order, first-call and
wrapper times, warmup policy, checks and every individual timing. No failing,
slower or contended cell was removed, and no performance rerun was selected.

Baseline held-out 481723's after inventory and held-out 926051's before
inventory show another worktree's CUDA tests on GPU 0. These baseline runs are
potentially contended and remain in the comparison. No other user's process was
stopped. A `gpu` requirement was declared, but no reservation API was available.
The candidate repeat was heavily contended by another user's FP8 GEMM test;
[the additional inventory](candidate/repeat-contention.log) records GPU 0 at
95% utilization. Its after inventory still shows that workload. The candidate
profile after inventory also has an unresolved `[No data]` PID; no timing claim
is drawn from profile durations. Fixed-math began while the FP8 test was still
visible, and fixed-scoring's before/after inventories were empty. Empty process
snapshots do not prove exclusive use between snapshots.

First-call times are fresh-wrapper observations with Dynamo reset per order,
not cold compiler-cache comparisons. Baseline and candidate each start their
sequence with fresh local Triton/Inductor/CUDA cache directories; profiling
warms native kernels before the unprofiled sequence. Caches persist across
cells/orders/runs within each side. No historical result is used as the paired
baseline. Fixed four-shape scoring and the 38-case corpus remain separate.

## Complete paired development data

All **96 cells passed correctness** across four baseline and four candidate
reports. [Accounting](comparison.json) verifies **59,520 raw calls**, both
orders, block/pooled medians, unchanged policies, input hashes and all first-call
observations. There are **20 native-latency regressions among 48 paired cells**;
the worst ratio is **0.0290x**. Per-cell reference drift spans **0.7776–21.4418x**.
All of these observations are retained, including the heavily regressing repeat.

The tables use geometric means of per-cell medians. Native speedup is baseline
native latency / candidate native latency; reference drift is candidate reference
latency / baseline reference latency, so drift above 1 means the reference slowed.

All 12 cells:

| Run | Baseline native µs | Candidate native µs | Native speedup | Reference drift |
| --- | ---: | ---: | ---: | ---: |
| primary | 285.36 | 265.72 | 1.0739x | 1.0359x |
| held-481723 | 232.35 | 202.53 | 1.1472x | 0.9979x |
| held-926051 | 232.78 | 208.45 | 1.1167x | 1.0325x |
| primary-repeat | 273.71 | 3947.59 | 0.0693x | 6.3291x |

The six composed cells affected by this change:

| Run | Baseline native µs | Candidate native µs | Native speedup | Reference drift |
| --- | ---: | ---: | ---: | ---: |
| primary | 414.86 | 336.97 | 1.2311x | 1.0214x |
| held-481723 | 349.36 | 274.87 | 1.2710x | 1.0184x |
| held-926051 | 353.15 | 274.47 | 1.2866x | 1.0319x |
| primary-repeat | 400.75 | 8669.01 | 0.0462x | 5.0397x |

Unchanged 12-cell capped diagnostic parity (not the four-shape score):

| Run | Baseline | Candidate | Raw baseline | Raw candidate |
| --- | ---: | ---: | --- | --- |
| primary | 81.0983% | 89.7065% | [report](baseline/primary.json) | [report](candidate/primary.json) |
| held-481723 | 75.4759% | 85.4462% | [report](baseline/held-481723.json) | [report](candidate/held-481723.json) |
| held-926051 | 75.8019% | 87.1436% | [report](baseline/held-926051.json) | [report](candidate/held-926051.json) |
| primary-repeat | 78.8802% | 37.3711% | [report](baseline/primary-repeat.json) | [report](candidate/primary-repeat.json) |

The first three pairs show development improvements in composed latency, but
**the complete paired data does not establish repeatable native acceleration**.
The repeat, baseline held-out contention and uncommitted candidate prevent a
clean milestone claim. Higher capped parity in the other pairs is not used as
proof of native acceleration. The unfused native `backend="eager"` path also
does not establish general Inductor parity.

## Checks

- [Focused regressions](candidate/focused-retry.log): 6/6 passed, including
  warm CPU/CUDA analysis reuse, per-call resolution, in-place capture data,
  helper/name/module/default/code changes, recursion/cross-module/closure
  rejection, inactive globals, code replacement to branches/exception handling,
  cache limits, metadata/method guards and launch-failure recovery.
- [Full CPU suite](candidate/cpu-suite.log): 5,590 run, 230 skips, no failures.
- [H100 compiler suite](candidate/h100-compiler.log): 142 run, six device-mask
  skips, no failures. [Two GPUs](candidate/two-gpu.log): all seven passed.
- [Python 3.14 compiler compatibility](candidate/python314-compiler.log):
  55 run, one hardware skip, no failures, using a separate local interpreter.
- [Rust](candidate/rust-tests.log): 397 passed; [fmt](candidate/rust-fmt.log)
  and [clippy](candidate/rust-clippy.log) passed. The separate native H100
  [bridge](candidate/rust-h100-bridge.log) and
  [matmul](candidate/rust-h100-matmul.log) tests passed.
- [Installed-wheel verification](candidate/wheel-verifier.log) passed.
- [Frozen compiler corpus](candidate/frozen-compiler.log): 38/38; separate
  [fixed math](candidate/fixed-math.json): 18/18 trials;
  [four-shape scoring](candidate/fixed-scoring.json): all four eligible,
  capped 100%, uncapped common-success ratio 1.3194x. These narrow scores
  do not measure this root-analysis change or establish broad parity.
- [Source/build audit](audit.log): every previous tracked file is byte-identical
  except the production edit; the only added code file is the focused test.
  Wheels, all installed Python source manifests and loaded runtime hashes match
  their respective captured sources. Prior publications are unchanged.

## Build and delivery boundary

Native CUDA uses driver-JIT embedded PTX and cuBLAS; nvcc 12.6.85 is recorded but
unused for those operations. The separate fixed four-shape scoring command
uses its original nvcc-built private kernels. Runtime paths, hashes, loaded
CUDA libraries, reference compiler and PyTorch configuration are in each raw
report. Native extension bytes match between baseline and both candidate
builds. Release builds use Rust 1.92.0, thin LTO, one codegen unit, locked offline
dependencies and the existing extension-module/abi3 configuration.

All environments, interpreters, wheels, caches and outputs are inside this
worktree. UV_PYTHON_BIN_DIR and UV_TOOL_BIN_DIR were explicitly redirected before
Python installation. No evaluator, workload, weight, denominator, dependency
lockfile, prior publication or managed history artifact was changed.

The baseline-wheel regression probe intentionally failed three reuse assertions.
The first candidate test run exposed only a new fixture cleanup error:
`unittest.mock` attempted to delete non-deletable function slots. Explicit
save/restore fixed the fixture; failed logs and both fixture versions are kept.

Before delivery, Burner still needs to commit the final implementation/tests,
run fresh empty-target **clean candidate** captures for all four declared runs,
publish evidence separately with exact implementation/publication commit IDs,
and complete independent exact-head review, all ten non-regressing evaluation
gates and exact-head CI. The agent's local checks are not those delivery gates.
Given the uncommitted candidate and observed baseline contention, this branch
does **not claim a repeatable clean performance milestone or general compiler
parity**, regardless of observed capped diagnostic parity.

## Identities, receipts and reproduction

[Identity record](identities.json) binds the complete baseline and final candidate
source manifests, including tests and unchanged harnesses. All commands have
adjacent `.receipt.json` files with argv, times, status, environment, source
identities, log hashes and GPU inventories. The three empty-target build records
are [baseline](baseline/build-capture/build-record.json),
[initial candidate](candidate/build-capture/build-record.json) and
[final measured candidate](candidate/final-build-capture/build-record.json).
The final candidate source/test snapshot is the measured implementation;
this evidence publication is a later working-tree addition, with no commit ID
for either stage until Burner creates them.

- Baseline production SHA-256: `5349cb16843b1f26de92c30516af7502f0a7ce52054377955ad4fbe9499c3aed`.
- Candidate production SHA-256: `19b0f414624dbbc6d3c5596bb0a990076e18173a7f5da4fbe24a55b8b1e97c7c`.
- Shared native extension SHA-256: `338651d909ae8edc5ae96d382494404c5a9a9d1f315a522df61337dd5781d235`.

Reproduction uses the unchanged `scripts/capture_depth_concat_build.py` and
`scripts/diagnose_compile_cuda_matmul.py`. Worktree-local environment settings
are in [env.sh](command-sources/env.sh.txt); the separate candidate cache paths
are in [candidate-env.sh](command-sources/candidate-env.sh.txt). The receipts
contain the exact commands used here, including `--allow-dirty` for development
candidate captures. After Burner commits the implementation, build into a new
absent target and omit `--allow-dirty` for each diagnostic. Keep the declared
four-run order and all 12 cells, including failures, both timing orders and raw
calls. Do not treat these development files as the required clean candidate run.

[Artifact hashes](artifact-hashes.json) verify byte-for-byte copies from this
worktree's `target/static-analysis` outputs. `command-sources/` preserves the
local orchestration and read-only accounting source; no repository evaluator,
workload, scoring corpus or managed progress generator was added or altered.
Wheel binaries and caches stay under `target`; their hashes and build commands
are published here.
