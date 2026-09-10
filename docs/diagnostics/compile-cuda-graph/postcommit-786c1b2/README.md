# Clean captured CUDA graph bridge evidence

Measured implementation: **`786c1b2e18ca83ecbe6e09606d33309d00187b2f`**, against the fresh merged-main
baseline **`fbb0aa0f5ad37fc0b868d0d153fb90aeeb41b6f3`**, on 2026-09-10 UTC. Build, correctness,
timings and accounting completed with clean Git status. This publication adds
only evidence and documentation; it is not independent approval, a ten-gate
evaluation or merge authorization.

## Complete unchanged diagnostic

| Run | Main capped parity | Candidate capped parity | Composed native geometric speedup |
| --- | ---: | ---: | ---: |
| primary | 79.7040% | 81.4367% | 1.0609x |
| primary-repeat | 79.8973% | 80.0108% | 0.9826x |
| held-481723 | 74.1541% | 75.5041% | 1.0054x |
| held-926051 | 74.9843% | 77.7223% | 1.0036x |

Each row contains all 12 cells of the unchanged diagnostic, including the six
composed programs. Primary and repeat use seed 798431; held-outs use the
previously declared seeds 481723 and 926051 and their deterministic shapes.
All 48 fresh candidate cells passed. The raw
[primary](primary.json), [repeat](primary-repeat.json),
[held-out 481723](held-481723.json), and [held-out 926051](held-926051.json)
reports retain both execution orders, all first-call/wrapper costs, every
individual timing, aggregate samples, dispersion, input-preservation checks and
changed-data checks. Matching baseline reports are under [baseline](baseline/primary.json)
and copied byte-for-byte; they measured actual merged main before optimization.
The earlier cbcbd84 measurements and dirty candidate development runs are not
used as this comparison's baseline or final candidate evidence.

[Verified comparison](verification.json) recomputes all **59,520**
individual calls across eight reports, including **29,760 newly measured calls**.
It verifies the fixed cell matrix, policies, seeds, regenerated input hashes,
block and pooled medians, capped aggregation, source/wheel/runtime identities
and command/log receipts. No cell or unsuccessful attempt was removed.
The clean paired data **does not establish the required repeatable native
latency improvement**. Composed geometric speedup was 1.0609x in the first
primary pair but 0.9826x in the repeat; held-out speedups were only 1.0054x and
1.0036x. Higher capped parity also reflects reference timing drift.
Correctness passed, but the performance milestone remains unproven and needs
independent review; no implementation change or favorable-only recapture was
made in this evidence step. The worst composed per-cell native speedup is
0.8172x; reference latency ratios span 0.8114–1.3870x across all pairs. Individual regressions and
all slower-than-reference cells remain visible. This native `backend="eager"`
capture path is unfused and does not establish general Inductor parity.

First-call observations are not cold disk-cache comparisons: each wrapper is
new and Dynamo resets per order, but compiler disk caches persist. All
observations, including the baseline-repeat 71.211256 ms native outlier, are
preserved; no first-call improvement is claimed. The native bridge still calls
the existing kernels/cuBLAS in order and synchronizes each operation.

## Build, checks and identities

- [Fresh build](build/build-record.json): absent target, locked offline release
  wheel build, then local wheel installation. Native extension SHA-256:
  `a03fa9df4ac2292bf2d2089884ed0bd2bf192522cfd9c81f511809ffe9881179`.
- Production source SHA-256: `979e3fb0b5aafc492cb45aa6ba573ab332210a933f62141c4351e79a1b5679b7`.
  [Source manifest](source-manifest.json), [complete diff](candidate.diff.txt),
  and [inspection](inspection.json) bind the exact committed code/tests/harness.
  Evaluators, frozen workloads, dependencies and managed progress artifacts
  were unchanged by the implementation.
- [Installed-wheel verification](wheel-verifier.log) passed under `-I`.
  Every timing report additionally checks all 59 installed package Python files
  against the checkout and the extension against the fresh build record.
- [CUDA regressions](cuda-regressions.log): 33 run, 30 passed, three device-mask
  skips. Covers numerical and wide-K/overflow/nonfinite behavior, new data,
  no-body/no-PyTorch execution from an isolated installed wheel, graph/native
  guards, aliases, offsets/overlap, lifetimes and recovery.
- [Separate GPUs 0,1](two-gpu.log): all three composed/compiled/eager matmul
  restoration tests passed, including mixed-device rejection.
- [Native bridge](rust-bridge.log) and [native matmul](rust-matmul.log): one and
  two focused Rust tests passed. Unrelated author-validated full suites were
  not repeated for this evidence-only step.
- [Fixed CUDA math](fixed-math.json): all six unchanged cases at all three
  prescribed seeds, 18/18 trials passed. [Fixed compiler corpus](frozen-compiler.log):
  38/38 reference-eligible cases passed.
- [Unchanged four-shape scoring](fixed-scoring.json): all four eligible;
  capped score 100.0%,
  common-success ratio 1.1984x.
  This remains a separate pre-existing private workload, not a score increase
  attributed to the composed bridge.

Local CPython 3.12.12, NumPy 2.5.1 and PyTorch 2.13.0+cu130 were reused from the
locked worktree environment. H100, driver 580.82.07, GPU 0 except the separate
two-GPU test. Native runtime and cuBLAS resolve from local nvidia/cu13 packages;
the reports include loaded paths, hashes, runtime versions, PyTorch build and
local Triton/PTX assembler identities. Rust 1.92.0, release, thin LTO, one codegen
unit, extension-module/abi3-py310. Native matmul uses cuBLAS and pointwise kernels
use driver-JIT PTX; nvcc 12.6.85 is recorded but unused for these operations.
The separate fixed scoring lane uses its existing nvcc-built private kernels.

The build target and capture temporary/CUDA/Triton/Inductor caches were new;
local Cargo downloads and Python packages were reused. CUDA driver caches were
warmed by the preceding correctness checks. Timing orders/cells/runs share their
new diagnostic compiler caches. Scoring used separate initially empty
Triton/Inductor caches and reused both unchanged native private-kernel libraries.
Their source checksums and actual binary hashes were verified and retained in
`verification.json`, alongside the scoring tool's reuse records. Baseline cache
states and first-call costs remain pinned to their original reports; they were
not rewritten.

## Receipts and reproduction

[Commands](commands.json) record actual argv, timestamps, exit status, clean
status before/after, source identities, environment and log hashes.
[Build commands](build/commands.json) record compiler and installer details.
[Capture completion](capture-complete.json) confirms unchanged sources.
[Copied-artifact hashes](artifact-hashes.json) bind publication bytes to the
original local captures. Orchestration and read-only audit source snapshots
are under `command-sources/`; the workload and build tools are unchanged files
from the measured commit.

The preserved [baseline profile](baseline/baseline-profile-retry.log) identified
per-node bridge and metadata overhead before implementation. Its temporary
script-name collision and retry remain recorded in `baseline/baseline-profile*`;
this historical profiling does not supply candidate timing credit. The source
`baseline/command-sources/profile_graph.py.txt` declared held-out seeds before
baseline measurements; `baseline/held-outs.json` was materialized later.

Reproduce using the [canonical capture procedure](../../../compile-cuda-matmul.md#reproduction-and-delivery)
and these actual command receipts, with new output/cache directories and a clean
checkout. No `--allow-dirty` option was used for this candidate capture. Final
independent exact-head review, all ten non-regressing gates and exact-head CI
remain Burner-owned prerequisites for managed merge.
