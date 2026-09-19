# Native compile method identity batch

**Composite validation pending.** The raw capture below identifies builds and
imports in the sibling `agent_c3ca04c7` worktree. Its measured implementation
`d58211d` matches the runtime and tests at composite `5ff110a`, but its paths do
not satisfy the composite's current-candidate provenance contract. The original
archives and measurements remain unchanged; this is an unresolved evidence
defect, not a historical relabeling or a fresh validation of the composite.
Replacement evidence requires clean exports and builds inside the composite,
followed by the declared correctness gates and complete timing sequence. That
capture is pending Burner's `gpu` and `cpu-heavy` resource leases; the composite
author has only the `composite-build` lease and cannot modify external lock state.

This candidate carries C's required R publication repair plus one stateless
native method-identity batch. Python retains the inventory and expected objects;
no ShapeGuards hash change or other runtime optimization is included. The
[guide](../../compile-pointwise-jit.md) describes the boundary and publication lifetime.

## Development validation

Fresh, separate release builds used Python 3.12.14 and 3.14.5, Rust 1.92,
PyO3 0.29.2/ABI3, PyTorch 2.13.0+cu130 and H100 GPU0
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`. The selected runtime and NVRTC
reported 13.0; `nvcc` was 12.6. Format, Cargo check, Clippy with warnings denied,
the unchanged native verifier and local interpreter/CUDA containment probes passed.

| Python | Complete pointwise suite | Compiler-owner suite |
|---|---|---|
| 3.12.14 | 516 passed, 9 skipped | 8 passed, 0 skipped |
| 3.14.5 | 516 passed, 9 skipped | 8 passed, 0 skipped |

Each pointwise inventory contains 525 ordered IDs: R's 519 tests plus six new
boundary tests, with one instrumentation test renamed. The eight imported
`tests.*` IDs are unchanged. Only L's exact nine two-device reservation skip
pairs were accepted; these runs do not establish two-device restoration.

[Raw development evidence](development.tar.gz) and its [manifest](manifest.json)
retain complete test IDs/results/logs, both native/default smoke histories,
build failures and successful receipts, source/wheel/installed/interpreter
inventories, recipes and the exact L-to-development recipe diff. The ordinary
144-line driver is unchanged and was not run. Full source, wheel, interpreter
and cache bundles remain in local `target/native-method-guard-trial/`; their
hashes/locations are recorded, but their bytes are omitted from the compact
archive and those local paths are not publicly durable.

## Committed-source capture

The final capture measured clean implementation commit
`d58211d07f15800971d7507426207af677514cfc` (T), against
`81a1e9606571914711491285ad8534451f7dcf3a` (C) and
`00bea94caaf70017cb347733fe7633a7b2b52637` (R).
Six fresh release exports passed the unchanged interpreter/CUDA containment
probe and native verifier. Each source root owns its interpreter, venv and build
outputs. C/R native inputs and binaries match; T changes the two Rust bridge
files and produces different native bytes. These comparisons measure the whole T
candidate. Toolchain, dependency, source, wheel and installed identities are retained.

Inputs were frozen at **2026-09-19 15:02:51 UTC**, before the four final gates.
Both interpreters passed **516 pointwise tests with exactly nine allowed skips**
and **all eight compiler-owner tests without skips**. The complete 525-ID
pointwise inventory, six additions, one instrumentation rename and unchanged
imported IDs are bound to execution. Both native and unchanged default
`torch.compile` smoke histories are retained. Correctness lanes ran concurrently;
all four gates passed before any ordinary timing.

All **28,944 ordinary samples** were retained from the unchanged 144-line driver:
C1/R1/T1/T2/R2/C2 on Python 3.12, then the same sequence on 3.14. Each leg used a
fresh process and distinct child-verified caches. The 12 cells retain their
numerical/alias × history 2/8 × newest/logical/ABI matrix, 31 warmups and 201
samples. No frozen input changed, phase failed, leg repeated or sample was removed.

[Final raw evidence](final.tar.gz) and its [manifest](final-manifest.json) retain
recipes and their exact L-to-T diff, freeze and build manifests, all command
receipts, complete test results, raw samples and nested smoke histories.
`summary.json` inside the archive contains each process and within-cell pooled
mean, median, min, p10, p90, p95, p99 and max, plus every comparison below at full
precision. The archive contains 570 files; it does **not** contain the full local
source, wheel, interpreter, dependency or cache bundles. Their hashes and local
locations under `target/native-method-guard-trial/final/` are recorded; these
local paths are not publicly durable hosting. The preceding development archive
remains unchanged and is not substituted for this clean capture.

### Ordinary-call ratios

Ratios divide candidate latency by comparison latency; **above 1 is slower**.
“Pooled” combines only the two legs of the same cell and interpreter. The two
process legs are the repetitions; 402 calls are not 402 independent experiments.
C drift is C2/C1, and R drift is R2/R1. No cells or interpreters are aggregated.

**Python 3.12.14**

| Cell (workload/history/mode) | T1/R1 | T2/R2 | T1/C1 | T2/C2 | Pooled T/R | Pooled T/C | C drift | R drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numerical/2/newest | 0.9888 | 0.9355 | 0.9739 | 1.0083 | 0.9611 | 0.9888 | 0.9636 | 1.0545 |
| numerical/2/logical | 1.0207 | 0.9334 | 1.0342 | 1.0082 | 0.9788 | 1.0175 | 0.9818 | 1.0465 |
| numerical/2/abi | 0.9585 | 0.9309 | 1.0600 | 1.0665 | 0.9430 | 1.0628 | 0.9928 | 1.0286 |
| numerical/8/newest | 0.9678 | 0.8789 | 0.9764 | 0.9932 | 0.9278 | 0.9862 | 0.9809 | 1.0986 |
| numerical/8/logical | 0.9627 | 0.9390 | 0.9892 | 0.9949 | 0.9510 | 0.9914 | 0.9985 | 1.0296 |
| numerical/8/abi | 0.9698 | 0.9509 | 1.0240 | 1.0669 | 0.9584 | 1.0425 | 0.9768 | 1.0379 |
| alias/2/newest | 0.9654 | 0.9389 | 0.9636 | 0.9975 | 0.9505 | 0.9785 | 0.9678 | 1.0301 |
| alias/2/logical | 0.9815 | 0.9364 | 0.9754 | 1.0083 | 0.9591 | 0.9912 | 0.9713 | 1.0523 |
| alias/2/abi | 0.9734 | 0.9584 | 1.0511 | 1.0654 | 0.9692 | 1.0605 | 1.0003 | 1.0299 |
| alias/8/newest | 0.9762 | 1.0124 | 1.0000 | 1.0336 | 0.9911 | 1.0127 | 1.0153 | 1.0119 |
| alias/8/logical | 0.9665 | 1.0516 | 0.9675 | 1.1186 | 0.9843 | 1.0203 | 0.9919 | 1.0540 |
| alias/8/abi | 1.0053 | 0.9517 | 1.0593 | 1.0614 | 0.9768 | 1.0611 | 0.9899 | 1.0479 |

**Python 3.14.5**

| Cell (workload/history/mode) | T1/R1 | T2/R2 | T1/C1 | T2/C2 | Pooled T/R | Pooled T/C | C drift | R drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numerical/2/newest | 0.9806 | 0.9839 | 1.0081 | 0.9892 | 0.9822 | 0.9959 | 1.0262 | 1.0037 |
| numerical/2/logical | 0.9818 | 0.9820 | 1.0414 | 0.9926 | 0.9831 | 1.0211 | 1.0193 | 0.9714 |
| numerical/2/abi | 1.0196 | 0.9570 | 1.0499 | 0.9831 | 0.9854 | 1.0135 | 1.0124 | 1.0099 |
| numerical/8/newest | 0.9880 | 0.9754 | 1.0186 | 0.9768 | 0.9796 | 1.0008 | 1.0121 | 0.9832 |
| numerical/8/logical | 0.9615 | 0.9523 | 1.0052 | 0.9575 | 0.9561 | 0.9853 | 1.0159 | 0.9770 |
| numerical/8/abi | 0.9607 | 0.9487 | 1.0228 | 1.0056 | 0.9569 | 1.0153 | 1.0115 | 1.0071 |
| alias/2/newest | 0.9696 | 0.9547 | 1.0306 | 0.9664 | 0.9627 | 0.9967 | 1.0391 | 0.9896 |
| alias/2/logical | 0.8835 | 0.9435 | 1.0375 | 0.9922 | 0.9333 | 1.0154 | 0.9942 | 0.8904 |
| alias/2/abi | 0.9733 | 0.9626 | 1.0500 | 1.0379 | 0.9694 | 1.0452 | 1.0024 | 1.0019 |
| alias/8/newest | 0.9360 | 0.9535 | 1.0147 | 0.9727 | 0.9400 | 0.9954 | 1.0247 | 0.9643 |
| alias/8/logical | 0.9520 | 0.9651 | 1.0350 | 1.0150 | 0.9570 | 1.0272 | 1.0099 | 0.9769 |
| alias/8/abi | 0.9782 | 0.9480 | 1.0717 | 1.0326 | 0.9609 | 1.0506 | 1.0058 | 1.0000 |

Pooled T/R medians are below one in every cell, but individual process pairs
include slower results. Against C, pooled T is slower in **7/12 Python 3.12
cells** and **8/12 Python 3.14 cells**, reaching ratios 1.0628 and 1.0506.
Reference drift is material: Python 3.12 R2/R1 reaches 1.0986, while Python 3.14
alias/history-2/logical is 0.8904. The raw distributions, tails and slower
controls are retained; these two repetitions do not establish a uniform win.

This is a warm native.compile comparison, not cold compilation, default-Inductor
performance parity, general accelerator parity or proof of two-device restoration.
The implementation commit is separate from this later evidence-only change;
runtime, tests, dependencies and benchmark harnesses were not changed in this step.
Burner still owns independent review, all ten evaluations and qualification.
