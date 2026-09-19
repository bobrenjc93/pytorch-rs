# Native compile method identity batch

The combined views, warm-admission and method-batch implementation is validated
at clean commit `e2dd9ec08535aecc8f7df0a98946e53708e5e0bb` (T). This capture replaces
the unresolved sibling-worktree evidence for the current composite. C is
`81a1e9606571914711491285ad8534451f7dcf3a`; R is
`00bea94caaf70017cb347733fe7633a7b2b52637`. Results measure the whole composite,
including its retained-admission repairs, not the method batch in isolation.

Python owns the ordered method inventory and expected objects; the private native
boundary performs fresh identity checks without retaining a table or decision.
The [guide](../../compile-pointwise-jit.md) describes this boundary, inactive-path
admission and transactional logical publication. Integration retains the bounded
prepared-owner scan; the incompatible eviction-only assertion was retired while
its ownership, recency, accounting and reset checks remain.

## Source-bound validation

Six fresh release exports and independently copied Python 3.12.14/3.14.5
interpreters, venvs and builds live inside this composite worktree. All six passed
the unchanged native verifier and nested interpreter/CUDA containment probe.
C/R native inputs match; T changes the two Rust bridge files. Source manifests,
wheel payloads, installed/native bytes and interpreter inventories bind each build
to its actual commit. Rust 1.92.0, PyO3 0.29.2/ABI3, PyTorch 2.13.0+cu130,
H100 GPU0 `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07,
runtime/NVRTC 13.0 and nvcc 12.6 were recorded under Burner's gpu/cpu-heavy leases.

| Interpreter | Pointwise suite | Compiler-owner suite |
|---|---|---|
| Python 3.12.14 | 529 passed, 9 skipped | 8 passed, no skips |
| Python 3.14.5 | 529 passed, 9 skipped | 8 passed, no skips |

Each complete pointwise inventory has 538 ordered IDs. Relative to the earlier
525-ID T inventory, the committed repairs add 13 IDs and remove none; two of the
additions also appear through the existing imported `tests.*` class. Relative to
R, the six native-boundary additions and one instrumentation rename remain.
The exact nine original two-device skip ID/reason pairs are unchanged. Portable
and real GPU cases both ran; these single-GPU suites do not prove two-device
restoration. All four gates passed before timing; native and unchanged default
`torch.compile` smoke histories are retained. Separately, all 485 Rust tests,
formatting and Clippy with warnings denied passed before freeze.

## Ordinary-call diagnostic

Source, recipes, inventories, skip contracts, build identities, paths and formulas
were frozen at **2026-09-19 18:08:57 UTC**. The unchanged 144-line R driver ran
C1/R1/T1/T2/R2/C2 on Python 3.12, then the same sequence on 3.14: 12 cells per leg,
257 float32 values, seed 20260919, 31 warmups and 201 samples per cell. Every call
checked exact output bits and alias behavior with matched synchronization.
All **28,944 samples** are retained. No frozen source changed, phase failed,
leg was repeated or sample removed. Every phase used a fresh process and distinct
child-verified caches; PID/birth/PPID, GPU snapshots, loaded libraries and thread
settings are retained.

Ratios are candidate/comparison median latency; **above 1 is slower**. Pooled
values combine only two process legs within one cell and interpreter. The process
legs are the repetitions, not 402 independent experiments. C drift is C2/C1;
R drift is R2/R1.

**Python 3.12.14**

| Cell (workload/history/mode) | T1/R1 | T2/R2 | T1/C1 | T2/C2 | Pooled T/R | Pooled T/C | C drift | R drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numerical/2/newest | 0.9253 | 0.9955 | 0.9641 | 0.9131 | 0.9660 | 0.9448 | 1.0651 | 0.9375 |
| numerical/2/logical | 0.9629 | 1.2458 | 1.0030 | 1.1174 | 1.0219 | 0.9950 | 1.0845 | 0.9339 |
| numerical/2/abi | 0.9658 | 1.0271 | 1.0522 | 0.9939 | 0.9971 | 1.0320 | 1.0690 | 0.9496 |
| numerical/8/newest | 1.0020 | 1.0520 | 0.9957 | 0.9301 | 1.0241 | 0.9612 | 1.1000 | 0.9786 |
| numerical/8/logical | 0.9874 | 1.0050 | 1.0289 | 0.9295 | 1.0020 | 0.9938 | 1.0752 | 0.9543 |
| numerical/8/abi | 0.9394 | 1.0293 | 1.0581 | 1.0170 | 0.9915 | 1.0503 | 1.0520 | 0.9229 |
| alias/2/newest | 0.9673 | 1.0183 | 1.0389 | 0.9149 | 0.9969 | 0.9855 | 1.1074 | 0.9263 |
| alias/2/logical | 0.9709 | 1.0310 | 1.0091 | 0.9381 | 1.0006 | 0.9907 | 1.1194 | 0.9800 |
| alias/2/abi | 0.9741 | 1.0278 | 1.0609 | 0.9348 | 1.0013 | 1.0301 | 1.1637 | 0.9719 |
| alias/8/newest | 0.9679 | 1.0097 | 1.0031 | 0.8759 | 0.9848 | 0.9635 | 1.1528 | 0.9649 |
| alias/8/logical | 0.9613 | 1.0025 | 1.0045 | 0.9301 | 0.9775 | 0.9733 | 1.0701 | 0.9502 |
| alias/8/abi | 0.9938 | 1.0110 | 1.0972 | 0.9931 | 1.0026 | 1.0532 | 1.0790 | 0.9600 |

**Python 3.14.5**

| Cell (workload/history/mode) | T1/R1 | T2/R2 | T1/C1 | T2/C2 | Pooled T/R | Pooled T/C | C drift | R drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numerical/2/newest | 0.8570 | 1.0507 | 0.9926 | 0.9980 | 0.9480 | 0.9918 | 1.0381 | 0.8513 |
| numerical/2/logical | 0.9058 | 0.9848 | 0.9970 | 0.9763 | 0.9507 | 0.9873 | 0.9970 | 0.8980 |
| numerical/2/abi | 0.8499 | 1.0002 | 1.0251 | 1.0021 | 0.9404 | 1.0124 | 1.0210 | 0.8481 |
| numerical/8/newest | 0.8579 | 1.0044 | 1.0021 | 0.9907 | 0.9561 | 0.9978 | 1.0054 | 0.8490 |
| numerical/8/logical | 0.8677 | 0.9913 | 1.0328 | 0.9906 | 0.9505 | 1.0168 | 1.0118 | 0.8494 |
| numerical/8/abi | 0.9110 | 1.0173 | 1.0513 | 1.0445 | 0.9801 | 1.0490 | 0.9987 | 0.8885 |
| alias/2/newest | 0.9330 | 0.9778 | 1.0400 | 0.9339 | 0.9810 | 1.0010 | 1.0400 | 0.8912 |
| alias/2/logical | 0.9394 | 0.9908 | 1.0738 | 0.9848 | 0.9860 | 1.0377 | 1.0278 | 0.8937 |
| alias/2/abi | 0.9208 | 1.0314 | 1.0933 | 1.0521 | 0.9924 | 1.0765 | 1.0082 | 0.8662 |
| alias/8/newest | 0.9468 | 1.0079 | 1.0731 | 1.0000 | 0.9914 | 1.0422 | 1.0065 | 0.8810 |
| alias/8/logical | 0.9307 | 1.0093 | 1.0916 | 0.9832 | 0.9987 | 1.0433 | 1.0554 | 0.8765 |
| alias/8/abi | 0.9451 | 0.9990 | 1.0985 | 1.0391 | 0.9678 | 1.0731 | 0.9989 | 0.8939 |

Pooled T is slower than C in **4/12 Python 3.12 cells** and **9/12 Python 3.14
cells**, reaching ratios 1.0532 and 1.0765. Against R, six Python 3.12 pooled
cells are slower; Python 3.14 pooled medians are lower, but its R2/R1 drift
ranges from 0.8481 to 0.8980. Python 3.12 C2/C1 reaches 1.1637, and an individual
T2/R2 comparison reaches 1.2458. These results do not establish a uniform win.
Full per-process and within-cell pooled distributions retain mean, median,
min, p10, p90, p95, p99 and max, including tails and slower controls.

This is a warm native.compile diagnostic, not a default-Inductor performance
comparison, cold-compile result, official score or general accelerator-parity
claim. No line/branch coverage or actual CPython 3.10 execution is claimed.
Burner owns independent review, all enabled evaluations and qualification.

## Evidence

The [composite raw bundle](composite.tar.gz) and [manifest](composite-manifest.json)
retain recipes, exact L-to-current and prior-final-to-composite diffs, freeze
bindings, ordered test results, command receipts/logs, smoke histories, raw timing
samples and `summary.json`. Full source, interpreter/stdlib, wheel, dependency
and cache bytes remain local under `target/native-method-guard-trial/composite/`;
the compact bundle contains their manifests and hashes, not those omitted bytes.
Those local bundle locations are not publicly durable hosting.

The preceding [development bundle](development.tar.gz)/[manifest](manifest.json)
and [original T capture](final.tar.gz)/[manifest](final-manifest.json) remain
unchanged at their original source/build identities. They are superseded for
current-composite validation by the fresh capture above; neither their timings
nor the source PR scores are credited to this composite. Earlier H/N negative
results and other historical archives remain untouched. This documentation and
evidence update changes no runtime, tests, dependencies or benchmark harness.
