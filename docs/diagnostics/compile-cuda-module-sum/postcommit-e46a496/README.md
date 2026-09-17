# Clean-commit positional CUDA row-sum capture

Measured commit: `e46a496f995c1491a305f3038886e15cbb017fcf`. This completes the clean capture
that was deferred in the [development record](../README.md). Those historical
baseline/development files remain byte-for-byte unchanged, including their
then-pending capture status. This evidence does not replace independent review
or Burner delivery and merge gates.

A detached checkout under this worktree's `target/module-sum-postcommit-e46a496/checkout`
remained clean throughout the run. The authorized CPython interpreter alone was
copied after verifying its complete file/mode/internal-link inventory. A fresh
local `.venv`, locked dev/reference dependencies, and exact-source release wheel
were built there with Rust 1.92.0. No project environment or wheel was copied.
The installed native/frontend sources were checked against the measured commit.

| Check | Measured result |
| --- | --- |
| Frozen non-scoring replay | 144 strict native/eager/Inductor cells passed; 96 capture gaps closed, 48 existing controls and four rebinding controls preserved |
| Complete compiler selection | 746 tests across 65 modules; 21 skips; no failures |
| Two-GPU restoration | 2 passed; no skips, mask `0,1` |
| Native eager/public/reference sums | 57 tests; 1 expected two-GPU skip under mask `0` |
| CUDA-hidden frontend/CPU boundaries | 59 tests; 1 hardware skip; empty mask |
| Rust default / Python bindings | 395 / 422 passed; no failures or ignored tests |
| Explicit native CUDA row sums / graph planner | 4 / 7 passed; nonzero selectors, no hardware early-return messages |
| Formatting / Clippy | Formatting and both default/Python-binding Clippy configurations passed with `-D warnings` |
| Documentation | 12 smoke tests, relative navigation and CUDA row-sum example passed |

The complete selection is every `tests/test_compile*.py`, including
`test_compiled*` and `test_compiler*`, plus `tests/test_top_level_compile.py`.
Two disjoint module queues ran the unchanged tests on GPU 0. Skips and warnings
are retained verbatim and are not credited as GPU execution. The dedicated
restoration run used GPU 0/1. Concurrent correctness runs and process splitting
provide no performance evidence or timing credit.

All 144 programs, shapes, input hashes and strict reference settings were checked
against the retained pre-edit baseline. The existing reduction's precision and
special-value policy is unchanged; these dyadic cases establish bounded
correctness, not general exact summation, compiler parity or a speed improvement.
No scoring corpus, evaluator, denominator or unsupported outcome was changed.

Python 3.12.14, PyTorch 2.13.0+cu130, NumPy 2.5.1 and native CUDA runtime 13.0
were used. The driver is 580.82.07; installed nvcc is 12.6.85, while reference
Triton 3.7.1 selects PTXAS 12.8.93. Native execution uses embedded PTX and driver
JIT. GPU 0 is `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`; GPU 1 is
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`, both NVIDIA H100s. Before/after
utilization and memory snapshots are observations, not reservations.

[Capture identity and totals](capture.json) bind the commit, source, native
extension, wheel, interpreter, dependencies and selected compilers.
[Command receipts](commands.json) retain timestamps, statuses, effective masks
(including leading `env` overrides), snapshots and warnings. The
[raw archive](raw-evidence.tar.gz) contains every log, per-module accounting,
input comparison, source manifest, exact capture recipes and recipe adaptations.
The original committed recipes are explicitly labeled `original-tools/`; they
are historical inputs, not current package identities. No Rust command is
attributed to a Python extension.

All executed test sources match the clean commit. Reconstruct each with
`git show e46a496f995c1491a305f3038886e15cbb017fcf:<path>` and verify against
archived `source-manifest.json`. Earlier failed test/fixture versions remain in
the unchanged development archive. Two setup-orchestration mistakes in this
capture are recorded in `orchestration-first-attempts.json`; neither executed a
workload. The measured code may precede a later evidence-only commit.
