# Clean-commit native ReLU validation

Measured implementation: `1abe675fc47215afba609f54716353730237c710`; base `main` was
`b74c2051b823518cc09d5a29f8237a3de9863516`. This completes the post-commit capture
explicitly deferred in the [development report](../README.md). All measurement
commands ran with a clean checkout before these reports and their documentation
link were added. The frozen 32-gap baseline, development failures and all earlier
evidence retain their original bytes and provenance.

A fresh locked release wheel was built using the committed
`build_cuda_add_diagnostic.py --revision 1abe675fc47215afba609f54716353730237c710`.
[Build provenance](build.json), [preflight](preflight.json), [command receipts](commands.json)
and [verification](verification.json) bind the source export, installed Python/native
files, interpreter, exact commands, UTC timestamps, inputs and GPU snapshots.
All Python executable/import and native build paths belong to this worktree;
external Rust/compiler installations were read-only.

| Check | Result |
| --- | --- |
| Complete compiler sweep (60 modules) | 678 run, 662 passed, 16 skipped |
| Focused CUDA/reference and native-owner startup regressions | 41 run, 37 passed, 4 skipped |
| CUDA-hidden/frontend/CPU reference regressions | 117 run, 88 passed, 29 skipped |
| Two-physical-GPU restoration | 4 run, 4 passed, 0 skipped |
| Rust default all-targets | 395 passed, 0 failed, 0 ignored |
| Rust Python-binding CUDA graph/planner | 15 passed, 0 failed, 0 ignored |
| Rust Python-binding graph/planner with CUDA hidden | 15 passed, 0 failed, 0 ignored |
| README/docs smoke | 12 run, 12 passed, 0 skipped |
| Formatting; default and Python-binding Clippy | Passed |
| Updated CUDA ReLU documentation example | Passed |

The complete selection is every `test_compile*.py` plus `test_top_level_compile.py`,
without exclusions. The 16 single-GPU sweep skips are device-mask guards;
restoration ran separately on GPUs 0 and 1. The CUDA-hidden Rust tests return
early from hardware sections, so their pass count is not GPU evidence. Test
function counts exclude generated subcases; the original 32 ReLU spelling cells
are checked under all four policies (128 combinations), with cold/repeated native
and Inductor calls. No assertion, workload, reference, supported boundary,
benchmark corpus, evaluator or denominator changed. All capture commands passed.

Python 3.12.14, Rust 1.92.0 and locked dev/reference dependencies were reused from
the verified worktree-local setup; no project environment or wheel was copied.
The complete interpreter inventory was rechecked against the original verified
distribution. The release build used an empty per-export Cargo target; download,
Python, CUDA, Inductor and Triton caches were warm. These are functional checks,
not timing or performance-parity measurements. Native/reference runtime was CUDA
13.0 with driver 580.82.07; nvcc 12.6.85 was installed, while native kernels used
driver JIT of embedded PTX.

Resources: `gpu` and `cpu-heavy`. GPU 0 was
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`; restoration also used GPU 1,
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. Both were NVIDIA H100, capability 9.0.
Before/after utilization and memory snapshots accompany every command; idle
snapshots are observations, not reservations. No other user's jobs were interrupted.

[Input references](inputs.json) reuse the unchanged development manifests and
seeded input recipe by hash. The full build-source manifest is reconstructed from
its existing shared base plus the recorded commit delta. Historical measurements
are not relabeled or used for current-candidate performance credit.

To reproduce, copy the unchanged [environment recipe](../recipes/env.sh.txt) to
`target/relu-postcommit/env.sh` and source it, then use the build command above
with a new destination name. Copy `run.py`, `preflight.py` and `replay.py` from
this capture's `recipes/*.txt` into `target/relu-postcommit`, along with the four
unchanged helpers listed in [recipe references](recipes.json). The recorder
requires this exact clean commit, captures new logs without overwriting prior
runs, and replays the committed test commands; only output paths and provenance
checks were added. The frozen rejection probe belongs only to the starting main.

The first publication attempt refused a filename collision between the outer
build-command log and the native compiler log. No measurements failed or were
rerun. The corrected publisher uses distinct log names; the [failed publication
inventory](publication-attempt1.json), initial recipe and audit error are retained.
A [second packaging retry](publication-attempt2.json) encountered that failed
audit's empty JSON output; publication now selects only the named command receipts.
Both failed attempts and their unchanged measurement-log hashes are preserved.

This capture does not approve the branch or replace independent review, later
revision testing, evaluations, delivery safeguards or fresh merge gates. Burner
owns subsequent artifact commits and delivery.
