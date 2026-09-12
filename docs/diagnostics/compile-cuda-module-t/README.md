# Trusted native t call validation

Non-scoring development evidence for exactly one-positional-argument native
`torch.t(x)` and genuine direct-import aliases. The frontend retains the genuine
PyO3 function at package startup and uses the existing CUDA rank-0–2 view node.
No kernels, backend semantics, methods, dtype, gradient or layout boundaries change.
See the [supported contract](../../compile-cuda-t.md).

Before edits, the unchanged frozen probe (SHA-256
`fecd393d78b70d7cba9dbfe7993fb94d7bf0c51057fd9a53018770291c79fc24`)
reproduced the audited clean-main
`9b9236bff0a7d6209da9ed75463151a603e248be` baseline using a fresh locked release
wheel. All 32 native compiled cold/repeated gaps, 12 existing-operation controls
and three unrelated-rebinding controls matched the audited proof, including
input hashes and exact failure messages. Native eager and reference Inductor
cold/repeated execution passed. Original bytes, recipes and failures are retained
as lossless gzip files with hashes, separately from candidate measurements.

The new tests cover all four policies, seeded scalar/vector/rectangular/empty/
singleton/odd and offset/noncontiguous inputs, raw IEEE bits, shared mutations,
parent deletion, nested/repeated wrapper identity, changed inputs across calls,
packed arithmetic, squeeze/ReLU and existing method consumers. Fresh processes
retain a genuine alias before jointly mutating/deleting public/native `t`, then
check cold/repeated capture, fake aliases, exact errors and cache recovery.
Per-used-field guards preserve unrelated squeeze/ReLU/add/neg/mul/matmul programs,
including lowered graphs and `recompile_limit=1`. Whole-graph prevalidation and
two-physical-GPU restoration remain required.

The old `m.t(x)` rejection placeholder in `tests/test_compile_cuda_t.py` now uses
`m.t(x, dim=0)` to retain its unsupported-binding purpose. Positive module and
alias coverage is in `tests/test_compile_cuda_module_t.py`. Earlier arithmetic,
ReLU and squeeze regression suites are unchanged.

## Results

| Check | Result |
| --- | --- |
| Exact-main frozen baseline | 32 expected gaps; 12 operation and 3 rebinding controls pass |
| Candidate replay, identical original inputs | All 32 gaps closed; all controls pass |
| Complete final-source compiler sweep, 62 modules | 708 run, 690 passed, 18 hardware-mask skips |
| Revised/added focused CUDA checks | 4 passed |
| CUDA-hidden/frontend/CPU/reference | 147 run, 94 passed, 53 hardware skips |
| Two-physical-GPU restoration | 5 passed |
| Rust default all-targets, CUDA hidden | 395 passed, 0 ignored |
| Rust Python-binding graph/planner, H100 | 15 passed |
| Rust Python-binding graph/planner, CUDA hidden | 15 passed; hardware sections return early |
| Formatting, default/Python-binding Clippy | Passed |
| Native extension/source verification | Passed |
| README/navigation and CUDA example | 12 tests and example passed |

## Development checks

The initial focused run recorded 67 tests: 59 passed, five two-device skips,
and three new-fixture failures. Its original source and log are retained. Two
assertions expected the wrong existing exception: malformed fixed-rank cached
metadata raises `ValueError`, and incompatible strided `view` raises `RuntimeError`.
The last comparison constructed a canonical `(7,1)` tensor instead of the
transpose view's surviving `(1,7)` strides. Those assertions now use exact
exceptions and a reference transpose; the same correction was applied to the
two-device expected view before that test ran. No production fix was necessary.
The four revised/added focused checks and all five restoration checks passed.
An evidence-packaging attempt also read its own empty redirected JSON output;
the recorded reproduction and original publisher are retained. Receipt selection
now requires a paired log. This did not change any measurement.

The candidate replay uses precisely the frozen baseline's 44 input/expression/
shape recipes and hashes: all 32 previously rejected cells and all 12 controls
pass cold/repeated native and reference Inductor execution. The three unrelated
rebindings also pass. The separate candidate script changes only baseline-state
assertions and requires successful execution; the frozen original remains intact.

[Command receipts](commands.json) record actual statuses, totals, GPU snapshots,
environment, native hashes and shared production/test-input manifest keys.
Full baseline manifests are stored once; later versions contain only deltas.
[Build records](builds.json) bind immutable source exports and locked release
wheels; baseline and development native binaries are byte-identical because
only the Python frontend changed. [Verification](verification.json) checks
production/native build identity. [External provenance](external-provenance.json)
binds the original proof and probe bytes. Command durations are not performance
measurements; test counts are test functions, not generated subcases.

The CUDA-hidden/CPU run used the first test snapshot: 147 tests, 94 passed and
53 hardware skips. All subsequent fixture edits affect CUDA-only cases; their
revised checks are covered by the final-source compiler sweep. Default Rust tests,
H100 and hidden Python-binding graph/planner tests, formatting, both Clippy
configurations and the README/navigation/CUDA example checks passed.

## Reproduction and delivery

Use the recorded environment and recipes with the worktree-local Python 3.12.14
installation, pinned Rust 1.92.0, locked dev/reference dependencies and fresh
source-export release wheels. Only the verified relocatable interpreter was
copied; no project environment, wheel or installed package was copied. All caches
and outputs stay inside this worktree. Resources are `gpu` and `cpu-heavy`;
ordinary checks use mask `0`, restoration uses `0,1`, and hidden checks use an
empty mask. UUID/utilization/memory snapshots are observations, not reservations.
Native/reference runtime is CUDA 13.0, driver 580.82.07; nvcc is 12.6.85, while
native kernels use driver JIT of embedded PTX.

These are uncommitted development measurements. The implementation agent is
explicitly forbidden to commit, push, create PRs or modify `.burner`. Therefore
Burner must make the managed implementation commit and then capture a separate
fresh clean-commit build and repeat the recorded final checks, using the existing
source-export builder with `--revision HEAD`. Bind that capture to the new clean
HEAD; the original baseline probe and development replay retain their original
main-based guards and measurement labels. That capture, independent review/
revision, same-branch draft PR, delivery and full fresh merge gates remain pending;
no successful results are assumed. Preserve review metadata and blocked-push
safeguards. PR1970/PR1971 remain separate unadopted human-review campaigns.

Frozen corpora, evaluator definitions/weights, benchmark evidence, observer,
hardware matrix and managed progress artifacts are unchanged. This does not
claim general torch.compile, Inductor, training, performance or hardware parity.
