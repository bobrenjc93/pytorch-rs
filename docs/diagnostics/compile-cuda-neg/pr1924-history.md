# PR #1924 historical validation

These two bundles preserve initial integration author records and a separate
fixed paired timing experiment. They do not measure the current checkout or the
later [clean df1964bd capture](../../compile-cuda-neg-validation.md#integrated-clean-commit-evidence).
They grant no current-candidate score credit, replace no Burner scores, and
must not be combined across environments into a single score. The existing
source-PR bundle and integrated clean-commit reports remain unchanged.

## Initial integration author records

[Original summary](initial-integration-1924/integration-checks/summary.md),
[build receipt](initial-integration-1924/integration-checks/build-record.json),
[build-start source snapshot](initial-integration-1924/integration-checks/build-start.json),
[command receipts](initial-integration-1924/integration-checks/checks-record.json),
[final GPU receipts](initial-integration-1924/integration-checks/final-gpu-record.json),
and [final audit](initial-integration-1924/integration-checks/final-audit.json)
bind the original source and installed native hashes. The native build began
and finished clean at `764f159b0ae08e3a9be4793bc7c528844b322380` in
`/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/composite_61b1dc8d`.
Subsequent measurements record dirty documentation and untracked diagnostic/test
files; the final audit records an empty production diff. This is not a clean
final-checkout capture. Historical absolute paths are provenance, not paths to
use in this checkout.

The first full suite and all four timing reports used CPython **3.12.14+meta**
through that worktree's `.venv/bin/python`, PyTorch 2.13.0+cu130, CUDA runtime
13.0, H100 and driver 580.82.07. The native release build used Rust/Cargo 1.92.0
and driver-JIT PTX; the private timing benchmark separately used nvcc 12.6.85.
The later cross-interpreter checks reused the native wheel, not fresh builds.

| Preserved check | Recorded result | Raw report/log |
| --- | --- | --- |
| Initial CUDA timing, fresh | 4/4 eligible passes; 100% | [report](initial-integration-1924/integration-checks/cuda-performance-fresh.json) |
| Initial CUDA timing, warm | 4/4 eligible passes; 92.10364622089932% | [report](initial-integration-1924/integration-checks/cuda-performance-warm.json) |
| Confirmation 1 | 4/4 eligible passes; 100% | [report](initial-integration-1924/integration-checks/cuda-performance-confirm-1.json) |
| Confirmation 2 | 4/4 eligible passes; 79.30190144231703% | [report](initial-integration-1924/integration-checks/cuda-performance-confirm-2.json) |
| Full Python 3.12.14+meta suite | 5,331 tests; OK, 11 skips | [log](initial-integration-1924/integration-checks/python-full.log) |
| Full Python 3.14.7 suite | 5,331 tests; OK, 11 skips | [log](initial-integration-1924/integration-checks/python314-full.log) |
| Fixed compile corpus | 38/38 eligible cases; 100% | [report](initial-integration-1924/integration-checks/compile-evaluation.json) |
| Fixed CUDA math | 2/6 cases across all three seeds; four zero-credit cases | [report](initial-integration-1924/integration-checks/cuda-math.json) |
| Neg/add diagnostic | GPU 0: 128 native passes, 40 unsupported; GPUs 0,1: 8 passes, 4 unsupported; all expectations met | [single](initial-integration-1924/integration-checks/diagnostic-single.json), [multi](initial-integration-1924/integration-checks/diagnostic-multi.json) |

Both full-suite logs end in `OK (skipped=11)`, corroborated by zero exit codes.
Interim counts misread expected error logging as failures; the final Python
3.14 result does **not** report nine suite failures. Actual failed setup and
focused reproduction logs are preserved too.

The [cross-interpreter receipts](initial-integration-1924/integration-checks/cross-interpreter-record.json)
and environment logs retain exact versions and interpreter/install paths:
[standard 3.12.14 candidate](initial-integration-1924/integration-checks/managed312-environment.log),
[standard 3.12.14 baseline](initial-integration-1924/integration-checks/baseline-managed312-environment.log),
and [3.14.7 candidate](initial-integration-1924/integration-checks/python314-environment.log).
The [baseline build](initial-integration-1924/integration-checks/baseline-build-record.json)
is `e4cddf0ecb45c23c683953fa2f8c45904b170a80`. On standard Python 3.12.14,
both [candidate](initial-integration-1924/integration-checks/managed312-known-failures.log)
and [baseline](initial-integration-1924/integration-checks/baseline-managed312-known-failures.log)
ran 13 tests and failed the same two noncanonical boolean-buffer cases.

The [factory seed receipts](initial-integration-1924/integration-checks/factory-seed-record.json)
preserve all 20 baseline/candidate runs at seeds 0–9: five tests passed in each
cold-cache run. The [cache receipts](initial-integration-1924/integration-checks/factory-cache-record.json)
and seed-0 warmup logs retain reference source/cache hashes. Reusing that bytecode
cache at seed 6 reproduced three ordering failures in five tests on both
[candidate](initial-integration-1924/integration-checks/factory-candidate-cache-test-seed6.log)
and [baseline](initial-integration-1924/integration-checks/factory-baseline-cache-test-seed6.log).
The candidate factory runs used `target/python312-managed-venv/bin/python`;
the baseline used `target/baseline-e4cddf0/source/.venv/bin/python` under the
original composite root. These bounded reproductions are distinct from the
successful full suites.

## Fixed paired timing validation

The [predeclared plan](paired-validation-1924/plan.json),
[baseline build receipt](paired-validation-1924/baseline/target/paired/reports/build-record.json),
[candidate build receipt](paired-validation-1924/candidate/target/paired/reports/build-record.json),
[run index](paired-validation-1924/results.json), and
[derived summary](paired-validation-1924/summary.json) preserve a separate matched
setup: Python **3.12.12**, PyTorch **2.13.0+cu130**, NumPy **2.5.1**, CPU affinity
**24**, and **H100 GPU 0**. CUDA runtime was 13.0, driver 580.82.07 and nvcc
12.6.85. Fresh native builds used baseline
`e4cddf0ecb45c23c683953fa2f8c45904b170a80` and reviewed candidate
`c41a850446c0cd94e70027b13d197a9c8413ff15`, with clean source receipts and matching
benchmark/dependency-lock hashes. Original interpreters were
`/tmp/pytorch-rs-1924-paired.9gl9RC/{baseline,candidate}/.venv/bin/python`.

The fixed order was [baseline 1](paired-validation-1924/baseline/target/paired/reports/run-1.json),
[candidate 2](paired-validation-1924/candidate/target/paired/reports/run-2.json),
[candidate 3](paired-validation-1924/candidate/target/paired/reports/run-3.json),
[baseline 4](paired-validation-1924/baseline/target/paired/reports/run-4.json),
[baseline 5](paired-validation-1924/baseline/target/paired/reports/run-5.json),
[candidate 6](paired-validation-1924/candidate/target/paired/reports/run-6.json).
All six scored 100%; all 24 native/reference-eligible workload outcomes passed.
Each checkout's first invocation used cold caches and subsequent invocations
reused its caches, with no retries or selection. Each shape retained five
warmups, 17 samples and three calls per sample.

| Shape | Candidate/base median native latency ratio |
| --- | ---: |
| 256 × 256 | 0.9430325515 |
| 1024 × 1024 | 0.9650679390 |
| 4096 × 256 | 0.9739054414 |
| 256 × 4096 | 1.0039280991 |

Ratios divide the candidate median of three run medians by the baseline median
of three run medians. These are bounded matched-setup observations, not proof
of universal parity or an explanation of the earlier unpinned Python
3.12.14+meta timing variation. Every earlier slow report remains preserved.
The archived validation scripts are historical source records with original
external paths; the portable audit below reads them as data and never executes
them, rebuilds environments, or reruns measurements.

## Audit the preserved bytes and accounting

The supplied read-only archives were SHA256-verified and inspected for safe
relative paths, duplicate entries and regular-file/directory-only contents
before extraction inside this worktree. Original member paths and file contents
are preserved; no environments, wheels, native binaries or caches were copied.
The nested `target/paired/reports` directories contain reports only.

| Archive | SHA256 | Extracted files and manifest |
| --- | --- | --- |
| `initial-integration-checks.tar.gz` | `aec0e3b5ec51c443c80160f2a42d25dfdc3ec73149a7ff5db9e3284efe77fa2f` | 109; [manifest](initial-integration-1924/artifact-sha256.json) |
| `paired-validation-reports.tar.gz` | `7de4d24e5b0143d0f323115c8ac4847047bdc0980873c5cddf1b4776e1c47689` | 42; [manifest](paired-validation-1924/artifact-sha256.json) |

Run from the repository root with Python 3.10+ (standard library only):

```bash
python -B -m unittest discover -s tests -p test_pr1924_historical_evidence.py -v
```

The [audit tests](../../../tests/test_pr1924_historical_evidence.py) verify the
complete file inventory and SHA256 hashes, original attribution, terminal suite
summaries, reproduction outcomes, exact score/case accounting, paired report
receipts and median ratios, and documentation links. They do not require the
original temporary archives, historical paths, PyTorch or a GPU.
