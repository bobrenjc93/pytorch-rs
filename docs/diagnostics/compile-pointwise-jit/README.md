# Native default pointwise JIT validation

Fresh coverage, CUDA-performance and generated-code captures measure clean
implementation commit `dbd1a0f6b398fde35ddd760018ee105a6758760b`, including all three rounds of review fixes,
on 2026-09-12. Both unchanged public-default-compile-v2 scoring commands report
`valid: true`, `diagnostic: false`, with no infrastructure error. The clean
baseline remains pinned to campaign base
`76738b39fd6884ffd43b4dff2f5292257a1c6e6b`, verified as the merge base with main.
See the [implementation contract](../../compile-pointwise-jit.md).

| Unchanged public-default-compile-v2 gate | Clean baseline | Clean candidate `dbd1a0f6` |
| --- | ---: | ---: |
| Weighted coverage | 0% (0/112 cells) | 6% (4/112 cells) |
| Weighted CUDA performance | 0% (0/56 cells) | 12% (4/56 cells) |
| CUDA-performance common-success geometric mean, reference/candidate | null | 2.1452513673646267 |

The measured gains over the baseline are 6 percentage points
of weighted coverage and 12 points of weighted CUDA
performance. Only the two arithmetic programs in both CUDA variants pass;
all other cells remain zero. Successful performance cells each reach the cap
of one, accounting for the arithmetic category's 12% weight. The uncapped ratio
describes only those four cells; the baseline has no common successes, so its
ratio is null. These are fixed-corpus results, not general Inductor parity.
The review fixes leave the earlier candidate's weighted scores unchanged.

## Current committed evidence

- [Coverage](postcommit-dbd1a0/candidate-coverage.json.gz) and
  [CUDA performance](postcommit-dbd1a0/candidate-cuda-perf.json.gz) are
  byte-preserving gzip copies of complete evaluator reports. They preserve
  all 112 coverage and 56 performance cells, both CUDA implementation orders,
  five warmups, 17 samples, one host thread, symmetric synchronization,
  changed-input checks, cold costs, unsupported outcomes and slow results.
  Every reference program passed in all five reference workers.
- The [receipt](postcommit-dbd1a0/postcommit.json) records commands, environment,
  setup timestamps, cache state, source/build/wheel identities and verification.
  [Logs](postcommit-dbd1a0/postcommit-logs.json.gz) retain both gate/build outputs,
  all ten workers, the focused checks, and the verification scripts. Raw output
  observations and wheels remain at verified worktree-local paths in the reports.
  All 123 baseline source hashes were verified against its commit; all 129
  candidate source hashes, wheel/native/interpreter identities, worker logs and
  raw-observation hashes were verified locally. Evaluator/corpus hashes, category
  weights, tolerances and denominators match the baseline.
- [Generated CUDA](postcommit-dbd1a0/kernel.cu),
  [PTX](postcommit-dbd1a0/kernel.ptx.gz),
  [provenance](postcommit-dbd1a0/provenance.json) and
  [source manifest](postcommit-dbd1a0/source-manifest.json.gz) were freshly
  captured with the existing [capture.py](capture.py) from the exact installed
  evaluation wheel. The provenance field named `base_commit` identifies the
  measured candidate commit. An independent ordinary public function exercises
  two shapes, changed values and fresh outputs, sharing one code module between
  two graph entries. The capture asserts that installed PyTorch was not imported.
  This establishes code-generation provenance, not a separate timing score.

The fresh committed wheel passed all six pointwise regression modules:
37 tests with one explicit two-device skip under `CUDA_VISIBLE_DEVICES=0`.
This includes both operator-added liveness/signature modules, overflow and
signed-zero regressions, repeated expressions/input identities, sign normalization,
shared products, Boolean/integer/float scalar distinctions, callback-free
admission, generated expression trees, guards, cache/reset,
concurrency, lifetimes and no-body/no-eager/no-PyTorch execution checks. The
checkout remained clean throughout both gates and these checks.

The native JIT selected NVRTC **13.0**, CUDA runtime **13000**, `compute_90`,
explicit FMA and gradual underflow, without fast math. Independently queried
`nvcc` **12.6** was not used for JIT generation. The GPU was H100 index 0,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver **580.82.07**. Python was
3.12.14, reference PyTorch 2.13.0+cu130, Rust 1.92.0, with release native builds.
The reports retain actual library paths, hashes and before/after GPU snapshots;
snapshots do not establish a scheduler reservation.

## Preserved earlier evidence

All earlier measured artifacts remain byte-for-byte unchanged:

- [Campaign baseline](baseline.json.gz): clean `76738b39`, zero coverage and
  performance, null common-success ratio. It is the comparison baseline for
  this candidate; unchanged historical workloads were not rerun.
- [Earlier coverage](candidate-coverage.json.gz),
  [earlier CUDA performance](candidate-cuda-perf.json.gz),
  [receipt](postcommit.json), [logs](postcommit-logs.json.gz), and root-level
  [codegen provenance](provenance.json), [source](kernel.cu),
  [PTX](kernel.ptx.gz), [manifest](source-manifest.json.gz): clean `5b93c983`,
  before review fixes. They do not measure the current candidate.
- [Original development diagnostic](candidate-diagnostic.json.gz): explicitly
  unscored dirty-source evidence, including original observations.
- [First-review coverage](postcommit-60abd/candidate-coverage.json.gz),
  [CUDA performance](postcommit-60abd/candidate-cuda-perf.json.gz),
  [receipt](postcommit-60abd/postcommit.json),
  [logs](postcommit-60abd/postcommit-logs.json.gz) and
  [codegen provenance](postcommit-60abd/provenance.json): clean `60abd863`,
  before the second numerical review fixes. These reports also record 6%/12%;
  their timings do not measure the current revision.
- [First-review validation](review-fixes.md) and [bundle](review-fixes.json.gz):
  development regressions and original failures before the second review fixes.
  Its exhaustive selection covered 69 compiler files and 805 cases.
- [Second-review coverage](postcommit-53c100/candidate-coverage.json.gz),
  [CUDA performance](postcommit-53c100/candidate-cuda-perf.json.gz),
  [receipt](postcommit-53c100/postcommit.json),
  [logs](postcommit-53c100/postcommit-logs.json.gz) and
  [codegen provenance](postcommit-53c100/provenance.json): clean `53c10058`,
  before the third review fixes. Those 6%/12% measurements remain pinned to
  their original code and build identities.
- [Second-review validation](review-round2.md) and [bundle](review-round2.json.gz):
  original 24 failing subcases, before/after numerical probes and checks at
  their original source identity, before the third review fixes.
- [Third-review validation](review-round3.md) and [bundle](review-round3.json.gz):
  original 30 failing subcases, intermediate failures, numerical probes and
  final checks. All 124 implementation-source hashes and six regression-module
  hashes match this committed revision. It records 406 default / 433 bindings
  Rust tests, 96 backend/entrypoint checks, Clippy, documentation and separate
  two-device checks. This remains unscored development evidence. Unrelated full
  suites were not repeated during this evidence refresh.
- Original author [inventory](validation.json), [logs](logs.json.gz) and
  [IEEE probes](initial-ieee-probes.json.gz) retain the initial validation,
  failures and subsequent repairs at their original source identities.

Historical paths identify the original captures and may no longer contain the
original installed build. Current-candidate credit uses only the fresh
`postcommit-dbd1a0` reports. No failed measurement was overwritten or promoted
as a score. This evidence step does not approve the branch or replace review.

## Reproduce

Run from a clean checkout with the locked worktree-local environment. Create
local cache/temp directories first; unset an inherited `CONDA_PREFIX` before
the wrapper supplies `VIRTUAL_ENV` to maturin. The wrapper sets local
`CARGO_HOME`, `CARGO_TARGET_DIR`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` and
the evaluation virtual environment, builds a release wheel and installs it.

```bash
unset CONDA_PREFIX
mkdir -p target/tmp target/xdg-cache target/default-compile-eval/cuda-cache
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/target/tmp"
export XDG_CACHE_HOME="$PWD/target/xdg-cache"
export CUDA_CACHE_PATH="$PWD/target/default-compile-eval/cuda-cache"
export CUDA_CACHE_DISABLE=1 UV_SYSTEM_CERTS=true HTTPS_PROXY=http://fwdproxy:8080
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric coverage --output target/postcommit-dbd1a0/coverage.json
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric cuda-perf --output target/postcommit-dbd1a0/cuda-perf.json
```

Choose new output paths when reproducing; the evaluator refuses to overwrite
evidence. The [receipt](postcommit-dbd1a0/postcommit.json) records the exact
codegen and six-module unittest commands using
`target/default-compile-eval/venv/bin/python`, with dedicated local
Inductor/Triton test caches. The unchanged
[gate documentation](../../torch-compile-default-evaluator.md) defines scoring.

Driver disk-code caching was disabled symmetrically for both implementations
in these candidate gates; the historical baseline did not set that flag.
These results are not cold-cache speed comparisons between builds. Worker
Inductor/Triton caches start separately fresh; dependency/native build caches
may be warm, as recorded in setup receipts. No implementation, dependency,
test, evaluator, corpus or managed progress artifact changed in this step.
