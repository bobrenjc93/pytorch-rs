# Native default pointwise JIT validation

The constant-folding precision and zero-origin repair changes implementation
after the latest clean capture. **A fresh post-commit capture is required** once
Burner commits this revision. [Repair validation](review-folding.md) records
development checks and original failures without assigning campaign scores.

The retained coverage, CUDA-performance and generated-code captures measure clean
implementation commit `0c836a491530d34b368bc13312c5695331634cf5` on 2026-09-12,
including the composed constant-tensor arithmetic fix. Both unchanged
public-default-compile-v2 scoring commands report
`valid: true`, `diagnostic: false`, with no infrastructure error. The clean
baseline remains pinned to campaign base
`76738b39fd6884ffd43b4dff2f5292257a1c6e6b`, verified as the merge base with main.
See the [implementation contract](../../compile-pointwise-jit.md).

| Unchanged public-default-compile-v2 gate | Clean baseline | Pre-repair candidate `0c836a49` |
| --- | ---: | ---: |
| Weighted coverage | 0% (0/112 cells) | 6% (4/112 cells) |
| Weighted CUDA performance | 0% (0/56 cells) | 12% (4/56 cells) |
| CUDA-performance common-success geometric mean, reference/candidate | null | 2.072811651004458 |

The measured gains over the baseline are 6 percentage points of
weighted coverage and 12 points of weighted CUDA performance.
Only the two arithmetic programs in both CUDA variants pass; all other cells
remain zero. The uncapped ratio describes only those four cells; the baseline
has no common successes, so its ratio is null. These are fixed-corpus results,
not general Inductor parity.

## Latest clean capture (refresh required)

- [Coverage](postcommit-0c836a/candidate-coverage.json.gz) and
  [CUDA performance](postcommit-0c836a/candidate-cuda-perf.json.gz) are
  byte-preserving gzip copies of complete evaluator reports. They preserve
  all 112 coverage and 56 performance cells, both CUDA implementation orders,
  five warmups, 17 samples, one host thread, symmetric synchronization,
  changed-input checks, cold costs, unsupported outcomes and slow results.
  Every reference program passed in all five reference workers.
- The [receipt](postcommit-0c836a/postcommit.json) records commands, environment,
  setup timestamps, cache state, source/build/wheel identities and verification.
  [Logs](postcommit-0c836a/postcommit-logs.json.gz) retain both gate/build outputs,
  all ten workers, the focused checks, and the verification scripts. Original
  raw-observation and wheel paths are recorded with hashes, but those temporary
  artifacts may be removed by worktree cleanup. The reports and compressed logs
  checked into this directory are the durable evidence.
  All 123 baseline source hashes were verified against its commit; all 129
  candidate source hashes, wheel/native/interpreter identities, worker logs and
  raw-observation hashes were verified locally. Evaluator/corpus hashes, category
  weights, tolerances and denominators match the baseline.
- [Generated CUDA](postcommit-0c836a/kernel.cu),
  [PTX](postcommit-0c836a/kernel.ptx.gz),
  [provenance](postcommit-0c836a/provenance.json) and
  [source manifest](postcommit-0c836a/source-manifest.json.gz) were freshly
  captured with the existing [capture.py](capture.py) from the exact installed
  evaluation wheel. The provenance field named `base_commit` identifies the
  measured candidate commit. An independent ordinary public function exercises
  two shapes, changed values and fresh outputs, sharing one code module between
  two graph entries. The capture asserts that installed PyTorch was not imported.
  This establishes code-generation provenance, not a separate timing score.

The `0c836a49` committed wheel passed all six pointwise regression modules:
38 tests with one explicit two-device skip under `CUDA_VISIBLE_DEVICES=0`.
This includes both operator-added liveness/signature modules, overflow and
signed-zero regressions, repeated expressions/input identities, sign normalization,
shared products, composed constant-tensor zero signs, scalar-type distinctions,
callback-free admission, generated expression trees, guards, cache/reset,
concurrency, lifetimes and no-body/no-eager/no-PyTorch execution checks. The
checkout remained clean throughout both gates and these checks.

[Repair validation](review-constant-tensors.md) preserves the original failures,
before/after numerical probes, Rust IR, Clippy and build checks. Its 124 source
hashes and six regression-module hashes match `0c836a49`;
those development runs remain distinct from these clean measurements.

The native JIT selected NVRTC **13.0**, CUDA runtime **13000**, `compute_90`,
explicit FMA and gradual underflow, without fast math. Independently queried
`nvcc` **12.6** was not used for JIT generation. The GPU was H100 index 0,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver **580.82.07**. Python was
3.12.14, reference PyTorch 2.13.0+cu130, Rust 1.92.0, with release native builds.
The reports retain actual library paths, hashes and before/after GPU snapshots;
snapshots do not establish a scheduler reservation.

## Baseline and history

The [campaign baseline](baseline.json.gz) measures clean `76738b39`: zero
coverage and performance, with a null common-success ratio. The
[evidence archive](archive.md) catalogs earlier implementations, review repairs,
and original failures. All captured artifacts retain their original bytes,
paths, and measured identities; none was relabeled as a current result.
These reports do not replace independent review or the merge gate.

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
export CUDA_CACHE_DISABLE=1 UV_SYSTEM_CERTS=true
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric coverage --output target/postcommit-0c836a/coverage.json
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric cuda-perf --output target/postcommit-0c836a/cuda-perf.json
```

Choose new output paths when reproducing; the evaluator refuses to overwrite
evidence. The [receipt](postcommit-0c836a/postcommit.json) records the exact
codegen and six-module unittest commands using
`target/default-compile-eval/venv/bin/python`, with dedicated local
Inductor/Triton test caches. The unchanged
[gate documentation](../../torch-compile-default-evaluator.md) defines scoring.

A proxy is not a project prerequisite. If your network requires one, set
`HTTPS_PROXY` to your own approved proxy before running these commands. The
receipts preserve the capture machine's proxy setting for provenance; do not
copy that machine-specific address into another environment.

Driver disk-code caching was disabled symmetrically for both implementations
in these candidate gates; the historical baseline did not set that flag.
These results are not cold-cache speed comparisons between builds. Worker
Inductor/Triton caches start separately fresh; dependency/native build caches
may be warm, as recorded in setup receipts. No implementation, dependency,
test, evaluator, corpus or managed progress artifact changed during that capture.
