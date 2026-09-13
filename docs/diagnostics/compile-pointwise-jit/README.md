# Native default pointwise JIT validation

The current product-contraction repair has
[development validation](review-product-priority.md). The clean captures below
predate that implementation change and do not measure the repaired candidate.
Fresh campaign and generated-code captures require Burner's next clean commit;
the original reports retain their measured identities.

The latest preserved coverage, CUDA-performance and generated-code captures measure clean
implementation commit `08e37fe500f2dfe2b53596335541dd6bf522ad60` on 2026-09-13,
including the restored public `torch.compile():` error prefix and its
CPU/non-Tensor rejection regression assertion. Both unchanged
public-default-compile-v2 scoring commands report
`valid: true`, `diagnostic: false`, with no infrastructure error. The clean
baseline remains pinned to campaign base
`76738b39fd6884ffd43b4dff2f5292257a1c6e6b`, verified as the merge base with main.
See the [implementation contract](../../compile-pointwise-jit.md).

| Unchanged public-default-compile-v2 gate | Clean baseline | Clean candidate `08e37fe5` |
| --- | ---: | ---: |
| Weighted coverage | 0% (0/112 cells) | 6% (4/112 cells) |
| Weighted CUDA performance | 0% (0/56 cells) | 12% (4/56 cells) |
| CUDA-performance common-success geometric mean, reference/candidate | null | 1.9202617174174663 |

The measured gains over the baseline are 6 percentage points of
weighted coverage and 12 points of weighted CUDA performance. The weighted
scores are unchanged from the preceding `74602c07` capture.
Only the two arithmetic programs in both CUDA variants pass; all other cells
remain zero. The uncapped ratio describes only those four cells; the baseline
has no common successes, so its ratio is null. These are fixed-corpus results,
not general Inductor parity.

## Latest clean capture, before the product-contraction repair

- [Coverage](postcommit-08e37f/candidate-coverage.json.gz) and
  [CUDA performance](postcommit-08e37f/candidate-cuda-perf.json.gz) are
  byte-preserving gzip copies of complete evaluator reports. They preserve
  all 112 coverage and 56 performance cells, both CUDA implementation orders,
  five warmups, 17 samples, one host thread, symmetric synchronization,
  changed-input checks, cold costs, unsupported outcomes and slow results.
  Every reference program passed in all five reference workers.
- The [receipt](postcommit-08e37f/postcommit.json) records commands, environment,
  setup timestamps, cache state, source/build/wheel identities and verification.
  [Logs](postcommit-08e37f/postcommit-logs.json.gz) retain both gate/build outputs,
  all ten workers, the focused checks, and the verification scripts. Original
  raw-observation and wheel paths are recorded with hashes, but those temporary
  artifacts may be removed by worktree cleanup. The reports and compressed logs
  checked into this directory are the durable evidence.
  All 123 baseline source hashes were verified against its commit; all 129
  candidate source hashes, wheel/native/interpreter identities, worker logs and
  raw-observation hashes were verified locally. Evaluator/corpus hashes, category
  weights, tolerances and denominators match the baseline.
- [Generated CUDA](postcommit-08e37f/kernel.cu),
  [PTX](postcommit-08e37f/kernel.ptx.gz),
  [provenance](postcommit-08e37f/provenance.json) and
  [source manifest](postcommit-08e37f/source-manifest.json.gz) were freshly
  captured with the existing [capture.py](capture.py) from the exact installed
  evaluation wheel. The provenance field named `base_commit` identifies the
  measured candidate commit. An independent ordinary public function exercises
  two shapes, changed values and fresh outputs, sharing one code module between
  two graph entries. The capture asserts that installed PyTorch was not imported.
  This establishes code-generation provenance, not a separate timing score.

The `08e37fe5` committed wheel passed all fifteen pointwise regression modules:
84 tests with two explicit two-device skips under `CUDA_VISIBLE_DEVICES=0`.
Ninety top-level/backend checks and sixteen archive/documentation checks also
passed. The rejection-message test verifies the public prefix and that
unsupported inputs execute no user callbacks. An in-memory old-prefix mutation
fails that assertion, as intended. The regression selection also includes the
operator-added liveness, signature, integer-scalar,
warm-binding, nonfinite-history and constant-transcendental modules,
scalar-zero contraction and offset cache transitions, callback-free globals-key
admission, static captured-zero transitions, constant-unary precision,
overflow and signed-zero regressions, repeated expressions/input identities, sign normalization,
shared-product consumer ordering, runtime-sine flushing, persistent per-binding
scalar promotion, constant-folding precision, zero-origin and scalar-type rules,
callback-free admission, generated expression trees, guards, cache/reset,
concurrency, lifetimes and no-body/no-eager/no-PyTorch execution checks. The
checkout remained clean throughout both gates and these checks.

[Repair validation](review-zero-boundaries.md) preserves the original failures,
before/after numerical probes, Rust IR, backend, Clippy and build checks. Its 124
source hashes and fifteen regression-module hashes match its historical
`42959e14` revision. Relative to that record, `74602c07` changes rejection
messages and their tests;
those development runs remain distinct from these clean measurements.
[Prefix-repair validation](review-error-prefix.md) retains the original failure
and documents the subsequent assertion and fresh capture.

The native JIT selected NVRTC **13.0**, CUDA runtime **13000**, `compute_90`,
explicit FMA and `--ftz=false`, without fast math. Runtime sine explicitly
flushes subnormal inputs to signed zero before accurate libdevice evaluation;
constant-only sine and other arithmetic retain gradual underflow. Constant-only
sine/cosine use double libdevice between float32 input/output boundaries. Independently
queried `nvcc` **12.6** was not used for JIT generation. The GPU was H100 index 0,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver **580.82.07**. Python was
3.12.14, reference PyTorch 2.13.0+cu130, Rust 1.92.0, with release native builds.
The reports retain actual library paths, hashes and before/after GPU snapshots;
snapshots do not establish a scheduler reservation.

## Baseline and history

The [campaign baseline](baseline.json.gz) measures clean `76738b39`: zero
coverage and performance, with a null common-success ratio. The
[evidence archive](archive.md) catalogs earlier implementations, review repairs,
and original failures. Seven superseded captures are consolidated in a
hash-indexed archive in this repository. All captured artifacts retain their
original bytes, logical paths and measured identities; none was relabeled as a
current result. The baseline and current capture remain directly accessible.
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
  --metric coverage --output target/postcommit-08e37f/coverage.json
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric cuda-perf --output target/postcommit-08e37f/cuda-perf.json
```

Choose new output paths when reproducing; the evaluator refuses to overwrite
evidence. The [receipt](postcommit-08e37f/postcommit.json) records the exact
codegen, fifteen-module regression, backend and archive/documentation commands using
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
