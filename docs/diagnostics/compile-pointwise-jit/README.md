# Native default pointwise JIT validation

The current coverage, CUDA-performance and generated-code captures measure clean
implementation commit `aab2fd711db784c26aa26e218e0e14443fb39bc2` on 2026-09-13,
including the libdevice product-order repair. Both unchanged
public-default-compile-v2 commands report `valid: true`, `diagnostic: false`,
with no infrastructure error. The source-bound campaign baseline remains
`76738b39fd6884ffd43b4dff2f5292257a1c6e6b`, verified as the merge base with main.
See the [implementation contract](../../compile-pointwise-jit.md).

| Unchanged public-default-compile-v2 gate | Clean baseline | Clean candidate `aab2fd7` |
| --- | ---: | ---: |
| Weighted coverage | 0% (0/112 cells) | 6% (4/112 cells) |
| Weighted CUDA performance | 0% (0/56 cells) | 12% (4/56 cells) |
| CUDA-performance common-success geometric mean, reference/candidate | null | 1.8926844887419676 |

The measured gains over the baseline are 6 percentage points of weighted
coverage and 12 points of weighted CUDA performance. The uncapped ratio
applies only to common-success cells; the baseline has no common successes,
so its ratio is null. The passing cells are `affine_relu` and `trigonometric`,
each in both CUDA variants. Weighted scores remain 6/12, as in the preceding
capture. Unsupported cells and slow results remain in the frozen
denominator. These are fixed-corpus results, not general Inductor parity.

## Current committed evidence

- [Coverage](postcommit-aab2fd/candidate-coverage.json.gz) and
  [CUDA performance](postcommit-aab2fd/candidate-cuda-perf.json.gz) preserve the
  complete evaluator reports: all 112/56 cells, both CUDA implementation orders,
  five warmups, 17 samples, one host thread, symmetric synchronization,
  fresh changed-input checks, cold costs, unsupported outcomes and slow results.
- The [receipt](postcommit-aab2fd/postcommit.json) records commands, environment,
  setup timestamps, cache state, source/build/wheel identities and verification.
  [Logs](postcommit-aab2fd/postcommit-logs.json.gz) retain gate/build outputs,
  all ten workers, checks and verification scripts. All 123 baseline source
  hashes were verified against its commit; candidate source, wheel, native,
  interpreter, worker-log and raw-observation hashes were verified locally.
  Frozen evaluator/corpus hashes, weights, tolerances and denominators match
  the baseline. Recorded temporary wheel and raw-observation paths may disappear
  during worktree cleanup; checked-in reports and compressed logs are durable.
- [Generated CUDA](postcommit-aab2fd/kernel.cu),
  [PTX](postcommit-aab2fd/kernel.ptx.gz),
  [provenance](postcommit-aab2fd/provenance.json) and
  [source manifest](postcommit-aab2fd/source-manifest.json.gz) were freshly
  captured using the existing [capture.py](capture.py) from the installed
  evaluation wheel. Its `base_commit` field identifies the measured candidate.
  An independent ordinary function exercises two shapes, changed values and
  fresh outputs, sharing one code module between two graph entries. The capture
  asserts that installed PyTorch was not imported. This is code-generation
  provenance, not another timing score.

The committed wheel passed all sixteen pointwise regression modules: 96 tests,
with two explicit two-device skips under `CUDA_VISIBLE_DEVICES=0`.
Ninety top-level/backend checks and sixteen archive/documentation checks also
passed. Both actual device-restoration tests and 26 Rust pointwise tests passed
in the preceding [repair validation](review-libdevice-order.md); this evidence
step did not repeat those unchanged checks. The clean selection includes the
operator's liveness, signature, integer-scalar, warm-binding, nonfinite-history,
offset-cache, constant-transcendental and product-priority regressions.
The checkout remained clean throughout both gates and all recorded checks.

Native JIT selected NVRTC **13.0**, CUDA runtime **13000**, `compute_90`, explicit
FMA and `--ftz=false`, without fast math. Runtime sine flushes subnormal inputs
to signed zero before accurate libdevice evaluation; constant sine/cosine use
double libdevice between float32 boundaries. Queried `nvcc` **12.6** was not used
for JIT generation. Hardware was H100 index 0,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver **580.82.07**; Python was
3.12.14 and reference PyTorch 2.13.0+cu130. Receipts preserve actual library
paths/hashes and before/after GPU snapshots. Snapshots are not reservations.

## Baseline and history

The [baseline](baseline.json.gz) remains unchanged. The
[evidence archive](archive.md) catalogs earlier captures, repair validations
and original failures at their actual source/build identities. The preceding
`08e37fe5` captures remain intact and receive no current-candidate performance
credit in the verified archive. The [libdevice repair](review-libdevice-order.md) preserves original
failures and development checks. Additional
[operator observations](operator-libdevice-compositions.json.gz) preserve the
32-program check, build attestation, reproduction script and original regression
failure log verbatim. The 32-program check used dirty base `6b73cbc`, an
isolated Python 3.12.12 install and GPU 7; the failure log used the earlier
native binary. Original external paths are historical provenance.
They are unscored development evidence, not this clean capture.

## Reproduce

Run from a clean checkout with the locked worktree-local environment. The wrapper
builds and installs a release wheel and sets local Cargo/uv/Python paths.

```bash
unset CONDA_PREFIX
mkdir -p target/tmp target/xdg-cache target/default-compile-eval/cuda-cache
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/target/tmp"
export XDG_CACHE_HOME="$PWD/target/xdg-cache"
export CUDA_CACHE_PATH="$PWD/target/default-compile-eval/cuda-cache"
export CUDA_CACHE_DISABLE=1 UV_SYSTEM_CERTS=true
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric coverage --output target/new-capture/coverage.json
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric cuda-perf --output target/new-capture/cuda-perf.json
```

Use new output paths; the evaluator refuses to overwrite evidence. The receipt
records exact codegen and regression commands with dedicated local test caches.
A proxy is not required by the project; configure your own approved proxy if
your network needs one. The [gate documentation](../../torch-compile-default-evaluator.md)
defines scoring. Driver disk-code caching was disabled symmetrically in these
candidate gates; the historical baseline did not set that flag. These are not
cold-cache speed comparisons between builds. Worker Inductor/Triton caches
start separately fresh; dependency/native build caches may be warm as recorded.

No implementation, dependency, test, evaluator, corpus or managed progress
artifact changed during capture. Independent review, CI and merge qualification
remain required.
