# Bounded direct Python helper capture

Current clean candidate measurements and two-H100 helper captures, including the
[warm-admission review fix](review-warm-admission.md), are recorded in
[postcommit-76ea91a7](postcommit-76ea91a7/README.md). The earlier
[a8211b07 captures](postcommit-a8211b07/README.md) and development records below
retain their original source identities and outcomes. The unchanged clean-main
measurement was revalidated for the new comparison.

This is development evidence for the frontend change based on
`fe7537251a53b54e1064309d28db2631605fb720`. It is not a clean-head evaluation,
a corpus score, or evidence of general Python/Inductor equivalence. Production
changes are confined to `python/torch_rs/_compile_pointwise.py`; the original
native IR, numerical admission, executors and reset owner are reused.

The [focused tests](../../../tests/test_compile_pointwise_helpers.py) cover
callback-free admission, data-only returns (including CPython 3.12
`RETURN_CONST`), helper identity, expanded budgets, source laziness, runtime
scalars, native input validation, failure-atomic publication, LRUs, reset and GC.
The decisive cache test retains F1/codeA, changes F1 to codeB, rebinds the root to
F2/codeA, and then introduces a new tensor ABI; its lowering still uses codeA.
Hardware tests additionally compare generic helper forms and the unchanged
`python_helper` corpus factory with ordinary `torch.compile(fn)`.

## Reproduction

Hold Burner's canonical `gpu` and `cpu-heavy` resources first. This author run's
read-only lock records named `idea_1f7d364a` for both resources. GPUs 0 and 1 were
explicitly selected; no other owner's jobs were interrupted. The evidence
contains their full UUIDs and before/after memory/utilization inventories.
An idle inventory alone is not a reservation.

Use a real Python 3.12 installation, dependencies, wheel and writable caches
inside the checkout. Set `TMPDIR`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`, `CARGO_HOME`, `CARGO_TARGET_DIR`, `CUDA_CACHE_PATH`,
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to worktree-local directories;
set `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
`CARGO_BUILD_JOBS=4`. The retained `env.sh` records the exact author environment.
The Python download failed through the proxy. The author copied an existing
Python 3.12.14 distribution and pinned package files into this worktree, checked
package compatibility, and replaced the copied native package with the fresh
source-built wheel before any tests. External sources were read only.

From the checkout root, using that local environment:

```bash
maturin build --release --locked --offline --out target/helper-work/wheels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/helper-work/wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -p 'test_compile*.py'
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_pointwise_helpers
CUDA_VISIBLE_DEVICES=0,1 cargo test --locked --offline --lib pointwise
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest discover -s tests -p 'test_compile*.py'
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_helpers.HelperHardware
.venv/bin/python docs/diagnostics/compile-pointwise-helpers/compare.py \
  --output target/helper-work/comparison-reproduction \
  --wheel target/helper-work/wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl \
  --devices 0,1
```

The output directory must be new and inside the checkout. The diagnostic runs
native/reference and reference/native in separate fresh processes and caches
on each selected GPU. Each wrapper persists through four shape/value/alias
states. Both sides use the same inputs, five warmups, 17 synchronized steady
samples, one host thread, and fixed `rtol=1e-5`, `atol=1e-6`, with explicit
NaN/infinity/signed-zero checks. Every cold duration is reported separately;
native cold admission includes body-policing profiling, so cold timings are
instrumented diagnostics. Steady samples have profiling disabled on both sides.
Cold and warm native calls are separately checked for body execution, and native
processes reject importing PyTorch. Default compiler configuration is untouched.
No reference-only reset, limit increase or favorable repeat selection is used.

The diagnostic covers three generic helper programs plus the unmodified frozen
helper case. These are focused measurements, not the full fixed coverage or
performance gate. Both native successes and any failures remain in its reports.
Generated sources/PTX identify the executor actually dispatched at each state.

## Evidence interpretation

The archives described below retain original logs, all detailed diagnostic
reports, source/build/wheel/import identities, synchronization runtime identity,
NVRTC options/version, compiler versions, GPU inventories and generated kernels.
The wheel's Python files and native extension were byte-compared with their
installed copies; Python sources were also compared with this checkout.

Initial helper hardware test failures came from invalid fixtures: this package
does not expose `float64` or `int64`, and CUDA `requires_grad_` is unsupported.
Fixtures now use constructible unsupported inputs (strided/CPU/CPU-gradient)
and retain native admission assertions. The failure logs are preserved.
The first diagnostic run was started with body profiling in steady timings;
the script was corrected while it was running. Its raw outcomes are preserved
but **comparison-v1 is invalid timing evidence** because it mixes script versions.
Comparison-v2 was run only after the script was fixed and uses unprofiled steady
samples. No outcome from v1 is substituted for v2.

Clean candidate/main qualification, independent review, exact-head scoring,
publication and merge remain with Burner and its existing normal owner. This
worker did not commit or modify evaluators, corpora, managed progress artifacts,
or the campaign observer. The operator's existing campaign-scoped pre-cleanup
observer must still preserve detailed frozen-evaluator reports; this focused
archive does not replace that observer or claim those evaluations ran.

## Recorded results

| Check | Result |
| --- | --- |
| Fresh locked release wheel; byte comparison against checkout/imports | Passed |
| Hardware-free compiler sweep (before four additional helper tests) | 922 tests, 354 skipped, no failures |
| Focused helper admission/cache/H100 tests | 28 passed |
| Final full compiler sweep with GPUs 0 and 1 visible | 926 tests, 201 skipped, no failures |
| Physical GPU 1 helper hardware tests; final two-device test | 3 passed, 1 explicit reservation skip; 1 passed |
| Rust pointwise regressions, GPUs hidden / GPUs 0 and 1 visible | 34 passed in each run |
| Fixed v2 isolated diagnostic, two GPUs and both framework orders | Eight successful legs; 64 paired histories passed |

The diagnostic's individual reference/native median latency ratios range from
0.527 to 1.301. Values below one mean the native call was slower. All raw samples
and every ratio remain in the archive; this range is not an aggregate score.

Python was 3.12.14; reference PyTorch was `2.13.0+cu130`. Driver 580.82.07 served
H100 UUIDs `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1` (index 0) and
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296` (index 1). Both processes synchronized
through the same local CUDA runtime 13.0 (`cudaRuntimeGetVersion=13000`);
native kernels used NVRTC 13.0 and `compute_90`, with `--fmad=true`, `--ftz=false`,
`--prec-div=true`, `--prec-sqrt=true`. The system `nvcc` was 12.6.85, not the
compiler used to generate these JIT kernels. Rust was 1.92.0 with the repository's
release thin-LTO/codegen-units=1 configuration and fresh worktree-local targets.

Frontend SHA256:
`1c4405368457998bee39d940e521e4fd283c4dcec44a39c241635f35a7d2fc95`.
Wheel SHA256:
`2c309de820fa4c5bb5bdcf4389c794d581a9aeef691624f1e1f371807a21303a`.
Native extension SHA256:
`3b3a17305660e40122191c5c10b6435412f026b1ef5508fa80a01298cef964d8`.

[comparisons.json.xz](comparisons.json.xz) holds 311 retained files: the two
complete diagnostic campaigns (including invalid v1), per-leg logs, all
outputs/timings, source manifests, generated CUDA/PTX, and the measured frontend,
diagnostic and focused-test sources. The measured test snapshot predates only
the final portable two-device availability skip.
[validation.json.xz](validation.json.xz) holds setup/build/test logs, identities,
reservation records, environment and final-source hashes. Files in each archive
are a mapping from original relative path to `text`, with SHA256 metadata.
Gzip originals are decoded losslessly in the comparison archive to improve
cross-report compression; their original and decoded hashes are retained.

Read a report without writing outside the checkout:

```python
import json, lzma
from pathlib import Path
archive = json.loads(lzma.decompress(Path(
    "docs/diagnostics/compile-pointwise-helpers/comparisons.json.xz").read_bytes()))
summary = json.loads(archive["files"]["comparison-v2/summary.json.gz"]["text"])
assert summary["passed"]
```
