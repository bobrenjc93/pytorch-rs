# Clean shape-branch evidence at be808651

Measured commit: `be808651fa90415c951e46180d3417e7939d61e8` (base
`fc5d39e3ca3889440950161177f78ff5bdfa4704`). The public `compile()` docstring
changed after the [df50471 capture](../postcommit-df50471e/README.md), changing the
packaged initializer's bytes. This refresh measures the new wheel. An AST check
confirmed that only docstrings changed in the initializer; historical evidence
remains pinned to its original revision and unchanged.

## Results

The same committed `tests/test_compile_pointwise_shape_branches.py` ran unchanged:

- H100, CPython 3.12.13: **22 passed, zero skips**.
- CPython 3.10.19, 3.11.15, 3.12.13, 3.13.13 and 3.14.5: **19 passed per version**,
  with the three CUDA tests explicitly skipped in portable runs.

The two ordinary default-Inductor comparison histories retain their fixed inputs,
shapes, changed-value checks, helpers, loops, scalar transitions and aliases.
Outputs are synchronized and materialized. The committed tests forbid native
Python-body execution, check native PTX, and require reference-generated kernels
with no graph breaks. Observations were respectively 9/5 reference graphs and
7/5 generated kernels, with no graph breaks. These are compiler observations,
not wrapper-entry counts, an official score, or a general coverage claim.

The [bounded support contract](../../../compile-pointwise-jit.md#bounded-root-shape-branches)
is unchanged. Unused-heavy signature reuse is checked; earlier comparative
overhead timings remain development evidence, not fresh latency measurements.
Unrelated full suites, historical workloads and official evaluations were not rerun.

## Provenance and reproduction

[Measurements](measurements.json.gz) contain clean before/after status, source
hashes, wheel members, timestamps, commands, per-test outcomes, interpreter/import
paths, loaded CUDA libraries and before/after GPU snapshots. All measured code
hashes remained unchanged during capture. Each venv's package initializer,
frontend and native extension matched the wheel bytes; the new public docstring
was verified in the installed package. No package-source overlay was used.

- Wheel SHA256: `2585c1f34cd57c57fddc315989a2e3cf8060d7b7bd21dcc7497edc71e3943fbe`.
- Locked release build: Rust 1.92.0, Maturin, fresh local Cargo target after setup.
- GPU 0: H100, UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07.
- PyTorch 2.13.0+cu130; loaded CUDA runtime 13000 and NVRTC 13.0.
  System `nvcc` is 12.6.85; the pointwise JIT uses NVRTC.
- GPU tests used `CUDA_VISIBLE_DEVICES=0` and fresh compiler caches. Portable
  tests used empty visibility; those skips are not GPU evidence.

Use the locked build procedure in [CONTRIBUTING.md](../../../../CONTRIBUTING.md)
from the measured clean revision, then run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_pointwise_shape_branches
```

The [raw archive](raw-captures.tar.gz) preserves setup/build/install/verification
logs, exact commands, environment settings and the existing stdlib capture runner.
Only that runner's expected-commit assertion was updated; its workloads and
compiler settings were unchanged. Runner hashes and that exact difference are
recorded. Use a new directory for any rerun rather than overwriting retained data.

## Retention and limits

The [manifest](raw-retention-manifest.json) hashes all 45 archived raw files and
records their canonical locations under
`target/default-compile-eval/shape-branches/postcommit-be808651/`. Initial bootstrap
failures caused by the absent previous venv were preserved before setup was
recreated inside this worktree. The subsequent capture runs passed without failures.

The previous checked-in archives and their hashes were verified unchanged.
Their 392 manifest-listed untracked development files were absent locally on
entry, as was the previous venv. This step does not establish external retention
or claim recovery of those bytes; the older checked-in 30-file raw archive remains
available. New raw captures and the new wheel are retained inside this worktree.

Only new evidence and its documentation changed in this step. Implementation,
tests, dependency definitions, benchmark harnesses, evaluators and managed progress
artifacts are unchanged. Independent review and Burner's normal merge gates remain
required. [Artifact verification](verification.log) records the archive/provenance checks.
