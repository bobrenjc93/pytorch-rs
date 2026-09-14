# Clean shape-branch validation

Measured code: `df50471e38ef73440f64545263c8f22e6c6a1125`, based on
`fc5d39e3ca3889440950161177f78ff5bdfa4704`. These measurements qualify that
implementation's [bounded subset](../../../compile-pointwise-jit.md#bounded-root-shape-branches),
not later code changes or general Inductor parity. This is focused validation;
no official evaluation or CUDA latency score was run.

## Results

The unchanged committed `tests/test_compile_pointwise_shape_branches.py` ran
against a newly packaged and installed release wheel:

| Execution | Passed | Skipped |
| --- | ---: | ---: |
| CPython 3.12.13, real H100 | 22 | 0 |
| CPython 3.10.19, portable | 19 | 3 CUDA tests |
| CPython 3.11.15, portable | 19 | 3 CUDA tests |
| CPython 3.12.13, portable | 19 | 3 CUDA tests |
| CPython 3.13.13, portable | 19 | 3 CUDA tests |
| CPython 3.14.5, portable | 19 | 3 CUDA tests |

The GPU tests use ordinary `torch_rs.compile(fn)` and `torch.compile(fn)` with
untouched defaults. Persistent histories include shapes 3, 5, 10, 12, 3, 0, 1,
changed values, IEEE inputs, and a second helper/loop/alias/scalar history. Outputs
are synchronized and materialized. Assertions check metadata, input preservation,
native PTX and forbidden native Python-body execution. The reference histories
reported respectively 9/5 graphs and 7/5 generated kernels, with no graph breaks;
these are compiler observations, not wrapper-entry counts or coverage scores.

Portable cases check both-arm admission, predicate origin, shape generalization,
negative axes, unbound locals, sequential-return bounds, inactive-helper mutation
and new-ABI readmission, descriptor/code guards, all cache lifetimes, failure
atomicity and unused-source reuse. The 120-unused-argument case checks guards and
reuse, not dispatch timing. Earlier unused-heavy timings remain precommit,
source-hash-bound development evidence; no fresh comparative timing is claimed.
Unrelated full suites and historical workloads were not rerun.

## Provenance and reproduction

[Measurements](measurements.json.gz) contain before/after clean git status,
tracked source hashes, timestamps, exact test commands, interpreter/import paths,
wheel member hashes, per-test outcomes, GPU inventories and runtime library hashes.
The package initializer, frontend and native extension imported from each local
venv were compared byte-for-byte with the wheel. Source hashes remained unchanged
through the captures. Package-source overlays were disabled.

- Wheel SHA256: `2d99bcb07987f5001f9acd8f6547ac3fc4ccd599ef3b6460cfa21603529667b8`.
- Build: locked release Maturin/Cargo, Rust 1.92.0, existing local Cargo artifacts;
  this was fresh wheel packaging, not a cold-build benchmark.
- GPU 0: NVIDIA H100, UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07.
- PyTorch: 2.13.0+cu130. Loaded CUDA runtime: 13000; NVRTC: 13.0.
  The separately installed `nvcc` is 12.6.85; pointwise JIT compilation uses NVRTC.
- GPU execution used `CUDA_VISIBLE_DEVICES=0` and fresh Inductor/Triton/CUDA cache
  directories. Portable execution used empty device visibility. Skips are not GPU evidence.

Use the locked build procedure in [CONTRIBUTING.md](../../../../CONTRIBUTING.md)
at the measured clean code revision, then run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_pointwise_shape_branches
```

The [raw archive](raw-captures.tar.gz) also includes the stdlib test runner,
environment recipe, build/install/verification logs, exact portable commands and
package inventory. The runner executes the committed test cases unchanged and
uses the committed CUDA diagnostic's runtime-provenance accessor; it introduces
no workload, tolerance, scoring or compiler-setting changes. For a new capture,
use a new report directory and compiler caches rather than overwriting retained
files. All build/test outputs stay inside the current worktree.

## Retention and limits

All 30 capture raw files are preserved byte-for-byte in the archive, with hashes
and original canonical-root locations in the [retention manifest](raw-retention-manifest.json).
The fresh runs had no failures. [Development retention](development-retention.json.gz)
records hashes and locations of 392 unchanged prior files, including original
setup, instruction-budget, recursion, import, provenance and fixture failures.
Those files remain under `target/default-compile-eval/shape-branches/`; this
manifest does not relabel them as clean-commit measurements. Initial/final author
wheels and the new clean wheel remain there too.

No implementation, tests, dependencies, benchmark harnesses, evaluators, corpora,
weights, denominators or managed progress artifacts changed in this evidence step.
[Artifact verification](verification.log) checked archive bytes, provenance, outcomes,
retained development hashes, links and the evidence-only diff. Independent review
and Burner's normal evaluation/merge gates remain required.
