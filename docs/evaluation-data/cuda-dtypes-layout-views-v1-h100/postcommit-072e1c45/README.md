# Clean-commit CUDA dtype/layout/view capture

Measured commit: `072e1c45e78b0e22a881d34807e842ce8f74b0f1`. This required post-commit capture uses the
unchanged committed evaluator and all six fixed cases on H100 GPU 0 at
evaluator-selected seeds `3293684309800391261` and `4185734436305625061`.
All build, capture and focused test commands began and ended with clean git
status. Outputs were staged under the worktree's ignored `target` directory
and copied here only after measurement and validation finished.

All **12 reference trials passed** on PyTorch `2.13.0+cu130`. All 12 candidate
trials executed in separate processes against a freshly built extension.
Candidate accounting remains **2/6**: transpose aliasing and offset slicing
pass at both seeds. Float64 and int64 fail with missing public dtype attributes;
transposed contiguous materialization and reshape requiring a copy raise
`NotImplementedError`. These four slots retain zero credit. This is correctness
evidence for the fixed denominator, with no implementation or headline-score gain.

The release build used Rust/Cargo 1.92.0, Maturin 1.14.1, CPython 3.14.5,
locked dependencies, thin LTO, one codegen unit, and the extension-module
feature. Its Cargo target started empty; the worktree-local Cargo download
cache and locked Python environment were already populated. The wheel's native
extension was extracted into this checkout's `python/torch_rs` as required by
the committed runner. The dependency check passed without installing packages.

Hardware: NVIDIA H100, UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`,
compute capability 9.0, driver 580.82.07, `CUDA_VISIBLE_DEVICES=0`.
Both reference and CUDA-initialized candidate workers loaded the local
`nvidia/cu13/lib/libcudart.so.13` and reported runtime version 13000. The
missing-dtype trials fail before loading CUDA; their empty runtime lists are
preserved. Installed nvcc reports 12.6.85 and was not invoked by this build or
these operations.

- [Raw results](results.json): every trial, observation, failure and fixed-slot verdict.
- [Build receipt](build-record.json) and [build log](build.log): actual fresh build and wheel/extension/source hashes.
- [Run receipt](run-receipt.json): actual command, timestamps, clean statuses and GPU utilization/memory before and after.
- [Environment](environment.json): interpreter identity, installed package versions and local cache/build configuration.
- [Command receipts](commands.json): dependency check, build, capture and focused tests, with elapsed time and exit status.
- [Focused tests](accounting-tests.log): all 11 accounting and isolation tests passed; hardware behavior is checked by the two-seed capture.
- [Validation](validation.json): raw accounting recomputation, 24 distinct worker processes, current source/evaluator/extension bindings, local package/runtime paths, and published artifact hashes.

The [original development capture](../README.md) remains unchanged and pinned
to its original source/build identities. It is not used as clean-commit evidence.
No implementation, dependency, test, matrix, evaluator or supported behavior
changed during this step. No unrelated full test suite was rerun.

To replay using an extension freshly built from the same source and its actual
build receipt:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B scripts/evaluate_cuda_dtypes_layout_views.py \
  --seed 3293684309800391261 --seed 4185734436305625061 \
  --build-record docs/evaluation-data/cuda-dtypes-layout-views-v1-h100/postcommit-072e1c45/build-record.json \
  --output target/dtypes-layout-views-postcommit-replay.json
```
