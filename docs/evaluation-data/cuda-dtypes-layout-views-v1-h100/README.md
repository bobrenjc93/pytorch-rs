# CUDA dtype/layout/view denominator validation

This capture validates the evaluator against unchanged production sources at
`37ea69ccc0db9f70fd914dc7b2277139da48f2d0`. The evaluator and matrix were
uncommitted additions during capture; their SHA-256 hashes are in `results.json`.
The source hash and empty production diff bind the fresh extension build. This
is evaluator evidence, not an implementation improvement or a headline score.

`build-record.json` and `build.log` record a fresh locked release wheel built in
an empty worktree-local Cargo target: Rust/Cargo 1.92.0, Maturin 1.14.1,
CPython 3.14.5, thin LTO, one codegen unit, and the extension-module feature.
The wheel's native extension was extracted into this checkout's `python/torch_rs`
for the isolated candidate workers. The pre-existing extension was not reused.
The default nvcc reports CUDA 12.6.85; this build and these operations did not
invoke nvcc. `setup.log` records the locked local Python dependency installation.

`run-receipt.json` records the exact command, elapsed time, selected environment,
and GPU inventory before/after. `results.json` retains every reference and
candidate trial, including errors and partial observations, independent semantic
validation, blocked imports, extension/source/evaluator hashes, reference build
configuration, and the actual mapped CUDA runtimes. GPU 0 is NVIDIA H100,
UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, compute capability 9.0,
driver 580.82.07. Only `CUDA_VISIBLE_DEVICES=0` was used.

The focused GPU regression command was:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B -m unittest \
  tests.test_cuda_dtypes_layout_views_evaluator \
  tests.test_cuda_transfers_evaluator tests.test_cuda_math_evaluator \
  tests.test_cuda_compilation_evaluator tests.test_hardware_heterogeneity_evaluation
```

All 64 tests passed (`regression-tests.log`). After adding a further regression
for malformed build receipts, the new suite's 12 tests passed with the single
hardware test explicitly skipped under an empty device mask
(`portable-tests.log`). That additional test is hardware independent. Synthetic
accounting fixtures do not constitute CUDA evidence. Syntax and `git diff
--check` also passed.

At evaluator-selected seeds `7444576324044628850` and `6784238651838885789`,
all **12 reference trials passed** on PyTorch `2.13.0+cu130`. All 12 candidate
trials ran in separate processes against the fresh extension. Candidate
accounting is **2/6** slots, with each credited case passing both seeds:

| Case | Candidate result at both seeds | Credit |
| --- | --- | --- |
| float64 roundtrip | Missing public `float64` attribute (`AttributeError`) | 0 |
| int64 roundtrip | Missing public `int64` attribute (`AttributeError`) | 0 |
| float32 transpose alias | Passed | 1 |
| float32 offset slice alias | Passed | 1 |
| transposed contiguous materialization | `NotImplementedError` | 0 |
| transposed reshape requiring a copy | `NotImplementedError` | 0 |

Both reference and CUDA-initialized candidate workers mapped the local locked
`nvidia/cu13/lib/libcudart.so.13`, reporting runtime version 13000. The two
missing-dtype cases fail before loading a CUDA runtime; their empty runtime
lists and failures remain in the report. Source, evaluator, matrix, helper and
extension hashes remained stable throughout execution. No unsupported behavior
was implemented, and the denominator remains six. The two passing slots
represent 5 points of CUDA's existing 15-point dtype/layout/view capability;
this capture does not assign headline or breadth credit.

Replay with the same freshly built, source-matched extension and receipt:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B scripts/evaluate_cuda_dtypes_layout_views.py \
  --seed 7444576324044628850 --seed 6784238651838885789 \
  --build-record docs/evaluation-data/cuda-dtypes-layout-views-v1-h100/build-record.json \
  --output target/dtypes-layout-views-replay.json
```
