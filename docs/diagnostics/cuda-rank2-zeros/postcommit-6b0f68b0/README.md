# Rank-2 CUDA zeros: historical clean-commit source-PR evidence

Measured implementation: `6b0f68b0d2408bd8bda5073ec332fd88f02a67fb` (base: `a60b059321a90d9ea9a0d64d777dbb8544383e11`).
This historical source-PR capture does not measure the combined candidate or
earn current-candidate credit. Original paths identify the source-PR worktree.
The build, evaluator, and focused tests all ran with an empty Git status from
this committed implementation. This directory contains evidence and its
explanation only; it does not approve the candidate or replace independent review.

The unchanged six-case CUDA transfer evaluator reports **5/6 cases passing**.
`cuda_f32_matrix_zero_roundtrip` passes at every evaluator-selected seed:
`8417970381438833507`, `3794413297053346249`, `2257496142127917571`.
At that source revision, the CUDA-to-CUDA copy case was unsupported and stays in the
denominator. This is correctness evidence for this capability only, with no
performance or overall hardware-score claim.

| Fixed case | Outcome |
| --- | --- |
| `cuda_f32_vector_zero_roundtrip` | Pass (3/3 seeds) |
| `cuda_f32_matrix_zero_roundtrip` | Pass (3/3 seeds) |
| `cuda_f32_contiguous_cpu_upload` | Pass (3/3 seeds) |
| `cuda_f32_strided_cpu_upload` | Pass (3/3 seeds) |
| `cuda_f32_view_to_cpu_copy` | Pass (3/3 seeds) |
| `cuda_f32_same_device_copy` | Unsupported (0/3 seeds) |

## Capture and provenance

- [Raw evaluator output](transfers.json) preserves all 18 trials, both workers'
  observations, unsupported outcomes, source and extension hashes, evaluator
  hashes, GPU inventory, and actual loaded CUDA runtime paths/versions.
- [Build receipt](build-record.json) and [build log](build.log) record the exact
  command, environment, timestamps, fresh Cargo target directory, cache state,
  release configuration, compiler versions, wheel hash, and extension hash.
- [Run receipt](run-record.json) records exact evaluator/test commands, times,
  exit codes, clean status before/after, and log hashes. [Evaluator stdout/stderr](transfers.log)
  is empty because the evaluator writes its report to the requested JSON file.
- [Focused test log](focused-tests.log) records four passing tests: generated
  rectangular/empty matrix differentials and the three CUDA zeros boundary
  checks. No unrelated full test suite was repeated.
- [Evidence verification](verification.json) records unchanged source/extension
  and evaluator hashes, accounting recomputed with the committed evaluator,
  local worker/package/runtime paths, and absence of candidate PyTorch imports.

The environment was Python 3.12.12 with PyTorch 2.13.0+cu130 on NVIDIA H100
(compute capability 9.0), driver 580.82.07, with `CUDA_VISIBLE_DEVICES=0`.
Both worker roles actually loaded CUDA runtime 13.0 (`cudaRuntimeGetVersion`
13000) from the worktree-local environment. Rust/Cargo 1.92.0 built the release
extension. Installed nvcc was 12.6.85; this allocation/transfer path invoked no
CUDA compiler. Python dependencies were materialized into a local virtual
environment without changing their versions; all recorded package/runtime
imports and virtual-environment executable paths are inside that original worktree.

## Reproduction

The exact absolute commands/environment are in the receipts. From this code
revision, use the documented Maturin release build with a new `CARGO_TARGET_DIR`,
`--locked --offline`, and the local Python environment. Extract the native
`torch_rs/torch_rs.abi3.so` member of the resulting wheel into `python/torch_rs/`
as required by the transfer evaluator's local-extension check. Produce the
build receipt using `source_provenance` and `sha256` from
`scripts/evaluate_cuda_math.py`, following the receipt procedure in
[the evaluator guide](../../../hardware-heterogeneity-evaluator.md).
Run `scripts/evaluate_cuda_transfers.py` with `CUDA_VISIBLE_DEVICES=0`, the
local reference/candidate Python, `--build-record`, and `--output`; omit
`--seed` to let the evaluator select three fresh seeds. The workload matrix,
PyTorch reference, and six-case denominator must remain unchanged.

The measurements precede publication of this evidence directory. An ensuing
evidence-only commit may retain this measured implementation identity.
Earlier uncommitted development runs are not used as final-candidate evidence;
historical reports elsewhere in the repository are unchanged.
