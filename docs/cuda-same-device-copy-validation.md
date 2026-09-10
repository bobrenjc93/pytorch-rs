# Native CUDA vector copy validation

The post-commit capture measured clean implementation commit
`78310d9277dcb54bb328c24cbf445f7d4e500cd5` on 2026-09-10 UTC.
The worktree was clean before the fresh build and after all measurements;
only this report and the preserved evidence were added afterward.
This replaces the earlier development capture as current-candidate evidence.

`cuda_f32_same_device_copy` received full slot credit at all three random seeds
selected by the unchanged transfer evaluator, exceeding the required two.

| Seed | Reference | Native candidate | Credit |
| --- | --- | --- | --- |
| `5106175624324498452` | passed | passed | 1 |
| `5380439577007405373` | passed | passed | 1 |
| `3733381193531632723` | passed | passed | 1 |

The fixed denominator remains six: 5/6 transfer slots passed.
Rank-2 CUDA zeros remains unsupported with zero credit. No cases, seeds, failures
or unsupported outcomes were removed. This capture measures correctness, not
performance or an overall hardware score.

## Preserved evidence

- [Complete unchanged-evaluator output](diagnostics/cuda-same-device-copy/postcommit-78310d92/transfers.json), including all
  six cases, reference/candidate observations, pointer checks and runtime provenance.
- [Build receipt](diagnostics/cuda-same-device-copy/postcommit-78310d92/build-record.json), binding the clean source and native
  extension hashes to the actual build configuration and worktree paths.
- [Capture receipt](diagnostics/cuda-same-device-copy/postcommit-78310d92/capture-record.json), recording command arguments,
  timestamps, exit codes, log hashes, source stability and empty before/after status.
- [Environment setup](diagnostics/cuda-same-device-copy/postcommit-78310d92/setup.log), [build log](diagnostics/cuda-same-device-copy/postcommit-78310d92/build.log), and
  [reference preflight](diagnostics/cuda-same-device-copy/postcommit-78310d92/preflight.log).

Raw artifacts were generated under `target/cuda-copy-postcommit-78310d92/` and
copied here byte-for-byte. The paths in the receipts describe their actual
capture locations in the current worktree. They were not rewritten.

## Build and hardware

- NVIDIA H100, compute capability 9.0, 97,871 MiB; driver 580.82.07.
  Reference GPU UUID: `8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`.
- PyTorch `2.13.0+cu130`, CUDA 13.0, Python 3.12.12.
  A new worktree-local `.venv` was installed with the existing locked dependencies:
  `uv sync --locked --no-install-project --group dev --group reference` using Python 3.12.
  Both evaluator workers use `.venv/bin/python`; reference package, native extension,
  selected CUDA runtime and build paths are rooted in this worktree.
- Native and reference runtime: `libcudart.so.13`, reported runtime version
  `13000`. `TORCH_RS_CUDART` selects
  `.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`.
- `rustc 1.92.0 (ded5c06cf 2025-12-08)`; `cargo 1.92.0 (344c4567c 2025-10-21)`.
- Available `nvcc`: CUDA 12.6, V12.6.85. It is unused by the copy path; no CUDA
  kernel is compiled. The selected runtime is recorded separately above.
- Fresh Cargo target `target/cuda-copy-postcommit-78310d92/build`, with the existing
  worktree-local Cargo dependency cache. Release profile: thin LTO, one codegen
  unit, `extension-module`, locked dependencies.
- Build command: `cargo build --release --features extension-module --locked --offline`. The resulting `libpytorch_rs.so`
  was copied to `python/torch_rs/torch_rs.abi3.so` before any copy tests or evaluation.
  All generated files, caches and environment installations remained in this worktree.

| Provenance | SHA-256 |
| --- | --- |
| `source_sha256` | `9f82cbdbed7a75d44a413d03f295798c01d76f13e070c6f5f90c254e1eae5a6d` |
| `production_diff_sha256` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `extension_sha256` | `f8cf93e95e09129d912c00076caa320e4c676ed7c34027498757d0e9fa619068` |
| `build-record.json` | `65495b9553488407825e1e3bede7765a1cce38a9a71319080d7ad2ac356aa74c` |
| `transfers.json` | `af6857c0dafe91af79f1f3bc94f20222f5184a5f181aac64c70cfe0fd67a2b1b` |

## Focused checks on the committed build

- [Rust vector copies](diagnostics/cuda-same-device-copy/postcommit-78310d92/rust-copy.log): 2 passed, including generated
  lengths, empty and offset vectors, layout, independent storage, source preservation,
  lifetimes, allocation reuse and unsupported boundaries.
- [Rust copy primitive](diagnostics/cuda-same-device-copy/postcommit-78310d92/rust-copy-bounds.log): 1 passed, checking bounds,
  empty offsets and exact signed-zero, subnormal, infinity and NaN bits.
- [H100 Python differentials](diagnostics/cuda-same-device-copy/postcommit-78310d92/python-copy.log): 5 passed with
  `CUDA_VISIBLE_DEVICES=0`; the two-device guard case was skipped for this mask.
- [Device guard check](diagnostics/cuda-same-device-copy/postcommit-78310d92/python-copy-guard.log): 1 passed with
  `CUDA_VISIBLE_DEVICES=0,1`; single-device cases were skipped for this mask.
  Both target ordinals were checked while the other device was current.
- The evaluator's existing `account()` recomputed the saved accounting exactly;
  source, evaluator and extension hashes were stable. Each candidate worker
  reported no blocked PyTorch imports and no loaded PyTorch modules.

Only the focused copy checks and required evaluator were rerun. Earlier broad
regression suites were not repeated or attributed to this clean-commit capture.
Implementation, dependencies, tests, supported behavior, evaluator scripts, the
fixed matrix, unrelated benchmark evidence and Burner-managed artifacts are unchanged.

The measured evaluator command is recorded in the capture receipt. Its equivalent
from the worktree root is:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_cuda_transfers.py \
  --build-record target/cuda-copy-postcommit-78310d92/build-record.json \
  --output target/cuda-copy-postcommit-78310d92/transfers.json
```

No seed arguments were supplied. For an exact replay, append:
`--seed 5106175624324498452 --seed 5380439577007405373 --seed 3733381193531632723`.
Future code builds require their own freshly generated source-matched build receipt.
