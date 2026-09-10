# Native CUDA vector copy validation

Validated on 2026-09-10 UTC. This change supports contiguous rank-1 float32
`clone()` and same-device `to(copy=True)` without gradients, with preserve-format
strides. Both APIs share a checked device-to-device `cudaMemcpy` primitive,
owned allocation, device guard and explicit completion. No CUDA kernel is added.

## Build and hardware

- NVIDIA H100, compute capability 9.0, 97,871 MiB; driver 580.82.07.
- Ordinary tests and evaluator: `CUDA_VISIBLE_DEVICES=0`. The device-guard test
  alone uses `CUDA_VISIBLE_DEVICES=0,1` and checks both target ordinals while the
  other device is current, including restoration after copies, errors and drops.
- Reference: PyTorch `2.13.0+cu130`, CUDA 13.0, Python 3.12.
- Selected native runtime: `libcudart.so.13` from
  `/data/users/bobren/a/pytorch-rs-burner/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/`,
  pinned with `TORCH_RS_CUDART`.
- Rust and Cargo 1.92.0; release profile, thin LTO, one codegen unit,
  `extension-module` feature, locked dependencies. A new worktree-local Cargo
  target was used; cached dependency sources were copied into this worktree.
- Available `nvcc`: CUDA 12.6, V12.6.85. It is unused by these copies; the runtime
  ABI is loaded dynamically. The compiler's version is not the selected runtime.
- Build command: `cargo build --release --features extension-module --locked --offline`.
  `target/build/release/libpytorch_rs.so` was copied to
  `python/torch_rs/torch_rs.abi3.so`; the evaluator loads this local extension.
  Build outputs, Python/CUDA caches and temporary files stay within the worktree.
  The reference environment was accessed read-only.

The receipt binds the uncommitted production source, not a clean HEAD claim:

| Field | Value |
| --- | --- |
| `commit` | `a60b059321a90d9ea9a0d64d777dbb8544383e11` |
| `source_sha256` | `9f82cbdbed7a75d44a413d03f295798c01d76f13e070c6f5f90c254e1eae5a6d` |
| `production_diff_sha256` | `a7d75bbbfbeca3826c65d374fd657f3802dacbf002ee03454ca21beac5efaebc` |
| `extension_sha256` | `f8cf93e95e09129d912c00076caa320e4c676ed7c34027498757d0e9fa619068` |

## Checks

- `cargo test --locked --offline --lib`: 165 passed on CUDA hardware, including
  direct copy bounds, empty offsets and bitwise signed-zero/subnormal/infinity/
  NaN-payload checks.
- `cargo test --locked --offline --test cuda_same_device_copy --test cuda_native_boundaries --test cuda_add --test cuda_mul_scalar`:
  10 passed. New copy tests cover generated lengths, empty and offset vectors,
  independent storage, source preservation, lifetime and allocation reuse.
- `python -m unittest discover -s tests -p 'test_cuda_same_device_copy.py' -v`:
  5 passed with mask `0`; the separate mask `0,1` run passed the device-guard test.
  Each run clearly skips cases requiring the other device mask.
- Python regression suites `test_cuda_host_transfer.py`, `test_cuda_native_views.py`,
  `test_cuda_zero_roundtrip.py`, `test_tensor_to.py`, `test_clone_channels_last.py`:
  45 passed, 2 skipped for the two-device mask. These include existing CPU copy
  autograd behavior and native transfer/view lifetime tests.
- `cargo clippy --locked --offline --all-targets --features python-bindings -- -D warnings`,
  `cargo fmt --all -- --check`, and `git diff --check`: passed.

The Python differentials compare exact float32 values and output strides with
PyTorch, cover relaxed-contiguous singleton/empty strides, mutate source storage
to prove independence, and prevent importing installed PyTorch in a subprocess.
Noncontiguous copies, other ranks, cross-device copies, dtype conversion,
non-preserve memory formats, asynchronous copies and CUDA autograd stay rejected.

## Unchanged transfer evaluator

Command (the evaluator chooses its default random seeds):

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cuda_transfers.py \
  --build-record target/copy-build-record.json \
  --output target/copy-transfers.json
```

The build receipt is generated from the existing evaluator's
`source_provenance()` plus the build command, toolchain versions and the local
extension SHA-256 above. Raw receipts and test logs reside under `target/`.
The evaluator scripts, fixed matrix, benchmark evidence and Burner-managed
progress artifacts are unchanged.

`cuda_f32_same_device_copy` received full slot credit at all three default
seeds selected by the unchanged evaluator (exceeding the required two):

| Seed | Reference | Native candidate | Credit |
| --- | --- | --- | --- |
| `7912991709410024047` | passed | passed | 1 |
| `4172382651939481785` | passed | passed | 1 |
| `8961035516691223039` | passed | passed | 1 |

The source, evaluator and extension hashes were stable throughout the run.
Both processes reported CUDA runtime version `13000` (13.0); the reference GPU
UUID was `8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`. Every candidate worker reported
no blocked import attempts and no loaded PyTorch modules.
The fixed transfer denominator remains six: five slots passed, and the existing
rank-2 CUDA zeros slot remains unsupported with zero credit. This is a transfer
correctness result, not a performance or overall hardware score.

Reproduce these seeds by appending
`--seed 7912991709410024047 --seed 4172382651939481785 --seed 8961035516691223039`
to the evaluator command. A fresh source-matched build receipt is required.

| Evidence / unchanged input | SHA-256 |
| --- | --- |
| `target/copy-build-record.json` | `b4bcdb9c0f904b5f65833c898b9b373da1d62b9735ab44f343257f3181192ead` |
| `target/copy-transfers.json` | `c2d96a214104071cedd18f7784ed10f1292f78bc17b41939ea98498d7411e877` |
| `evaluate_cuda_transfers.py` | `a73e5f44e7ff47c1b6de2df77d14a9c5450eef3f28f0435215ea1f3112bec3c3` |
| `evaluate_cuda_math.py` | `29a97d07bad0d82c261e28aa880de68cdf46df0731d5ff0aa9802ec5bb8e605b` |
| `hardware-heterogeneity-matrix-v1.json` | `a077f00f5eb50e2dceb2bd41a8453977955a8114fed203b7235f3a8befb24698` |
