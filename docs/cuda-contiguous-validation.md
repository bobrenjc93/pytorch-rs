# Native CUDA contiguous materialization: H100 correctness

This precommit capture is based on verified remote `main`
`b8622bc673e6fc5e728871cf89eaeabab0eebb2f`, plus the exact
[production patch](diagnostics/cuda-contiguous/precommit-b8622bc/production.patch).
No commit was created: the build receipt explicitly records a dirty worktree and
`precommit-diagnostic`, not a clean-head evaluation. The
[source manifest](diagnostics/cuda-contiguous/precommit-b8622bc/source-manifest.json)
and [release receipt](diagnostics/cuda-contiguous/precommit-b8622bc/build-record.json)
bind the actual production files and native extension used by these tests.

| Artifact | SHA-256 |
| --- | --- |
| Production source | `7c4d4950f5c99d24513f184ee596e5aed2e72b02181e0288be09f792b72f6f05` |
| Production tracked diff | `9fe75c82a06fc0d5d5681f255ea86fc43176bef681a6e76f057f3a2f7ae328b5` |
| Native extension (installed wheel and source package) | `cc4bf845fece98275489f9d9665c4dc3ae68a74d7687a8d0f7ad9a8851c6eb9e` |

The source digest includes the new PTX file; the tracked-diff digest alone does
not. Source and both extension hashes were unchanged through final validation.
[Inventory](diagnostics/cuda-contiguous/precommit-b8622bc/inventory.json) also binds
the final tests, raw logs, receipts and fixture snapshots.

## Build and hardware

Fresh canonical worktree-local `.venv`, Python 3.12.13, locked PyTorch
`2.13.0+cu130` and NumPy 2.5.1. Rust/Cargo 1.92.0; fresh Cargo release target,
thin LTO, one codegen unit, `extension-module`. The unchanged build capture used
`maturin build --release --locked --offline`, then installed that wheel into
`.venv` and copied the identical extension into the source package.
All installations, caches, temporary files and outputs stayed in this worktree.

Single-GPU tests used `CUDA_VISIBLE_DEVICES=0`: NVIDIA H100, compute capability
9.0, UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, 97,871 MiB reported by
`nvidia-smi`, driver 580.82.07. Native runtime was explicitly selected with
`TORCH_RS_CUDART=.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`
(resolved absolute path in receipts), runtime version 13000. PyTorch reports CUDA
13.0. Installed nvcc is 12.6.85 and was **not used**: the driver JIT compiles the
embedded PTX. Two-device restoration tests alone used `CUDA_VISIBLE_DEVICES=0,1`.
See [environment](diagnostics/cuda-contiguous/precommit-b8622bc/environment.log)
and GPU snapshots for exact properties, UUIDs, memory use and utilization.

## Checks

- 197 Rust tests passed: all 178 library tests and the selected CUDA integration
  suites, including the new packing primitive and public API tests. Formatting
  and Clippy (`--all-targets --features python-bindings -- -D warnings`) passed.
  The documentation check passed with zero doctests.
- 85 existing CPU layout/view/reshape/clone tests passed, including autograd and
  memory-format differentials.
- Six new single-GPU Python tests passed; the two-device test was explicitly
  skipped under the single-device mask and passed separately. The existing
  same-device-copy restoration test also passed with two devices. With CUDA
  hidden, all seven hardware tests skip clearly (no correctness credit).
- The existing CUDA regression run passed 137 tests, with 19 mask-specific
  skips (156 total). It covers transfers, storage, views, vector
  copies, arithmetic, reductions, matmul and eager graph capture. Its complete
  results and mask-specific skips are retained in
  [the log](diagnostics/cuda-contiguous/precommit-b8622bc/python-cuda-regressions.log).

The new tests compare non-square transposes, offset slices and strided column
views over fixed and seeded shapes, including a grid-stride tail beyond 1M
elements. They check logical values, float32/device/index, strides/offsets,
source nonmutation, independent storage, alias/copy mutation behavior, reshape
requiring and avoiding a copy, empty/scalar/singleton identity, lifetime/thread
behavior and blocked PyTorch imports. Raw CUDA byte readback checks signed zero,
subnormals, infinities and quiet/signaling NaN payloads. Rust tests cover checked
extent/offset overflow and invalid rank/stride plans; Python tests cover formats,
unsupported ranks/dtypes/gradients and unchanged clone/`to(copy=True)` boundaries.

Two unsuccessful fixture attempts are preserved, with their exact test source
snapshots. The first had 38 setup errors from unsupported multi-slice tuple
indexing and absent native dtype constants. Equivalent chained slices and
reference dtype objects corrected setup. The second had three incorrect
zero-offset assertions for already-contiguous singleton views; both backends
correctly retained offset 8. The final fixture checks that alias contract.
**No production code or native binary changed between these attempts.**

## Reproduction

From the worktree root, use the local-cache exports in
[environment.sh.txt](diagnostics/cuda-contiguous/precommit-b8622bc/environment.sh.txt),
then run:

```sh
uv sync --locked --no-install-project --group dev --group reference --python /usr/bin/python3.12
cargo fetch --locked
.venv/bin/python scripts/capture_depth_concat_build.py --allow-dirty --output target/cuda-contiguous-release
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_cuda_contiguous
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_cuda_contiguous.ContiguousDeviceGuardTests tests.test_cuda_same_device_copy.CopyDeviceGuardTests
```

Use a fresh output directory for each build capture. Exact Rust, CPU and existing
CUDA regression commands are in `initial-commands.json` and the per-run
`*.receipt.json` files. Raw artifacts were copied byte-for-byte from `target`;
receipt paths retain their original capture locations.

This is bounded native layout correctness evidence, with no performance,
compiler-parity or evaluator-score claim. Campaign evaluators, cases/weights,
benchmark evidence, `.burner` definitions and managed progress are unchanged.
PR1970/1971 remain separate unadopted evaluator campaigns.
