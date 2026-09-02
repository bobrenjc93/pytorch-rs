# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: current composite worktree after the review update that
packs owned non-contiguous rank-2 matmul operands and reuses the row-blocked
contiguous kernel for skinny non-empty products.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/review-release-timings-pass*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`), so setup used the
worktree-local `.venv`. Cargo registry data stayed inside `target/cargo-home`,
and Cargo ran offline.

```bash
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets matmul
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo clippy --locked --offline --all-targets -- -D warnings
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline \
  --out target/review-wheels
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps \
  target/review-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_matmul tests.test_matmul_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/review_release_timings.py \
  > target/review-release-timings-pass1.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  REVIEW_TIMING_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/review_release_timings.py \
  > target/review-release-timings-pass2.json
```

Focused results: native formatting, clippy, and matmul tests passed. The Python
matmul implementation and PyTorch 2.13 differential tests passed 19 tests.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0; CUDA disabled for this CPU timing
  run with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from the wheel-installed local release build
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Build time: the final offline release extension build completed in 28.24s

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes:
the first pass measured `torch_rs` before PyTorch, and the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks. A
block repeated the operation according to the table's `Repeats` column. Times
below are median microseconds per operation; reported medians are medians of
the two per-process medians.

Before timing each supported cell, the driver checked output shape, stride,
storage offset, contiguity, dtype, device, `requires_grad`, leaf status, NaN
classifications, sign bits, and values against PyTorch with `rtol=2e-5`,
`atol=1e-5`, and `equal_nan=True`. After every warmup and measured block, the
driver consumed the last output as a BLAKE2b checksum over tensor metadata and
logical bytes.

## Supported Timed Cells

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

Review rerun summary:

- All supported cells: 1.83x uncapped, 1.83x capped
- Square contiguous cells: 3.34x
- Rectangular contiguous cells: 3.18x
- Skinny contiguous cells: 2.38x
- Empty-dimension cells: 0.38x
- Offset contiguous cells: 2.78x
- Non-contiguous transpose cells: 4.55x
- `no_grad` cells: 2.74x

Compared with the prior timing snapshot, the supported-cell capped geomean
improved from 2.07x to 1.83x, and the non-contiguous transpose cells improved
from 34.82x to 4.55x. Unsupported matmul variants remain separated below and
are not included in these supported-cell geomeans.

| Workload | Category | API | Output | Repeats | `torch_rs` median +/- MAD | PyTorch median +/- MAD | `torch_rs` / PyTorch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `square_128x128_by_128x128` | square contiguous | `left @ right` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 3 | 150.204 +/- 2.532 us | 45.140 +/- 0.262 us | 3.33x |
| `square_128x128_by_128x128` | square contiguous | `left.matmul(right)` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 3 | 150.544 +/- 2.337 us | 45.435 +/- 0.429 us | 3.31x |
| `square_128x128_by_128x128` | square contiguous | `torch.matmul(left, right)` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 3 | 152.107 +/- 3.938 us | 45.167 +/- 0.224 us | 3.37x |
| `rectangular_192x64_by_64x96` | rectangular contiguous | `left @ right` | `(192, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 5 | 85.851 +/- 2.391 us | 27.050 +/- 0.270 us | 3.17x |
| `rectangular_192x64_by_64x96` | rectangular contiguous | `left.matmul(right)` | `(192, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 5 | 85.713 +/- 2.294 us | 26.873 +/- 0.124 us | 3.19x |
| `rectangular_192x64_by_64x96` | rectangular contiguous | `torch.matmul(left, right)` | `(192, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 5 | 84.873 +/- 1.621 us | 26.724 +/- 0.093 us | 3.18x |
| `skinny_1024x8_by_8x64` | skinny contiguous | `left @ right` | `(1024, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 8 | 41.634 +/- 0.448 us | 17.617 +/- 0.158 us | 2.36x |
| `skinny_1024x8_by_8x64` | skinny contiguous | `left.matmul(right)` | `(1024, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 8 | 42.583 +/- 1.279 us | 17.512 +/- 0.102 us | 2.43x |
| `skinny_1024x8_by_8x64` | skinny contiguous | `torch.matmul(left, right)` | `(1024, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 8 | 42.624 +/- 0.948 us | 18.222 +/- 0.603 us | 2.34x |
| `empty_rows_0x256_by_256x128` | empty dimension | `left @ right` | `(0, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 2000 | 0.242 +/- 0.004 us | 0.971 +/- 0.006 us | 0.25x |
| `empty_rows_0x256_by_256x128` | empty dimension | `left.matmul(right)` | `(0, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 2000 | 0.303 +/- 0.005 us | 0.958 +/- 0.006 us | 0.32x |
| `empty_rows_0x256_by_256x128` | empty dimension | `torch.matmul(left, right)` | `(0, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 2000 | 0.391 +/- 0.003 us | 0.873 +/- 0.006 us | 0.45x |
| `empty_inner_128x0_by_0x64` | empty dimension | `left @ right` | `(128, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 200 | 0.596 +/- 0.002 us | 1.576 +/- 0.027 us | 0.38x |
| `empty_inner_128x0_by_0x64` | empty dimension | `left.matmul(right)` | `(128, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 200 | 0.674 +/- 0.005 us | 1.552 +/- 0.019 us | 0.43x |
| `empty_inner_128x0_by_0x64` | empty dimension | `torch.matmul(left, right)` | `(128, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 200 | 0.747 +/- 0.003 us | 1.437 +/- 0.012 us | 0.52x |
| `offset_contiguous_96x80_by_80x72` | offset | `left @ right` | `(96, 72)`, stride `(72, 1)`, offset 0, requires_grad=False | 6 | 39.747 +/- 0.056 us | 14.379 +/- 0.051 us | 2.76x |
| `offset_contiguous_96x80_by_80x72` | offset | `left.matmul(right)` | `(96, 72)`, stride `(72, 1)`, offset 0, requires_grad=False | 6 | 39.770 +/- 0.101 us | 14.424 +/- 0.065 us | 2.76x |
| `offset_contiguous_96x80_by_80x72` | offset | `torch.matmul(left, right)` | `(96, 72)`, stride `(72, 1)`, offset 0, requires_grad=False | 6 | 40.218 +/- 0.086 us | 14.224 +/- 0.037 us | 2.83x |
| `noncontig_transpose_96x128_by_128x64` | noncontiguous | `left @ right` | `(96, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 5 | 89.202 +/- 1.815 us | 19.009 +/- 0.109 us | 4.69x |
| `noncontig_transpose_96x128_by_128x64` | noncontiguous | `left.matmul(right)` | `(96, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 5 | 88.791 +/- 1.543 us | 21.001 +/- 1.423 us | 4.23x |
| `noncontig_transpose_96x128_by_128x64` | noncontiguous | `torch.matmul(left, right)` | `(96, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 5 | 89.742 +/- 2.451 us | 18.906 +/- 0.096 us | 4.75x |
| `no_grad_requires_grad_96x64_by_64x80` | no_grad | `left @ right` | `(96, 80)`, stride `(80, 1)`, offset 0, requires_grad=False | 6 | 35.357 +/- 0.212 us | 12.525 +/- 0.055 us | 2.82x |
| `no_grad_requires_grad_96x64_by_64x80` | no_grad | `left.matmul(right)` | `(96, 80)`, stride `(80, 1)`, offset 0, requires_grad=False | 6 | 35.298 +/- 0.141 us | 14.578 +/- 0.128 us | 2.42x |
| `no_grad_requires_grad_96x64_by_64x80` | no_grad | `torch.matmul(left, right)` | `(96, 80)`, stride `(80, 1)`, offset 0, requires_grad=False | 6 | 37.305 +/- 1.961 us | 12.402 +/- 0.066 us | 3.01x |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `operator_rank1_dot` | `RuntimeError: matmul currently requires two rank-2 tensors, got [3] and [3]` | supported shape `()`, stride `()`, dtype `torch.float32` | zero |
| `tensor_matmul_matrix_vector` | `RuntimeError: matmul currently requires two rank-2 tensors, got [32, 64] and [64]` | supported shape `(32,)`, stride `(1,)`, dtype `torch.float32` | zero |
| `torch_matmul_batched_rank3` | `RuntimeError: matmul currently requires two rank-2 tensors, got [3, 4, 5] and [3, 5, 2]` | supported shape `(3, 4, 2)`, stride `(8, 2, 1)`, dtype `torch.float32` | zero |
| `torch_matmul_out_rank2` | `TypeError: matmul() got an unexpected keyword argument 'out'` | supported shape `(2, 2)`, stride `(2, 1)`, dtype `torch.float32` | zero |
| `operator_full_sum_backward` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported left_grad `(2, 3)`; right_grad `(3, 2)` | zero |
| `tensor_matmul_full_sum_backward` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported left_grad `(2, 3)`; right_grad `(3, 2)` | zero |
| `torch_matmul_full_sum_backward` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported left_grad `(2, 3)`; right_grad `(3, 2)` | zero |
