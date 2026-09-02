# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`9e94ad2e3776fd637bc6bed37430ba0af1fe08d4`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/rank2-matmul-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`), so setup used a
worktree-local `.venv`. Cargo registry data was copied read-only from the
existing user cache into `target/cargo-home`, then Cargo ran offline so build
artifacts and dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
mkdir -p target/rank2-matmul-wheels
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  /usr/bin/time -p .venv/bin/maturin build --release --locked --offline \
  --out target/rank2-matmul-wheels
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps target/rank2-matmul-wheels/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_matmul tests.test_matmul_reference
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
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/rank2_matmul_release_timings.py \
  > target/rank2-matmul-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  RANK2_MATMUL_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/rank2_matmul_release_timings.py \
  > target/rank2-matmul-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_matmul tests.test_matmul_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets matmul
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 19 tests. The README/docs smoke test passed 7 tests. `cargo fmt
--check` passed. The focused native Rust `matmul` filter passed 6 library
tests and 3 `tensor_baseline` tests. `git diff --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core, 384 logical CPUs
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0, from
  `.venv/lib/python3.12/site-packages/torch`; `torch.cuda.is_available()` was
  `False` under `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Float32 matmul precision: PyTorch reported `"highest"`
- Dependency installation: locked `uv sync` resolved in 26 ms, prepared
  packages in 16.26s, and installed in 941 ms
- Build time: successful offline release extension build completed in 36.53s
  (`real 36.78`); the release wheel reinstall resolved in 2 ms, prepared in
  40 ms, and installed in 20 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported cell, the driver checked output shape, stride,
storage offset, contiguity, dtype, device, `requires_grad`, and leaf status,
then checked NaN classifications, sign bits, and values against PyTorch with
`rtol=2e-5`, `atol=1e-5`, and `equal_nan=True`. After every warmup and
measured block, the driver consumed the last output as a 64-bit BLAKE2b rolling
checksum over tensor metadata and logical bytes. Floating-point matmul
accumulation is not bit-identical for every non-empty cell, so checksum pairs
can differ between implementations after the allclose correctness gate. The
checksum column shows the final rolling sink from one pass as
`torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.42x uncapped, 2.07x capped
- `@` operator cells: 2.29x uncapped, 1.96x capped
- `Tensor.matmul` cells: 2.33x uncapped, 2.00x capped
- `torch.matmul` cells: 2.65x uncapped, 2.25x capped
- Square contiguous cells: 3.33x uncapped, 3.33x capped
- Rectangular contiguous cells: 3.15x uncapped, 3.15x capped
- Skinny contiguous cells: 3.03x uncapped, 3.03x capped
- Empty-dimension cells: 0.36x uncapped, 0.36x capped
- Offset cells: 2.93x uncapped, 2.93x capped
- Noncontiguous transpose cells: 34.82x uncapped, 10.00x capped
- `no_grad` cells: 2.73x uncapped, 2.73x capped

No backward-through-full-`sum` matmul cells are supported in this snapshot:
rank-2 matmul with `requires_grad=True` returns a leaf output without a
`grad_fn`. The three API-level backward cells are therefore listed below as
zero-credit unsupported cells. Including all unsupported cells below as
zero-credit denominator entries with a 10.00x capped penalty gives a combined
capped aggregate of 2.95x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `square_128x128_by_128x128` | square contiguous | `left @ right` | left/right (128, 128), stride (128, 1), offset 0 | (128, 128), stride (128, 1), offset 0, requires_grad=False | 3 | 150.182 us +/- 2.774, var 125.529 | 46.605 us +/- 1.903, var 14.498 | 3.22x | `1272492015363886211`/`4416486515723700608` |
| `square_128x128_by_128x128` | square contiguous | `left.matmul(right)` | left/right (128, 128), stride (128, 1), offset 0 | (128, 128), stride (128, 1), offset 0, requires_grad=False | 3 | 151.320 us +/- 4.657, var 74.518 | 44.878 us +/- 0.372, var 12.227 | 3.37x | `1272492015363886211`/`4416486515723700608` |
| `square_128x128_by_128x128` | square contiguous | `torch.matmul(left, right)` | left/right (128, 128), stride (128, 1), offset 0 | (128, 128), stride (128, 1), offset 0, requires_grad=False | 3 | 151.082 us +/- 3.978, var 299.230 | 44.589 us +/- 0.259, var 15.818 | 3.39x | `1272492015363886211`/`4416486515723700608` |
| `rectangular_192x64_by_64x96` | rectangular contiguous | `left @ right` | left (192, 64), stride (64, 1); right (64, 96), stride (96, 1) | (192, 96), stride (96, 1), offset 0, requires_grad=False | 5 | 88.974 us +/- 2.594, var 58.842 | 27.779 us +/- 0.246, var 2.753 | 3.20x | `8921478535050720123`/`10059273008971090750` |
| `rectangular_192x64_by_64x96` | rectangular contiguous | `left.matmul(right)` | left (192, 64), stride (64, 1); right (64, 96), stride (96, 1) | (192, 96), stride (96, 1), offset 0, requires_grad=False | 5 | 87.464 us +/- 1.083, var 8.189 | 27.858 us +/- 0.236, var 2.069 | 3.14x | `8921478535050720123`/`10059273008971090750` |
| `rectangular_192x64_by_64x96` | rectangular contiguous | `torch.matmul(left, right)` | left (192, 64), stride (64, 1); right (64, 96), stride (96, 1) | (192, 96), stride (96, 1), offset 0, requires_grad=False | 5 | 86.319 us +/- 2.617, var 27.221 | 27.662 us +/- 0.916, var 2.826 | 3.12x | `8921478535050720123`/`10059273008971090750` |
| `skinny_1024x8_by_8x64` | skinny contiguous | `left @ right` | left (1024, 8), stride (8, 1); right (8, 64), stride (64, 1) | (1024, 64), stride (64, 1), offset 0, requires_grad=False | 8 | 48.530 us +/- 1.368, var 18.691 | 17.733 us +/- 0.400, var 2.516 | 2.74x | `2984979620803967732`/`13073546940570444636` |
| `skinny_1024x8_by_8x64` | skinny contiguous | `left.matmul(right)` | left (1024, 8), stride (8, 1); right (8, 64), stride (64, 1) | (1024, 64), stride (64, 1), offset 0, requires_grad=False | 8 | 51.311 us +/- 2.475, var 23.069 | 17.373 us +/- 0.143, var 1.304 | 2.95x | `2984979620803967732`/`13073546940570444636` |
| `skinny_1024x8_by_8x64` | skinny contiguous | `torch.matmul(left, right)` | left (1024, 8), stride (8, 1); right (8, 64), stride (64, 1) | (1024, 64), stride (64, 1), offset 0, requires_grad=False | 8 | 60.009 us +/- 2.056, var 57.698 | 17.353 us +/- 0.239, var 1.654 | 3.46x | `2984979620803967732`/`13073546940570444636` |
| `empty_rows_0x256_by_256x128` | empty dimension | `left @ right` | left zeros((0, 256)); right (256, 128), stride (128, 1) | (0, 128), stride (128, 1), offset 0, requires_grad=False | 2000 | 0.282 us +/- 0.010, var 0.000 | 1.162 us +/- 0.023, var 0.014 | 0.24x | `6004968643818944219`/`6004968643818944219` |
| `empty_rows_0x256_by_256x128` | empty dimension | `left.matmul(right)` | left zeros((0, 256)); right (256, 128), stride (128, 1) | (0, 128), stride (128, 1), offset 0, requires_grad=False | 2000 | 0.281 us +/- 0.006, var 0.001 | 1.165 us +/- 0.054, var 0.027 | 0.24x | `6004968643818944219`/`6004968643818944219` |
| `empty_rows_0x256_by_256x128` | empty dimension | `torch.matmul(left, right)` | left zeros((0, 256)); right (256, 128), stride (128, 1) | (0, 128), stride (128, 1), offset 0, requires_grad=False | 2000 | 0.354 us +/- 0.006, var 0.000 | 1.020 us +/- 0.098, var 0.035 | 0.35x | `6004968643818944219`/`6004968643818944219` |
| `empty_inner_128x0_by_0x64` | empty dimension | `left @ right` | left ones((128, 0)); right zeros((0, 64)) | (128, 64), stride (64, 1), offset 0, requires_grad=False | 200 | 0.643 us +/- 0.016, var 0.002 | 1.563 us +/- 0.025, var 0.021 | 0.41x | `14366504465922197046`/`14366504465922197046` |
| `empty_inner_128x0_by_0x64` | empty dimension | `left.matmul(right)` | left ones((128, 0)); right zeros((0, 64)) | (128, 64), stride (64, 1), offset 0, requires_grad=False | 200 | 0.780 us +/- 0.004, var 0.004 | 1.858 us +/- 0.101, var 0.062 | 0.42x | `14366504465922197046`/`14366504465922197046` |
| `empty_inner_128x0_by_0x64` | empty dimension | `torch.matmul(left, right)` | left ones((128, 0)); right zeros((0, 64)) | (128, 64), stride (64, 1), offset 0, requires_grad=False | 200 | 0.877 us +/- 0.005, var 0.007 | 1.393 us +/- 0.008, var 0.004 | 0.63x | `14366504465922197046`/`14366504465922197046` |
| `offset_contiguous_96x80_by_80x72` | offset | `left @ right` | left tensor((3, 96, 80))[1] -> (96, 80), offset 7680; right tensor((2, 80, 72))[1] -> (80, 72), offset 5760 | (96, 72), stride (72, 1), offset 0, requires_grad=False | 6 | 41.224 us +/- 1.338, var 6.435 | 14.279 us +/- 0.050, var 0.747 | 2.89x | `15246320616318125486`/`17232460849795164269` |
| `offset_contiguous_96x80_by_80x72` | offset | `left.matmul(right)` | left tensor((3, 96, 80))[1] -> (96, 80), offset 7680; right tensor((2, 80, 72))[1] -> (80, 72), offset 5760 | (96, 72), stride (72, 1), offset 0, requires_grad=False | 6 | 44.585 us +/- 3.240, var 23.971 | 15.410 us +/- 0.072, var 0.873 | 2.89x | `15246320616318125486`/`17232460849795164269` |
| `offset_contiguous_96x80_by_80x72` | offset | `torch.matmul(left, right)` | left tensor((3, 96, 80))[1] -> (96, 80), offset 7680; right tensor((2, 80, 72))[1] -> (80, 72), offset 5760 | (96, 72), stride (72, 1), offset 0, requires_grad=False | 6 | 46.478 us +/- 1.353, var 11.781 | 15.358 us +/- 0.070, var 3.579 | 3.03x | `15246320616318125486`/`17232460849795164269` |
| `noncontig_transpose_96x128_by_128x64` | noncontiguous | `left @ right` | left tensor((128, 96)).transpose(0, 1) -> (96, 128), stride (1, 96); right tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128) | (96, 64), stride (64, 1), offset 0, requires_grad=False | 5 | 677.841 us +/- 6.406, var 1569.329 | 20.055 us +/- 0.101, var 2.702 | 33.80x | `2336782377594427688`/`10708052186225123967` |
| `noncontig_transpose_96x128_by_128x64` | noncontiguous | `left.matmul(right)` | left tensor((128, 96)).transpose(0, 1) -> (96, 128), stride (1, 96); right tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128) | (96, 64), stride (64, 1), offset 0, requires_grad=False | 5 | 689.910 us +/- 8.224, var 1043.399 | 19.947 us +/- 0.138, var 3.211 | 34.59x | `2336782377594427688`/`10708052186225123967` |
| `noncontig_transpose_96x128_by_128x64` | noncontiguous | `torch.matmul(left, right)` | left tensor((128, 96)).transpose(0, 1) -> (96, 128), stride (1, 96); right tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128) | (96, 64), stride (64, 1), offset 0, requires_grad=False | 5 | 720.106 us +/- 12.448, var 542.670 | 19.937 us +/- 0.103, var 2.720 | 36.12x | `2336782377594427688`/`10708052186225123967` |
| `no_grad_requires_grad_96x64_by_64x80` | no_grad | `left @ right` | left/right leaves, requires_grad=True; operation inside no_grad | (96, 80), stride (80, 1), offset 0, requires_grad=False | 6 | 41.326 us +/- 0.243, var 23.371 | 15.228 us +/- 0.073, var 0.738 | 2.71x | `13557601489413660982`/`4150218836828107213` |
| `no_grad_requires_grad_96x64_by_64x80` | no_grad | `left.matmul(right)` | left/right leaves, requires_grad=True; operation inside no_grad | (96, 80), stride (80, 1), offset 0, requires_grad=False | 6 | 41.297 us +/- 0.333, var 6.826 | 15.052 us +/- 0.110, var 1.382 | 2.74x | `13557601489413660982`/`4150218836828107213` |
| `no_grad_requires_grad_96x64_by_64x80` | no_grad | `torch.matmul(left, right)` | left/right leaves, requires_grad=True; operation inside no_grad | (96, 80), stride (80, 1), offset 0, requires_grad=False | 6 | 41.343 us +/- 0.412, var 3.542 | 15.047 us +/- 0.056, var 2.058 | 2.75x | `13557601489413660982`/`4150218836828107213` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `operator_rank1_dot` | `RuntimeError: matmul currently requires two rank-2 tensors, got [3] and [3]` | supported shape (), stride (), dtype torch.float32 | zero |
| `tensor_matmul_matrix_vector` | `RuntimeError: matmul currently requires two rank-2 tensors, got [32, 64] and [64]` | supported shape (32,), stride (1,), dtype torch.float32 | zero |
| `torch_matmul_batched_rank3` | `RuntimeError: matmul currently requires two rank-2 tensors, got [3, 4, 5] and [3, 5, 2]` | supported shape (3, 4, 2), stride (8, 2, 1), dtype torch.float32 | zero |
| `torch_matmul_out_rank2` | `TypeError: matmul() got an unexpected keyword argument 'out'` | supported shape (2, 2), stride (2, 1), dtype torch.float32 | zero |
| `operator_full_sum_backward` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported left_grad (2, 3); right_grad (3, 2) | zero |
| `tensor_matmul_full_sum_backward` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported left_grad (2, 3); right_grad (3, 2) | zero |
| `torch_matmul_full_sum_backward` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported left_grad (2, 3); right_grad (3, 2) | zero |
