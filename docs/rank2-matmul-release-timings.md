# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`4d3914c86bda5bb4f8a1f21f433296191c7c3f5f`. This branch adds timing evidence
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
wheel_dir="$(mktemp -d "$PWD/target/rank2-matmul-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/rank2-matmul-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/rank2-matmul-wheel-dir.txt)"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
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
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
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
git diff --check
```

Checks run for this evidence:

```bash
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
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 19 tests. `cargo fmt --check` passed. The focused native Rust `matmul`
filter passed 9 tests across the library and tensor-baseline targets. The
README/docs smoke test passed 7 tests, and `git diff --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0, from
  `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit; `torch.version.debug` and `torch_rs.version.debug` both reported
  `False`
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 26 ms, prepared
  packages in 15.94s, and installed in 1.36s
- Build time: successful offline release extension build completed in 34.60s;
  the release wheel reinstall resolved in 2 ms, prepared in 45 ms, and
  installed in 18 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported forward cell, the driver compared `torch_rs`
output with PyTorch, checking shape, stride, storage offset, contiguity, dtype,
device, `requires_grad`, and leaf status exactly. It checked values with
`rtol=2e-6`, `atol=1e-5`, and equal nonfinite classifications, and the table
reports the maximum absolute difference observed during those gates.
After every warmup and measured block, the driver consumed the last output as a
64-bit BLAKE2b rolling checksum over tensor metadata and materialized logical
bytes. The checksum column shows the final rolling sink from one pass as
`torch_rs`/PyTorch; both process passes produced the same sink pairs.

There are no supported backward-through-full-`sum` rank-2 matmul cells in this
revision: for grad-requiring inputs, `@`, `Tensor.matmul`, and `torch.matmul`
produce a leaf output with `requires_grad=False`. The equivalent PyTorch 2.13
operations support leaf gradients, so those cells are preserved as zero-credit
unsupported rows below instead of being timed.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.36x uncapped, 2.17x capped
- `@` cells: 2.23x uncapped, 2.05x capped
- `Tensor.matmul` cells: 2.35x uncapped, 2.16x capped
- `torch.matmul` cells: 2.52x uncapped, 2.31x capped
- Square contiguous cells: 3.28x uncapped, 3.28x capped
- Rectangular contiguous cells: 3.94x uncapped, 3.94x capped
- Skinny contiguous cells: 2.61x uncapped, 2.61x capped
- Empty-dimension cells: 0.33x uncapped, 0.33x capped
- Empty-inner-dimension cells: 0.45x uncapped, 0.45x capped
- Offset cells: 2.87x uncapped, 2.87x capped
- Noncontiguous cells: 19.73x uncapped, 10.00x capped
- `no_grad` cells: 3.39x uncapped, 3.39x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 2.70x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums | Max abs diff |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `square_contiguous_128x128` | square contiguous | `@` | left `(128, 128)`, stride `(128, 1)`; right `(128, 128)`, stride `(128, 1)` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 2 | 147.105 us +/- 0.774 us, var 139.290 | 45.078 us +/- 0.210 us, var 9.282 | 3.26x | `1539344778545015372`/`1040892007285659261` | 2.38e-06 |
| `square_contiguous_128x128` | square contiguous | `Tensor.matmul` | left `(128, 128)`, stride `(128, 1)`; right `(128, 128)`, stride `(128, 1)` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 2 | 149.195 us +/- 1.707 us, var 203.141 | 45.504 us +/- 0.273 us, var 11.333 | 3.28x | `1539344778545015372`/`1040892007285659261` | 2.38e-06 |
| `square_contiguous_128x128` | square contiguous | `torch.matmul` | left `(128, 128)`, stride `(128, 1)`; right `(128, 128)`, stride `(128, 1)` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 2 | 149.806 us +/- 2.497 us, var 218.113 | 45.433 us +/- 0.210 us, var 6.217 | 3.30x | `1539344778545015372`/`1040892007285659261` | 2.38e-06 |
| `rectangular_contiguous_257x263_by_263x127` | rectangular contiguous | `@` | left `(257, 263)`, stride `(263, 1)`; right `(263, 127)`, stride `(127, 1)` | `(257, 127)`, stride `(127, 1)`, offset 0, requires_grad=False | 1 | 914.251 us +/- 5.483 us, var 5459.107 | 233.148 us +/- 1.052 us, var 46.458 | 3.92x | `3499818643637156113`/`9054179867479613139` | 1.05e-05 |
| `rectangular_contiguous_257x263_by_263x127` | rectangular contiguous | `Tensor.matmul` | left `(257, 263)`, stride `(263, 1)`; right `(263, 127)`, stride `(127, 1)` | `(257, 127)`, stride `(127, 1)`, offset 0, requires_grad=False | 1 | 918.703 us +/- 3.290 us, var 618.290 | 233.058 us +/- 1.047 us, var 42.225 | 3.94x | `3499818643637156113`/`9054179867479613139` | 1.05e-05 |
| `rectangular_contiguous_257x263_by_263x127` | rectangular contiguous | `torch.matmul` | left `(257, 263)`, stride `(263, 1)`; right `(263, 127)`, stride `(127, 1)` | `(257, 127)`, stride `(127, 1)`, offset 0, requires_grad=False | 1 | 926.370 us +/- 12.253 us, var 499.333 | 233.854 us +/- 1.828 us, var 77.173 | 3.96x | `3499818643637156113`/`9054179867479613139` | 1.05e-05 |
| `skinny_inner_1024x8_by_8x64` | skinny contiguous | `@` | left `(1024, 8)`, stride `(8, 1)`; right `(8, 64)`, stride `(64, 1)` | `(1024, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 2 | 47.990 us +/- 0.423 us, var 12.671 | 18.660 us +/- 0.746 us, var 5.467 | 2.57x | `12371554046374008237`/`14602358208796006328` | 4.77e-07 |
| `skinny_inner_1024x8_by_8x64` | skinny contiguous | `Tensor.matmul` | left `(1024, 8)`, stride `(8, 1)`; right `(8, 64)`, stride `(64, 1)` | `(1024, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 2 | 47.785 us +/- 0.385 us, var 12.963 | 18.183 us +/- 0.230 us, var 2.382 | 2.63x | `12371554046374008237`/`14602358208796006328` | 4.77e-07 |
| `skinny_inner_1024x8_by_8x64` | skinny contiguous | `torch.matmul` | left `(1024, 8)`, stride `(8, 1)`; right `(8, 64)`, stride `(64, 1)` | `(1024, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 2 | 48.371 us +/- 0.593 us, var 25.810 | 18.353 us +/- 0.243 us, var 2.465 | 2.64x | `12371554046374008237`/`14602358208796006328` | 4.77e-07 |
| `empty_rows_0x257_by_257x31` | empty dimension | `@` | left `(0, 257)`, stride `(257, 1)`; right `(257, 31)`, stride `(31, 1)` | `(0, 31)`, stride `(31, 1)`, offset 0, requires_grad=False | 5000 | 0.247 us +/- 0.002 us, var 0.000 | 0.947 us +/- 0.008 us, var 0.002 | 0.26x | `4911983151691592547`/`4911983151691592547` | 0 |
| `empty_rows_0x257_by_257x31` | empty dimension | `Tensor.matmul` | left `(0, 257)`, stride `(257, 1)`; right `(257, 31)`, stride `(31, 1)` | `(0, 31)`, stride `(31, 1)`, offset 0, requires_grad=False | 5000 | 0.303 us +/- 0.003 us, var 0.000 | 0.939 us +/- 0.006 us, var 0.001 | 0.32x | `4911983151691592547`/`4911983151691592547` | 0 |
| `empty_rows_0x257_by_257x31` | empty dimension | `torch.matmul` | left `(0, 257)`, stride `(257, 1)`; right `(257, 31)`, stride `(31, 1)` | `(0, 31)`, stride `(31, 1)`, offset 0, requires_grad=False | 5000 | 0.371 us +/- 0.002 us, var 0.001 | 0.831 us +/- 0.004 us, var 0.001 | 0.45x | `4911983151691592547`/`4911983151691592547` | 0 |
| `empty_inner_257x0_by_0x31` | empty inner dimension | `@` | left `(257, 0)`, stride `(1, 1)`; right `(0, 31)`, stride `(31, 1)` | `(257, 31)`, stride `(31, 1)`, offset 0, requires_grad=False | 200 | 0.596 us +/- 0.002 us, var 0.006 | 1.547 us +/- 0.016 us, var 0.002 | 0.39x | `17846138844049691346`/`17846138844049691346` | 0 |
| `empty_inner_257x0_by_0x31` | empty inner dimension | `Tensor.matmul` | left `(257, 0)`, stride `(1, 1)`; right `(0, 31)`, stride `(31, 1)` | `(257, 31)`, stride `(31, 1)`, offset 0, requires_grad=False | 200 | 0.661 us +/- 0.002 us, var 0.001 | 1.523 us +/- 0.015 us, var 0.005 | 0.43x | `17846138844049691346`/`17846138844049691346` | 0 |
| `empty_inner_257x0_by_0x31` | empty inner dimension | `torch.matmul` | left `(257, 0)`, stride `(1, 1)`; right `(0, 31)`, stride `(31, 1)` | `(257, 31)`, stride `(31, 1)`, offset 0, requires_grad=False | 200 | 0.758 us +/- 0.002 us, var 0.002 | 1.405 us +/- 0.016 us, var 0.001 | 0.54x | `17846138844049691346`/`17846138844049691346` | 0 |
| `offset_contiguous_67x71_by_71x59` | offset | `@` | left `tensor((3, 67, 71))[1]` -> `(67, 71)`, stride `(71, 1)`, offset 4757; right `tensor((3, 71, 59))[2]` -> `(71, 59)`, stride `(59, 1)`, offset 8378 | `(67, 59)`, stride `(59, 1)`, offset 0, requires_grad=False | 5 | 32.067 us +/- 0.034 us, var 2.550 | 11.270 us +/- 0.031 us, var 0.450 | 2.85x | `4991956766735916514`/`17923137154081457241` | 1.67e-06 |
| `offset_contiguous_67x71_by_71x59` | offset | `Tensor.matmul` | left `tensor((3, 67, 71))[1]` -> `(67, 71)`, stride `(71, 1)`, offset 4757; right `tensor((3, 71, 59))[2]` -> `(71, 59)`, stride `(59, 1)`, offset 8378 | `(67, 59)`, stride `(59, 1)`, offset 0, requires_grad=False | 5 | 32.141 us +/- 0.059 us, var 3.159 | 11.218 us +/- 0.032 us, var 0.269 | 2.87x | `4991956766735916514`/`17923137154081457241` | 1.67e-06 |
| `offset_contiguous_67x71_by_71x59` | offset | `torch.matmul` | left `tensor((3, 67, 71))[1]` -> `(67, 71)`, stride `(71, 1)`, offset 4757; right `tensor((3, 71, 59))[2]` -> `(71, 59)`, stride `(59, 1)`, offset 8378 | `(67, 59)`, stride `(59, 1)`, offset 0, requires_grad=False | 5 | 32.093 us +/- 0.117 us, var 1.483 | 11.092 us +/- 0.023 us, var 0.736 | 2.89x | `4991956766735916514`/`17923137154081457241` | 1.67e-06 |
| `noncontig_transpose_128x64_by_64x96` | noncontiguous | `@` | left `tensor((64, 128)).transpose(0, 1)` -> `(128, 64)`, stride `(1, 128)`; right `tensor((96, 64)).transpose(0, 1)` -> `(64, 96)`, stride `(1, 64)` | `(128, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 1 | 387.731 us +/- 1.607 us, var 223.533 | 19.684 us +/- 0.114 us, var 1.887 | 19.70x | `15561673016148123495`/`2481061242902906307` | 1.43e-06 |
| `noncontig_transpose_128x64_by_64x96` | noncontiguous | `Tensor.matmul` | left `tensor((64, 128)).transpose(0, 1)` -> `(128, 64)`, stride `(1, 128)`; right `tensor((96, 64)).transpose(0, 1)` -> `(64, 96)`, stride `(1, 64)` | `(128, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 1 | 388.618 us +/- 1.688 us, var 1563.551 | 19.740 us +/- 0.126 us, var 3.486 | 19.69x | `15561673016148123495`/`2481061242902906307` | 1.43e-06 |
| `noncontig_transpose_128x64_by_64x96` | noncontiguous | `torch.matmul` | left `tensor((64, 128)).transpose(0, 1)` -> `(128, 64)`, stride `(1, 128)`; right `tensor((96, 64)).transpose(0, 1)` -> `(64, 96)`, stride `(1, 64)` | `(128, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 1 | 389.925 us +/- 3.200 us, var 958.278 | 19.680 us +/- 0.145 us, var 3.316 | 19.81x | `15561673016148123495`/`2481061242902906307` | 1.43e-06 |
| `no_grad_requires_grad_127x131_by_131x67` | no_grad | `@` | left `(127, 131)`, stride `(131, 1)`, requires_grad=True; right `(131, 67)`, stride `(67, 1)`, requires_grad=True; operation inside `no_grad` | `(127, 67)`, stride `(67, 1)`, offset 0, requires_grad=False | 2 | 124.045 us +/- 0.313 us, var 29.562 | 37.121 us +/- 0.205 us, var 3.977 | 3.34x | `7760013269833554144`/`7585788148992230720` | 1.91e-06 |
| `no_grad_requires_grad_127x131_by_131x67` | no_grad | `Tensor.matmul` | left `(127, 131)`, stride `(131, 1)`, requires_grad=True; right `(131, 67)`, stride `(67, 1)`, requires_grad=True; operation inside `no_grad` | `(127, 67)`, stride `(67, 1)`, offset 0, requires_grad=False | 2 | 126.369 us +/- 0.464 us, var 45.179 | 36.991 us +/- 0.200 us, var 6.834 | 3.42x | `7760013269833554144`/`7585788148992230720` | 1.91e-06 |
| `no_grad_requires_grad_127x131_by_131x67` | no_grad | `torch.matmul` | left `(127, 131)`, stride `(131, 1)`, requires_grad=True; right `(131, 67)`, stride `(67, 1)`, requires_grad=True; operation inside `no_grad` | `(127, 67)`, stride `(67, 1)`, offset 0, requires_grad=False | 2 | 126.066 us +/- 0.336 us, var 29.641 | 36.806 us +/- 0.175 us, var 3.142 | 3.43x | `7760013269833554144`/`7585788148992230720` | 1.91e-06 |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `@_backward_sum_32x33_by_33x17` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported leaf gradients | zero |
| `Tensor.matmul_backward_sum_32x33_by_33x17` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported leaf gradients | zero |
| `torch.matmul_backward_sum_32x33_by_33x17` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported leaf gradients | zero |
| `torch_matmul_out_2x3_by_3x2` | `TypeError: matmul() got an unexpected keyword argument 'out'` | supported out tensor | zero |
