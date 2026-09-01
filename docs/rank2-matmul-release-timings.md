# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`199196b80bd8dd1a7a6c6569d472339078a40145`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/rank2-matmul-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_SHLVL=0`), so setup used a
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
  .venv/bin/python -m unittest \
  tests.test_matmul tests.test_matmul_reference \
  tests.test_transpose tests.test_transpose_reference \
  tests.test_set_float32_matmul_precision \
  tests.test_set_float32_matmul_precision_reference
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
  cargo test --locked --offline --all-targets matrix
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
  .venv/bin/python -m unittest \
  tests.test_matmul tests.test_matmul_reference \
  tests.test_transpose tests.test_transpose_reference \
  tests.test_set_float32_matmul_precision \
  tests.test_set_float32_matmul_precision_reference
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
  cargo test --locked --offline --all-targets matrix
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 49 tests. `cargo fmt --check` passed. The filtered native Rust `matmul`
tests passed 9 tests, and the filtered native Rust `matrix` tests passed 2
tests. The README/docs smoke test passed, and `git diff --check` passed.

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
  unit; `torch_rs.version.debug` reported `False`
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 30 ms, prepared
  packages in 15.56s, and installed in 1.07s
- Build time: successful offline release extension build completed in 35.81s;
  the release wheel reinstall resolved in 2 ms, prepared in 36 ms, and
  installed in 12 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported forward cell, the driver checked `torch_rs`
against PyTorch for shape, stride, storage offset, contiguity, dtype, device,
`requires_grad`, leaf status, NaN classification, and values with
`rtol=1e-4`, `atol=1e-5`, and `equal_nan=True`. After every warmup and
measured block, the driver consumed the last output as a 64-bit BLAKE2b
rolling checksum over tensor metadata and logical bytes. Different checksum
values between `torch_rs` and PyTorch are expected for dense non-empty cells
because their accumulation order is not bit-identical; the correctness gate
above was applied before timing. The checksum column shows the final rolling
sink from one pass as `torch_rs`/PyTorch; both process passes produced stable
sink pairs for each implementation.

No active-autograd rank-2 matmul backward-through-full-`sum` cells are
supported in this snapshot: `torch_rs` returns a detached leaf for requires-grad
matmul while PyTorch records a `grad_fn`. Those equivalent PyTorch cells are
kept in the zero-credit unsupported table instead of being timed as supported
workloads.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.34x uncapped, 2.11x capped
- `@` operator cells: 2.20x uncapped, 1.99x capped
- `Tensor.matmul` cells: 2.30x uncapped, 2.08x capped
- `torch.matmul` cells: 2.53x uncapped, 2.28x capped
- Square contiguous cells: 3.33x uncapped, 3.33x capped
- Rectangular contiguous cells: 4.48x uncapped, 4.48x capped
- Skinny contiguous cells: 3.03x uncapped, 3.03x capped
- Empty-dimension cells: 0.29x uncapped, 0.29x capped
- Offset contiguous cells: 3.50x uncapped, 3.50x capped
- Noncontiguous cells: 22.49x uncapped, 10.00x capped
- `no_grad` cells: 2.91x uncapped, 2.91x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 3.00x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `square_128x128` | square contiguous | `left @ other` | left/right (128, 128), stride (128, 1), requires_grad=False | (128, 128), stride (128, 1), offset 0, requires_grad=False, is_leaf=True | 3 | 151.395 us +/- 3.602 us, var 170.501 | 45.464 us +/- 0.988 us, var 17.387 | 3.33x | `1121397423969788280`/`9857879494003972976` |
| `square_128x128` | square contiguous | `left.matmul(other)` | left/right (128, 128), stride (128, 1), requires_grad=False | (128, 128), stride (128, 1), offset 0, requires_grad=False, is_leaf=True | 3 | 151.544 us +/- 3.111 us, var 528.392 | 45.511 us +/- 1.110 us, var 19.439 | 3.33x | `1121397423969788280`/`9857879494003972976` |
| `square_128x128` | square contiguous | `torch.matmul(left, other)` | left/right (128, 128), stride (128, 1), requires_grad=False | (128, 128), stride (128, 1), offset 0, requires_grad=False, is_leaf=True | 3 | 150.950 us +/- 3.799 us, var 219.210 | 45.284 us +/- 1.008 us, var 19.263 | 3.33x | `1121397423969788280`/`9857879494003972976` |
| `rectangular_257x129_by_129x263` | rectangular contiguous | `left @ other` | left (257, 129), right (129, 263), canonical row-major strides | (257, 263), stride (263, 1), offset 0, requires_grad=False, is_leaf=True | 1 | 914.773 us +/- 3.034 us, var 624.828 | 203.938 us +/- 1.657 us, var 46.638 | 4.49x | `1543793376217475051`/`14641382051020048483` |
| `rectangular_257x129_by_129x263` | rectangular contiguous | `left.matmul(other)` | left (257, 129), right (129, 263), canonical row-major strides | (257, 263), stride (263, 1), offset 0, requires_grad=False, is_leaf=True | 1 | 916.530 us +/- 3.295 us, var 315.004 | 204.334 us +/- 1.317 us, var 95.560 | 4.49x | `1543793376217475051`/`14641382051020048483` |
| `rectangular_257x129_by_129x263` | rectangular contiguous | `torch.matmul(left, other)` | left (257, 129), right (129, 263), canonical row-major strides | (257, 263), stride (263, 1), offset 0, requires_grad=False, is_leaf=True | 1 | 915.563 us +/- 4.352 us, var 538.365 | 204.599 us +/- 1.897 us, var 65.215 | 4.47x | `1543793376217475051`/`14641382051020048483` |
| `skinny_1024x8_by_8x16` | skinny contiguous | `left @ other` | left (1024, 8), right (8, 16), canonical row-major strides | (1024, 16), stride (16, 1), offset 0, requires_grad=False, is_leaf=True | 20 | 18.513 us +/- 0.109 us, var 1.902 | 6.226 us +/- 0.075 us, var 0.661 | 2.97x | `7050339990616449437`/`15775378655190309413` |
| `skinny_1024x8_by_8x16` | skinny contiguous | `left.matmul(other)` | left (1024, 8), right (8, 16), canonical row-major strides | (1024, 16), stride (16, 1), offset 0, requires_grad=False, is_leaf=True | 20 | 18.561 us +/- 0.156 us, var 0.799 | 6.186 us +/- 0.058 us, var 0.072 | 3.00x | `7050339990616449437`/`15775378655190309413` |
| `skinny_1024x8_by_8x16` | skinny contiguous | `torch.matmul(left, other)` | left (1024, 8), right (8, 16), canonical row-major strides | (1024, 16), stride (16, 1), offset 0, requires_grad=False, is_leaf=True | 20 | 18.726 us +/- 0.184 us, var 2.753 | 6.004 us +/- 0.050 us, var 0.086 | 3.12x | `7050339990616449437`/`15775378655190309413` |
| `empty_rows_0x64_by_64x32` | empty dimension | `left @ other` | left zeros((0, 64)); right (64, 32), canonical row-major strides | (0, 32), stride (32, 1), offset 0, requires_grad=False, is_leaf=True | 5000 | 0.225 us +/- 0.002 us, var 0.000 | 0.955 us +/- 0.005 us, var 0.001 | 0.24x | `11743028092850095163`/`11743028092850095163` |
| `empty_rows_0x64_by_64x32` | empty dimension | `left.matmul(other)` | left zeros((0, 64)); right (64, 32), canonical row-major strides | (0, 32), stride (32, 1), offset 0, requires_grad=False, is_leaf=True | 5000 | 0.278 us +/- 0.002 us, var 0.000 | 0.962 us +/- 0.006 us, var 0.004 | 0.29x | `11743028092850095163`/`11743028092850095163` |
| `empty_rows_0x64_by_64x32` | empty dimension | `torch.matmul(left, other)` | left zeros((0, 64)); right (64, 32), canonical row-major strides | (0, 32), stride (32, 1), offset 0, requires_grad=False, is_leaf=True | 5000 | 0.343 us +/- 0.009 us, var 0.002 | 0.846 us +/- 0.005 us, var 0.004 | 0.41x | `11743028092850095163`/`11743028092850095163` |
| `empty_inner_32x0_by_0x16` | empty dimension | `left @ other` | left zeros((32, 0)); right zeros((0, 16)), canonical row-major strides | (32, 16), stride (16, 1), offset 0, requires_grad=False, is_leaf=True | 1000 | 0.277 us +/- 0.006 us, var 0.000 | 1.205 us +/- 0.009 us, var 0.001 | 0.23x | `1152609824216190470`/`1152609824216190470` |
| `empty_inner_32x0_by_0x16` | empty dimension | `left.matmul(other)` | left zeros((32, 0)); right zeros((0, 16)), canonical row-major strides | (32, 16), stride (16, 1), offset 0, requires_grad=False, is_leaf=True | 1000 | 0.322 us +/- 0.006 us, var 0.001 | 1.188 us +/- 0.011 us, var 0.013 | 0.27x | `1152609824216190470`/`1152609824216190470` |
| `empty_inner_32x0_by_0x16` | empty dimension | `torch.matmul(left, other)` | left zeros((32, 0)); right zeros((0, 16)), canonical row-major strides | (32, 16), stride (16, 1), offset 0, requires_grad=False, is_leaf=True | 1000 | 0.399 us +/- 0.008 us, var 0.003 | 1.078 us +/- 0.010 us, var 0.024 | 0.37x | `1152609824216190470`/`1152609824216190470` |
| `offset_contiguous_97x65_by_65x33` | offset contiguous | `left @ other` | left tensor((3, 97, 65))[1] -> (97, 65), offset 6305; right tensor((3, 65, 33))[1] -> (65, 33), offset 2145 | (97, 33), stride (33, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 28.655 us +/- 0.056 us, var 0.988 | 8.155 us +/- 0.021 us, var 0.404 | 3.51x | `2127067911345725205`/`5959261106102650017` |
| `offset_contiguous_97x65_by_65x33` | offset contiguous | `left.matmul(other)` | left tensor((3, 97, 65))[1] -> (97, 65), offset 6305; right tensor((3, 65, 33))[1] -> (65, 33), offset 2145 | (97, 33), stride (33, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 28.297 us +/- 0.048 us, var 6.131 | 8.159 us +/- 0.017 us, var 0.223 | 3.47x | `2127067911345725205`/`5959261106102650017` |
| `offset_contiguous_97x65_by_65x33` | offset contiguous | `torch.matmul(left, other)` | left tensor((3, 97, 65))[1] -> (97, 65), offset 6305; right tensor((3, 65, 33))[1] -> (65, 33), offset 2145 | (97, 33), stride (33, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 28.151 us +/- 0.056 us, var 1.041 | 7.996 us +/- 0.016 us, var 0.326 | 3.52x | `2127067911345725205`/`5959261106102650017` |
| `noncontig_transpose_128x64_by_64x48` | noncontiguous | `left @ other` | left tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128); right tensor((48, 64)).transpose(0, 1) -> (64, 48), stride (1, 64) | (128, 48), stride (48, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 234.404 us +/- 2.515 us, var 195.141 | 10.562 us +/- 0.081 us, var 0.348 | 22.19x | `15578249923024148880`/`10429815600264253031` |
| `noncontig_transpose_128x64_by_64x48` | noncontiguous | `left.matmul(other)` | left tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128); right tensor((48, 64)).transpose(0, 1) -> (64, 48), stride (1, 64) | (128, 48), stride (48, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 233.017 us +/- 1.359 us, var 164.586 | 10.344 us +/- 0.030 us, var 0.350 | 22.53x | `15578249923024148880`/`10429815600264253031` |
| `noncontig_transpose_128x64_by_64x48` | noncontiguous | `torch.matmul(left, other)` | left tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128); right tensor((48, 64)).transpose(0, 1) -> (64, 48), stride (1, 64) | (128, 48), stride (48, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 232.300 us +/- 1.126 us, var 92.628 | 10.209 us +/- 0.048 us, var 1.477 | 22.75x | `15578249923024148880`/`10429815600264253031` |
| `no_grad_requires_grad_96x64_by_64x48` | no_grad | `left @ other` | left/right leaves with requires_grad=True; operation inside no_grad | (96, 48), stride (48, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 27.855 us +/- 0.092 us, var 2.146 | 9.679 us +/- 0.043 us, var 1.520 | 2.88x | `14761302374450055690`/`10830632428278797067` |
| `no_grad_requires_grad_96x64_by_64x48` | no_grad | `left.matmul(other)` | left/right leaves with requires_grad=True; operation inside no_grad | (96, 48), stride (48, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 27.900 us +/- 0.105 us, var 4.685 | 9.636 us +/- 0.030 us, var 0.349 | 2.90x | `14761302374450055690`/`10830632428278797067` |
| `no_grad_requires_grad_96x64_by_64x48` | no_grad | `torch.matmul(left, other)` | left/right leaves with requires_grad=True; operation inside no_grad | (96, 48), stride (48, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 27.979 us +/- 0.099 us, var 1.216 | 9.444 us +/- 0.032 us, var 0.218 | 2.96x | `14761302374450055690`/`10830632428278797067` |

## Zero-Credit Unsupported Cells

These cells are not timed as supported matmul performance because `torch_rs`
cannot execute the equivalent PyTorch behavior. They are preserved as
zero-credit cells instead of being removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `operator_active_autograd_forward` | returned shape=(2, 2), requires_grad=False, is_leaf=True | returned shape=(2, 2), requires_grad=True, is_leaf=False | zero |
| `operator_backward_full_sum` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | returned `None` | zero |
| `tensor_matmul_backward_full_sum` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | returned `None` | zero |
| `torch_matmul_backward_full_sum` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | returned `None` | zero |
| `top_level_torch_matmul_out` | `TypeError: matmul() got an unexpected keyword argument 'out'` | returned shape=(2, 2), requires_grad=False, is_leaf=True | zero |
| `operator_rank1_by_rank2` | `RuntimeError: matmul currently requires two rank-2 tensors, got [3] and [3, 2]` | returned shape=(2,), requires_grad=False, is_leaf=True | zero |
| `top_level_batched_rank3_by_rank2` | `RuntimeError: matmul currently requires two rank-2 tensors, got [4, 2, 3] and [3, 2]` | returned shape=(4, 2, 2), requires_grad=False, is_leaf=True | zero |
