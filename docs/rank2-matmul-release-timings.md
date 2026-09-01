# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`8c05958c435aa3a3cecb44555cc73e13d0ac7b0f`. This branch adds timing
evidence only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/rank2-matmul-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`, `CONDA_SHLVL=0`), so
setup used a worktree-local `.venv`. Cargo registry data was copied read-only
from the existing user cache into `target/cargo-home`, then Cargo ran offline
so build artifacts and dependency state stayed inside this worktree.

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
git add --intent-to-add docs/rank2-matmul-release-timings.md
git diff --check
git reset -q -- docs/rank2-matmul-release-timings.md
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 19 tests. The README/docs smoke test passed 7 tests. `cargo fmt
--check` passed. The filtered native Rust matmul tests passed 6 library tests
and 3 integration tests. `git diff --check` passed.

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
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 26 ms, prepared
  packages in 15.96s, and installed in 940 ms; the release wheel reinstall
  resolved in 1 ms, prepared in 41 ms, and installed in 12 ms
- Build time: successful offline release extension build completed in 35.03s

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported forward cell, the driver compared `torch_rs` with
PyTorch for shape, stride, storage offset, contiguity, dtype, device,
`requires_grad`, leaf status, NaN classifications, sign bits for non-NaN
values, and numeric values with `rtol=2e-6`, `atol=1e-5`, and
`equal_nan=True`. After every warmup and measured block, it consumed the last
output as a 64-bit BLAKE2b rolling checksum over tensor metadata and logical
bytes. The checksum column shows the final rolling sink from one pass as
`torch_rs`/PyTorch; both process passes produced the same sink pairs. Non-empty
matmul outputs are tolerance-equal but not necessarily bit-identical, so their
materialized checksums can differ while the correctness gate still passes.

No backward-through-full-`sum` matmul cells are supported at this revision:
`torch_rs` returns detached matmul outputs for `requires_grad=True` operands.
Those cells are preserved below as zero-credit unsupported entries rather than
being removed from the denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.12x uncapped, 2.05x capped
- `@` operator cells: 2.07x uncapped, 2.00x capped
- `Tensor.matmul` cells: 2.10x uncapped, 2.02x capped
- `torch.matmul` cells: 2.21x uncapped, 2.13x capped
- Square contiguous cells: 3.18x uncapped, 3.18x capped
- Rectangular contiguous cells: 3.22x uncapped, 3.22x capped
- Skinny contiguous cells: 3.24x uncapped, 3.24x capped
- Empty-dimension cells: 0.47x uncapped, 0.47x capped
- Offset contiguous cells: 3.14x uncapped, 3.14x capped
- Noncontiguous cells: 13.23x uncapped, 10.00x capped
- `no_grad` cells: 1.34x uncapped, 1.34x capped

Including the unsupported backward-through-full-`sum` cells below as
zero-credit denominator entries with a 10.00x capped penalty gives a combined
capped aggregate of 2.44x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums | Max abs diff |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `square_96x96` | square contiguous | `left @ right` | left `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False; right `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 74.257 us +/- 11.109 us, var 367.715 | 23.456 us +/- 1.905 us, var 9.330 | 3.17x | `8371638168758384640`/`7069418618342349280` | 5.36442e-07 |
| `square_96x96` | square contiguous | `left.matmul(right)` | left `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False; right `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 69.595 us +/- 6.215 us, var 268.170 | 21.257 us +/- 0.173 us, var 14.924 | 3.27x | `8371638168758384640`/`7069418618342349280` | 5.36442e-07 |
| `square_96x96` | square contiguous | `torch.matmul(left, right)` | left `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False; right `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | `(96, 96)`, stride `(96, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 65.860 us +/- 2.251 us, var 239.474 | 21.262 us +/- 0.283 us, var 6.134 | 3.10x | `8371638168758384640`/`7069418618342349280` | 5.36442e-07 |
| `rectangular_97x131_by_131x61` | rectangular contiguous | `left @ right` | left `(97, 131)`, stride `(131, 1)`, offset 0, requires_grad=False; right `(131, 61)`, stride `(61, 1)`, offset 0, requires_grad=False | `(97, 61)`, stride `(61, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 91.055 us +/- 8.052 us, var 162.944 | 27.682 us +/- 0.949 us, var 11.796 | 3.29x | `10241960832616584960`/`6018192590736400128` | 4.76837e-07 |
| `rectangular_97x131_by_131x61` | rectangular contiguous | `left.matmul(right)` | left `(97, 131)`, stride `(131, 1)`, offset 0, requires_grad=False; right `(131, 61)`, stride `(61, 1)`, offset 0, requires_grad=False | `(97, 61)`, stride `(61, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 89.070 us +/- 4.074 us, var 598.738 | 27.079 us +/- 0.336 us, var 12.886 | 3.29x | `10241960832616584960`/`6018192590736400128` | 4.76837e-07 |
| `rectangular_97x131_by_131x61` | rectangular contiguous | `torch.matmul(left, right)` | left `(97, 131)`, stride `(131, 1)`, offset 0, requires_grad=False; right `(131, 61)`, stride `(61, 1)`, offset 0, requires_grad=False | `(97, 61)`, stride `(61, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 86.391 us +/- 3.758 us, var 54.147 | 28.100 us +/- 1.420 us, var 11.641 | 3.07x | `10241960832616584960`/`6018192590736400128` | 4.76837e-07 |
| `skinny_2048x8_by_8x32` | skinny contiguous | `left @ right` | left `(2048, 8)`, stride `(8, 1)`, offset 0, requires_grad=False; right `(8, 32)`, stride `(32, 1)`, offset 0, requires_grad=False | `(2048, 32)`, stride `(32, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 66.713 us +/- 5.861 us, var 157.132 | 19.432 us +/- 0.504 us, var 14.002 | 3.43x | `9197254599447372768`/`12249346293893813472` | 1.19209e-07 |
| `skinny_2048x8_by_8x32` | skinny contiguous | `left.matmul(right)` | left `(2048, 8)`, stride `(8, 1)`, offset 0, requires_grad=False; right `(8, 32)`, stride `(32, 1)`, offset 0, requires_grad=False | `(2048, 32)`, stride `(32, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 61.300 us +/- 1.565 us, var 168.075 | 19.322 us +/- 0.526 us, var 8.346 | 3.17x | `9197254599447372768`/`12249346293893813472` | 1.19209e-07 |
| `skinny_2048x8_by_8x32` | skinny contiguous | `torch.matmul(left, right)` | left `(2048, 8)`, stride `(8, 1)`, offset 0, requires_grad=False; right `(8, 32)`, stride `(32, 1)`, offset 0, requires_grad=False | `(2048, 32)`, stride `(32, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 62.975 us +/- 2.652 us, var 234.266 | 20.261 us +/- 1.533 us, var 34.093 | 3.11x | `9197254599447372768`/`12249346293893813472` | 1.19209e-07 |
| `empty_rows_0x257_by_257x13` | empty dimension | `left @ right` | left `(0, 257)`, stride `(257, 1)`, offset 0, requires_grad=False; right `(257, 13)`, stride `(13, 1)`, offset 0, requires_grad=False | `(0, 13)`, stride `(13, 1)`, offset 0, requires_grad=False, is_leaf=True | 5000 | 0.569 us +/- 0.006 us, var 0.000 | 1.391 us +/- 0.011 us, var 0.002 | 0.41x | `11530685533628284128`/`11530685533628284128` | 0 |
| `empty_rows_0x257_by_257x13` | empty dimension | `left.matmul(right)` | left `(0, 257)`, stride `(257, 1)`, offset 0, requires_grad=False; right `(257, 13)`, stride `(13, 1)`, offset 0, requires_grad=False | `(0, 13)`, stride `(13, 1)`, offset 0, requires_grad=False, is_leaf=True | 5000 | 0.615 us +/- 0.005 us, var 0.000 | 1.392 us +/- 0.013 us, var 0.014 | 0.44x | `11530685533628284128`/`11530685533628284128` | 0 |
| `empty_rows_0x257_by_257x13` | empty dimension | `torch.matmul(left, right)` | left `(0, 257)`, stride `(257, 1)`, offset 0, requires_grad=False; right `(257, 13)`, stride `(13, 1)`, offset 0, requires_grad=False | `(0, 13)`, stride `(13, 1)`, offset 0, requires_grad=False, is_leaf=True | 5000 | 0.701 us +/- 0.006 us, var 0.001 | 1.246 us +/- 0.010 us, var 0.001 | 0.56x | `11530685533628284128`/`11530685533628284128` | 0 |
| `empty_inner_257x0_by_0x13` | empty dimension | `left @ right` | left `(257, 0)`, stride `(1, 1)`, offset 0, requires_grad=False; right `(0, 13)`, stride `(13, 1)`, offset 0, requires_grad=False | `(257, 13)`, stride `(13, 1)`, offset 0, requires_grad=False, is_leaf=True | 50 | 0.736 us +/- 0.008 us, var 0.004 | 1.724 us +/- 0.010 us, var 0.014 | 0.43x | `10877989294195986656`/`10877989294195986656` | 0 |
| `empty_inner_257x0_by_0x13` | empty dimension | `left.matmul(right)` | left `(257, 0)`, stride `(1, 1)`, offset 0, requires_grad=False; right `(0, 13)`, stride `(13, 1)`, offset 0, requires_grad=False | `(257, 13)`, stride `(13, 1)`, offset 0, requires_grad=False, is_leaf=True | 50 | 0.778 us +/- 0.008 us, var 0.006 | 1.687 us +/- 0.011 us, var 0.022 | 0.46x | `10877989294195986656`/`10877989294195986656` | 0 |
| `empty_inner_257x0_by_0x13` | empty dimension | `torch.matmul(left, right)` | left `(257, 0)`, stride `(1, 1)`, offset 0, requires_grad=False; right `(0, 13)`, stride `(13, 1)`, offset 0, requires_grad=False | `(257, 13)`, stride `(13, 1)`, offset 0, requires_grad=False, is_leaf=True | 50 | 0.888 us +/- 0.014 us, var 0.014 | 1.577 us +/- 0.008 us, var 0.044 | 0.56x | `10877989294195986656`/`10877989294195986656` | 0 |
| `offset_89x55_by_55x34` | offset contiguous | `left @ right` | left `(89, 55)`, stride `(55, 1)`, offset 4895, requires_grad=False; right `(55, 34)`, stride `(34, 1)`, offset 1870, requires_grad=False | `(89, 34)`, stride `(34, 1)`, offset 0, requires_grad=False, is_leaf=True | 3 | 22.926 us +/- 0.215 us, var 5.253 | 7.264 us +/- 0.035 us, var 0.856 | 3.16x | `10681775018771063648`/`13353900284209125824` | 3.57628e-07 |
| `offset_89x55_by_55x34` | offset contiguous | `left.matmul(right)` | left `(89, 55)`, stride `(55, 1)`, offset 4895, requires_grad=False; right `(55, 34)`, stride `(34, 1)`, offset 1870, requires_grad=False | `(89, 34)`, stride `(34, 1)`, offset 0, requires_grad=False, is_leaf=True | 3 | 22.007 us +/- 0.097 us, var 15.368 | 7.223 us +/- 0.039 us, var 0.751 | 3.05x | `10681775018771063648`/`13353900284209125824` | 3.57628e-07 |
| `offset_89x55_by_55x34` | offset contiguous | `torch.matmul(left, right)` | left `(89, 55)`, stride `(55, 1)`, offset 4895, requires_grad=False; right `(55, 34)`, stride `(34, 1)`, offset 1870, requires_grad=False | `(89, 34)`, stride `(34, 1)`, offset 0, requires_grad=False, is_leaf=True | 3 | 22.813 us +/- 0.185 us, var 5.288 | 7.059 us +/- 0.042 us, var 0.825 | 3.23x | `10681775018771063648`/`13353900284209125824` | 3.57628e-07 |
| `noncontig_transpose_79x113_by_113x47` | noncontiguous | `left @ right` | left `(79, 113)`, stride `(1, 79)`, offset 0, requires_grad=False; right `(113, 47)`, stride `(1, 113)`, offset 0, requires_grad=False | `(79, 47)`, stride `(47, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 252.409 us +/- 5.854 us, var 131.308 | 19.191 us +/- 0.070 us, var 5.234 | 13.15x | `1978048977741433312`/`9814321649685865952` | 3.57628e-07 |
| `noncontig_transpose_79x113_by_113x47` | noncontiguous | `left.matmul(right)` | left `(79, 113)`, stride `(1, 79)`, offset 0, requires_grad=False; right `(113, 47)`, stride `(1, 113)`, offset 0, requires_grad=False | `(79, 47)`, stride `(47, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 253.471 us +/- 4.206 us, var 60.214 | 19.191 us +/- 0.098 us, var 7.960 | 13.21x | `1978048977741433312`/`9814321649685865952` | 3.57628e-07 |
| `noncontig_transpose_79x113_by_113x47` | noncontiguous | `torch.matmul(left, right)` | left `(79, 113)`, stride `(1, 79)`, offset 0, requires_grad=False; right `(113, 47)`, stride `(1, 113)`, offset 0, requires_grad=False | `(79, 47)`, stride `(47, 1)`, offset 0, requires_grad=False, is_leaf=True | 2 | 253.260 us +/- 5.540 us, var 123.472 | 18.996 us +/- 0.078 us, var 4.594 | 13.33x | `1978048977741433312`/`9814321649685865952` | 3.57628e-07 |
| `no_grad_requires_grad_31x37_by_37x29` | no_grad | `left @ right` | left `(31, 37)`, stride `(37, 1)`, offset 0, requires_grad=True; right `(37, 29)`, stride `(29, 1)`, offset 0, requires_grad=True | `(31, 29)`, stride `(29, 1)`, offset 0, requires_grad=False, is_leaf=True | 5 | 6.381 us +/- 0.017 us, var 0.492 | 4.933 us +/- 0.033 us, var 0.724 | 1.29x | `6528823500333074656`/`4084368776410915296` | 1.78814e-07 |
| `no_grad_requires_grad_31x37_by_37x29` | no_grad | `left.matmul(right)` | left `(31, 37)`, stride `(37, 1)`, offset 0, requires_grad=True; right `(37, 29)`, stride `(29, 1)`, offset 0, requires_grad=True | `(31, 29)`, stride `(29, 1)`, offset 0, requires_grad=False, is_leaf=True | 5 | 6.513 us +/- 0.026 us, var 0.712 | 4.890 us +/- 0.047 us, var 0.581 | 1.33x | `6528823500333074656`/`4084368776410915296` | 1.78814e-07 |
| `no_grad_requires_grad_31x37_by_37x29` | no_grad | `torch.matmul(left, right)` | left `(31, 37)`, stride `(37, 1)`, offset 0, requires_grad=True; right `(37, 29)`, stride `(29, 1)`, offset 0, requires_grad=True | `(31, 29)`, stride `(29, 1)`, offset 0, requires_grad=False, is_leaf=True | 5 | 6.517 us +/- 0.038 us, var 0.387 | 4.700 us +/- 0.023 us, var 0.508 | 1.39x | `6528823500333074656`/`4084368776410915296` | 1.78814e-07 |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | API | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `backward_full_sum_17x19_by_19x23` | `left @ right` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported; output `requires_grad=True`; left/right grad checksums `12589315674427152259`/`16592137621549209632` | zero |
| `backward_full_sum_17x19_by_19x23` | `left.matmul(right)` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported; output `requires_grad=True`; left/right grad checksums `12589315674427152259`/`16592137621549209632` | zero |
| `backward_full_sum_17x19_by_19x23` | `torch.matmul(left, right)` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported; output `requires_grad=True`; left/right grad checksums `12589315674427152259`/`16592137621549209632` | zero |
