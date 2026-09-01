# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`0a6fe63470fd7787b226c1a1384e185d7e7d00a4`. This branch adds timing evidence
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
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
printf '%s\n' "$wheel_dir" > target/rank2-matmul-wheel-dir.txt
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
passed 19 tests. `cargo fmt --check` passed. The filtered native Rust matmul
tests passed 9 tests: 6 library tests and 3 integration tests. The README/docs
smoke test passed 7 tests, and `git diff --check` passed.

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
  packages in 16.11s, installed in 1.34s, and took 17.60s wall time
- Build time: successful offline release extension build completed in 35.62s;
  the release wheel reinstall resolved in 2 ms, prepared in 44 ms, installed
  in 13 ms, and took 0.17s wall time

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported cell, the driver checked shape, stride, storage
offset, contiguity, dtype, device, `requires_grad`, and leaf status against
PyTorch. Values were checked with matching NaN classifications, matching
non-NaN sign bits, and `np.testing.assert_allclose(..., rtol=2e-6,
atol=1e-6, equal_nan=True)`. After every warmup and measured block, the driver
consumed the last output as a 64-bit BLAKE2b rolling checksum over tensor
metadata and logical bytes. The checksum column shows final rolling sinks from
pass 1 and pass 2 as `torch_rs:pass1,pass2; PyTorch:pass1,pass2`.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.20x uncapped, 1.98x capped
- `@` operator cells: 2.06x uncapped, 1.85x capped
- `Tensor.matmul` cells: 2.18x uncapped, 1.96x capped
- `torch.matmul` cells: 2.36x uncapped, 2.13x capped
- Square contiguous cells: 3.34x uncapped, 3.34x capped
- Rectangular contiguous cells: 4.42x uncapped, 4.42x capped
- Skinny contiguous cells: 3.28x uncapped, 3.28x capped
- Empty-dimension cells: 0.32x uncapped, 0.32x capped
- Empty-inner cells: 0.66x uncapped, 0.66x capped
- Offset contiguous cells: 4.43x uncapped, 4.43x capped
- Noncontiguous transpose cells: 26.29x uncapped, 10.00x capped
- `no_grad` cells: 3.21x uncapped, 3.21x capped
- Backward-through-full-`sum` cells: no supported matmul cells in the current
  contract; attempted cells are listed as zero-credit unsupported below

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 2.65x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `square_contiguous_128` | square contiguous | `left @ right` | left `(128, 128)`, stride `(128, 1)`, offset 0; right `(128, 128)`, stride `(128, 1)`, offset 0 | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 8 | 148.880 us +/- 1.010 us, var 100.023 | 45.329 us +/- 0.993 us, var 14.542 | 3.28x | `torch_rs:9334941618108809664,9334941618108809664; PyTorch:8567547243513667776,8567547243513667776` |
| `square_contiguous_128` | square contiguous | `left.matmul(right)` | left `(128, 128)`, stride `(128, 1)`, offset 0; right `(128, 128)`, stride `(128, 1)`, offset 0 | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 8 | 150.215 us +/- 0.938 us, var 8.883 | 44.405 us +/- 0.246 us, var 1.359 | 3.38x | `torch_rs:9334941618108809664,9334941618108809664; PyTorch:8567547243513667776,8567547243513667776` |
| `square_contiguous_128` | square contiguous | `torch.matmul(left, right)` | left `(128, 128)`, stride `(128, 1)`, offset 0; right `(128, 128)`, stride `(128, 1)`, offset 0 | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 8 | 149.417 us +/- 0.784 us, var 5.112 | 44.581 us +/- 0.444 us, var 1.504 | 3.35x | `torch_rs:9334941618108809664,9334941618108809664; PyTorch:8567547243513667776,8567547243513667776` |
| `rectangular_contiguous_257x129_by_129x263` | rectangular contiguous | `left @ right` | left `(257, 129)`, stride `(129, 1)`, offset 0; right `(129, 263)`, stride `(263, 1)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 3 | 915.515 us +/- 4.014 us, var 49.139 | 207.916 us +/- 2.450 us, var 28.437 | 4.40x | `torch_rs:5402242938982849504,5402242938982849504; PyTorch:8780333234460141312,8780333234460141312` |
| `rectangular_contiguous_257x129_by_129x263` | rectangular contiguous | `left.matmul(right)` | left `(257, 129)`, stride `(129, 1)`, offset 0; right `(129, 263)`, stride `(263, 1)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 3 | 916.902 us +/- 3.550 us, var 471.198 | 208.188 us +/- 2.287 us, var 17.085 | 4.40x | `torch_rs:5402242938982849504,5402242938982849504; PyTorch:8780333234460141312,8780333234460141312` |
| `rectangular_contiguous_257x129_by_129x263` | rectangular contiguous | `torch.matmul(left, right)` | left `(257, 129)`, stride `(129, 1)`, offset 0; right `(129, 263)`, stride `(263, 1)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 3 | 923.036 us +/- 4.664 us, var 221.053 | 207.070 us +/- 2.605 us, var 23.958 | 4.46x | `torch_rs:5402242938982849504,5402242938982849504; PyTorch:8780333234460141312,8780333234460141312` |
| `skinny_contiguous_4096x8_by_8x32` | skinny contiguous | `left @ right` | left `(4096, 8)`, stride `(8, 1)`, offset 0; right `(8, 32)`, stride `(32, 1)`, offset 0 | `(4096, 32)`, stride `(32, 1)`, offset 0, requires_grad=False | 16 | 121.019 us +/- 2.399 us, var 16.776 | 37.399 us +/- 1.304 us, var 5.559 | 3.24x | `torch_rs:9220741011861238656,9220741011861238656; PyTorch:18012629966419162080,18012629966419162080` |
| `skinny_contiguous_4096x8_by_8x32` | skinny contiguous | `left.matmul(right)` | left `(4096, 8)`, stride `(8, 1)`, offset 0; right `(8, 32)`, stride `(32, 1)`, offset 0 | `(4096, 32)`, stride `(32, 1)`, offset 0, requires_grad=False | 16 | 119.037 us +/- 0.970 us, var 6.116 | 36.046 us +/- 0.570 us, var 2.820 | 3.30x | `torch_rs:9220741011861238656,9220741011861238656; PyTorch:18012629966419162080,18012629966419162080` |
| `skinny_contiguous_4096x8_by_8x32` | skinny contiguous | `torch.matmul(left, right)` | left `(4096, 8)`, stride `(8, 1)`, offset 0; right `(8, 32)`, stride `(32, 1)`, offset 0 | `(4096, 32)`, stride `(32, 1)`, offset 0, requires_grad=False | 16 | 118.290 us +/- 0.684 us, var 4.134 | 35.840 us +/- 0.484 us, var 0.800 | 3.30x | `torch_rs:9220741011861238656,9220741011861238656; PyTorch:18012629966419162080,18012629966419162080` |
| `empty_rows_0x257_by_257x263` | empty dimension | `left @ right` | left `(0, 257)`, stride `(257, 1)`, offset 0; right `(257, 263)`, stride `(263, 1)`, offset 0 | `(0, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 5000 | 0.231 us +/- 0.002 us, var 0.000 | 0.931 us +/- 0.005 us, var 0.001 | 0.25x | `torch_rs:616029249794733280,616029249794733280; PyTorch:616029249794733280,616029249794733280` |
| `empty_rows_0x257_by_257x263` | empty dimension | `left.matmul(right)` | left `(0, 257)`, stride `(257, 1)`, offset 0; right `(257, 263)`, stride `(263, 1)`, offset 0 | `(0, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 5000 | 0.276 us +/- 0.002 us, var 0.001 | 0.936 us +/- 0.005 us, var 0.000 | 0.30x | `torch_rs:616029249794733280,616029249794733280; PyTorch:616029249794733280,616029249794733280` |
| `empty_rows_0x257_by_257x263` | empty dimension | `torch.matmul(left, right)` | left `(0, 257)`, stride `(257, 1)`, offset 0; right `(257, 263)`, stride `(263, 1)`, offset 0 | `(0, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 5000 | 0.352 us +/- 0.003 us, var 0.000 | 0.807 us +/- 0.006 us, var 0.000 | 0.44x | `torch_rs:616029249794733280,616029249794733280; PyTorch:616029249794733280,616029249794733280` |
| `empty_columns_257x263_by_263x0` | empty dimension | `left @ right` | left `(257, 263)`, stride `(263, 1)`, offset 0; right `(263, 0)`, stride `(1, 1)`, offset 0 | `(257, 0)`, stride `(1, 1)`, offset 0, requires_grad=False | 5000 | 0.233 us +/- 0.003 us, var 0.000 | 0.931 us +/- 0.006 us, var 0.004 | 0.25x | `torch_rs:15538049934065561088,15538049934065561088; PyTorch:15538049934065561088,15538049934065561088` |
| `empty_columns_257x263_by_263x0` | empty dimension | `left.matmul(right)` | left `(257, 263)`, stride `(263, 1)`, offset 0; right `(263, 0)`, stride `(1, 1)`, offset 0 | `(257, 0)`, stride `(1, 1)`, offset 0, requires_grad=False | 5000 | 0.276 us +/- 0.002 us, var 0.000 | 0.933 us +/- 0.007 us, var 0.001 | 0.30x | `torch_rs:15538049934065561088,15538049934065561088; PyTorch:15538049934065561088,15538049934065561088` |
| `empty_columns_257x263_by_263x0` | empty dimension | `torch.matmul(left, right)` | left `(257, 263)`, stride `(263, 1)`, offset 0; right `(263, 0)`, stride `(1, 1)`, offset 0 | `(257, 0)`, stride `(1, 1)`, offset 0, requires_grad=False | 5000 | 0.352 us +/- 0.003 us, var 0.000 | 0.810 us +/- 0.004 us, var 0.003 | 0.44x | `torch_rs:15538049934065561088,15538049934065561088; PyTorch:15538049934065561088,15538049934065561088` |
| `empty_inner_257x0_by_0x263` | empty inner dimension | `left @ right` | left `(257, 0)`, stride `(1, 1)`, offset 0; right `(0, 263)`, stride `(263, 1)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 128 | 2.842 us +/- 0.030 us, var 0.017 | 4.600 us +/- 0.078 us, var 0.086 | 0.62x | `torch_rs:4500670932617572096,4500670932617572096; PyTorch:4500670932617572096,4500670932617572096` |
| `empty_inner_257x0_by_0x263` | empty inner dimension | `left.matmul(right)` | left `(257, 0)`, stride `(1, 1)`, offset 0; right `(0, 263)`, stride `(263, 1)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 128 | 2.991 us +/- 0.042 us, var 0.009 | 4.344 us +/- 0.053 us, var 0.013 | 0.69x | `torch_rs:4500670932617572096,4500670932617572096; PyTorch:4500670932617572096,4500670932617572096` |
| `empty_inner_257x0_by_0x263` | empty inner dimension | `torch.matmul(left, right)` | left `(257, 0)`, stride `(1, 1)`, offset 0; right `(0, 263)`, stride `(263, 1)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 128 | 3.011 us +/- 0.055 us, var 0.050 | 4.475 us +/- 0.052 us, var 0.064 | 0.67x | `torch_rs:4500670932617572096,4500670932617572096; PyTorch:4500670932617572096,4500670932617572096` |
| `offset_contiguous_257x129_by_129x263` | offset contiguous | `left @ right` | left `(257, 129)`, stride `(129, 1)`, offset 33153; right `(129, 263)`, stride `(263, 1)`, offset 67854 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 3 | 919.447 us +/- 3.526 us, var 376.420 | 206.850 us +/- 1.806 us, var 25.219 | 4.45x | `torch_rs:4891835137536927616,4891835137536927616; PyTorch:10550242359678505440,10550242359678505440` |
| `offset_contiguous_257x129_by_129x263` | offset contiguous | `left.matmul(right)` | left `(257, 129)`, stride `(129, 1)`, offset 33153; right `(129, 263)`, stride `(263, 1)`, offset 67854 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 3 | 913.704 us +/- 2.669 us, var 69.237 | 206.941 us +/- 2.293 us, var 10.510 | 4.42x | `torch_rs:4891835137536927616,4891835137536927616; PyTorch:10550242359678505440,10550242359678505440` |
| `offset_contiguous_257x129_by_129x263` | offset contiguous | `torch.matmul(left, right)` | left `(257, 129)`, stride `(129, 1)`, offset 33153; right `(129, 263)`, stride `(263, 1)`, offset 67854 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 3 | 914.862 us +/- 3.540 us, var 1029.760 | 207.077 us +/- 1.926 us, var 25.127 | 4.42x | `torch_rs:4891835137536927616,4891835137536927616; PyTorch:10550242359678505440,10550242359678505440` |
| `noncontig_transposed_257x129_by_129x263` | noncontiguous transpose | `left @ right` | left `(257, 129)`, stride `(1, 257)`, offset 0; right `(129, 263)`, stride `(1, 129)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 2 | 5447.605 us +/- 31.943 us, var 172411.786 | 206.575 us +/- 2.764 us, var 17.457 | 26.37x | `torch_rs:13496987231262377728,13496987231262377728; PyTorch:3936886919654069216,3936886919654069216` |
| `noncontig_transposed_257x129_by_129x263` | noncontiguous transpose | `left.matmul(right)` | left `(257, 129)`, stride `(1, 257)`, offset 0; right `(129, 263)`, stride `(1, 129)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 2 | 5446.273 us +/- 29.096 us, var 147374.767 | 206.497 us +/- 2.749 us, var 55.051 | 26.37x | `torch_rs:13496987231262377728,13496987231262377728; PyTorch:3936886919654069216,3936886919654069216` |
| `noncontig_transposed_257x129_by_129x263` | noncontiguous transpose | `torch.matmul(left, right)` | left `(257, 129)`, stride `(1, 257)`, offset 0; right `(129, 263)`, stride `(1, 129)`, offset 0 | `(257, 263)`, stride `(263, 1)`, offset 0, requires_grad=False | 2 | 5466.309 us +/- 50.804 us, var 784091.528 | 209.164 us +/- 4.842 us, var 91.246 | 26.13x | `torch_rs:13496987231262377728,13496987231262377728; PyTorch:3936886919654069216,3936886919654069216` |
| `no_grad_requires_grad_128` | no_grad | `left @ right` | left `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=True; right `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=True; operation inside `no_grad` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 8 | 150.121 us +/- 1.110 us, var 35.116 | 46.443 us +/- 0.334 us, var 1.280 | 3.23x | `torch_rs:15229858161126193088,15229858161126193088; PyTorch:8844262383134964672,8844262383134964672` |
| `no_grad_requires_grad_128` | no_grad | `left.matmul(right)` | left `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=True; right `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=True; operation inside `no_grad` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 8 | 149.714 us +/- 0.996 us, var 43.363 | 46.555 us +/- 0.383 us, var 1.448 | 3.22x | `torch_rs:15229858161126193088,15229858161126193088; PyTorch:8844262383134964672,8844262383134964672` |
| `no_grad_requires_grad_128` | no_grad | `torch.matmul(left, right)` | left `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=True; right `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=True; operation inside `no_grad` | `(128, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 8 | 150.228 us +/- 1.346 us, var 40.327 | 47.238 us +/- 1.111 us, var 4.551 | 3.18x | `torch_rs:15229858161126193088,15229858161126193088; PyTorch:8844262383134964672,8844262383134964672` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `@_backward_full_sum_16x17_by_17x19` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported loss shape `()`, left grad `(16, 17)`, right grad `(17, 19)` | zero |
| `Tensor.matmul_backward_full_sum_16x17_by_17x19` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported loss shape `()`, left grad `(16, 17)`, right grad `(17, 19)` | zero |
| `torch.matmul_backward_full_sum_16x17_by_17x19` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported loss shape `()`, left grad `(16, 17)`, right grad `(17, 19)` | zero |
| `operator_rank1_dot_17` | `RuntimeError: matmul currently requires two rank-2 tensors, got [17] and [17]` | supported output `()` | zero |
| `tensor_matmul_rank3_by_rank2` | `RuntimeError: matmul currently requires two rank-2 tensors, got [2, 3, 4] and [4, 5]` | supported output `(2, 3, 5)` | zero |
| `torch_matmul_out_keyword` | `TypeError: matmul() got an unexpected keyword argument 'out'` | supported output `(2, 2)` | zero |
