# Creation Factory Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot based on
`123ad635c137da09765334b22b8dc29345dd4fa0`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/creation-factory-release-timings-pass*.json`. No
Conda environment was active in the shell (`CONDA_PREFIX=`), so setup used a
worktree-local `.venv`. Cargo registry data was copied read-only from the
existing user cache into `target/cargo-home`, then Cargo ran offline so build
artifacts and dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  /usr/bin/time -f 'elapsed=%e' uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  /usr/bin/time -f 'elapsed=%e' \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry
/usr/bin/time -f 'copy_elapsed=%e' \
  cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
wheel_dir="$(mktemp -d "$PWD/target/creation-factory-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/creation-factory-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  /usr/bin/time -f 'build_elapsed=%e' \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/creation-factory-wheel-dir.txt)"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  /usr/bin/time -f 'install_elapsed=%e' \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_zeros tests.test_zeros_reference \
  tests.test_ones tests.test_ones_reference \
  tests.test_full_reference \
  tests.test_zeros_like tests.test_zeros_like_reference \
  tests.test_ones_like tests.test_ones_like_reference \
  tests.test_full_like tests.test_full_like_reference \
  tests.test_as_tensor tests.test_as_tensor_reference \
  tests.test_asarray tests.test_asarray_reference \
  tests.test_arange tests.test_arange_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_python_api tests.test_tensor_buffer tests.test_tensor_buffer_reference \
  tests.test_eye tests.test_eye_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets full
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets eye
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/creation_factory_release_timings.py \
  > target/creation-factory-release-timings-pass1.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  CREATION_FACTORY_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/creation_factory_release_timings.py \
  > target/creation-factory-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_zeros tests.test_zeros_reference \
  tests.test_ones tests.test_ones_reference \
  tests.test_full_reference \
  tests.test_zeros_like tests.test_zeros_like_reference \
  tests.test_ones_like tests.test_ones_like_reference \
  tests.test_full_like tests.test_full_like_reference \
  tests.test_as_tensor tests.test_as_tensor_reference \
  tests.test_asarray tests.test_asarray_reference \
  tests.test_arange tests.test_arange_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_python_api tests.test_tensor_buffer tests.test_tensor_buffer_reference \
  tests.test_eye tests.test_eye_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets full
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets eye
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python creation implementation and PyTorch 2.13
differential tests passed 156 tests with 2 skips. The supplemental constructor
and `eye` tests passed 111 tests. `cargo fmt --check` passed. The filtered
native Rust tests passed 7 `full`-matched tests, including one unrelated
`full_slice` name match, and 3 `eye` tests. The README/docs smoke test passed
7 tests, and `git diff --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core; 384 logical CPUs online
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
- Dependency installation: `.venv` creation completed in 0.27s; locked
  `uv sync` resolved in 26 ms, prepared packages in 16.21s, installed in
  1.35s, and completed in 17.72s wall time; Cargo registry cache copy into
  `target/cargo-home` completed in 0.15s
- Build/install time: the successful offline release extension build completed
  in 36.76s wall time; the release wheel reinstall resolved in 1 ms, prepared
  in 42 ms, installed in 18 ms, and completed in 0.18s wall time

Inputs and source Python containers were created outside the timed region with
NumPy seed `20260902`. Each implementation used the same CPU `float32` values,
shapes, layouts, and thread settings. Every timing cell ran in two pinned
process passes. The first pass measured `torch_rs` before PyTorch; the second
pass reversed that order. Each pass used 15 untimed warmup blocks and 81
measured blocks. A block repeated the operation according to the table's
`Repeats` column; times below are median microseconds per operation. Reported
medians are medians of the two per-process medians. MAD and variance are the
medians of the per-process MAD and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch and checked shape, stride, storage offset, element count,
dtype, device, layout, contiguity, `requires_grad`, and leaf state. Identity
conversion cells also checked Python-object identity and storage aliasing.
Fresh-output cells checked distinct Python objects and, for nonempty outputs,
fresh storage across repeated calls; like-factory cells also checked that the
result did not alias the source. After every warmup and measured block, the
driver consumed the last output as a 64-bit BLAKE2b rolling checksum over tensor
metadata and logical float32 bit patterns. The checksum column shows the final
rolling sink from one pass as `torch_rs`/PyTorch; both process passes produced
the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.43x uncapped, 0.44x capped
- Requested factory/conversion/arange subset: 0.46x uncapped, 0.46x capped
- `torch.tensor` constructor cells: 0.16x uncapped, 0.24x capped
- `torch.zeros` cells: 0.43x uncapped, 0.43x capped
- `torch.ones` cells: 0.46x uncapped, 0.46x capped
- `torch.full` cells: 0.55x uncapped, 0.55x capped
- Like-factory cells: 0.42x uncapped, 0.42x capped
- Identity conversion cells: 0.54x uncapped, 0.54x capped
- Fresh conversion cells: 0.18x uncapped, 0.18x capped
- `torch.arange` cells: 0.93x uncapped, 0.93x capped
- `torch.eye` cells: 0.40x uncapped, 0.40x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 1.09x across all
48 listed cells.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `tensor_nested_sequence_64x65` | tensor constructor | `torch.tensor` | prebuilt exact Python float nested list, shape `(64, 65)` | `(64, 65), stride (65, 1), offset 0` | 128 | 107.391 us +/- 0.958, var 10.353 | 182.845 us +/- 1.274, var 6.986 | 0.59x | `6ceede5f3234b467`/`6ceede5f3234b467` |
| `tensor_memoryview_float32_65536` | tensor constructor | `torch.tensor` | prebuilt one-dimensional float32 memoryview, 65,536 elements | `(65536,), stride (1,), offset 0` | 64 | 171.908 us +/- 1.432, var 542.904 | 3819.066 us +/- 37.036, var 14607.404 | 0.05x | `8dd7f8b40afb6307`/`8dd7f8b40afb6307` |
| `zeros_scalar` | zeros | `torch.zeros` | shape `()` | `(), stride (), offset 0` | 20000 | 0.365 us +/- 0.004, var 0.002 | 1.420 us +/- 0.005, var 0.000 | 0.26x | `08a47b76a7a089eb`/`08a47b76a7a089eb` |
| `zeros_empty_2x0x3` | zeros | `torch.zeros` | shape `(2, 0, 3)` | `(2, 0, 3), stride (3, 3, 1), offset 0` | 20000 | 0.426 us +/- 0.004, var 0.000 | 1.682 us +/- 0.009, var 0.002 | 0.25x | `3b9923e6a6cca0bb`/`3b9923e6a6cca0bb` |
| `zeros_medium_257x263` | zeros | `torch.zeros` | shape `(257, 263)` | `(257, 263), stride (263, 1), offset 0` | 64 | 3.187 us +/- 0.117, var 0.058 | 4.983 us +/- 0.118, var 0.280 | 0.64x | `201b39ae29ac9323`/`201b39ae29ac9323` |
| `zeros_large_1024x1024` | zeros | `torch.zeros` | shape `(1024, 1024)` | `(1024, 1024), stride (1024, 1), offset 0` | 4 | 103.389 us +/- 10.044, var 580.625 | 122.709 us +/- 5.294, var 121.148 | 0.84x | `af12b872282d8822`/`af12b872282d8822` |
| `ones_scalar` | ones | `torch.ones` | shape `()` | `(), stride (), offset 0` | 20000 | 0.351 us +/- 0.002, var 0.000 | 1.471 us +/- 0.009, var 0.006 | 0.24x | `b801f8e14cd85de2`/`b801f8e14cd85de2` |
| `ones_empty_2x0x3` | ones | `torch.ones` | shape `(2, 0, 3)` | `(2, 0, 3), stride (3, 3, 1), offset 0` | 20000 | 0.427 us +/- 0.008, var 0.000 | 1.676 us +/- 0.021, var 0.003 | 0.25x | `3b9923e6a6cca0bb`/`3b9923e6a6cca0bb` |
| `ones_medium_257x263` | ones | `torch.ones` | shape `(257, 263)` | `(257, 263), stride (263, 1), offset 0` | 64 | 5.329 us +/- 0.086, var 0.095 | 5.129 us +/- 0.146, var 0.116 | 1.04x | `ef008b6623a8d4cc`/`ef008b6623a8d4cc` |
| `ones_large_1024x1024` | ones | `torch.ones` | shape `(1024, 1024)` | `(1024, 1024), stride (1024, 1), offset 0` | 4 | 114.186 us +/- 5.565, var 310.403 | 166.962 us +/- 17.077, var 7575.051 | 0.68x | `03ccb9ece667a3c3`/`03ccb9ece667a3c3` |
| `full_scalar_signed_zero` | full | `torch.full` | shape `()`, fill `-0.0` | `(), stride (), offset 0` | 20000 | 0.360 us +/- 0.005, var 0.003 | 1.184 us +/- 0.013, var 0.010 | 0.30x | `d0c8a207c6faa5fe`/`d0c8a207c6faa5fe` |
| `full_empty_2x0x3` | full | `torch.full` | shape `(2, 0, 3)`, fill `7.0` | `(2, 0, 3), stride (3, 3, 1), offset 0` | 20000 | 0.462 us +/- 0.003, var 0.001 | 1.399 us +/- 0.007, var 0.000 | 0.33x | `3b9923e6a6cca0bb`/`3b9923e6a6cca0bb` |
| `full_medium_257x263` | full | `torch.full` | shape `(257, 263)`, fill `1.25` | `(257, 263), stride (263, 1), offset 0` | 64 | 5.376 us +/- 0.101, var 0.811 | 4.812 us +/- 0.115, var 0.297 | 1.12x | `23e5760922a0d1a8`/`23e5760922a0d1a8` |
| `full_large_1024x1024` | full | `torch.full` | shape `(1024, 1024)`, fill `-2.5` | `(1024, 1024), stride (1024, 1), offset 0` | 4 | 122.739 us +/- 5.972, var 216.592 | 145.892 us +/- 6.022, var 181.574 | 0.84x | `91c23a1df7233e49`/`91c23a1df7233e49` |
| `zeros_like_scalar` | like factory | `torch.zeros_like` | source shape `()` | `(), stride (), offset 0` | 20000 | 0.279 us +/- 0.002, var 0.000 | 0.888 us +/- 0.006, var 0.006 | 0.31x | `08a47b76a7a089eb`/`08a47b76a7a089eb` |
| `zeros_like_empty_2x0x3` | like factory | `torch.zeros_like` | source shape `(2, 0, 3)` | `(2, 0, 3), stride (3, 3, 1), offset 0` | 20000 | 0.303 us +/- 0.002, var 0.001 | 1.067 us +/- 0.006, var 0.009 | 0.28x | `3b9923e6a6cca0bb`/`3b9923e6a6cca0bb` |
| `zeros_like_offset_257x263` | like factory | `torch.zeros_like` | source shape `(257, 263)`, storage offset `67591` | `(257, 263), stride (263, 1), offset 0` | 64 | 2.996 us +/- 0.022, var 0.055 | 4.493 us +/- 0.102, var 0.036 | 0.67x | `201b39ae29ac9323`/`201b39ae29ac9323` |
| `ones_like_scalar` | like factory | `torch.ones_like` | source shape `()` | `(), stride (), offset 0` | 20000 | 0.278 us +/- 0.002, var 0.000 | 0.849 us +/- 0.006, var 0.022 | 0.33x | `b801f8e14cd85de2`/`b801f8e14cd85de2` |
| `ones_like_empty_2x0x3` | like factory | `torch.ones_like` | source shape `(2, 0, 3)` | `(2, 0, 3), stride (3, 3, 1), offset 0` | 20000 | 0.301 us +/- 0.006, var 0.000 | 0.994 us +/- 0.010, var 0.006 | 0.30x | `3b9923e6a6cca0bb`/`3b9923e6a6cca0bb` |
| `ones_like_offset_257x263` | like factory | `torch.ones_like` | source shape `(257, 263)`, storage offset `67591` | `(257, 263), stride (263, 1), offset 0` | 64 | 5.305 us +/- 0.140, var 0.302 | 4.698 us +/- 0.222, var 0.722 | 1.13x | `ef008b6623a8d4cc`/`ef008b6623a8d4cc` |
| `full_like_scalar` | like factory | `torch.full_like` | source shape `()`, fill `-3.5` | `(), stride (), offset 0` | 20000 | 0.277 us +/- 0.001, var 0.000 | 1.118 us +/- 0.011, var 0.008 | 0.25x | `e11d5e6e13b86742`/`e11d5e6e13b86742` |
| `full_like_empty_2x0x3` | like factory | `torch.full_like` | source shape `(2, 0, 3)`, fill `inf` | `(2, 0, 3), stride (3, 3, 1), offset 0` | 20000 | 0.299 us +/- 0.002, var 0.000 | 1.243 us +/- 0.007, var 0.001 | 0.24x | `3b9923e6a6cca0bb`/`3b9923e6a6cca0bb` |
| `full_like_offset_257x263` | like factory | `torch.full_like` | source shape `(257, 263)`, storage offset `67591`, fill `1.25` | `(257, 263), stride (263, 1), offset 0` | 64 | 5.292 us +/- 0.115, var 0.703 | 4.972 us +/- 0.132, var 0.335 | 1.06x | `23e5760922a0d1a8`/`23e5760922a0d1a8` |
| `as_tensor_identity_strided_view` | conversion identity | `torch.as_tensor` | prebuilt noncontiguous native Tensor view, shape `(4, 3)`, stride `(5, 20)` | `(4, 3), stride (5, 20), offset 1` | 20000 | 0.095 us +/- 0.001, var 0.000 | 0.310 us +/- 0.002, var 0.000 | 0.31x | `febedc535065c815`/`febedc535065c815` |
| `as_tensor_float_scalar` | conversion create | `torch.as_tensor` | Python float `-3.25` | `(), stride (), offset 0` | 20000 | 0.196 us +/- 0.003, var 0.000 | 1.269 us +/- 0.013, var 0.045 | 0.15x | `8b8a2b2d6f854516`/`8b8a2b2d6f854516` |
| `as_tensor_sequence_64x65` | conversion create | `torch.as_tensor` | prebuilt exact Python float nested list, shape `(64, 65)` | `(64, 65), stride (65, 1), offset 0` | 128 | 103.500 us +/- 0.982, var 4.544 | 420.636 us +/- 2.887, var 41.207 | 0.25x | `6ceede5f3234b467`/`6ceede5f3234b467` |
| `asarray_identity_strided_view` | conversion identity | `torch.asarray` | prebuilt noncontiguous native Tensor view, shape `(4, 3)`, stride `(5, 20)`, `copy=False` | `(4, 3), stride (5, 20), offset 1` | 20000 | 1.753 us +/- 0.013, var 0.009 | 1.866 us +/- 0.012, var 0.003 | 0.94x | `febedc535065c815`/`febedc535065c815` |
| `asarray_float_scalar` | conversion create | `torch.asarray` | Python float `-3.25` | `(), stride (), offset 0` | 20000 | 0.202 us +/- 0.002, var 0.000 | 1.334 us +/- 0.009, var 0.008 | 0.15x | `8b8a2b2d6f854516`/`8b8a2b2d6f854516` |
| `arange_empty_float_end` | arange | `torch.arange` | `end=0.0` | `(0,), stride (1,), offset 0` | 20000 | 0.242 us +/- 0.001, var 0.000 | 1.166 us +/- 0.006, var 0.003 | 0.21x | `c01069d7ef55472f`/`c01069d7ef55472f` |
| `arange_float_65537` | arange | `torch.arange` | `end=65537.0` | `(65537,), stride (1,), offset 0` | 64 | 36.957 us +/- 0.212, var 0.677 | 24.245 us +/- 0.182, var 0.100 | 1.52x | `b31bc0b61cac2b96`/`b31bc0b61cac2b96` |
| `arange_numpy_float32_65537` | arange | `torch.arange` | `end=np.float32(65537.0), dtype=float32` | `(65537,), stride (1,), offset 0` | 64 | 38.250 us +/- 0.297, var 1.314 | 24.782 us +/- 0.187, var 0.245 | 1.54x | `b31bc0b61cac2b96`/`b31bc0b61cac2b96` |
| `arange_int_explicit_float32_65537` | arange | `torch.arange` | `end=65537, dtype=float32` | `(65537,), stride (1,), offset 0` | 64 | 37.171 us +/- 0.249, var 0.225 | 24.308 us +/- 0.181, var 0.913 | 1.53x | `b31bc0b61cac2b96`/`b31bc0b61cac2b96` |
| `eye_zero_rows_0x257` | eye | `torch.eye` | `n=0, m=257` | `(0, 257), stride (257, 1), offset 0` | 20000 | 0.314 us +/- 0.002, var 0.000 | 1.452 us +/- 0.009, var 0.001 | 0.22x | `c3ac04962f7b45ec`/`c3ac04962f7b45ec` |
| `eye_rectangular_513x257` | eye | `torch.eye` | `n=513, m=257` | `(513, 257), stride (257, 1), offset 0` | 32 | 6.911 us +/- 0.174, var 0.117 | 9.404 us +/- 0.204, var 0.315 | 0.73x | `8630afad27ccb234`/`8630afad27ccb234` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent CPU
float32 PyTorch operation. They are preserved as zero-credit cells instead of
being removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `empty_float32_257x263` | `AttributeError: module 'torch_rs' has no attribute 'empty'` | supported shape `(257, 263)`, dtype `torch.float32`, device `cpu` | zero |
| `empty_like_contiguous_257x263` | `AttributeError: module 'torch_rs' has no attribute 'empty_like'` | supported shape `(257, 263)`, dtype `torch.float32`, device `cpu` | zero |
| `zeros_out_float32_257x263` | `RuntimeError: zeros(): the 'out' argument is not supported` | supported shape `(257, 263)`, dtype `torch.float32`, device `cpu` | zero |
| `ones_out_float32_257x263` | `RuntimeError: ones(): the 'out' argument is not supported` | supported shape `(257, 263)`, dtype `torch.float32`, device `cpu` | zero |
| `full_out_float32_257x263` | `RuntimeError: full(): the 'out' argument is not supported` | supported shape `(257, 263)`, dtype `torch.float32`, device `cpu` | zero |
| `zeros_like_noncontiguous_3x2` | `NotImplementedError: zeros_like(): only exact native CPU float32 row-major contiguous Tensor inputs are supported` | supported shape `(3, 2)`, dtype `torch.float32`, device `cpu` | zero |
| `ones_like_noncontiguous_3x2` | `NotImplementedError: ones_like(): only exact native CPU float32 row-major contiguous Tensor inputs are supported` | supported shape `(3, 2)`, dtype `torch.float32`, device `cpu` | zero |
| `full_like_noncontiguous_3x2` | `NotImplementedError: full_like(): only exact native CPU float32 row-major contiguous Tensor inputs are supported` | supported shape `(3, 2)`, dtype `torch.float32`, device `cpu` | zero |
| `as_tensor_numpy_array_64x65` | `NotImplementedError: as_tensor(): only exact native CPU float32 Tensor inputs, Python float scalars, or exact list/tuple sequences of Python floats are supported; NumPy arrays/scalars, integer and boolean inference, and other conversions are not implemented` | supported shape `(64, 65)`, dtype `torch.float32`, device `cpu` | zero |
| `asarray_sequence_2` | `NotImplementedError: asarray(): only exact native CPU float32 Tensor inputs or Python float scalars are supported; Python sequences, NumPy arrays/scalars, and non-float scalar conversions are not implemented` | supported shape `(2,)`, dtype `torch.float32`, device `cpu` | zero |
| `asarray_tensor_copy_true_2x3` | `NotImplementedError: asarray(): copy=True requires a copy and is not supported` | supported shape `(2, 3)`, dtype `torch.float32`, device `cpu` | zero |
| `arange_start_end_float32` | `TypeError: arange(): start and step overloads are not supported; pass one exact Python float endpoint` | supported shape `(65537,)`, dtype `torch.float32`, device `cpu` | zero |
| `arange_start_end_step_float32` | `TypeError: arange(): start and step overloads are not supported; pass one exact Python float endpoint` | supported shape `(32769,)`, dtype `torch.float32`, device `cpu` | zero |
| `eye_out_float32_257x263` | `TypeError: eye() got an unexpected keyword argument 'out'` | supported shape `(257, 263)`, dtype `torch.float32`, device `cpu` | zero |
