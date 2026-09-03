# `Tensor.mean` and `torch.mean` Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot based on
`2443c632b1c7e2499c5b917d8eb3ebeb23f465b0`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-mean-release-timings*.json`. No Conda
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
mkdir -p target/cargo-home/registry && \
  cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
wheel_dir="$(mktemp -d "$PWD/target/tensor-mean-wheels.XXXXXX")" && \
  printf '%s\n' "$wheel_dir" > target/tensor-mean-wheel-dir.txt && \
  /usr/bin/time -f 'wall_time_seconds=%e' \
  env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
    CARGO_HOME="$PWD/target/cargo-home" \
    CARGO_TARGET_DIR="$PWD/target" \
    TMPDIR="$PWD/target" \
    VIRTUAL_ENV="$PWD/.venv" \
    PYO3_PYTHON="$PWD/.venv/bin/python" \
    .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/tensor-mean-wheel-dir.txt)" && \
  /usr/bin/time -f 'wall_time_seconds=%e' \
  env UV_CACHE_DIR="$PWD/target/uv-cache" \
    UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
    uv pip install --python "$PWD/.venv/bin/python" \
    --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_tensor_mean tests.test_tensor_mean_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_mean_release_timings.py \
  > target/tensor-mean-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  MEAN_TIMING_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/tensor_mean_release_timings.py \
  > target/tensor-mean-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets mean
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_tensor_mean tests.test_tensor_mean_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 14 tests. The focused Rust `mean` filter passed 1 test, `cargo fmt
--check` passed, the README/docs smoke test passed 7 tests, and `git diff
--check` passed.

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
- Dependency installation: locked `uv sync` resolved in 32 ms, prepared
  packages in 15.82s, and installed in 1.40s
- Build/install time: successful offline release extension build completed in
  36.62s wall time; the release wheel reinstall resolved in 1 ms, prepared in
  58 ms, installed in 12 ms, and took 0.19s wall time

Inputs were created outside the timed region with NumPy seed `20260902`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported cell, the driver checked output shape, stride,
storage offset, contiguity, dtype, device, `requires_grad`, and leaf status
against PyTorch, then checked values with `rtol=1e-5`, `atol=1e-6`, and
`equal_nan=True`. Scalar and empty cells were bit-identical; larger finite
reductions used the tolerance gate to allow equivalent float32 reduction-order
differences. After every warmup and measured block, the driver consumed the
last output as a 64-bit BLAKE2b rolling checksum over tensor metadata and
logical bytes. The checksum column shows the final rolling sink from one pass
as `torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 3.01x uncapped, 2.21x capped
- `Tensor.mean` cells: 2.73x uncapped, 2.01x capped
- `torch.mean` cells: 3.31x uncapped, 2.44x capped
- `keepdim=False` cells: 2.93x uncapped, 2.16x capped
- `keepdim=True` cells: 3.09x uncapped, 2.27x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 2.85x.

| Workload | Category | API | Keepdim | Input | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar` | scalar | `Tensor.mean` | `False` | `(), stride (), offset 0, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 10000 | 0.624 us +/- 0.003 us, var 0.000 | 3.540 us +/- 0.018 us, var 0.003 | 0.18x | `15076791099251742393`/`15076791099251742393` |
| `scalar` | scalar | `torch.mean` | `False` | `(), stride (), offset 0, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 10000 | 1.014 us +/- 0.008 us, var 0.001 | 3.461 us +/- 0.016 us, var 0.023 | 0.29x | `15076791099251742393`/`15076791099251742393` |
| `scalar` | scalar | `Tensor.mean` | `True` | `(), stride (), offset 0, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 10000 | 0.680 us +/- 0.003 us, var 0.001 | 3.378 us +/- 0.020 us, var 0.005 | 0.20x | `15076791099251742393`/`15076791099251742393` |
| `scalar` | scalar | `torch.mean` | `True` | `(), stride (), offset 0, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 10000 | 1.072 us +/- 0.005 us, var 0.002 | 3.313 us +/- 0.019 us, var 0.003 | 0.32x | `15076791099251742393`/`15076791099251742393` |
| `empty` | empty | `Tensor.mean` | `False` | `(0, 2), stride (3, 3), offset 1, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 5000 | 0.627 us +/- 0.005 us, var 0.000 | 3.666 us +/- 0.078 us, var 0.022 | 0.17x | `7354914679711560234`/`7354914679711560234` |
| `empty` | empty | `torch.mean` | `False` | `(0, 2), stride (3, 3), offset 1, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 5000 | 1.005 us +/- 0.008 us, var 0.005 | 3.592 us +/- 0.076 us, var 0.021 | 0.28x | `7354914679711560234`/`7354914679711560234` |
| `empty` | empty | `Tensor.mean` | `True` | `(0, 2), stride (3, 3), offset 1, requires_grad=False` | `(1, 1), stride (1, 1), offset 0, requires_grad=False` | 5000 | 0.745 us +/- 0.004 us, var 0.000 | 3.641 us +/- 0.076 us, var 0.017 | 0.20x | `5254175592101868473`/`5254175592101868473` |
| `empty` | empty | `torch.mean` | `True` | `(0, 2), stride (3, 3), offset 1, requires_grad=False` | `(1, 1), stride (1, 1), offset 0, requires_grad=False` | 5000 | 1.127 us +/- 0.015 us, var 0.003 | 3.587 us +/- 0.088 us, var 0.156 | 0.31x | `5254175592101868473`/`5254175592101868473` |
| `contiguous_257x263` | contiguous | `Tensor.mean` | `False` | `(257, 263), stride (263, 1), offset 0, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 32 | 56.432 us +/- 0.172 us, var 0.200 | 6.089 us +/- 0.048 us, var 0.057 | 9.27x | `9421408590394125848`/`10654781370006770425` |
| `contiguous_257x263` | contiguous | `torch.mean` | `False` | `(257, 263), stride (263, 1), offset 0, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 32 | 56.739 us +/- 0.115 us, var 0.130 | 5.995 us +/- 0.031 us, var 0.036 | 9.46x | `9421408590394125848`/`10654781370006770425` |
| `contiguous_257x263` | contiguous | `Tensor.mean` | `True` | `(257, 263), stride (263, 1), offset 0, requires_grad=False` | `(1, 1), stride (1, 1), offset 0, requires_grad=False` | 32 | 56.474 us +/- 0.127 us, var 0.131 | 6.102 us +/- 0.078 us, var 0.178 | 9.25x | `8042894671230192062`/`14522807788861939360` |
| `contiguous_257x263` | contiguous | `torch.mean` | `True` | `(257, 263), stride (263, 1), offset 0, requires_grad=False` | `(1, 1), stride (1, 1), offset 0, requires_grad=False` | 32 | 56.956 us +/- 0.158 us, var 0.173 | 6.030 us +/- 0.071 us, var 0.123 | 9.45x | `8042894671230192062`/`14522807788861939360` |
| `offset_transposed_521x509` | offset | `Tensor.mean` | `False` | `(521, 509), stride (1, 521), offset 265189, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 5 | 220.453 us +/- 0.750 us, var 19.149 | 17.846 us +/- 0.310 us, var 1.304 | 12.35x | `1409585842495144373`/`5916078585191094536` |
| `offset_transposed_521x509` | offset | `torch.mean` | `False` | `(521, 509), stride (1, 521), offset 265189, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 5 | 220.954 us +/- 0.758 us, var 31.521 | 18.056 us +/- 0.280 us, var 1.131 | 12.24x | `1409585842495144373`/`5916078585191094536` |
| `offset_transposed_521x509` | offset | `Tensor.mean` | `True` | `(521, 509), stride (1, 521), offset 265189, requires_grad=False` | `(1, 1), stride (1, 1), offset 0, requires_grad=False` | 5 | 220.607 us +/- 0.761 us, var 101.547 | 17.788 us +/- 0.157 us, var 0.683 | 12.40x | `10028973358795473437`/`13275932074158337198` |
| `offset_transposed_521x509` | offset | `torch.mean` | `True` | `(521, 509), stride (1, 521), offset 265189, requires_grad=False` | `(1, 1), stride (1, 1), offset 0, requires_grad=False` | 5 | 220.787 us +/- 0.482 us, var 69.679 | 17.851 us +/- 0.201 us, var 1.617 | 12.37x | `10028973358795473437`/`13275932074158337198` |
| `noncontig_transpose_512x1024` | noncontiguous | `Tensor.mean` | `False` | `(512, 1024), stride (1, 512), offset 0, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 5 | 1052.679 us +/- 3.000 us, var 199.473 | 27.627 us +/- 0.116 us, var 1.161 | 38.10x | `608022005218124146`/`15047293425289646882` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.mean` | `False` | `(512, 1024), stride (1, 512), offset 0, requires_grad=False` | `(), stride (), offset 0, requires_grad=False` | 5 | 1052.935 us +/- 3.379 us, var 41.701 | 28.122 us +/- 0.157 us, var 1.855 | 37.44x | `608022005218124146`/`15047293425289646882` |
| `noncontig_transpose_512x1024` | noncontiguous | `Tensor.mean` | `True` | `(512, 1024), stride (1, 512), offset 0, requires_grad=False` | `(1, 1), stride (1, 1), offset 0, requires_grad=False` | 5 | 1049.156 us +/- 2.810 us, var 28.831 | 27.985 us +/- 0.115 us, var 2.303 | 37.49x | `7147316750816051310`/`10279209046983331364` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.mean` | `True` | `(512, 1024), stride (1, 512), offset 0, requires_grad=False` | `(1, 1), stride (1, 1), offset 0, requires_grad=False` | 5 | 1050.427 us +/- 3.228 us, var 91.202 | 27.940 us +/- 0.137 us, var 1.825 | 37.60x | `7147316750816051310`/`10279209046983331364` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `tensor_mean_dim0` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `(3,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_mean_dim_tuple` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `(), stride (), offset 0, requires_grad=False` | zero |
| `tensor_mean_dtype_float64` | `TypeError: mean() received an invalid combination of arguments - got (dtype=torch.dtype, ), but expected one of: * (*, torch.dtype dtype = None) * (tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None)` | supported `(), stride (), offset 0, requires_grad=False` | zero |
| `top_level_torch_mean_out_tensor` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `(), stride (), offset 0, requires_grad=False` | zero |
