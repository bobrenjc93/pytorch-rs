# `Tensor.mean` and `torch.mean` Full-Reduction Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot based on
`bef83cb588242ec6e8010540fb4368906fca5ffd`. This branch adds timing evidence
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
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
wheel_dir="$(mktemp -d "$PWD/target/tensor-mean-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/tensor-mean-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_tensor_mean tests.test_tensor_mean_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_mean_release_timings.py \
  > target/tensor-mean-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
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
--check` passed, the README/docs smoke test passed, and `git diff --check`
passed.

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
- Release build/install mode: `maturin build --release --locked --offline`
  wheel, force-reinstalled into the worktree `.venv` with `uv pip install
  --force-reinstall --no-deps`
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  packages in 17.63s, and installed in 1.11s
- Build time: successful offline release extension build completed in 37.52s
  wall time; Cargo reported 37.33s in the release profile. The release wheel
  reinstall resolved in 1 ms, prepared in 41 ms, installed in 15 ms, and took
  0.18s wall time.

Inputs were created outside the timed region with NumPy seed `20260902`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported cell, the driver compared `torch_rs` output
values with PyTorch using `rtol=1e-5`, `atol=1e-6`, and `equal_nan=True`, then
checked shape, stride, storage offset, contiguity, dtype, device,
`requires_grad`, and leaf status. After every warmup and measured block, it
consumed the last output as a 64-bit BLAKE2b rolling checksum over tensor
metadata and logical bytes. The checksum column shows the final rolling sink
from one pass as `torch_rs`/PyTorch; both process passes produced the same sink
pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.
The method rows timed `source.mean(dim=None, keepdim=..., dtype=module.float32)`;
the top-level rows timed
`module.mean(source, dim=None, keepdim=..., dtype=module.float32, out=None)`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.97x uncapped, 2.18x capped
- `Tensor.mean` cells: 2.69x uncapped, 1.98x capped
- `torch.mean` cells: 3.27x uncapped, 2.40x capped
- Scalar cells: 0.21x uncapped, 0.21x capped
- Scalar `keepdim=True` cells: 0.24x uncapped, 0.24x capped
- Empty cells: 0.21x uncapped, 0.21x capped
- Empty `keepdim=True` cells: 0.24x uncapped, 0.24x capped
- Contiguous cells: 9.36x uncapped, 9.36x capped
- Contiguous `keepdim=True` cells: 9.51x uncapped, 9.51x capped
- Offset cells: 12.24x uncapped, 10.00x capped
- Offset `keepdim=True` cells: 12.27x uncapped, 10.00x capped
- Noncontiguous cells: 37.71x uncapped, 10.00x capped
- Noncontiguous `keepdim=True` cells: 38.62x uncapped, 10.00x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 3.10x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar` | scalar | `Tensor.mean` | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.598 us +/- 0.004 us, var 0.000 | 3.540 us +/- 0.025 us, var 0.159 | 0.17x | `12760630757868178341`/`12760630757868178341` |
| `scalar` | scalar | `torch.mean` | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.932 us +/- 0.006 us, var 0.000 | 3.429 us +/- 0.015 us, var 0.010 | 0.27x | `12760630757868178341`/`12760630757868178341` |
| `scalar_keepdim` | scalar keepdim | `Tensor.mean` | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.655 us +/- 0.003 us, var 0.000 | 3.359 us +/- 0.020 us, var 0.034 | 0.19x | `12760630757868178341`/`12760630757868178341` |
| `scalar_keepdim` | scalar keepdim | `torch.mean` | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.990 us +/- 0.005 us, var 0.000 | 3.276 us +/- 0.022 us, var 0.114 | 0.30x | `12760630757868178341`/`12760630757868178341` |
| `empty_transposed_3x0x2` | empty | `Tensor.mean` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 5000 | 0.597 us +/- 0.007 us, var 0.006 | 3.675 us +/- 0.079 us, var 0.042 | 0.16x | `11155667488977603928`/`11155667488977603928` |
| `empty_transposed_3x0x2` | empty | `torch.mean` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 5000 | 0.986 us +/- 0.006 us, var 0.001 | 3.506 us +/- 0.064 us, var 0.007 | 0.28x | `11155667488977603928`/`11155667488977603928` |
| `empty_transposed_3x0x2_keepdim` | empty keepdim | `Tensor.mean` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (1, 1, 1), stride (1, 1, 1), offset 0, requires_grad=False | 5000 | 0.714 us +/- 0.007 us, var 0.000 | 3.656 us +/- 0.076 us, var 0.011 | 0.20x | `692090448826161100`/`692090448826161100` |
| `empty_transposed_3x0x2_keepdim` | empty keepdim | `torch.mean` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (1, 1, 1), stride (1, 1, 1), offset 0, requires_grad=False | 5000 | 1.078 us +/- 0.008 us, var 0.002 | 3.527 us +/- 0.066 us, var 0.009 | 0.31x | `692090448826161100`/`692090448826161100` |
| `contiguous_257x263` | contiguous | `Tensor.mean` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 32 | 56.346 us +/- 0.107 us, var 0.164 | 6.087 us +/- 0.053 us, var 0.457 | 9.26x | `8461811684698743521`/`9259035234526454357` |
| `contiguous_257x263` | contiguous | `torch.mean` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 32 | 56.666 us +/- 0.107 us, var 0.150 | 5.994 us +/- 0.046 us, var 0.055 | 9.45x | `8461811684698743521`/`9259035234526454357` |
| `contiguous_257x263_keepdim` | contiguous keepdim | `Tensor.mean` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (1, 1), stride (1, 1), offset 0, requires_grad=False | 32 | 56.498 us +/- 0.150 us, var 0.175 | 5.997 us +/- 0.035 us, var 0.086 | 9.42x | `329852768945211398`/`6975144948838318628` |
| `contiguous_257x263_keepdim` | contiguous keepdim | `torch.mean` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (1, 1), stride (1, 1), offset 0, requires_grad=False | 32 | 56.929 us +/- 0.168 us, var 0.234 | 5.924 us +/- 0.042 us, var 0.039 | 9.61x | `329852768945211398`/`6975144948838318628` |
| `offset_contiguous_509x521` | offset | `Tensor.mean` | (509, 521), stride (521, 1), offset 265189, requires_grad=False | (), stride (), offset 0, requires_grad=False | 5 | 220.010 us +/- 0.697 us, var 5.230 | 18.093 us +/- 0.229 us, var 2.783 | 12.16x | `8410860385309216503`/`3839850855607379565` |
| `offset_contiguous_509x521` | offset | `torch.mean` | (509, 521), stride (521, 1), offset 265189, requires_grad=False | (), stride (), offset 0, requires_grad=False | 5 | 220.319 us +/- 0.516 us, var 4.502 | 17.879 us +/- 0.254 us, var 0.879 | 12.32x | `8410860385309216503`/`3839850855607379565` |
| `offset_contiguous_509x521_keepdim` | offset keepdim | `Tensor.mean` | (509, 521), stride (521, 1), offset 265189, requires_grad=False | (1, 1), stride (1, 1), offset 0, requires_grad=False | 5 | 219.768 us +/- 0.594 us, var 4.300 | 17.875 us +/- 0.243 us, var 0.863 | 12.29x | `17755586067310200311`/`10620088473552867507` |
| `offset_contiguous_509x521_keepdim` | offset keepdim | `torch.mean` | (509, 521), stride (521, 1), offset 265189, requires_grad=False | (1, 1), stride (1, 1), offset 0, requires_grad=False | 5 | 220.046 us +/- 0.401 us, var 5.417 | 17.956 us +/- 0.274 us, var 1.515 | 12.25x | `17755586067310200311`/`10620088473552867507` |
| `noncontig_transpose_512x1024` | noncontiguous | `Tensor.mean` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 5 | 1064.711 us +/- 3.381 us, var 140.782 | 28.285 us +/- 0.142 us, var 2.084 | 37.64x | `9224539793122911896`/`11540415007128187415` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.mean` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 5 | 1063.709 us +/- 2.828 us, var 35.431 | 28.161 us +/- 0.381 us, var 2.086 | 37.77x | `9224539793122911896`/`11540415007128187415` |
| `noncontig_transpose_512x1024_keepdim` | noncontiguous keepdim | `Tensor.mean` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (1, 1), stride (1, 1), offset 0, requires_grad=False | 5 | 1064.566 us +/- 3.305 us, var 49.397 | 27.564 us +/- 0.113 us, var 1.482 | 38.62x | `10913179105518255984`/`17668076794816289708` |
| `noncontig_transpose_512x1024_keepdim` | noncontiguous keepdim | `torch.mean` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (1, 1), stride (1, 1), offset 0, requires_grad=False | 5 | 1065.235 us +/- 2.867 us, var 40.376 | 27.588 us +/- 0.083 us, var 1.410 | 38.61x | `10913179105518255984`/`17668076794816289708` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `tensor_mean_dim0` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `(3,), stride (1,), offset 0, requires_grad=False` | zero |
| `tensor_mean_tuple_dim_keepdim` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `(1, 1), stride (1, 1), offset 0, requires_grad=False` | zero |
| `tensor_mean_dtype_float64` | `TypeError: mean() received an invalid combination of arguments - got (dtype=torch.dtype, ), but expected one of:` | supported float64 `(), stride (), offset 0, requires_grad=False` | zero |
| `top_level_torch_mean_dim0` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `(3,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_mean_out_tensor` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `(), stride (), offset 0, requires_grad=False` | zero |
| `top_level_torch_mean_dtype_float64` | `TypeError: mean() received an invalid combination of arguments - got (Tensor, dtype=torch.dtype), but expected one of:` | supported float64 `(), stride (), offset 0, requires_grad=False` | zero |
