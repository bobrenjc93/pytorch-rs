# Full-Reduction `keepdim=True` Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot based on
`8d32a1e9b5105cb3da27fe6b2aef40347a41340d`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/full-reduction-keepdim-release-timings*.json`. No
Conda environment was active in the shell (`CONDA_PREFIX=`), so setup used a
worktree-local `.venv`. Cargo registry data was copied read-only from the
existing user cache into `target/cargo-home`, then Cargo ran offline so build
artifacts and dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  /usr/bin/time -p uv sync --locked --no-install-project \
  --group dev --group reference
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
wheel_dir="$(mktemp -d "$PWD/target/full-reduction-keepdim-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/full-reduction-keepdim-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  /usr/bin/time -p .venv/bin/maturin build --release --locked --offline \
  --out "$wheel_dir"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  /usr/bin/time -p uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_tensor_sum \
  tests.test_tensor_sum_reference tests.test_tensor_mean \
  tests.test_tensor_mean_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets sum
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets mean
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python \
  target/full_reduction_keepdim_release_timings.py \
  > target/full-reduction-keepdim-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python \
  target/full_reduction_keepdim_release_timings.py \
  > target/full-reduction-keepdim-release-timings-pass2.json
git diff --check
```

Checks run for this evidence:

- Focused Python implementation and PyTorch 2.13 differential tests passed 37
  tests across `tests.test_tensor_sum`, `tests.test_tensor_sum_reference`,
  `tests.test_tensor_mean`, and `tests.test_tensor_mean_reference`.
- `cargo fmt --check` passed.
- `cargo test --locked --offline --all-targets sum` passed 35 filtered Rust
  tests.
- `cargo test --locked --offline --all-targets mean` passed 1 filtered Rust
  test.
- The README/docs smoke test passed 7 tests.
- `git diff --check` passed after the documentation edits.

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
- Dependency installation: locked `uv sync` resolved in 28 ms, prepared
  packages in 15.55s, and installed in 0.904s
- Build time: successful offline release extension build completed in 35.98s;
  the release wheel reinstall resolved in 2 ms, prepared in 48 ms, and
  installed in 17 ms

Inputs were created outside the timed region with NumPy seed `20260902`. Each
implementation used the same CPU `float32` values, shapes, layouts, grad mode,
and thread settings. Every timing cell ran in two pinned process passes. Each
pass used 15 untimed warmup blocks and 81 measured blocks. A block repeated the
operation according to the table's `Repeats` column; times below are median
microseconds per operation. Reported medians are medians of the two per-process
medians. MAD and variance are medians of the two per-process MAD and sample
variance values.

Before timing each supported forward cell, the driver compared `torch_rs`
output values with PyTorch using `rtol=1e-5`, `atol=1e-6`, and
`equal_nan=True`, checked signed-zero bits where zeros were present, and
checked shape, stride, storage offset, contiguity, dtype, device,
`requires_grad`, and leaf status. Before timing each backward cell, it checked
the rank-preserving `keepdim=True` output metadata and value, the final scalar
loss metadata and value, and the leaf gradient values and metadata. Backward
timings used fresh leaf/view inputs for every measured invocation, constructed
outside the timed interval for that block. After every warmup and measured
block, the driver consumed the last forward output or the last backward scalar
loss plus leaf gradient as a byte-level rolling checksum. The checksum column
shows the final rolling sink from one pass as `torch_rs`/PyTorch; both process
passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.36x uncapped, 1.84x capped
- `Tensor.sum` cells: 2.77x uncapped, 2.03x capped
- `torch.sum` cells: 3.23x uncapped, 2.37x capped
- `Tensor.mean` cells: 1.75x uncapped, 1.44x capped
- `torch.mean` cells: 1.99x uncapped, 1.65x capped
- Scalar cells: 0.31x uncapped, 0.31x capped
- Empty cells: 0.31x uncapped, 0.31x capped
- Contiguous cells: 11.63x uncapped, 9.69x capped
- Offset cells: 11.34x uncapped, 9.58x capped
- Non-contiguous cells: 40.59x uncapped, 10.00x capped
- `no_grad` cells: 4.76x uncapped, 4.76x capped
- Backward-through-final-scalar cells: 0.17x uncapped, 0.17x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 2.68x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar` | scalar | `Tensor.sum` | (), stride (), offset 0, requires_grad=False | (), stride (), requires_grad=False | 10000 | 0.498 us +/- 0.003 us, var 0.001 | 1.481 us +/- 0.008 us, var 0.001 | 0.34x | `8506298670402141664`/`8506298670402141664` |
| `scalar` | scalar | `torch.sum` | (), stride (), offset 0, requires_grad=False | (), stride (), requires_grad=False | 10000 | 0.774 us +/- 0.005 us, var 0.003 | 1.392 us +/- 0.013 us, var 0.006 | 0.56x | `8506298670402141664`/`8506298670402141664` |
| `scalar` | scalar | `Tensor.mean` | (), stride (), offset 0, requires_grad=False | (), stride (), requires_grad=False | 10000 | 0.596 us +/- 0.004 us, var 0.001 | 3.395 us +/- 0.039 us, var 0.008 | 0.18x | `8506298670402141664`/`8506298670402141664` |
| `scalar` | scalar | `torch.mean` | (), stride (), offset 0, requires_grad=False | (), stride (), requires_grad=False | 10000 | 0.884 us +/- 0.006 us, var 0.001 | 3.194 us +/- 0.017 us, var 0.003 | 0.28x | `8506298670402141664`/`8506298670402141664` |
| `empty` | empty | `Tensor.sum` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (1, 1, 1), stride (1, 1, 1), requires_grad=False | 5000 | 0.548 us +/- 0.003 us, var 0.000 | 1.536 us +/- 0.008 us, var 0.007 | 0.36x | `16554107875731086272`/`16554107875731086272` |
| `empty` | empty | `torch.sum` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (1, 1, 1), stride (1, 1, 1), requires_grad=False | 5000 | 0.830 us +/- 0.005 us, var 0.004 | 1.479 us +/- 0.015 us, var 0.005 | 0.56x | `16554107875731086272`/`16554107875731086272` |
| `empty` | empty | `Tensor.mean` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (1, 1, 1), stride (1, 1, 1), requires_grad=False | 5000 | 0.640 us +/- 0.006 us, var 0.001 | 3.627 us +/- 0.023 us, var 0.029 | 0.18x | `12192658935001407200`/`12192658935001407200` |
| `empty` | empty | `torch.mean` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (1, 1, 1), stride (1, 1, 1), requires_grad=False | 5000 | 0.922 us +/- 0.005 us, var 0.000 | 3.516 us +/- 0.018 us, var 0.016 | 0.26x | `12192658935001407200`/`12192658935001407200` |
| `contiguous_257x263` | contiguous | `Tensor.sum` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 32 | 56.391 us +/- 0.140 us, var 0.168 | 3.966 us +/- 0.013 us, var 0.022 | 14.22x | `13177377812645297664`/`13177377812645297664` |
| `contiguous_257x263` | contiguous | `torch.sum` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 32 | 56.729 us +/- 0.153 us, var 0.616 | 3.883 us +/- 0.015 us, var 0.032 | 14.61x | `13177377812645297664`/`13177377812645297664` |
| `contiguous_257x263` | contiguous | `Tensor.mean` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 32 | 56.494 us +/- 0.143 us, var 46.454 | 6.004 us +/- 0.035 us, var 0.061 | 9.41x | `8813841727591912288`/`8813841727591912288` |
| `contiguous_257x263` | contiguous | `torch.mean` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 32 | 56.723 us +/- 0.119 us, var 0.401 | 6.066 us +/- 0.090 us, var 0.139 | 9.35x | `8813841727591912288`/`8813841727591912288` |
| `offset_251x257` | offset | `Tensor.sum` | (251, 257), stride (257, 1), offset 64507, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 32 | 53.793 us +/- 0.155 us, var 0.143 | 3.908 us +/- 0.017 us, var 0.051 | 13.76x | `9774034036901710816`/`9774034036901710816` |
| `offset_251x257` | offset | `torch.sum` | (251, 257), stride (257, 1), offset 64507, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 32 | 54.125 us +/- 0.153 us, var 0.218 | 3.798 us +/- 0.017 us, var 0.023 | 14.25x | `9774034036901710816`/`9774034036901710816` |
| `offset_251x257` | offset | `Tensor.mean` | (251, 257), stride (257, 1), offset 64507, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 32 | 54.176 us +/- 0.409 us, var 8.009 | 5.967 us +/- 0.031 us, var 0.034 | 9.08x | `1594572519236815072`/`1594572519236815072` |
| `offset_251x257` | offset | `torch.mean` | (251, 257), stride (257, 1), offset 64507, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 32 | 54.256 us +/- 0.157 us, var 0.249 | 5.839 us +/- 0.039 us, var 0.087 | 9.29x | `1594572519236815072`/`1594572519236815072` |
| `noncontig_transpose_512x1024` | noncontiguous | `Tensor.sum` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 5 | 1083.512 us +/- 3.337 us, var 3542.892 | 24.636 us +/- 0.330 us, var 1.386 | 43.98x | `3694799405524908800`/`3694799405524908800` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.sum` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 5 | 1059.947 us +/- 3.906 us, var 646.354 | 25.413 us +/- 0.743 us, var 14.081 | 41.71x | `3694799405524908800`/`3694799405524908800` |
| `noncontig_transpose_512x1024` | noncontiguous | `Tensor.mean` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 5 | 1087.535 us +/- 4.605 us, var 1587.449 | 27.386 us +/- 0.236 us, var 1.792 | 39.71x | `10735597623273256384`/`10735597623273256384` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.mean` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (1, 1), stride (1, 1), requires_grad=False | 5 | 1044.945 us +/- 14.834 us, var 2301.512 | 28.046 us +/- 0.098 us, var 1.051 | 37.26x | `10735597623273256384`/`10735597623273256384` |
| `no_grad_127x131` | no_grad | `Tensor.sum` | (127, 131), stride (131, 1), offset 0, requires_grad=True | (1, 1), stride (1, 1), requires_grad=False | 100 | 14.321 us +/- 0.073 us, var 0.089 | 2.252 us +/- 0.032 us, var 0.006 | 6.36x | `16174551241767787712`/`16174551241767787712` |
| `no_grad_127x131` | no_grad | `torch.sum` | (127, 131), stride (131, 1), offset 0, requires_grad=True | (1, 1), stride (1, 1), requires_grad=False | 100 | 14.629 us +/- 0.072 us, var 0.024 | 2.160 us +/- 0.024 us, var 0.015 | 6.77x | `16174551241767787712`/`16174551241767787712` |
| `no_grad_127x131` | no_grad | `Tensor.mean` | (127, 131), stride (131, 1), offset 0, requires_grad=True | (1, 1), stride (1, 1), requires_grad=False | 100 | 14.395 us +/- 0.074 us, var 0.016 | 4.295 us +/- 0.078 us, var 1.096 | 3.35x | `13589152873600316384`/`13589152873600316384` |
| `no_grad_127x131` | no_grad | `torch.mean` | (127, 131), stride (131, 1), offset 0, requires_grad=True | (1, 1), stride (1, 1), requires_grad=False | 100 | 14.723 us +/- 0.087 us, var 0.047 | 4.137 us +/- 0.058 us, var 0.805 | 3.56x | `13589152873600316384`/`13589152873600316384` |
| `backward_transpose_32x33` | backward-through-final-scalar | `Tensor.sum` | (33, 32), stride (1, 33), offset 0, requires_grad=True | kept (1, 1), stride (1, 1); loss (); grad (32, 33), stride (33, 1) | 10 | 5.502 us +/- 0.078 us, var 0.144 | 29.206 us +/- 0.591 us, var 5.027 | 0.19x | `9035479817622323648`/`9035479817622323648` |
| `backward_transpose_32x33` | backward-through-final-scalar | `torch.sum` | (33, 32), stride (1, 33), offset 0, requires_grad=True | kept (1, 1), stride (1, 1); loss (); grad (32, 33), stride (33, 1) | 10 | 5.781 us +/- 0.062 us, var 0.387 | 29.197 us +/- 0.588 us, var 1.566 | 0.20x | `9035479817622323648`/`9035479817622323648` |
| `backward_transpose_32x33` | backward-through-final-scalar | `Tensor.mean` | (33, 32), stride (1, 33), offset 0, requires_grad=True | kept (1, 1), stride (1, 1); loss (); grad (32, 33), stride (33, 1) | 10 | 5.743 us +/- 0.076 us, var 0.258 | 40.521 us +/- 1.089 us, var 4.102 | 0.14x | `5099509856607269312`/`5099509856607269312` |
| `backward_transpose_32x33` | backward-through-final-scalar | `torch.mean` | (33, 32), stride (1, 33), offset 0, requires_grad=True | kept (1, 1), stride (1, 1); loss (); grad (32, 33), stride (33, 1) | 10 | 6.069 us +/- 0.070 us, var 0.134 | 40.609 us +/- 0.963 us, var 4.515 | 0.15x | `5099509856607269312`/`5099509856607269312` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `tensor_sum_dim0_keepdim_true` | `TypeError: sum() received an invalid combination of arguments - got (int, bool)` | supported `tensor([[2., 2., 2.]])` | zero |
| `torch_sum_dim0_keepdim_true` | `NotImplementedError: sum(): only full reductions with dim=None support keepdim; dim and out reductions are not supported` | supported `tensor([[2., 2., 2.]])` | zero |
| `tensor_mean_dim0_keepdim_true` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `tensor([[1., 1., 1.]])` | zero |
| `torch_mean_dim0_keepdim_true` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `tensor([[1., 1., 1.]])` | zero |
| `torch_sum_keepdim_true_concrete_out` | `NotImplementedError: sum(): only full reductions with dim=None support keepdim; dim and out reductions are not supported` | supported `tensor([[6.]])` | zero |
| `torch_mean_keepdim_true_concrete_out` | `NotImplementedError: mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported` | supported `tensor([[1.]])` | zero |
| `tensor_sum_keepdim_true_float64_dtype` | no native `torch_rs.float64`; passing PyTorch's `float64` is rejected as a non-native dtype | supported `tensor([[6.]], dtype=torch.float64)` | zero |
| `tensor_mean_keepdim_true_float64_dtype` | no native `torch_rs.float64`; passing PyTorch's `float64` is rejected as a non-native dtype | supported `tensor([[1.]], dtype=torch.float64)` | zero |
