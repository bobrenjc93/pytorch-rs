# `torch.nn.functional.linear` Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot at
`31f650b2fdb62c1328c6f884d0b53ff483d282b8`. The working tree changes for
this report are documentation-only; the timed runtime code is that commit's
release build.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/functional-linear-release-timings-pass*.json`. No
Conda environment was active in the shell (`CONDA_PREFIX=`), so setup used the
worktree-local `.venv`. Cargo registry data was pre-populated into
`target/cargo-home` from an existing read-only local cache, and accepted Cargo
commands ran offline.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home
cp -a /home/bobren/.cargo/registry target/cargo-home/
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets linear
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  /usr/bin/time -p .venv/bin/maturin build --release --locked --offline \
  --out target/functional-linear-wheels
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps \
  target/functional-linear-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_linear \
  tests.test_nn_functional_linear_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo clippy --locked --offline --all-targets -- -D warnings
env PATH="$PWD/.venv/bin:/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/functional_linear_release_timings.py \
  > target/functional-linear-release-timings-pass1.json
env PATH="$PWD/.venv/bin:/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  FUNCTIONAL_LINEAR_TIMING_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/functional_linear_release_timings.py \
  > target/functional-linear-release-timings-pass2.json
```

Focused results: `cargo fmt --check` passed; `cargo test --locked --offline
--all-targets linear` built successfully and matched no native Rust tests; the
wheel-installed functional-linear Python implementation and PyTorch 2.13
differential tests passed 34 tests; `cargo clippy --locked --offline
--all-targets -- -D warnings` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0; CUDA disabled for this CPU timing
  run with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from the wheel-installed local release build under
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Release build/install mode: `maturin build --release --locked --offline`
  built `target/functional-linear-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl`;
  `uv pip install --force-reinstall --no-deps` installed that exact wheel into
  `.venv`
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  packages in 15.89s, and installed in 1.27s; the local wheel reinstall
  resolved in 2 ms, prepared in 59 ms, and installed in 15 ms
- Build time: the release extension build completed in 36.85s
  (`/usr/bin/time -p` real time 37.04s)

Inputs were generated with NumPy seed `20260903` and created outside the timed
region. Each implementation used equivalent CPU `float32` values, shapes,
layouts, bias arguments, grad mode, and thread settings. Every timing cell ran
in two pinned process passes: pass 1 measured `torch_rs` before PyTorch, and
pass 2 reversed that order. Each pass used 15 untimed warmup blocks and 81
measured blocks. A block repeated the operation according to the table's
`Repeats` column. Times below are median microseconds per operation; reported
medians are medians of the two per-process medians.

Before timing each supported cell, the driver checked output shape, stride,
storage offset, contiguity, dtype, device, `requires_grad`, leaf status, NaN
classifications, sign bits, and values against PyTorch with `rtol=2e-5`,
`atol=1e-5`, and `equal_nan=True`. After every warmup and measured block, the
driver consumed the last output as a BLAKE2b checksum over tensor metadata and
logical bytes. The checksum table reports those materialization digests for
both process passes.

## Supported Timed Cells

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

Geometric mean `torch_rs / PyTorch` slowdown:

- All supported cells: 1.39x uncapped, 1.39x capped
- Rank-1 cells: 1.53x uncapped, 1.51x capped
- Rank-2 cells: 1.61x uncapped, 1.61x capped
- Bias-free rank-3 cells: 1.07x uncapped, 1.07x capped
- Non-empty contiguous cells: 5.49x uncapped, 5.32x capped
- Offset/non-contiguous cells: 1.95x uncapped, 1.95x capped
- Empty cells: 0.19x uncapped, 0.19x capped
- Singleton-bias cells: 0.14x uncapped, 0.14x capped
- `no_grad` cells: 4.30x uncapped, 4.30x capped

Unsupported module, broader-bias, and active-autograd rows remain separated
below and are not included in these supported-cell geomeans.

| Workload | Category | API | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `rank1_contiguous_no_bias_257_to131` | rank-1 contiguous | `functional.linear(input, weight)` | `(131,)`, stride `(1,)`, offset 0, requires_grad=False | 128 | 58.103 +/- 1.937 us, var 23.921 | 4.948 +/- 0.075 us, var 0.091 | 11.74x |
| `rank1_contiguous_bias_257_to131` | rank-1 contiguous bias | `functional.linear(input, weight, bias)` | `(131,)`, stride `(1,)`, offset 0, requires_grad=False | 128 | 54.910 +/- 0.382 us, var 8.880 | 5.602 +/- 0.049 us, var 0.020 | 9.80x |
| `rank1_offset_noncontig_bias_257_to131` | rank-1 offset/noncontiguous bias | `functional.linear(input, weight, bias)` | `(131,)`, stride `(1,)`, offset 0, requires_grad=False | 96 | 5.267 +/- 0.206 us, var 0.193 | 8.162 +/- 0.263 us, var 0.351 | 0.65x |
| `rank1_empty_features_singleton_bias_0_to131` | rank-1 empty singleton-bias | `functional.linear(input, weight, bias)` | `(131,)`, stride `(1,)`, offset 0, requires_grad=False | 5000 | 0.440 +/- 0.006 us, var 0.001 | 4.820 +/- 0.032 us, var 0.081 | 0.09x |
| `rank1_zero_outputs_257_to0` | rank-1 empty output | `functional.linear(input, weight, bias)` | `(0,)`, stride `(1,)`, offset 0, requires_grad=False | 5000 | 0.415 +/- 0.003 us, var 0.000 | 2.202 +/- 0.013 us, var 0.004 | 0.19x |
| `rank1_no_grad_requires_grad_bias_257_to131` | rank-1 no_grad | `with no_grad(): functional.linear(input, weight, bias)` | `(131,)`, stride `(1,)`, offset 0, requires_grad=False | 96 | 55.000 +/- 0.388 us, var 1.566 | 5.524 +/- 0.068 us, var 0.012 | 9.96x |
| `rank2_contiguous_no_bias_64x128_to96` | rank-2 contiguous | `functional.linear(input, weight)` | `(64, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 12 | 75.403 +/- 0.568 us, var 4.017 | 19.083 +/- 0.099 us, var 0.801 | 3.95x |
| `rank2_contiguous_bias_64x128_to96` | rank-2 contiguous bias | `functional.linear(input, weight, bias)` | `(64, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 12 | 75.615 +/- 0.286 us, var 5.019 | 20.035 +/- 0.156 us, var 0.856 | 3.77x |
| `rank2_offset_contiguous_bias_63x127_to95` | rank-2 offset bias | `functional.linear(input, weight, bias)` | `(63, 95)`, stride `(95, 1)`, offset 0, requires_grad=False | 12 | 108.672 +/- 5.126 us, var 133.362 | 27.744 +/- 0.602 us, var 1.461 | 3.92x |
| `rank2_offset_noncontig_bias_63x127_to95` | rank-2 offset/noncontiguous bias | `functional.linear(input, weight, bias)` | `(63, 95)`, stride `(95, 1)`, offset 0, requires_grad=False | 10 | 98.590 +/- 1.292 us, var 18.553 | 30.210 +/- 0.195 us, var 1.154 | 3.26x |
| `rank2_empty_rows_singleton_bias_0x256_to128` | rank-2 empty singleton-bias | `functional.linear(input, weight, bias)` | `(0, 128)`, stride `(128, 1)`, offset 0, requires_grad=False | 5000 | 0.321 +/- 0.007 us, var 0.000 | 1.588 +/- 0.008 us, var 0.001 | 0.20x |
| `rank2_empty_inner_bias_128x0_to64` | rank-2 empty inner bias | `functional.linear(input, weight, bias)` | `(128, 64)`, stride `(64, 1)`, offset 0, requires_grad=False | 2000 | 0.937 +/- 0.025 us, var 0.001 | 4.918 +/- 0.032 us, var 0.139 | 0.19x |
| `rank2_no_grad_requires_grad_bias_64x128_to96` | rank-2 no_grad | `with no_grad(): functional.linear(input, weight, bias)` | `(64, 96)`, stride `(96, 1)`, offset 0, requires_grad=False | 10 | 77.345 +/- 0.523 us, var 25.027 | 20.272 +/- 0.172 us, var 1.051 | 3.82x |
| `rank3_contiguous_no_bias_8x16x64_to32` | rank-3 contiguous bias-free | `functional.linear(input, weight)` | `(8, 16, 32)`, stride `(512, 32, 1)`, offset 0, requires_grad=False | 24 | 25.272 +/- 0.277 us, var 0.337 | 8.685 +/- 0.076 us, var 0.206 | 2.91x |
| `rank3_offset_contiguous_no_bias_7x9x63_to31` | rank-3 offset bias-free | `functional.linear(input, weight)` | `(7, 9, 31)`, stride `(279, 31, 1)`, offset 0, requires_grad=False | 24 | 19.262 +/- 0.370 us, var 0.269 | 8.914 +/- 0.045 us, var 0.090 | 2.16x |
| `rank3_offset_noncontig_no_bias_7x9x63_to31` | rank-3 offset/noncontiguous bias-free | `functional.linear(input, weight)` | `(7, 9, 31)`, stride `(279, 31, 1)`, offset 0, requires_grad=False | 16 | 20.504 +/- 0.162 us, var 0.414 | 13.066 +/- 0.089 us, var 0.478 | 1.57x |
| `rank3_empty_sequence_no_bias_8x0x64_to32` | rank-3 empty bias-free | `functional.linear(input, weight)` | `(8, 0, 32)`, stride `(32, 32, 1)`, offset 0, requires_grad=False | 5000 | 0.459 +/- 0.004 us, var 0.000 | 1.965 +/- 0.010 us, var 0.003 | 0.23x |
| `rank3_empty_inner_no_bias_8x16x0_to32` | rank-3 empty inner bias-free | `functional.linear(input, weight)` | `(8, 16, 32)`, stride `(512, 32, 1)`, offset 0, requires_grad=False | 2000 | 0.712 +/- 0.006 us, var 0.001 | 2.346 +/- 0.014 us, var 0.003 | 0.30x |
| `rank3_no_grad_requires_grad_no_bias_4x8x64_to32` | rank-3 no_grad bias-free | `with no_grad(): functional.linear(input, weight)` | `(4, 8, 32)`, stride `(256, 32, 1)`, offset 0, requires_grad=False | 32 | 9.137 +/- 0.071 us, var 0.176 | 4.356 +/- 0.017 us, var 0.044 | 2.10x |

## Materialized Checksums

The digest columns are BLAKE2b-128 checksums accumulated from the materialized
last output of each warmup and measured block. Each entry is `pass1/pass2`.

| Workload | `torch_rs` materialized digest | PyTorch materialized digest |
| --- | --- | --- |
| `rank1_contiguous_no_bias_257_to131` | `b144f57829b6dfbf4e411f3aa0b7f9de/b144f57829b6dfbf4e411f3aa0b7f9de` | `6afd903e8f433269a3c08ee027afbb63/6afd903e8f433269a3c08ee027afbb63` |
| `rank1_contiguous_bias_257_to131` | `90bd797cccb885effa1420cb02dd981a/90bd797cccb885effa1420cb02dd981a` | `143740b2461ebdd31e7f95c75fd9a146/143740b2461ebdd31e7f95c75fd9a146` |
| `rank1_offset_noncontig_bias_257_to131` | `4e27a5a968422e3e7e13e18b971f6ae6/4e27a5a968422e3e7e13e18b971f6ae6` | `c91c1936a7387253e194ef8d319d3ba6/c91c1936a7387253e194ef8d319d3ba6` |
| `rank1_empty_features_singleton_bias_0_to131` | `7ee7551458d8621fd5849fd5ff579abf/7ee7551458d8621fd5849fd5ff579abf` | `7ee7551458d8621fd5849fd5ff579abf/7ee7551458d8621fd5849fd5ff579abf` |
| `rank1_zero_outputs_257_to0` | `df499fe64c31d97bcb9d3821191e30ea/df499fe64c31d97bcb9d3821191e30ea` | `df499fe64c31d97bcb9d3821191e30ea/df499fe64c31d97bcb9d3821191e30ea` |
| `rank1_no_grad_requires_grad_bias_257_to131` | `90bd797cccb885effa1420cb02dd981a/90bd797cccb885effa1420cb02dd981a` | `143740b2461ebdd31e7f95c75fd9a146/143740b2461ebdd31e7f95c75fd9a146` |
| `rank2_contiguous_no_bias_64x128_to96` | `00dc5938f4af726007099130aa9b5c1c/00dc5938f4af726007099130aa9b5c1c` | `84f4e04a743779f542e58c96a8841ed7/84f4e04a743779f542e58c96a8841ed7` |
| `rank2_contiguous_bias_64x128_to96` | `febe380e6c261bd3f156fe55eaeceafe/febe380e6c261bd3f156fe55eaeceafe` | `8cdcb724b991d38f6b07f6f59ff9016d/8cdcb724b991d38f6b07f6f59ff9016d` |
| `rank2_offset_contiguous_bias_63x127_to95` | `eb3a803bf1851d01389277687f916c89/eb3a803bf1851d01389277687f916c89` | `8c2d4cac155f0291ebecc7808ae6badb/8c2d4cac155f0291ebecc7808ae6badb` |
| `rank2_offset_noncontig_bias_63x127_to95` | `2658f2600ab0dea12ec95f055a6f52db/2658f2600ab0dea12ec95f055a6f52db` | `47dcca9fb05cc29f2775ce34ed951377/47dcca9fb05cc29f2775ce34ed951377` |
| `rank2_empty_rows_singleton_bias_0x256_to128` | `bcb23c548f734241305829aff7e5b582/bcb23c548f734241305829aff7e5b582` | `bcb23c548f734241305829aff7e5b582/bcb23c548f734241305829aff7e5b582` |
| `rank2_empty_inner_bias_128x0_to64` | `cba1daab785621a0462278893f97326a/cba1daab785621a0462278893f97326a` | `cba1daab785621a0462278893f97326a/cba1daab785621a0462278893f97326a` |
| `rank2_no_grad_requires_grad_bias_64x128_to96` | `febe380e6c261bd3f156fe55eaeceafe/febe380e6c261bd3f156fe55eaeceafe` | `8cdcb724b991d38f6b07f6f59ff9016d/8cdcb724b991d38f6b07f6f59ff9016d` |
| `rank3_contiguous_no_bias_8x16x64_to32` | `db59ec861a4499a7a66c34565125448b/db59ec861a4499a7a66c34565125448b` | `270dcd7aff58b94e18ae7f394eb4c984/270dcd7aff58b94e18ae7f394eb4c984` |
| `rank3_offset_contiguous_no_bias_7x9x63_to31` | `68a9b5056ea871c8664071b945348908/68a9b5056ea871c8664071b945348908` | `6d5299e4cfe91a510b575bbc916c6398/6d5299e4cfe91a510b575bbc916c6398` |
| `rank3_offset_noncontig_no_bias_7x9x63_to31` | `7e6b7e79a47326b68c8a154f815edcdf/7e6b7e79a47326b68c8a154f815edcdf` | `96bff329d21c4a67e0d7d587c2c6b8be/96bff329d21c4a67e0d7d587c2c6b8be` |
| `rank3_empty_sequence_no_bias_8x0x64_to32` | `209ee0ddddd3d26e2ee75df687d75a03/209ee0ddddd3d26e2ee75df687d75a03` | `209ee0ddddd3d26e2ee75df687d75a03/209ee0ddddd3d26e2ee75df687d75a03` |
| `rank3_empty_inner_no_bias_8x16x0_to32` | `f8f8827b72650dae6a71ef52ef70e60e/f8f8827b72650dae6a71ef52ef70e60e` | `f8f8827b72650dae6a71ef52ef70e60e/f8f8827b72650dae6a71ef52ef70e60e` |
| `rank3_no_grad_requires_grad_no_bias_4x8x64_to32` | `eac1d079271488c7eaee68004825f5d7/eac1d079271488c7eaee68004825f5d7` | `c4903a438a665c8a33fc64092be9a54e/c4903a438a665c8a33fc64092be9a54e` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They remain explicit zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `module_rank2_forward` | `AttributeError: module 'torch_rs.nn' has no attribute 'Linear'` | supported shape `(3, 4)`, stride `(4, 1)`, dtype `torch.float32` | zero |
| `functional_rank3_with_rank1_bias` | `NotImplementedError: torch_rs.nn.functional.linear only supports bias for rank-1 or rank-2 input` | supported shape `(2, 3, 4)`, stride `(12, 4, 1)`, dtype `torch.float32` | zero |
| `functional_rank2_with_rank2_bias` | `NotImplementedError: torch_rs.nn.functional.linear only supports a rank-1 bias tensor` | supported shape `(3, 4)`, stride `(4, 1)`, dtype `torch.float32` | zero |
| `functional_rank2_with_scalar_bias` | `NotImplementedError: torch_rs.nn.functional.linear only supports a rank-1 bias tensor` | supported shape `(3, 4)`, stride `(4, 1)`, dtype `torch.float32` | zero |
| `functional_rank1_active_autograd_backward` | `RuntimeError: linear(): autograd recording is not supported` | supported backward execution | zero |
| `functional_rank2_active_autograd_backward` | `RuntimeError: linear(): autograd recording is not supported` | supported backward execution | zero |
| `functional_rank3_active_autograd_backward` | `RuntimeError: linear(): autograd recording is not supported` | supported backward execution | zero |
