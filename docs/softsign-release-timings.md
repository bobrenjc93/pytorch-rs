# `torch.nn.functional.softsign` Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on
`41baf18c86b9a2dfa7342daebb53be99e7189df9`, plus the worktree changes that add
the native fused inference path for `torch.nn.functional.softsign`.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/softsign-release-timings*.json`. No Conda
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
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked --offline
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/softsign_release_timings.py \
  > target/softsign-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  SOFTSIGN_TIMING_IMPL_ORDER=pytorch,torch_rs_composition,torch_rs \
  taskset -c 24 .venv/bin/python target/softsign_release_timings.py \
  > target/softsign-release-timings-pass2.json
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
  cargo clippy --locked --offline --all-targets -- -D warnings
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  RUSTFLAGS="-L native=/usr/local/fbcode/platform010/lib -l dylib=python3.12" \
  cargo test --locked --offline --all-targets --features extension-module softsign
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --locked --offline --all-targets --features python-bindings -- -D warnings
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_softsign \
  tests.test_nn_functional_softsign_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 14 tests. The full native Rust suite passed 306 tests, the focused
`extension-module` Rust softsign filter passed, both Clippy configurations
passed, the README/docs smoke test passed 9 tests, `cargo fmt --check` passed,
and `git diff --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0
- `torch_rs`: 0.1.0 from the release editable install
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  packages in 16.05s, and installed in 4.68s
- Build time: successful offline release extension build completed in 36.72s

Inputs were created outside the timed region with NumPy seed `20260903`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. The timed implementations were the fused
`torch_rs.nn.functional.softsign`, the previous `torch_rs` primitive
composition `input / (input.abs() + 1)`, and PyTorch 2.13
`torch.nn.functional.softsign`. Every timing cell ran in two pinned process
passes. The first pass measured `torch_rs`, the composition, then PyTorch; the
second pass reversed that order. Each pass used 15 untimed warmup blocks and 81
measured blocks. A block repeated the operation according to the table's
`Repeats` column; times below are median microseconds per operation. Reported
medians are medians of the two per-process medians. MAD and variance are the
medians of the per-process MAD and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch and checked shape, stride, storage offset, contiguity,
channels-last contiguity, dtype, device, `requires_grad`, leaf status, fresh
output storage, and input nonmutation. After every warmup and measured block,
the driver consumed the last output as a 64-bit BLAKE2b rolling checksum over
tensor metadata and logical bytes. The checksum column shows the final rolling
sink from one pass as `torch_rs`/PyTorch; both process passes produced the same
sink pairs.

`torch_rs / PyTorch` and `torch_rs / composition` are slowdown ratios, so lower
is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean slowdown for the supported inference cells:

- Fused `torch_rs` / PyTorch: 0.28x uncapped, 0.28x capped
- Prior `torch_rs` composition / PyTorch: 0.44x uncapped, 0.45x capped
- Fused `torch_rs` / prior `torch_rs` composition: 0.62x uncapped, 0.62x
  capped

| Workload | Category | Input | Output | Repeats | fused `torch_rs` median +/- MAD, variance | prior composition median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | `torch_rs` / composition | Materialized checksums |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `scalar` | scalar | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.550 us +/- 0.006, var 0.002 | 0.456 us +/- 0.004, var 0.010 | 5.252 us +/- 0.036, var 0.011 | 0.10x | 1.21x | `5518593338568906336`/`5518593338568906336` |
| `empty` | empty | (0, 2), stride (3, 3), offset 1, requires_grad=False | (0, 2), stride (2, 1), offset 0, requires_grad=False | 5000 | 0.565 us +/- 0.004, var 0.002 | 0.524 us +/- 0.003, var 0.000 | 5.010 us +/- 0.036, var 0.032 | 0.11x | 1.08x | `8855064225997338195`/`8855064225997338195` |
| `contiguous_257x263` | contiguous | (257, 263), stride (263, 1), offset 0, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 15.068 us +/- 0.223, var 0.121 | 26.886 us +/- 0.294, var 0.622 | 28.606 us +/- 0.186, var 0.372 | 0.53x | 0.56x | `9025014186034785185`/`9025014186034785185` |
| `offset_257x263` | offset | (257, 263), stride (263, 1), offset 67591, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 15.050 us +/- 0.196, var 0.076 | 27.248 us +/- 0.215, var 0.376 | 27.942 us +/- 0.197, var 0.556 | 0.54x | 0.55x | `16680188594917605937`/`16680188594917605937` |
| `noncontig_transpose_512x1024` | noncontiguous | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 117.892 us +/- 1.786, var 13.187 | 405.695 us +/- 6.700, var 135.098 | 184.506 us +/- 5.052, var 169.434 | 0.64x | 0.29x | `8075477398820509548`/`8075477398820509548` |
| `channels_last_8x15x31x33` | channels_last | (8, 15, 31, 33), stride (15345, 1, 495, 15), offset 0, requires_grad=False | (8, 15, 31, 33), stride (15345, 1, 495, 15), offset 0, requires_grad=False | 8 | 27.112 us +/- 0.305, var 6.070 | 91.360 us +/- 1.470, var 13.530 | 47.847 us +/- 0.800, var 3.090 | 0.57x | 0.30x | `9884608429938328812`/`9884608429938328812` |
| `numerical_edges` | edge_values | (20,), stride (1,), offset 0, requires_grad=False | (20,), stride (1,), offset 0, requires_grad=False | 10000 | 0.599 us +/- 0.005, var 0.000 | 0.567 us +/- 0.003, var 0.000 | 5.740 us +/- 0.202, var 0.580 | 0.10x | 1.06x | `4154604552554360487`/`4154604552554360487` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` explicitly does not support the
equivalent PyTorch operation. They are preserved as zero-credit cells instead
of being removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `functional_softsign_active_autograd` | `RuntimeError: softsign(): autograd recording is not supported` | supported first-order autograd | zero |
| `module_nn_softsign` | `AttributeError: module 'torch_rs.nn' has no attribute 'Softsign'` | supported module wrapper | zero |
