# `torch.div` and `torch.divide` Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot based on
`aaedebeb52012450276ebf4515bd8d022853a84c`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/top-level-division-release-timings*.json`. No Conda
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
  cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/ && \
  wheel_dir="$(mktemp -d "$PWD/target/top-level-division-wheels.XXXXXX")" && \
  printf '%s\n' "$wheel_dir" > target/top-level-division-wheel-dir.txt && \
  env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
    CARGO_HOME="$PWD/target/cargo-home" \
    CARGO_TARGET_DIR="$PWD/target" \
    TMPDIR="$PWD/target" \
    VIRTUAL_ENV="$PWD/.venv" \
    PYO3_PYTHON="$PWD/.venv/bin/python" \
    .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/top-level-division-wheel-dir.txt)" && \
  env UV_CACHE_DIR="$PWD/target/uv-cache" \
    UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
    uv pip install --python "$PWD/.venv/bin/python" \
    --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_tensor_div tests.test_tensor_div_reference \
  tests.test_top_level_div tests.test_top_level_div_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/top_level_division_release_timings.py \
  > target/top-level-division-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  DIVISION_TIMING_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/top_level_division_release_timings.py \
  > target/top-level-division-release-timings-pass2.json
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
  cargo test --locked --offline --all-targets division
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_tensor_div tests.test_tensor_div_reference \
  tests.test_top_level_div tests.test_top_level_div_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 18 tests. The focused Rust `division` filter passed 7 tests, `cargo fmt
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
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  packages in 16.09s, and installed in 1.87s
- Build time: successful offline release extension build completed in 35.74s;
  the release wheel reinstall resolved in 1 ms, prepared in 43 ms, and
  installed in 13 ms

Inputs were created outside the timed region with NumPy seed `20260902`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch and checked shape, stride, storage offset, contiguity,
dtype, device, `requires_grad`, and leaf status. The `no_grad` cells used
pre-created leaf tensors with `requires_grad=True` and timed the top-level
operation inside the `no_grad` context; outputs were required to be fresh
leaf tensors with `requires_grad=False`. After every warmup and measured block,
the driver materialized the last output as a 64-bit BLAKE2b rolling checksum
over output metadata and logical bytes. The checksum column shows the final
rolling sink from one pass as `torch_rs`/PyTorch; both process passes produced
the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 1.02x uncapped, 1.02x capped
- `torch.div` cells: 1.03x uncapped, 1.03x capped
- `torch.divide` cells: 1.01x uncapped, 1.01x capped
- Tensor/tensor cells: 1.03x uncapped, 1.03x capped
- Tensor/scalar cells: 0.99x uncapped, 0.99x capped
- Scalar/tensor cells: 1.05x uncapped, 1.05x capped
- Scalar cells: 0.33x uncapped, 0.33x capped
- Tensor/tensor contiguous cells: 1.42x uncapped, 1.42x capped
- Tensor/scalar contiguous cells: 1.86x uncapped, 1.86x capped
- Scalar/tensor contiguous cells: 1.96x uncapped, 1.96x capped
- Broadcasting cells: 1.95x uncapped, 1.95x capped
- Empty cells: 0.31x uncapped, 0.31x capped
- Offset tensor/tensor cells: 4.02x uncapped, 4.02x capped
- Offset tensor/scalar cells: 1.80x uncapped, 1.80x capped
- Noncontiguous cells: 4.01x uncapped, 4.01x capped
- Signed-zero/NaN/inf cells: 0.39x uncapped, 0.39x capped
- `no_grad` tensor/tensor cells: 1.30x uncapped, 1.30x capped
- `no_grad` tensor/scalar cells: 1.32x uncapped, 1.32x capped
- `no_grad` scalar/tensor cells: 1.32x uncapped, 1.32x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 2.04x.

| Workload | Category | API | Operand path | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_tensor_tensor` | scalar | `torch.div` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | (), stride (), offset 0, requires_grad=False | 10000 | 0.308 us +/- 0.002 us, var 0.002 | 1.174 us +/- 0.013 us, var 0.010 | 0.26x | `16222328004266668340`/`16222328004266668340` |
| `scalar_tensor_scalar` | scalar | `torch.div` | tensor/scalar | left scalar tensor, shape (), stride (); scalar 2.25 | (), stride (), offset 0, requires_grad=False | 10000 | 1.134 us +/- 0.007 us, var 0.010 | 2.555 us +/- 0.051 us, var 0.046 | 0.44x | `16222328004266668340`/`16222328004266668340` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.div` | tensor/tensor | left/right (257, 263), stride (263, 1), nonzero float32 divisor | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 15.059 us +/- 0.297 us, var 0.173 | 10.590 us +/- 0.320 us, var 1.465 | 1.42x | `3351274874555059250`/`3351274874555059250` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.div` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 108.654 us +/- 1.751 us, var 10.650 | 57.725 us +/- 0.678 us, var 5.068 | 1.88x | `10735279501772474769`/`10735279501772474769` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.div` | scalar/tensor | scalar -2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 108.596 us +/- 1.317 us, var 12.745 | 56.301 us +/- 0.513 us, var 2.899 | 1.93x | `10231642340614551885`/`10231642340614551885` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.div` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 16 | 108.151 us +/- 2.421 us, var 29.757 | 55.564 us +/- 0.477 us, var 0.957 | 1.95x | `9089072239104929759`/`9089072239104929759` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.div` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 0.394 us +/- 0.002 us, var 0.000 | 1.243 us +/- 0.006 us, var 0.001 | 0.32x | `11498465976707186792`/`11498465976707186792` |
| `offset_transposed_521x509` | offset | `torch.div` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 153.861 us +/- 3.659 us, var 86.392 | 39.601 us +/- 0.802 us, var 1.922 | 3.89x | `18291836575230787745`/`18291836575230787745` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.div` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar 1.75 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 59.672 us +/- 1.117 us, var 6.471 | 33.799 us +/- 0.427 us, var 2.889 | 1.77x | `1231849374558012098`/`1231849374558012098` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.div` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 309.172 us +/- 5.236 us, var 323.590 | 73.620 us +/- 1.834 us, var 15.408 | 4.20x | `1944881246454062374`/`1944881246454062374` |
| `signed_zero_nan_inf` | signed-zero NaN/inf | `torch.div` | tensor/tensor | special float32 bit patterns [0, -0, 1, -1, inf, -inf, qnan] divided by [1, -1, 0, -0, inf, -inf, 2] | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 0.334 us +/- 0.002 us, var 0.000 | 1.171 us +/- 0.010 us, var 0.013 | 0.29x | `5684299589417147291`/`5684299589417147291` |
| `signed_zero_nan_inf_tensor_scalar` | signed-zero NaN/inf | `torch.div` | tensor/scalar | special float32 bit patterns [0, -0, 1, -1, inf, -inf, qnan] divided by scalar -0.0 | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 1.371 us +/- 0.043 us, var 0.019 | 2.650 us +/- 0.048 us, var 0.099 | 0.52x | `4798341315228508333`/`4798341315228508333` |
| `signed_zero_nan_inf_scalar_tensor` | signed-zero NaN/inf | `torch.div` | scalar/tensor | scalar NaN divided by special float32 bit patterns [1, -1, 0, -0, inf, -inf, 2] | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 1.188 us +/- 0.008 us, var 0.001 | 2.645 us +/- 0.045 us, var 0.039 | 0.45x | `12266681747092274798`/`12266681747092274798` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.div` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 16.209 us +/- 0.301 us, var 0.517 | 12.418 us +/- 0.213 us, var 0.213 | 1.31x | `12816167244662236081`/`12816167244662236081` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.div` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar 2.0; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 16.671 us +/- 0.256 us, var 0.248 | 12.495 us +/- 0.261 us, var 3.004 | 1.33x | `5497398482399487088`/`5497398482399487088` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.div` | scalar/tensor | scalar 2.0; right leaf (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 16.547 us +/- 0.261 us, var 0.176 | 12.616 us +/- 0.243 us, var 0.210 | 1.31x | `4440349670192522211`/`4440349670192522211` |
| `scalar_tensor_tensor` | scalar | `torch.divide` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | (), stride (), offset 0, requires_grad=False | 10000 | 0.306 us +/- 0.002 us, var 0.000 | 1.258 us +/- 0.015 us, var 0.009 | 0.24x | `16222328004266668340`/`16222328004266668340` |
| `scalar_tensor_scalar` | scalar | `torch.divide` | tensor/scalar | left scalar tensor, shape (), stride (); scalar 2.25 | (), stride (), offset 0, requires_grad=False | 10000 | 1.141 us +/- 0.008 us, var 0.001 | 2.634 us +/- 0.047 us, var 0.046 | 0.43x | `16222328004266668340`/`16222328004266668340` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.divide` | tensor/tensor | left/right (257, 263), stride (263, 1), nonzero float32 divisor | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 15.115 us +/- 0.226 us, var 0.184 | 10.652 us +/- 0.099 us, var 0.111 | 1.42x | `3351274874555059250`/`3351274874555059250` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.divide` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 108.199 us +/- 0.868 us, var 3.887 | 58.583 us +/- 0.767 us, var 7.280 | 1.85x | `10735279501772474769`/`10735279501772474769` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.divide` | scalar/tensor | scalar -2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 113.424 us +/- 4.380 us, var 36.664 | 57.216 us +/- 0.595 us, var 13.655 | 1.98x | `10231642340614551885`/`10231642340614551885` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.divide` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 16 | 107.509 us +/- 1.204 us, var 5.121 | 54.968 us +/- 0.668 us, var 20.830 | 1.96x | `9089072239104929759`/`9089072239104929759` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.divide` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 0.394 us +/- 0.003 us, var 0.001 | 1.274 us +/- 0.008 us, var 0.004 | 0.31x | `11498465976707186792`/`11498465976707186792` |
| `offset_transposed_521x509` | offset | `torch.divide` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 159.397 us +/- 4.246 us, var 63.832 | 38.355 us +/- 0.627 us, var 2.873 | 4.16x | `18291836575230787745`/`18291836575230787745` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.divide` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar 1.75 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 61.651 us +/- 1.484 us, var 15.965 | 33.546 us +/- 0.539 us, var 3.462 | 1.84x | `1231849374558012098`/`1231849374558012098` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.divide` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 357.533 us +/- 9.793 us, var 1050.662 | 93.399 us +/- 5.160 us, var 229.109 | 3.83x | `1944881246454062374`/`1944881246454062374` |
| `signed_zero_nan_inf` | signed-zero NaN/inf | `torch.divide` | tensor/tensor | special float32 bit patterns [0, -0, 1, -1, inf, -inf, qnan] divided by [1, -1, 0, -0, inf, -inf, 2] | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 0.334 us +/- 0.001 us, var 0.000 | 1.242 us +/- 0.010 us, var 0.028 | 0.27x | `5684299589417147291`/`5684299589417147291` |
| `signed_zero_nan_inf_tensor_scalar` | signed-zero NaN/inf | `torch.divide` | tensor/scalar | special float32 bit patterns [0, -0, 1, -1, inf, -inf, qnan] divided by scalar -0.0 | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 1.210 us +/- 0.010 us, var 0.001 | 2.706 us +/- 0.024 us, var 0.076 | 0.45x | `4798341315228508333`/`4798341315228508333` |
| `signed_zero_nan_inf_scalar_tensor` | signed-zero NaN/inf | `torch.divide` | scalar/tensor | scalar NaN divided by special float32 bit patterns [1, -1, 0, -0, inf, -inf, 2] | (7,), stride (1,), offset 0, requires_grad=False | 10000 | 1.207 us +/- 0.017 us, var 0.002 | 2.701 us +/- 0.023 us, var 0.086 | 0.45x | `12266681747092274798`/`12266681747092274798` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.divide` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 16.303 us +/- 0.259 us, var 0.174 | 12.647 us +/- 0.240 us, var 0.256 | 1.29x | `12816167244662236081`/`12816167244662236081` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.divide` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar 2.0; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 16.488 us +/- 0.231 us, var 0.245 | 12.559 us +/- 0.192 us, var 0.211 | 1.31x | `5497398482399487088`/`5497398482399487088` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.divide` | scalar/tensor | scalar 2.0; right leaf (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 16.530 us +/- 0.221 us, var 0.123 | 12.471 us +/- 0.198 us, var 0.158 | 1.33x | `4440349670192522211`/`4440349670192522211` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `top_level_torch_div_out_tensor` | `RuntimeError: div(): the 'out' argument is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_div_rounding_mode_floor` | `NotImplementedError: div(): non-None rounding_mode is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_div_rounding_mode_trunc` | `NotImplementedError: div(): non-None rounding_mode is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_div_scalar_scalar` | `TypeError: div(): scalar-scalar division is not supported; at least one operand must be Tensor` | supported `(), stride (), offset 0, requires_grad=False` | zero |
| `top_level_torch_div_active_autograd_tensor_tensor` | `RuntimeError: div(): autograd recording is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=True` | zero |
| `top_level_torch_div_active_autograd_tensor_scalar` | `RuntimeError: div(): autograd recording is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=True` | zero |
| `top_level_torch_div_active_autograd_scalar_tensor` | `RuntimeError: div(): autograd recording is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=True` | zero |
| `top_level_torch_divide_out_tensor` | `RuntimeError: divide(): the 'out' argument is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_divide_rounding_mode_floor` | `NotImplementedError: divide(): non-None rounding_mode is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_divide_rounding_mode_trunc` | `NotImplementedError: divide(): non-None rounding_mode is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_divide_scalar_scalar` | `TypeError: divide(): scalar-scalar division is not supported; at least one operand must be Tensor` | supported `(), stride (), offset 0, requires_grad=False` | zero |
| `top_level_torch_divide_active_autograd_tensor_tensor` | `RuntimeError: divide(): autograd recording is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=True` | zero |
| `top_level_torch_divide_active_autograd_tensor_scalar` | `RuntimeError: divide(): autograd recording is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=True` | zero |
| `top_level_torch_divide_active_autograd_scalar_tensor` | `RuntimeError: divide(): autograd recording is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=True` | zero |
