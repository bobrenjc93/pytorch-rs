# True-Division Release Timings

Date: 2026-08-31

Candidate provenance: release wheel built from source snapshot
`d8f3bd62ca6766ccf4b6b702cc007ba7e0ad76c3`. This branch adds only this
report, its index link, and the existing smoke-test index entry.

Exact commands, run from the repository root:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" \
  uv venv --clear --python 3.12
UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/tensor-div-release-timings/cargo-home/registry \
  target/tensor-div-release-timings/tmp-hermetic \
  target/tensor-div-release-timings/wheels-hermetic
cp -a /home/bobren/.cargo/registry/. \
  target/tensor-div-release-timings/cargo-home/registry/
PATH=/home/bobren/.cargo/bin:$PATH \
  CARGO_HOME="$PWD/target/tensor-div-release-timings/cargo-home" \
  CARGO_NET_OFFLINE=true \
  TMPDIR="$PWD/target/tensor-div-release-timings/tmp-hermetic" \
  VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked \
  --out target/tensor-div-release-timings/wheels-hermetic
UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" \
  uv pip install --force-reinstall --no-deps \
  target/tensor-div-release-timings/wheels-hermetic/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= taskset -c 24 \
  .venv/bin/python - <<'PY'
# Inline timing driver defined the workload matrix below, constructed inputs
# before timing, ran the correctness/materialization gates below, then timed
# `/`, `Tensor.div`, and `Tensor.divide` with 15 warmup blocks and 81 samples.
PY
```

The timing driver measured eager CPU `float32` true division after imports and
input construction. Each supported `torch_rs` output was checked before timing
against the equivalent PyTorch 2.13.0 output for shape, stride, storage offset,
contiguity, and bitwise `float32` values; the focused public tests also checked
dtype and device parity. The driver then consumed the last output after every
warmup and measured block by materializing it through NumPy and reading
first/last values; empty outputs consumed their zero element count.

Checks run for this report:

```bash
PATH=/home/bobren/.cargo/bin:$PATH cargo fmt --check
git diff --check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_tensor_div \
  tests.test_tensor_div_reference
```

Results: the focused division Python tests passed 9 tests.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from
  `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the force-installed release wheel at
  `.venv/lib/python3.12/site-packages/torch_rs`
- Build mode: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU `float32`; `CUDA_VISIBLE_DEVICES=` hid GPUs for the timing
  run
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: `uv venv` completed in 0.43s; locked `uv sync`
  resolved in 30 ms, prepared packages in 16.44s, installed in 1.04s, and took
  17.65s wall time
- Build/install time: the recorded release wheel build used the worktree-local
  offline Cargo home above and completed in 34.65s; force installing that wheel
  took 0.19s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`. Unsupported cells are listed explicitly and receive zero
performance credit.

Geometric mean `torch_rs / PyTorch` slowdown:

- Supported cells, uncapped: 5.77x
- Supported cells, capped to `[0.10x, 10.00x]`: 4.53x
- Score-counted capped aggregate with two unsupported reflected-method cells
  assigned the 10.00x worst cap: 4.95x

Geometric mean by supported public surface:

- `/`: 5.49x uncapped, 4.44x capped
- `Tensor.div`: 5.95x uncapped, 4.59x capped
- `Tensor.divide`: 5.93x uncapped, 4.59x capped

## Workloads

All tensors are CPU `float32`. Shapes, strides, and storage offsets are shown
as `(shape), stride (stride), offset N`.

| Workload | Operand A | Operand B | Output | Repeats |
| --- | --- | --- | --- | ---: |
| `contiguous_tensor_tensor` | `(257, 263), stride (263, 1), offset 0` | `(257, 263), stride (263, 1), offset 0` | `(257, 263), stride (263, 1), offset 0` | 32 |
| `contiguous_tensor_scalar` | `(257, 263), stride (263, 1), offset 0` | scalar `2.5` | `(257, 263), stride (263, 1), offset 0` | 32 |
| `reflected_scalar_contiguous` | scalar `2.5` | `(257, 263), stride (263, 1), offset 0` | `(257, 263), stride (263, 1), offset 0` | 32 |
| `broadcast_vector_denominator` | `(640, 768), stride (768, 1), offset 0` | `(768,), stride (1,), offset 0` | `(640, 768), stride (768, 1), offset 0` | 16 |
| `empty_broadcast` | `(0, 4096), stride (4096, 1), offset 0` | `(1, 4096), stride (4096, 1), offset 0` | `(0, 4096), stride (4096, 1), offset 0` | 5000 |
| `offset_noncontiguous_tensor_tensor` | `(65536,), stride (17,), offset 5` | `(65536,), stride (17,), offset 7` | `(65536,), stride (1,), offset 0` | 16 |

## Supported Timed Cells

| Surface | Workload | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | ---: | ---: | ---: |
| `/` | `contiguous_tensor_tensor` | 94.287 us +/- 0.578, var 1.800 | 11.215 us +/- 0.248, var 0.286 | 8.41x |
| `Tensor.div` | `contiguous_tensor_tensor` | 95.001 us +/- 0.520, var 99.807 | 11.492 us +/- 0.107, var 0.141 | 8.27x |
| `Tensor.divide` | `contiguous_tensor_tensor` | 94.807 us +/- 0.488, var 0.914 | 10.973 us +/- 0.090, var 0.086 | 8.64x |
| `/` | `contiguous_tensor_scalar` | 94.371 us +/- 0.212, var 0.568 | 10.496 us +/- 0.168, var 0.219 | 8.99x |
| `Tensor.div` | `contiguous_tensor_scalar` | 95.754 us +/- 0.601, var 6.875 | 10.748 us +/- 0.096, var 0.190 | 8.91x |
| `Tensor.divide` | `contiguous_tensor_scalar` | 95.387 us +/- 0.937, var 2.044 | 10.779 us +/- 0.137, var 0.128 | 8.85x |
| `/` | `reflected_scalar_contiguous` | 94.792 us +/- 0.251, var 0.586 | 19.405 us +/- 0.227, var 0.149 | 4.89x |
| `/` | `broadcast_vector_denominator` | 1294.635 us +/- 13.755, var 2699.465 | 55.650 us +/- 0.214, var 0.954 | 23.26x |
| `Tensor.div` | `broadcast_vector_denominator` | 1335.690 us +/- 34.569, var 2025.710 | 55.680 us +/- 0.568, var 2.004 | 23.99x |
| `Tensor.divide` | `broadcast_vector_denominator` | 1291.115 us +/- 9.124, var 880.628 | 55.398 us +/- 0.160, var 0.588 | 23.31x |
| `/` | `empty_broadcast` | 0.243 us +/- 0.003, var 0.000264 | 1.173 us +/- 0.008, var 0.000903 | 0.21x |
| `Tensor.div` | `empty_broadcast` | 0.323 us +/- 0.002, var 0.000173 | 1.174 us +/- 0.007, var 0.006292 | 0.28x |
| `Tensor.divide` | `empty_broadcast` | 0.323 us +/- 0.002, var 0.000031 | 1.211 us +/- 0.007, var 0.009418 | 0.27x |
| `/` | `offset_noncontiguous_tensor_tensor` | 1542.828 us +/- 17.747, var 3337.250 | 100.906 us +/- 1.328, var 28.421 | 15.29x |
| `Tensor.div` | `offset_noncontiguous_tensor_tensor` | 1561.946 us +/- 29.563, var 2390.637 | 101.852 us +/- 0.956, var 3.310 | 15.34x |
| `Tensor.divide` | `offset_noncontiguous_tensor_tensor` | 1545.222 us +/- 35.787, var 6008.312 | 100.274 us +/- 0.841, var 5362.868 | 15.41x |

## Unsupported Zero-Credit Cells

| Cell | Status | Score treatment |
| --- | --- | --- |
| `Tensor.div` reflected scalar, equivalent to scalar `/ tensor` | Unsupported: Python reflected scalar division is exposed through `/`; tensor methods require a Tensor receiver. | Zero credit; counted at the 10.00x worst cap in the score-counted aggregate above. |
| `Tensor.divide` reflected scalar, equivalent to scalar `/ tensor` | Unsupported: Python reflected scalar division is exposed through `/`; tensor methods require a Tensor receiver. | Zero credit; counted at the 10.00x worst cap in the score-counted aggregate above. |
| Top-level `torch_rs.div` / `torch_rs.divide` | Unsupported and outside this branch's timed public surface; both names are absent. | Zero credit in any broader top-level-API coverage matrix. |
| `Tensor.div` / `Tensor.divide` with non-`None` `rounding_mode`, `out=`, or in-place `div_` / `divide_` | Unsupported by the current contract and covered by the focused division tests. | Zero credit if those workload cells are included by an evaluator. |
