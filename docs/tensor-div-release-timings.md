# True-Division Tensor Release Timings

Date: 2026-08-31

Candidate provenance: source snapshot based on
`b95bcf787c93f700a29e41155faf9b3704068c66`.

Exact commands, from the repository root:

```bash
UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
PATH="/home/bobren/.cargo/bin:$PATH" \
  CARGO_NET_OFFLINE=true \
  RUSTUP_HOME="/home/bobren/.rustup" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 \
  .venv/bin/python target/tensor-div-release-timings/bench_tensor_div.py
```

The timing driver was an ignored run artifact under `target/`. It constructed
all inputs before timing, set both PyTorch thread counts to 1, checked each
`torch_rs` result bitwise against the equivalent PyTorch 2.13 result, then ran
15 warmup blocks and 81 measured blocks per implementation. Every warmup and
measured block consumed the last output through a uint32 bit checksum inside
the timed block so outputs were materialized symmetrically and deferred work
could not be hidden. Correctness gates compared shape, stride, storage offset,
contiguity, dtype, device, values, and non-aliasing for non-empty outputs
before timing. Inputs were CPU `float32` tensors generated from fixed NumPy
seed `20260831`.

Checks run for this report:

```bash
PATH="/home/bobren/.cargo/bin:$PATH" \
  CARGO_NET_OFFLINE=true \
  /home/bobren/.cargo/bin/cargo fmt --check
git diff --check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_tensor_div \
  tests.test_tensor_div_reference \
  tests.test_readme_quickstart
PATH="/home/bobren/.cargo/bin:$PATH" \
  CARGO_NET_OFFLINE=true \
  /home/bobren/.cargo/bin/cargo test --all-targets
```

Results: `cargo fmt --check` and `git diff --check` passed. The focused
Python division/reference and README/docs smoke tests passed 16 tests. The Rust
suite passed 291 tests.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the release `maturin develop` install at
  `python/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 29 ms, prepared
  packages in 17.50s, and installed in 1.71s
- Build time: release extension build and editable install completed in 35.37s;
  Cargo registry dependencies were resolved from the existing locked offline
  cache with `CARGO_NET_OFFLINE=true`

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each supported per-cell ratio
to `[0.10x, 10.00x]`.

Geometric mean `torch_rs / PyTorch` slowdown for supported true-division cells:

- `/`: 6.80x uncapped, 4.37x capped
- `Tensor.div`: 7.55x uncapped, 4.48x capped
- `Tensor.divide`: 7.68x uncapped, 4.44x capped
- All supported cells: 7.29x uncapped, 4.43x capped

Category geomeans for supported cells:

- Contiguous tensor/tensor: 8.45x uncapped, 8.45x capped
- Contiguous tensor/scalar: 8.50x uncapped, 8.50x capped
- Reflected scalar: 4.82x uncapped, 4.82x capped
- Broadcasting: 48.81x uncapped, 10.00x capped
- Empty broadcast: 0.23x uncapped, 0.23x capped
- Noncontiguous/offset: 29.44x uncapped, 10.00x capped

## Supported Cells

| Workload | API | Input | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Checksum |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `slash_tensor_tensor_contiguous` | `/` | `(257, 263) / (257, 263)`, stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 32 | 94.139 us +/- 0.341, var 5.637 | 10.952 us +/- 0.058, var 0.116 | 8.60x | 71852224488317 |
| `slash_tensor_scalar_contiguous` | `/` | `(257, 263) / 3.25`, stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 32 | 94.103 us +/- 0.306, var 0.600 | 10.984 us +/- 0.065, var 0.181 | 8.57x | 72054334538415 |
| `slash_reflected_scalar_contiguous` | `/` | `3.25 / (257, 263)`, stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 32 | 94.562 us +/- 0.350, var 0.572 | 19.611 us +/- 0.254, var 0.113 | 4.82x | 71902467645171 |
| `slash_tensor_tensor_broadcast_vector` | `/` | `(640, 768) / (768,)`, strides `(768, 1)` and `(1,)` | `(640, 768)`, stride `(768, 1)` | 5 | 3967.714 us +/- 57.142, var 57769.256 | 81.267 us +/- 1.771, var 10.282 | 48.82x | 522566311625339 |
| `slash_tensor_tensor_empty_strided_broadcast` | `/` | transposed `(3, 0, 2) / (1, 1, 2)` empty broadcast | `(3, 0, 2)`, stride `(1, 3, 0)` | 5000 | 0.245 us +/- 0.003, var 0.000362 | 1.256 us +/- 0.006, var 0.001790 | 0.20x | 0 |
| `slash_tensor_tensor_offset_transposed` | `/` | offset transposed `(521, 509) / (521, 509)`, stride `(1, 521)`, input storage offset `265189` | `(521, 509)`, stride `(1, 521)` | 5 | 2243.251 us +/- 14.063, var 1234.733 | 77.187 us +/- 2.350, var 8.882 | 29.06x | 281908711944261 |
| `div_tensor_tensor_contiguous` | `Tensor.div` | `(257, 263) / (257, 263)`, stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 32 | 94.941 us +/- 0.365, var 3.269 | 11.501 us +/- 0.149, var 0.185 | 8.25x | 71852224488317 |
| `div_tensor_scalar_contiguous` | `Tensor.div` | `(257, 263).div(3.25)`, stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 32 | 96.148 us +/- 0.565, var 3.109 | 11.221 us +/- 0.116, var 0.073 | 8.57x | 72054334538415 |
| `div_tensor_tensor_broadcast_vector` | `Tensor.div` | `(640, 768).div((768,))`, strides `(768, 1)` and `(1,)` | `(640, 768)`, stride `(768, 1)` | 5 | 3913.406 us +/- 36.611, var 42401.384 | 82.763 us +/- 1.867, var 5.656 | 47.28x | 522566311625339 |
| `div_tensor_tensor_empty_strided_broadcast` | `Tensor.div` | transposed `(3, 0, 2).div((1, 1, 2))` empty broadcast | `(3, 0, 2)`, stride `(1, 3, 0)` | 5000 | 0.321 us +/- 0.002, var 0.000064 | 1.262 us +/- 0.009, var 0.008406 | 0.25x | 0 |
| `div_tensor_tensor_offset_transposed` | `Tensor.div` | offset transposed `(521, 509).div((521, 509))`, stride `(1, 521)`, input storage offset `265189` | `(521, 509)`, stride `(1, 521)` | 5 | 2243.504 us +/- 11.455, var 949.381 | 78.120 us +/- 1.422, var 3.240 | 28.72x | 281908711944261 |
| `divide_tensor_tensor_contiguous` | `Tensor.divide` | `(257, 263) / (257, 263)`, stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 32 | 94.743 us +/- 0.520, var 101.000 | 11.151 us +/- 0.075, var 0.120 | 8.50x | 71852224488317 |
| `divide_tensor_scalar_contiguous` | `Tensor.divide` | `(257, 263).divide(3.25)`, stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 32 | 95.451 us +/- 0.525, var 1.381 | 11.425 us +/- 0.265, var 0.219 | 8.35x | 72054334538415 |
| `divide_tensor_tensor_broadcast_vector` | `Tensor.divide` | `(640, 768).divide((768,))`, strides `(768, 1)` and `(1,)` | `(640, 768)`, stride `(768, 1)` | 5 | 4015.620 us +/- 57.545, var 17157.761 | 79.745 us +/- 1.164, var 3.538 | 50.36x | 522566311625339 |
| `divide_tensor_tensor_empty_strided_broadcast` | `Tensor.divide` | transposed `(3, 0, 2).divide((1, 1, 2))` empty broadcast | `(3, 0, 2)`, stride `(1, 3, 0)` | 5000 | 0.317 us +/- 0.003, var 0.000034 | 1.298 us +/- 0.006, var 0.000779 | 0.24x | 0 |
| `divide_tensor_tensor_offset_transposed` | `Tensor.divide` | offset transposed `(521, 509).divide((521, 509))`, stride `(1, 521)`, input storage offset `265189` | `(521, 509)`, stride `(1, 521)` | 5 | 2269.062 us +/- 15.936, var 1415.241 | 74.228 us +/- 2.035, var 7.510 | 30.57x | 281908711944261 |

## Zero-Credit Unsupported Cells

| Cell | Credit | Reason |
| --- | ---: | --- |
| `Tensor.div_reflected_scalar` | 0 | Python method form requires the Tensor as receiver; the supported reflected scalar division surface is `scalar / tensor` through `/`. |
| `Tensor.divide_reflected_scalar` | 0 | Python method form requires the Tensor as receiver; the supported reflected scalar division surface is `scalar / tensor` through `/`. |
| `torch_rs.div` / `torch_rs.divide` | 0 | Top-level true-division callables are not exposed by the current supported surface. |
