# `torch.nn.functional.mse_loss` Release Timings

Date: 2026-08-30

Review updates: 2026-09-01, 2026-09-03

Candidate provenance: source snapshot based on
`2231dec5e208f3545c05484d497b32b3981f640d`

## Review Update: `reduction="mean"` No-Grad Contiguous

The 2026-09-03 review rerun measured only the optimized
`mse_loss(reduction="mean")` cell for same-shape row-major contiguous CPU
`float32` tensors that do not record gradients. The worktree was based on
`b708c018869fb678a15014fb7355143dfcea3f22` with this MSE mean fast-path change.

The release extension was built and installed into the current worktree's
`.venv` with `maturin develop --release --locked`. The ignored one-off timing
driver was `target/mse_mean_release_timing.py`, with JSON outputs under
`target/mse-mean-release-timing-pass*.json`.

Inputs were created outside the timed region from NumPy seed-free deterministic
`linspace` buffers, `CUDA_VISIBLE_DEVICES=` was set, PyTorch and `torch_rs`
thread counts were fixed to one, and runs were pinned with `taskset -c 24`.
Each process pass used 15 warmup blocks and 81 measured blocks, with 64 calls
per block. The first pass measured `torch_rs` before PyTorch; the second pass
reversed that order. Values below are medians of the two per-process medians.
The scalar output was consumed with `.item()` after each block as the
materialization guard.

Correctness was checked before timing for output shape, stride, storage offset,
contiguity, dtype, device, `requires_grad`, leaf status, and values against
PyTorch 2.13. The large reduction value used `rtol=1e-5`, `atol=1e-5`, and
`equal_nan=True` to allow equivalent float32 accumulation grouping; the focused
native and differential tests retain stricter scalar, empty, signed-zero,
NaN/inf, active-autograd, broadcast, noncontiguous, metadata, and nonmutation
coverage.

Focused checks for this update:

```bash
PATH="$PWD/target/rust-env/bin:$PATH" \
  cargo fmt --check
PATH="$PWD/target/rust-env/bin:$PATH" \
  cargo clippy --all-targets -- -D warnings
PATH="$PWD/target/rust-env/bin:$PATH" \
  cargo test --all-targets
PATH="$PWD/target/rust-env/bin:$PATH" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --all-targets --features python-bindings -- -D warnings
PY_LIBDIR="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
PATH="$PWD/target/rust-env/bin:$PATH" \
LD_LIBRARY_PATH="$PY_LIBDIR:${LD_LIBRARY_PATH:-}" \
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --all-targets --features python-bindings
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
```

Results: Rust tests passed 303 tests without Python bindings and 314 tests with
Python bindings. The focused MSE Python implementation and PyTorch 2.13
differential tests passed 48 tests.

Geometric mean `torch_rs / PyTorch` slowdown for the optimized
`reduction="mean"` cell:

- Uncapped: 2.67x
- Capped to `[0.10x, 10.00x]` per cell: 2.67x

| Workload | Category | Output | Repeats | `torch_rs` median +/- MAD | PyTorch median +/- MAD | `torch_rs` / PyTorch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `mse_mean_contiguous_255x256` | same-shape row-major contiguous no-grad | `()`, stride `()`, offset 0, requires_grad=False | 64 | 54.137 +/- 0.122 us | 20.314 +/- 0.173 us | 2.67x |

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.14.5
- NumPy: 2.5.1
- Rust: `rustc 1.92.0-nightly (2300c2aef 2025-10-12)`,
  `cargo 1.92.0-nightly (81c3f77a4 2025-10-10)`
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.14/site-packages/torch`
- `torch_rs`: 0.1.0 from the release editable install in
  `.venv/lib/python3.14/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 29 ms, prepared
  packages in 15.84s, and installed in 861 ms
- Build time: the first successful release rebuild after the code change
  completed in 29.73s; the final benchmark reinstall reused cached artifacts
  and completed in 0.02s

## Review Update: `reduction="sum"`

The 2026-09-01 review rerun used the current composite worktree's release
wheel, built and installed under `target/review-wheels`, and the ignored
one-off timing driver `target/review_release_timings.py`. The driver emitted
JSON under `target/review-release-timings-pass*.json`.

The same run also refreshed rank-2 matmul and generalized `unbind` evidence.
For `mse_loss(reduction="sum")`, it used CPU `float32` inputs created outside
the timed region with NumPy seed `20260901`, `CUDA_VISIBLE_DEVICES=`, one
PyTorch thread, one `torch_rs` thread, `taskset -c 24`, 15 warmup blocks, and
81 measured blocks in each of two process passes. The first pass measured
`torch_rs` before PyTorch; the second pass reversed that order. Values below
are medians of the two per-process medians.

Correctness was checked against PyTorch 2.13 before timing for output shape,
stride, storage offset, contiguity, dtype, device, `requires_grad`, leaf
status, NaN classifications, sign bits, and values. Larger reduction timing
inputs used `rtol=1e-4`, `atol=1e-4`, and `equal_nan=True` to allow equivalent
float32 sums with different accumulation grouping; the focused reference tests
continue to check the smaller semantic contract within one ULP.

Focused checks for this update:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
```

Result: the focused MSE Python implementation and PyTorch 2.13 differential
tests passed 37 tests.

Geometric mean `torch_rs / PyTorch` slowdown for the supported
`reduction="sum"` cells:

- Uncapped: 1.05x
- Capped to `[0.10x, 10.00x]` per cell: 1.05x

| Workload | Category | Output | Repeats | `torch_rs` median +/- MAD | PyTorch median +/- MAD | `torch_rs` / PyTorch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `mse_sum_scalar` | scalar | `()`, stride `()`, offset 0, requires_grad=False | 10000 | 1.957 +/- 0.012 us | 6.727 +/- 0.040 us | 0.29x |
| `mse_sum_empty_transposed` | empty | `()`, stride `()`, offset 0, requires_grad=False | 5000 | 1.982 +/- 0.009 us | 6.941 +/- 0.072 us | 0.29x |
| `mse_sum_broadcasted_256x384_by_384` | broadcasted | `()`, stride `()`, offset 0, requires_grad=False | 8 | 96.266 +/- 0.476 us | 28.698 +/- 0.578 us | 3.35x |
| `mse_sum_offset_96x80` | offset | `()`, stride `()`, offset 0, requires_grad=False | 16 | 9.486 +/- 0.027 us | 8.439 +/- 0.067 us | 1.12x |
| `mse_sum_noncontig_transpose_256x512` | noncontiguous | `()`, stride `()`, offset 0, requires_grad=False | 4 | 137.977 +/- 2.332 us | 34.458 +/- 0.168 us | 4.00x |

Command shape: from the repository root, `uv venv --clear --python 3.12`,
locked `uv sync --locked --no-install-project --group dev --group reference`,
then a release wheel build and install through `./scripts/test-python.sh`. The
timing driver ran against that installed wheel after imports and input
construction, with 9 warmup blocks and 51 measured blocks per implementation.
Inputs were CPU `float32` tensors. Broadcast size-mismatch warning parity was
checked before timing, then `UserWarning` was ignored symmetrically for both
implementations inside the measured region.

The primary timings below measure eager `mse_loss(reduction="none")`
construction and consume the last output after each measured block as a
dead-code and deferred-work guard. The full-output checksum table consumes every
fresh result with `output.sum().item()` inside the timed loop; those numbers are
kept as a conservative end-to-end guard and are dominated by the current
`torch_rs.sum` implementation for non-empty outputs.

Checks run before timing:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --all-targets --features python-bindings -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --all-targets --features python-bindings
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
UV_CACHE_DIR="$PWD/.uv-cache" \
  ./scripts/test-python.sh
```

Results: the focused MSE Python tests passed 29 tests. The wheel-installed full
Python suite passed 4196 tests with 3 skips.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.12
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 29 ms and installed in
  887 ms
- Build time: first successful release extension build completed in 31.93s; the
  later wheel-based test build reused cached artifacts and reported 0.01s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## MSE Construction

Relative to the prior 2026-08-30 MSE report, the optimized transposed control
changed from 244.610 us to 97.753 us (-60.0%). Existing scalar-broadcast and
contiguous controls did not regress by more than 5%; their worst construction
movement was `scalar_target_2d_heldout`, from 45.603 us to 47.096 us (+3.3%).

Geometric mean `torch_rs / PyTorch` slowdown for the scalar-broadcast held-out
cells:

- Uncapped: 0.67x
- Capped to `[0.10x, 10.00x]` per cell: 0.67x

Geometric mean `torch_rs / PyTorch` slowdown for the held-out same-stride
non-contiguous cells:

- Uncapped: 1.08x
- Capped to `[0.10x, 10.00x]` per cell: 1.08x

Geometric mean `torch_rs / PyTorch` slowdown for all construction cells:

- Uncapped: 0.80x
- Capped to `[0.10x, 10.00x]` per cell: 0.80x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `scalar_input_2d_heldout` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 2 | 46.500 us +/- 0.611, var 2.765 | 52.339 us +/- 0.290, var 1.773 | 0.89x |
| `scalar_target_2d_heldout` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 2 | 47.096 us +/- 0.245, var 1.822 | 53.105 us +/- 0.180, var 13.041 | 0.89x |
| `scalar_input_3d_heldout` | `()` / `(17, 257, 263)` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 101.494 us +/- 1.942, var 22.674 | 112.731 us +/- 1.003, var 11.102 | 0.90x |
| `scalar_target_3d_heldout` | `(17, 257, 263)` / `()` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 102.726 us +/- 1.311, var 43.991 | 116.145 us +/- 0.921, var 15.130 | 0.88x |
| `scalar_target_empty_contiguous` | `(0, 4096)` / `()` | `(0, 4096)`, stride `(4096, 1)` | 5000 | 1.019 us +/- 0.003, var 0.004 | 4.896 us +/- 0.017, var 0.001 | 0.21x |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 16 | 11.794 us +/- 0.068, var 0.130 | 15.559 us +/- 0.205, var 1.045 | 0.76x |
| `same_noncontiguous_transpose_control` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 97.753 us +/- 2.969, var 21.541 | 81.474 us +/- 0.350, var 2.320 | 1.20x |
| `heldout_offset_transposed_509x521` | offset transposed `(509, 521)` / `(509, 521)`, stride `(1, 509)` | `(509, 521)`, stride `(1, 509)` | 2 | 45.849 us +/- 0.236, var 2.314 | 42.609 us +/- 0.286, var 1.571 | 1.08x |
| `heldout_channels_last_8x15x31x33` | channels-last `(8, 15, 31, 33)` / `(8, 15, 31, 33)` | `(8, 15, 31, 33)`, stride `(15345, 1, 495, 15)` | 4 | 23.866 us +/- 0.193, var 1.525 | 22.046 us +/- 0.128, var 0.398 | 1.08x |

## Full-Output Checksum Guard

Relative to the prior 2026-08-30 report, the same-shape transposed checksum
control changed from 1284.220 us to 1141.751 us (-11.1%). Existing
scalar-broadcast and contiguous checksum controls did not regress by more than
5%; the largest movement was `same_contiguous_prime_control`, from 65.968 us to
68.391 us (+3.7%).

Geometric mean `torch_rs / PyTorch` slowdown for the scalar-broadcast held-out
cells:

- Uncapped: 3.10x
- Capped to `[0.10x, 10.00x]` per cell: 3.10x

Geometric mean `torch_rs / PyTorch` slowdown for the held-out same-stride
non-contiguous cells:

- Uncapped: 4.40x
- Capped to `[0.10x, 10.00x]` per cell: 4.40x

Geometric mean `torch_rs / PyTorch` slowdown for all full-output checksum cells:

- Uncapped: 3.86x
- Capped to `[0.10x, 10.00x]` per cell: 3.83x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `scalar_input_2d_heldout` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 2 | 449.063 us +/- 1.537, var 6.775 | 78.048 us +/- 0.506, var 2.426 | 5.75x |
| `scalar_target_2d_heldout` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 2 | 451.522 us +/- 1.137, var 6.549 | 77.537 us +/- 0.360, var 2.903 | 5.82x |
| `scalar_input_3d_heldout` | `()` / `(17, 257, 263)` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 1050.038 us +/- 2.914, var 43.913 | 169.917 us +/- 1.152, var 12.480 | 6.18x |
| `scalar_target_3d_heldout` | `(17, 257, 263)` / `()` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 1056.417 us +/- 5.278, var 1206.038 | 169.826 us +/- 1.233, var 14.846 | 6.22x |
| `scalar_target_empty_contiguous` | `(0, 4096)` / `()` | `(0, 4096)`, stride `(4096, 1)` | 5000 | 1.119 us +/- 0.005, var 0.001 | 5.036 us +/- 0.013, var 0.003 | 0.22x |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 16 | 68.391 us +/- 0.199, var 0.276 | 21.310 us +/- 0.122, var 0.064 | 3.21x |
| `same_noncontiguous_transpose_control` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 1141.751 us +/- 3.811, var 39.105 | 107.392 us +/- 0.526, var 3.595 | 10.63x |
| `heldout_offset_transposed_509x521` | offset transposed `(509, 521)` / `(509, 521)`, stride `(1, 509)` | `(509, 521)`, stride `(1, 509)` | 2 | 265.892 us +/- 1.918, var 10.517 | 56.826 us +/- 0.851, var 2.801 | 4.68x |
| `heldout_channels_last_8x15x31x33` | channels-last `(8, 15, 31, 33)` / `(8, 15, 31, 33)` | `(8, 15, 31, 33)`, stride `(15345, 1, 495, 15)` | 4 | 124.728 us +/- 0.784, var 1.949 | 30.096 us +/- 0.128, var 0.469 | 4.14x |
