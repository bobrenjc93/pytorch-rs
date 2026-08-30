# `torch.nn.functional.l1_loss(reduction="none")` Release Timings

## 2026-08-30 Scalar-Broadcast Fast Path

Revision under test: uncommitted worktree based on
`977ed0531ca513267ff1a5793207b3409d8e3b42`.

Command shape: worktree-local `uv venv --clear --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, then
release wheel builds through `maturin build --release --locked` and
installation with `uv pip install --force-reinstall --no-deps`. The clean base
wheel was built from a `git archive HEAD` snapshot under
`target/l1-scalar-benchmark.b5vSYN/base-src`; candidate wheels were built from
this worktree. The final timing run pinned each process with `taskset -c 24`,
used 15 warmup blocks and 101 measured blocks per implementation, and ran with
`CUDA_VISIBLE_DEVICES=` plus one-thread BLAS/OpenMP environment settings.
Inputs were created before timing as CPU `float32` tensors. Each `torch_rs`
result was checked against PyTorch 2.13 for values, shape, strides, and
broadcast warning text before timing. `UserWarning` was ignored symmetrically
inside the measured region, and each block consumed the last output's metadata
and representative values as a dead-code and deferred-work guard.

Checks run before timing:

```bash
/home/bobren/.cargo/bin/cargo fmt --check
git diff --check
/home/bobren/.cargo/bin/cargo clippy --all-targets -- -D warnings
/home/bobren/.cargo/bin/cargo test --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" \
  /home/bobren/.cargo/bin/cargo clippy --all-targets --features python-bindings -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  /home/bobren/.cargo/bin/cargo test --all-targets --features python-bindings
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
PATH="/home/bobren/.cargo/bin:$PATH" \
  UV_CACHE_DIR="$PWD/.uv-cache" \
  ./scripts/test-python.sh
```

Results: focused L1 Python tests passed 22 tests. The wheel-installed full
Python suite passed 4205 tests with 3 skips.

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
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  31 packages in 15.69s, and installed in 1.44s
- Build time: clean `HEAD` base release wheel build completed in 30.85s; final
  candidate release wheel build completed in 23.83s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

Relative to the clean `HEAD` base, the held-out scalar-broadcast L1 cells
improved by a geometric mean of 32.0%. The largest single-cell improvement was
`scalar_target_2d_heldout`, from 109.913 us to 59.028 us (-46.3%).

Geometric mean `torch_rs / PyTorch` slowdown for scalar-broadcast held-out
cells:

- Uncapped: 0.57x
- Capped to `[0.10x, 10.00x]` per cell: 0.57x

Same-shape contiguous controls regressed by 0.8% geometric mean, with worst
single-cell movement of +2.0%. Noncontiguous controls regressed by 0.3%
geometric mean, with worst single-cell movement of +2.0%. No same-shape
contiguous or noncontiguous control regressed by more than 5%.

Geometric mean `torch_rs / PyTorch` slowdown for same-shape contiguous
controls:

- Uncapped: 0.65x
- Capped to `[0.10x, 10.00x]` per cell: 0.65x

Geometric mean `torch_rs / PyTorch` slowdown for noncontiguous controls:

- Uncapped: 2.45x
- Capped to `[0.10x, 10.00x]` per cell: 2.45x

| Workload | Category | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scalar_input_2d_heldout` | scalar broadcast | `(640, 768)`, stride `(768, 1)` | 10 | 87.133 us +/- 1.298, var 60.545 | 95.442 us +/- 0.278, var 0.887 | 0.91x | 135.059 us | -35.5% |
| `scalar_target_2d_heldout` | scalar broadcast | `(640, 768)`, stride `(768, 1)` | 10 | 59.028 us +/- 1.821, var 15.915 | 95.638 us +/- 0.402, var 3.345 | 0.62x | 109.913 us | -46.3% |
| `scalar_input_3d_heldout` | scalar broadcast | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 299.543 us +/- 11.998, var 515.612 | 221.074 us +/- 5.208, var 65.747 | 1.35x | 546.476 us | -45.2% |
| `scalar_target_3d_heldout` | scalar broadcast | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 287.235 us +/- 9.724, var 338.175 | 214.154 us +/- 3.515, var 36.552 | 1.34x | 527.749 us | -45.6% |
| `scalar_input_empty_contiguous` | scalar broadcast | `(0, 4096)`, stride `(4096, 1)` | 5000 | 1.050 us +/- 0.004, var 0.005 | 5.814 us +/- 0.020, var 0.006 | 0.18x | 1.071 us | -2.0% |
| `scalar_target_empty_contiguous` | scalar broadcast | `(0, 4096)`, stride `(4096, 1)` | 5000 | 1.043 us +/- 0.003, var 0.000 | 5.876 us +/- 0.063, var 0.033 | 0.18x | 1.067 us | -2.3% |
| `same_contiguous_prime_control` | same-shape contiguous control | `(257, 263)`, stride `(263, 1)` | 16 | 12.160 us +/- 0.126, var 1.304 | 21.399 us +/- 0.519, var 0.412 | 0.57x | 11.927 us | +2.0% |
| `same_contiguous_bandwidth_control` | same-shape contiguous control | `(2048, 2048)`, stride `(2048, 1)` | 1 | 1504.425 us +/- 19.138, var 1109.055 | 2000.556 us +/- 19.099, var 4956.900 | 0.75x | 1510.715 us | -0.4% |
| `noncontig_transpose_control` | noncontiguous control | `(512, 1024)`, stride `(1, 512)` | 5 | 340.271 us +/- 8.865, var 276.822 | 124.322 us +/- 1.803, var 47.580 | 2.74x | 340.858 us | -0.2% |
| `noncontig_offset_transposed_control` | noncontiguous control | `(509, 521)`, stride `(1, 509)` | 5 | 161.715 us +/- 4.018, var 45.787 | 64.301 us +/- 0.565, var 4.255 | 2.51x | 158.482 us | +2.0% |
| `noncontig_channels_last_control` | noncontiguous control | `(8, 15, 31, 33)`, stride `(15345, 1, 495, 15)` | 8 | 71.425 us +/- 0.860, var 19.074 | 33.252 us +/- 0.198, var 2.082 | 2.15x | 72.055 us | -0.9% |

## 2026-08-30 Same-Shape Contiguous Fast Path

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`46fa2fa2a66f1e38e331e8011d83611a33b59f82`

Command shape: worktree-local `uv venv --clear --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, then
release wheel builds through `maturin build --release --locked` and
installation with `uv pip install --force-reinstall --no-deps`. The clean base
wheel was built from a `git archive HEAD` snapshot under
`target/l1-loss-base-src`; candidate wheels were built from this worktree. The
timing driver ran against the installed wheels after imports and input
construction, with 15 warmup blocks and 81 measured blocks per implementation.
Inputs were CPU `float32` tensors. The focused test suite checked broadcast
size-mismatch warning parity, and `UserWarning` was ignored symmetrically for
both implementations inside the measured region.

The timings below measure eager `l1_loss(reduction="none")` construction. The
driver materialized and bit-compared each result against PyTorch before timing
and consumed the last output after every warmup and measured block as a
dead-code and deferred-work guard.

Checks run before timing:

```bash
/home/bobren/.cargo/bin/cargo fmt --check
git diff --check
/home/bobren/.cargo/bin/cargo clippy --all-targets -- -D warnings
/home/bobren/.cargo/bin/cargo test --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" \
  /home/bobren/.cargo/bin/cargo clippy --all-targets --features python-bindings -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  /home/bobren/.cargo/bin/cargo test --all-targets --features python-bindings
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
PATH="/home/bobren/.cargo/bin:$PATH" \
  UV_CACHE_DIR="$PWD/.uv-cache" \
  ./scripts/test-python.sh
```

Results: the focused L1 Python tests passed 19 tests. The wheel-installed full
Python suite passed 4202 tests with 3 skips.

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
- Dependency installation: locked `uv sync` resolved in 30 ms, prepared
  31 packages in 17.44s, and installed in 3.69s
- Build time: first successful candidate release extension build completed in
  31.53s; clean `HEAD` base release wheel build completed in 30.62s; the final
  cached candidate wheel rebuild completed in 0.01s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Same-Shape Contiguous

Relative to the clean `HEAD` base, the same-shape contiguous held-out cells
improved by a geometric mean of 27.5%. The largest single-cell improvement was
the bandwidth-sized `(2048, 2048)` case, from 2382.644 us to 1518.897 us
(-36.3%).

Geometric mean `torch_rs / PyTorch` slowdown for these same-shape contiguous
cells:

- Uncapped: 0.20x
- Capped to `[0.10x, 10.00x]` per cell: 0.22x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_scalar` | `()` / `()` | `()`, stride `()` | 50,000 | 0.441 us +/- 0.005, var 0.000 | 3.983 us +/- 0.025, var 0.002 | 0.11x | 0.529 us | -16.6% |
| `same_empty_0x4096` | `(0, 4096)` / `(0, 4096)` | `(0, 4096)`, stride `(4096, 1)` | 20,000 | 0.267 us +/- 0.002, var 0.000 | 3.760 us +/- 0.011, var 0.001 | 0.07x | 0.342 us | -21.9% |
| `same_small_17x19` | `(17, 19)` / `(17, 19)` | `(17, 19)`, stride `(19, 1)` | 2,000 | 0.371 us +/- 0.002, var 0.000 | 4.336 us +/- 0.030, var 0.010 | 0.09x | 0.549 us | -32.4% |
| `same_prime_257x263` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 32 | 12.290 us +/- 0.201, var 1.224 | 22.475 us +/- 0.076, var 0.363 | 0.55x | 17.161 us | -28.4% |
| `same_bandwidth_2048x2048` | `(2048, 2048)` / `(2048, 2048)` | `(2048, 2048)`, stride `(2048, 1)` | 1 | 1518.897 us +/- 32.689, var 7091.694 | 1745.229 us +/- 49.506, var 8893.894 | 0.87x | 2382.644 us | -36.3% |

## Broadcast And Noncontiguous Controls

Relative to the clean `HEAD` base, no existing broadcast or noncontiguous L1
control regressed by more than 5%. The worst movement was
`broadcast_scalar_input_640x768`, from 124.282 us to 129.423 us (+4.1%). The
combined control geometric mean improved by 12.4%.

Geometric mean `torch_rs / PyTorch` slowdown for broadcast controls:

- Uncapped: 0.94x
- Capped to `[0.10x, 10.00x]` per cell: 0.94x

Geometric mean `torch_rs / PyTorch` slowdown for noncontiguous controls:

- Uncapped: 2.35x
- Capped to `[0.10x, 10.00x]` per cell: 2.35x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `broadcast_scalar_input_640x768` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 10 | 129.423 us +/- 2.992, var 139.741 | 95.669 us +/- 0.343, var 20.298 | 1.35x | 124.282 us | +4.1% |
| `broadcast_scalar_target_640x768` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 10 | 101.985 us +/- 2.815, var 19.953 | 96.795 us +/- 0.394, var 20.976 | 1.05x | 100.738 us | +1.2% |
| `broadcast_vector_target_257x263` | `(257, 263)` / `(263,)` | `(257, 263)`, stride `(263, 1)` | 64 | 18.578 us +/- 0.123, var 0.099 | 22.590 us +/- 0.208, var 0.500 | 0.82x | 18.811 us | -1.2% |
| `broadcast_column_target_257x263` | `(257, 263)` / `(257, 1)` | `(257, 263)`, stride `(263, 1)` | 64 | 14.476 us +/- 0.074, var 0.169 | 21.955 us +/- 0.105, var 0.700 | 0.66x | 15.267 us | -5.2% |
| `noncontig_transpose_512x1024` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 5 | 327.884 us +/- 5.847, var 192.785 | 127.160 us +/- 1.627, var 116.498 | 2.58x | 440.712 us | -25.6% |
| `noncontig_offset_transposed_509x521` | offset transposed `(509, 521)` / `(509, 521)`, stride `(1, 509)` | `(509, 521)`, stride `(1, 509)` | 5 | 156.921 us +/- 4.865, var 101.797 | 66.298 us +/- 0.785, var 32.601 | 2.37x | 220.305 us | -28.8% |
| `noncontig_channels_last_8x15x31x33` | channels-last `(8, 15, 31, 33)` / `(8, 15, 31, 33)` | `(8, 15, 31, 33)`, stride `(15345, 1, 495, 15)` | 8 | 73.641 us +/- 2.087, var 98.383 | 34.577 us +/- 0.274, var 11.290 | 2.13x | 97.010 us | -24.1% |
