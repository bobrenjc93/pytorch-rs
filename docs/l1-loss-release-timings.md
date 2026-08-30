# `torch.nn.functional.l1_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`88074b85d49b863227ce8a546ee3394b51fbdf44`.

Command shape: worktree-local `uv venv --clear --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, then
release wheel builds through `maturin build --release --locked` and
installation with `uv pip install --force-reinstall --no-deps`. The clean base
wheel was built from a `git archive HEAD` snapshot under
`target/l1-loss-bench-artifacts`; candidate wheels were built from this
worktree. The timing driver ran against the installed wheels after imports and
input construction, pinned to CPU 24 with `taskset -c 24`, with 15 warmup blocks
and 81 measured blocks per implementation. Inputs were CPU `float32` tensors.
`UserWarning` was ignored symmetrically for both implementations inside the
measured region.

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

Results: the focused L1 Python tests passed 23 tests. The wheel-installed full
Python suite passed 4206 tests with 3 skips.

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
  31 packages in 15.47s, and installed in 1.24s
- Build time: clean `HEAD` base release wheel build completed in 30.76s; final
  candidate release wheel build completed in 24.23s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Same-Shape Contiguous Controls

Relative to the clean `HEAD` base, the same-shape contiguous controls changed
by a geometric mean of -0.46%. No same-shape contiguous control regressed by
more than 2.31%.

Geometric mean `torch_rs / PyTorch` slowdown for these same-shape contiguous
cells:

- Uncapped: 0.18x
- Capped to `[0.10x, 10.00x]` per cell: 0.21x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_scalar` | `()` / `()` | `()`, stride `()` | 50,000 | 0.402 us +/- 0.004, var 0.000 | 3.918 us +/- 0.025, var 0.011 | 0.10x | 0.413 us | -2.74% |
| `same_empty_0x4096` | `(0, 4096)` / `(0, 4096)` | `(0, 4096)`, stride `(4096, 1)` | 200,000 | 0.220 us +/- 0.001, var 0.000 | 3.727 us +/- 0.022, var 0.024 | 0.06x | 0.222 us | -0.54% |
| `same_small_17x19` | `(17, 19)` / `(17, 19)` | `(17, 19)`, stride `(19, 1)` | 2,000 | 0.343 us +/- 0.004, var 0.000 | 4.275 us +/- 0.020, var 0.004 | 0.08x | 0.335 us | +2.31% |
| `same_prime_257x263` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 32 | 12.533 us +/- 0.160, var 0.134 | 22.855 us +/- 0.147, var 0.226 | 0.55x | 12.732 us | -1.56% |
| `same_bandwidth_2048x2048` | `(2048, 2048)` / `(2048, 2048)` | `(2048, 2048)`, stride `(2048, 1)` | 1 | 1489.964 us +/- 17.096, var 1405.488 | 2287.450 us +/- 59.791, var 11254.406 | 0.65x | 1485.407 us | +0.31% |

## Broadcast Controls

Relative to the clean `HEAD` base, the broadcast controls changed by a
geometric mean of +0.56%. No broadcast control regressed by more than 3.95%.

Geometric mean `torch_rs / PyTorch` slowdown for broadcast controls:

- Uncapped: 0.86x
- Capped to `[0.10x, 10.00x]` per cell: 0.86x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `broadcast_scalar_input_640x768` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 40 | 121.012 us +/- 1.800, var 13.850 | 105.456 us +/- 0.305, var 0.685 | 1.15x | 123.528 us | -2.04% |
| `broadcast_scalar_target_640x768` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 40 | 100.680 us +/- 1.816, var 257.943 | 105.473 us +/- 0.309, var 113.905 | 0.95x | 103.404 us | -2.63% |
| `broadcast_vector_target_257x263` | `(257, 263)` / `(263,)` | `(257, 263)`, stride `(263, 1)` | 256 | 19.949 us +/- 0.075, var 0.632 | 25.018 us +/- 0.949, var 1.178 | 0.80x | 19.341 us | +3.14% |
| `broadcast_column_target_257x263` | `(257, 263)` / `(257, 1)` | `(257, 263)`, stride `(263, 1)` | 256 | 15.196 us +/- 0.207, var 1.136 | 23.721 us +/- 0.056, var 0.218 | 0.64x | 14.619 us | +3.95% |

## Held-Out Matching-Dense Noncontiguous

Relative to the clean `HEAD` base, the same-shape matching-dense
noncontiguous cells improved by a geometric mean of 69.70%. These are the
held-out L1 cells targeted by the new fast path.

Geometric mean `torch_rs / PyTorch` slowdown for these noncontiguous cells:

- Uncapped: 0.61x
- Capped to `[0.10x, 10.00x]` per cell: 0.61x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `noncontig_transpose_512x1024` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 5 | 112.805 us +/- 2.456, var 44.817 | 134.950 us +/- 1.584, var 10.193 | 0.84x | 363.969 us | -69.01% |
| `noncontig_offset_transposed_509x521` | offset transposed `(509, 521)` / `(509, 521)`, stride `(1, 509)` | `(509, 521)`, stride `(1, 509)` | 5 | 50.438 us +/- 1.667, var 16.429 | 70.172 us +/- 1.701, var 6.003 | 0.72x | 171.125 us | -70.53% |
| `noncontig_channels_last_8x15x31x33` | channels-last `(8, 15, 31, 33)` / `(8, 15, 31, 33)` | `(8, 15, 31, 33)`, stride `(15345, 1, 495, 15)` | 8 | 21.749 us +/- 0.244, var 2.542 | 36.048 us +/- 0.147, var 2.640 | 0.60x | 74.241 us | -70.70% |
| `noncontig_singleton_131x1x127` | permuted singleton `(127, 1, 131)` / `(127, 1, 131)`, stride `(1, 127, 127)` | `(127, 1, 131)`, stride `(1, 127, 127)` | 128 | 3.109 us +/- 0.014, var 0.041 | 7.966 us +/- 0.048, var 0.034 | 0.39x | 9.867 us | -68.49% |

## Control Regression Summary

Across the existing same-shape contiguous and broadcast controls, the geometric
mean changed by -0.01%. The worst control movement was
`broadcast_column_target_257x263`, from 14.619 us to 15.196 us (+3.95%), which
is below the 5% regression threshold.
