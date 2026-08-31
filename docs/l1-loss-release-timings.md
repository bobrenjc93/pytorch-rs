# `torch.nn.functional.l1_loss(reduction="none")` Release Timings

Date: 2026-08-30

Candidate provenance: source snapshot based on
`14dbf2f4df31972fd2d08d39533530ee483390bd`.

Command shape: from the repository root, `uv venv --clear --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, then
release wheel builds through `maturin build --release --locked` and
installation with `uv pip install --force-reinstall --no-deps`. The clean base
wheel was built from a `git archive HEAD` snapshot under a repo-local
`target/l1-scalar-broadcast.*/base-src` directory; the candidate wheel was
built from the source snapshot under test. The timing driver ran against the
installed wheels after imports and input construction, with 15 warmup blocks
and 81 measured blocks per implementation. Inputs were CPU `float32` tensors.
Broadcast size-mismatch warning parity was checked before timing, then
`UserWarning` was ignored symmetrically for both implementations inside the
measured region.

The timings below measure eager `l1_loss(reduction="none")` construction. The
driver materialized and bit-compared each result against PyTorch before timing
and consumed the last output after every warmup and measured block as a
dead-code and deferred-work guard.

Checks run before timing:

```bash
cargo fmt --check
git diff --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --all-targets --features python-bindings -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --all-targets --features python-bindings
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
.venv/bin/python -m unittest tests.test_readme_quickstart
UV_CACHE_DIR="$PWD/.uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" \
  ./scripts/test-python.sh
```

Results: the focused L1 Python tests passed 23 tests. The wheel-installed full
Python suite passed 4208 tests with 3 skips.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 32 ms, prepared
  31 packages in 15.57s, and installed in 1.44s
- Build time: clean `HEAD` base release wheel build completed in 30.96s; the
  final candidate release wheel rebuild completed in 24.69s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Scalar-Broadcast Held-Out

Relative to the clean `HEAD` base, scalar-broadcast L1 held-out cells improved
by a geometric mean of 32.3%. The largest non-empty improvement was
`scalar_target_2d_heldout`, from 103.331 us to 56.467 us (-45.4%).

Geometric mean `torch_rs / PyTorch` slowdown for scalar-broadcast held-out
cells:

- Uncapped: 0.53x
- Capped to `[0.10x, 10.00x]` per cell: 0.53x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scalar_input_2d_heldout` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 10 | 94.835 us +/- 1.827, var 68.547 | 96.892 us +/- 1.032, var 3.243 | 0.98x | 122.121 us | -22.3% |
| `scalar_target_2d_heldout` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 10 | 56.467 us +/- 1.648, var 14.124 | 97.034 us +/- 0.450, var 4.148 | 0.58x | 103.331 us | -45.4% |
| `scalar_input_3d_heldout` | `()` / `(17, 257, 263)` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 294.566 us +/- 7.300, var 396.657 | 231.630 us +/- 2.293, var 19.719 | 1.27x | 524.113 us | -43.8% |
| `scalar_target_3d_heldout` | `(17, 257, 263)` / `()` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 286.143 us +/- 7.621, var 425.572 | 231.319 us +/- 4.486, var 271.710 | 1.24x | 513.577 us | -44.3% |
| `scalar_input_empty_contiguous` | `()` / `(0, 4096)` | `(0, 4096)`, stride `(4096, 1)` | 20,000 | 0.970 us +/- 0.002, var 0.000 | 6.297 us +/- 0.043, var 0.003 | 0.15x | 1.151 us | -15.7% |
| `scalar_target_empty_contiguous` | `(0, 4096)` / `()` | `(0, 4096)`, stride `(4096, 1)` | 20,000 | 0.983 us +/- 0.003, var 0.000 | 6.310 us +/- 0.023, var 0.004 | 0.16x | 1.139 us | -13.7% |

## Same-Shape And Noncontiguous Controls

Relative to the clean `HEAD` base, no same-shape contiguous or noncontiguous L1
control regressed by more than 5%. The largest control movement was
`same_contiguous_prime_control`, from 11.939 us to 11.439 us (-4.2%). The
largest positive movement was `noncontig_offset_transposed_509x521_control`,
from 153.552 us to 157.068 us (+2.3%).

Geometric mean `torch_rs / PyTorch` slowdown for same-shape contiguous
controls:

- Uncapped: 0.26x
- Capped to `[0.10x, 10.00x]` per cell: 0.26x

Geometric mean `torch_rs / PyTorch` slowdown for noncontiguous controls:

- Uncapped: 2.29x
- Capped to `[0.10x, 10.00x]` per cell: 2.29x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 32 | 11.439 us +/- 0.126, var 0.044 | 22.320 us +/- 0.108, var 0.123 | 0.51x | 11.939 us | -4.2% |
| `same_contiguous_bandwidth_control` | `(2048, 2048)` / `(2048, 2048)` | `(2048, 2048)`, stride `(2048, 1)` | 1 | 1491.415 us +/- 27.162, var 1797.746 | 11120.218 us +/- 174.424, var 90598.592 | 0.13x | 1500.389 us | -0.6% |
| `noncontig_transpose_512x1024_control` | `(512, 1024)` / `(512, 1024)` | `(512, 1024)`, stride `(1, 512)` | 5 | 328.124 us +/- 2.610, var 163.245 | 132.498 us +/- 3.297, var 54.977 | 2.48x | 332.589 us | -1.3% |
| `noncontig_offset_transposed_509x521_control` | `(509, 521)` / `(509, 521)` | `(509, 521)`, stride `(1, 509)` | 5 | 157.068 us +/- 3.075, var 62.690 | 67.470 us +/- 0.689, var 1.011 | 2.33x | 153.552 us | +2.3% |
| `noncontig_channels_last_8x15x31x33_control` | `(8, 15, 31, 33)` / `(8, 15, 31, 33)` | `(8, 15, 31, 33)`, stride `(15345, 1, 495, 15)` | 8 | 71.925 us +/- 0.973, var 4.355 | 34.736 us +/- 0.348, var 0.392 | 2.07x | 71.706 us | +0.3% |
