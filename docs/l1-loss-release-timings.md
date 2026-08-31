# `torch.nn.functional.l1_loss(reduction="none")` Release Timings

Date: 2026-08-31

Candidate provenance: source snapshot based on
`df21e2bbb9432940e5152103f92e9e64ecac6491`, plus the worktree changes that add
the rank-4 channels-last L1 fast path.

Command shape: from the repository root, locked
`uv sync --locked --no-install-project --group dev --group reference`, then
release wheel builds through `maturin build --release --locked` and
installation with `uv pip install --force-reinstall --no-deps`. The clean base
wheel was built from a `git archive HEAD` snapshot under the repo-local
`target/l1-channels-last-bench.1prRpr/base-src` directory; the candidate wheel
was built from the source snapshot under test. The timing driver ran against
the installed wheels after imports and input construction, with 15 warmup
blocks and 81 measured blocks per implementation. Inputs were CPU `float32`
tensors. Broadcast size-mismatch warning parity was checked before timing, then
`UserWarning` was ignored symmetrically for both implementations inside the
measured region.

The timings below measure eager `l1_loss(reduction="none")` construction. The
driver materialized and bit-compared each result against PyTorch before timing
and consumed the last output after every warmup and measured block as a
dead-code and deferred-work guard. Two pinned process passes were run for the
base wheel, candidate wheel, and PyTorch reference. Reported medians are the
medians of those per-process medians; MAD and variance columns report the
median per-process MAD and variance.

Checks run before timing:

```bash
cargo fmt --check
git diff --check
cargo test absolute_difference --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --all-targets --features python-bindings -- -D warnings
RUSTFLAGS="-L native=/usr/local/fbcode/platform010/lib -l dylib=python3.12" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --all-targets --features extension-module
UV_CACHE_DIR="$PWD/.uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" \
  ./scripts/test-python.sh
```

Results: the focused L1 Python tests passed 27 tests. The wheel-installed full
Python suite passed 4243 tests with 3 skips.

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
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 29 ms, prepared
  packages in 15.98s, and installed in 2.20s
- Build time: clean `HEAD` base release wheel build completed in 32.46s; the
  final candidate release wheel rebuild completed in 0.40s using existing
  release artifacts from the preceding checks

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Channels-Last Held-Out

Relative to the clean `HEAD` base, the held-out rank-4 channels-last L1 cells
improved by a geometric mean of 74.7%. Including the existing channels-last
control cell, all channels-last cells improved by a geometric mean of 75.5%.
The existing `noncontig_channels_last_8x15x31x33_control` moved from 99.956 us
to 21.648 us (-78.3%), changing its `torch_rs / PyTorch` ratio from a
noncontiguous fallback-like result to 0.60x.

Geometric mean `torch_rs / PyTorch` slowdown for held-out rank-4 channels-last
cells:

- Uncapped: 0.30x
- Capped to `[0.10x, 10.00x]` per cell: 0.30x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `noncontig_channels_last_8x15x31x33_control` | `(8, 15, 31, 33), stride (15345, 1, 495, 15)` / `(8, 15, 31, 33), stride (15345, 1, 495, 15)` | `(8, 15, 31, 33), stride (15345, 1, 495, 15)` | 8 | 21.648 us +/- 0.641, var 4.637 | 36.168 us +/- 0.525, var 1.756 | 0.60x | 99.956 us | -78.3% |
| `channels_last_7x13x29x31_heldout` | `(7, 13, 29, 31), stride (11687, 1, 403, 13)` / `(7, 13, 29, 31), stride (11687, 1, 403, 13)` | `(7, 13, 29, 31), stride (11687, 1, 403, 13)` | 16 | 14.349 us +/- 0.163, var 1.256 | 27.439 us +/- 0.395, var 0.583 | 0.52x | 65.600 us | -78.1% |
| `channels_last_3x5x17x19_heldout` | `(3, 5, 17, 19), stride (1615, 1, 95, 5)` / `(3, 5, 17, 19), stride (1615, 1, 95, 5)` | `(3, 5, 17, 19), stride (1615, 1, 95, 5)` | 128 | 1.235 us +/- 0.003, var 0.001 | 6.346 us +/- 0.041, var 0.017 | 0.19x | 4.340 us | -71.5% |
| `channels_last_singleton_h_11x17x1x23_heldout` | `(11, 17, 1, 23), stride (391, 1, 391, 17)` / `(11, 17, 1, 23), stride (391, 1, 391, 17)` | `(11, 17, 1, 23), stride (391, 1, 391, 17)` | 128 | 1.170 us +/- 0.006, var 0.003 | 6.227 us +/- 0.029, var 0.064 | 0.19x | 3.919 us | -70.1% |
| `channels_last_tail_2x9x37x41_heldout` | `(2, 9, 37, 41), stride (13653, 1, 369, 9)` / `(2, 9, 37, 41), stride (13653, 1, 369, 9)` | `(2, 9, 37, 41), stride (13653, 1, 369, 9)` | 32 | 4.779 us +/- 0.019, var 0.088 | 11.256 us +/- 0.129, var 0.139 | 0.42x | 21.620 us | -77.9% |

## Regression Controls

Relative to the clean `HEAD` base, no existing contiguous, scalar-broadcast,
transposed, or non-scalar broadcast control regressed by more than 5%. Across
those non-channels-last controls, current vs base improved by a geometric mean
of 3.6%; the largest positive movement was
`column_target_2d_broadcast_control`, from 92.720 us to 95.866 us (+3.4%).

Geometric mean `torch_rs / PyTorch` slowdown by control category:

- Same-shape contiguous controls: 0.61x uncapped, 0.61x capped
- Scalar-broadcast controls: 0.95x uncapped, 0.95x capped
- Non-scalar broadcast controls: 1.01x uncapped, 1.01x capped
- Rank-2 transposed controls: 2.89x uncapped, 2.89x capped

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_prime_control` | `(257, 263), stride (263, 1)` / `(257, 263), stride (263, 1)` | `(257, 263), stride (263, 1)` | 32 | 12.823 us +/- 0.291, var 0.707 | 23.002 us +/- 0.173, var 0.297 | 0.56x | 12.943 us | -0.9% |
| `same_contiguous_bandwidth_control` | `(2048, 2048), stride (2048, 1)` / `(2048, 2048), stride (2048, 1)` | `(2048, 2048), stride (2048, 1)` | 1 | 1519.533 us +/- 44.457, var 5224.588 | 2286.667 us +/- 154.694, var 50105.593 | 0.66x | 1788.834 us | -15.1% |
| `scalar_input_2d_heldout` | `(), stride ()` / `(640, 768), stride (768, 1)` | `(640, 768), stride (768, 1)` | 10 | 94.732 us +/- 1.073, var 9.332 | 99.258 us +/- 0.428, var 1.922 | 0.95x | 99.931 us | -5.2% |
| `scalar_target_2d_heldout` | `(640, 768), stride (768, 1)` / `(), stride ()` | `(640, 768), stride (768, 1)` | 10 | 57.429 us +/- 1.365, var 13.970 | 100.839 us +/- 0.449, var 1.732 | 0.57x | 63.137 us | -9.0% |
| `scalar_input_3d_heldout` | `(), stride ()` / `(17, 257, 263), stride (67591, 263, 1)` | `(17, 257, 263), stride (67591, 263, 1)` | 1 | 303.603 us +/- 9.208, var 523.977 | 235.185 us +/- 3.670, var 147.163 | 1.29x | 305.236 us | -0.5% |
| `scalar_target_3d_heldout` | `(17, 257, 263), stride (67591, 263, 1)` / `(), stride ()` | `(17, 257, 263), stride (67591, 263, 1)` | 1 | 279.418 us +/- 11.001, var 1577.155 | 237.118 us +/- 4.853, var 87.688 | 1.18x | 308.616 us | -9.5% |
| `vector_target_2d_broadcast_control` | `(640, 768), stride (768, 1)` / `(768,), stride (1,)` | `(640, 768), stride (768, 1)` | 32 | 121.972 us +/- 1.612, var 25.821 | 107.733 us +/- 0.398, var 1.441 | 1.13x | 119.515 us | +2.1% |
| `column_target_2d_broadcast_control` | `(640, 768), stride (768, 1)` / `(640, 1), stride (1, 1)` | `(640, 768), stride (768, 1)` | 32 | 95.866 us +/- 2.045, var 13.118 | 106.970 us +/- 0.680, var 1.489 | 0.90x | 92.720 us | +3.4% |
| `noncontig_transpose_512x1024_control` | `(512, 1024), stride (1, 512)` / `(512, 1024), stride (1, 512)` | `(512, 1024), stride (1, 512)` | 5 | 441.212 us +/- 5.403, var 590.317 | 159.586 us +/- 1.310, var 14.304 | 2.76x | 443.047 us | -0.4% |
| `noncontig_offset_transposed_509x521_control` | `(509, 521), stride (1, 509)` / `(509, 521), stride (1, 509)` | `(509, 521), stride (1, 509)` | 5 | 216.200 us +/- 2.691, var 163.695 | 71.592 us +/- 1.177, var 2.247 | 3.02x | 214.375 us | +0.9% |
