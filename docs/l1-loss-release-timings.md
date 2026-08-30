# `torch.nn.functional.l1_loss(reduction="none")` Release Timings

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
