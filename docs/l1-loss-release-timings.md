# `torch.nn.functional.l1_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`ae55c0b1273f277e57961be230582a661d0ff1da`

Command shape: worktree-local `uv venv --clear --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, then
release wheels for a clean `HEAD` export and the current worktree. The timing
driver ran after imports and input construction, with 9 warmup blocks and 51
measured blocks per implementation. Inputs were CPU `float32` tensors. Every
`torch_rs` cell was checked against PyTorch 2.13 for output shape, stride, and
bitwise values before timing. Broadcast size-mismatch warnings were ignored
symmetrically inside timed regions.

The timings below measure eager `l1_loss(reduction="none")` construction and
consume the last output after each measured block as a dead-code and
deferred-work guard.

Checks run before timing:

```bash
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true \
  cargo fmt --check
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true \
  cargo clippy --all-targets -- -D warnings
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true \
  cargo test --all-targets
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --all-targets --features python-bindings -- -D warnings
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  ./scripts/test-python.sh
```

Results: the focused L1 Python tests passed 21 tests. The wheel-installed full
Python suite passed 4200 tests with 3 skips.

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
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1 in the timed `torch_rs` processes
- Dependency installation: locked `uv sync` resolved in 29 ms and installed in
  1.55s
- Build time: base release extension build completed in 31.07s; the first
  current release extension build after the final source change completed in
  24.76s; the later wheel-based test build reused cached artifacts and reported
  0.01s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. `Current vs Base` compares this worktree against a
clean `HEAD` export built in the same benchmark run.

## L1 Construction

The same-shape contiguous held-out cells improve by a geometric mean of 31.2%
versus base and run at a geometric mean `torch_rs / PyTorch` slowdown of 0.14x
uncapped, 0.21x capped to `[0.10x, 10.00x]` per cell.

Existing broadcast and noncontiguous L1 controls did not regress by more than
5%. Broadcast controls moved by +2.8% and +3.6%; noncontiguous controls moved
by -22.7% and -25.6%. The control geometric mean moved by -11.5%.

| Workload | Input / target | Output | Repeats | Current `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | Current / PyTorch | Base `torch_rs` median | Current vs Base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_scalar_heldout` | `()` / `()` | `()`, stride `()` | 5000 | 0.206 us +/- 0.001, var 0.000 | 4.872 us +/- 0.046, var 0.056 | 0.04x | 0.257 us | -20.1% |
| `same_contiguous_empty_0x4099` | `(0, 4099)` / `(0, 4099)` | `(0,4099)`, stride `(4099,1)` | 5000 | 0.223 us +/- 0.002, var 0.000 | 4.611 us +/- 0.049, var 0.062 | 0.05x | 0.322 us | -30.9% |
| `same_contiguous_small_7x13` | `(7, 13)` / `(7, 13)` | `(7,13)`, stride `(13,1)` | 5000 | 0.267 us +/- 0.002, var 0.000 | 5.172 us +/- 0.010, var 0.003 | 0.05x | 0.364 us | -26.6% |
| `same_contiguous_prime_509x521` | `(509, 521)` / `(509, 521)` | `(509,521)`, stride `(521,1)` | 16 | 45.898 us +/- 1.371, var 3.548 | 65.153 us +/- 0.896, var 2.608 | 0.70x | 69.925 us | -34.4% |
| `same_contiguous_bandwidth_4096x4096` | `(4096, 4096)` / `(4096, 4096)` | `(4096,4096)`, stride `(4096,1)` | 1 | 29941.882 us +/- 461.467, var 687627.262 | 48013.992 us +/- 756.804, var 2029068.260 | 0.62x | 51512.995 us | -41.9% |
| `broadcast_vector_target_control` | `(640, 768)` / `(768,)` | `(640,768)`, stride `(768,1)` | 4 | 154.421 us +/- 4.154, var 138.189 | 100.169 us +/- 1.350, var 14.892 | 1.54x | 150.169 us | +2.8% |
| `broadcast_scalar_target_control` | `(640, 768)` / `()` | `(640,768)`, stride `(768,1)` | 4 | 133.334 us +/- 3.823, var 502.596 | 95.159 us +/- 0.518, var 1.345 | 1.40x | 128.644 us | +3.6% |
| `noncontiguous_transpose_control` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512,1024)`, stride `(1,512)` | 4 | 349.438 us +/- 4.860, var 65.800 | 133.242 us +/- 1.357, var 25.736 | 2.62x | 452.344 us | -22.7% |
| `offset_transposed_control` | offset transposed `(509, 521)` / `(509, 521)`, stride `(1, 509)` | `(509,521)`, stride `(1,509)` | 16 | 154.827 us +/- 2.069, var 48.955 | 66.814 us +/- 0.264, var 3.033 | 2.32x | 208.090 us | -25.6% |
