# `torch.nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`ba614d70727f67898eba3ff5ea1ba0b926ee1950`.

Build command: release `maturin develop --release --locked` from the current
worktree, installed into the worktree-local `.venv`. Cargo dependency cache,
temporary files, and build output were kept under `target/`.

Correctness and lint checks run before timing:

```bash
env CARGO_HOME="$PWD/target/cargo-home" CARGO_NET_OFFLINE=true \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo fmt --check

env CARGO_HOME="$PWD/target/cargo-home" CARGO_NET_OFFLINE=true \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo clippy --all-targets -- -D warnings

env CARGO_HOME="$PWD/target/cargo-home" CARGO_NET_OFFLINE=true \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo clippy --all-targets --features python-bindings -- -D warnings

env CARGO_HOME="$PWD/target/cargo-home" CARGO_NET_OFFLINE=true \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo test --all-targets

env CARGO_HOME="$PWD/target/cargo-home" CARGO_NET_OFFLINE=true \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo test --all-targets --features python-bindings

env PYTHONPATH="$PWD/python" \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
```

Results: Rust test suites passed, including 273 tests without Python bindings
and 284 tests with Python bindings. The focused native and PyTorch 2.13
differential MSE Python suites passed 25 tests.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.13
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0, imported from the worktree
  `.venv`
- `torch_rs`: 0.1.0 from `python/torch_rs`, native extension built in release
  profile with Cargo `[profile.release]` thin LTO and one codegen unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for timing
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1

The timing driver built inputs once, performed 13 warmup blocks and 71 measured
blocks per implementation, and compared equivalent `torch_rs` and PyTorch
calls with the same dtype, device, shape, stride, and thread settings. Before
timing each workload, the `torch_rs` output was bitwise-checked against PyTorch
2.13.0, including output shape and stride metadata. During timing, each eager
output was consumed through a data-pointer/metadata sink. Broadcast
size-mismatch warnings were ignored inside the timed loops for both
implementations; the focused differential tests above cover warning parity.

Times below are median microseconds per call. MAD is median absolute deviation
in microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. The capped ratio clamps each per-cell ratio to
`[0.10x, 10.00x]` before geometric aggregation.

Geometric mean `torch_rs / PyTorch` slowdown for held-out scalar-broadcast
cells:

- Uncapped: 0.63x
- Capped to `[0.10x, 10.00x]` per cell: 0.63x

Geometric mean `torch_rs / PyTorch` slowdown for all cells in the table:

- Uncapped: 0.83x
- Capped to `[0.10x, 10.00x]` per cell: 0.83x

## Eager Output Creation

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Capped ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)`, contiguous | `(257, 263)`, stride `(263, 1)` | 64 | 12.433 us +/- 0.045, var 0.126 | 15.243 us +/- 0.568, var 0.977 | 0.82x | 0.82x |
| `same_contiguous_bandwidth_control` | `(1536, 1536)` / `(1536, 1536)`, contiguous | `(1536, 1536)`, stride `(1536, 1)` | 4 | 667.246 us +/- 23.288, var 4149.682 | 569.195 us +/- 11.753, var 330.049 | 1.17x | 1.17x |
| `same_noncontiguous_transpose_control` | transposed `(512, 1024)` / `(512, 1024)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 4 | 244.369 us +/- 2.153, var 7.945 | 79.022 us +/- 1.558, var 5.453 | 3.09x | 3.09x |
| `scalar_input_contiguous_heldout_513x1021` | `()` / `(513, 1021)` | `(513, 1021)`, stride `(1021, 1)` | 8 | 52.910 us +/- 1.219, var 88.325 | 56.244 us +/- 0.418, var 1.214 | 0.94x | 0.94x |
| `scalar_target_contiguous_heldout_513x1021` | `(513, 1021)` / `()` | `(513, 1021)`, stride `(1021, 1)` | 8 | 48.599 us +/- 0.662, var 1.011 | 56.547 us +/- 0.985, var 1.910 | 0.86x | 0.86x |
| `scalar_input_contiguous_heldout_1537x257` | `()` / `(1537, 257)` | `(1537, 257)`, stride `(257, 1)` | 8 | 37.760 us +/- 0.606, var 1.247 | 47.609 us +/- 0.759, var 6754.235 | 0.79x | 0.79x |
| `scalar_target_contiguous_heldout_1537x257` | `(1537, 257)` / `()` | `(1537, 257)`, stride `(257, 1)` | 8 | 40.268 us +/- 0.571, var 0.853 | 48.878 us +/- 0.458, var 0.328 | 0.82x | 0.82x |
| `scalar_input_empty_contiguous_heldout` | `()` / `(17, 0, 19)` | `(17, 0, 19)`, stride `(19, 19, 1)` | 10000 | 2.278 us +/- 0.006, var 0.000 | 6.636 us +/- 0.031, var 0.060 | 0.34x | 0.34x |
| `scalar_target_empty_contiguous_heldout` | `(17, 0, 19)` / `()` | `(17, 0, 19)`, stride `(19, 19, 1)` | 10000 | 2.274 us +/- 0.006, var 0.000 | 6.668 us +/- 0.019, var 0.002 | 0.34x | 0.34x |

## Control Regression Guard

A clean `HEAD` snapshot was exported into `target/baseline_head_1788089334`,
built in release mode, installed into the same local `.venv`, and timed with
the identical eager-output driver. Positive percentages mean the candidate was
slower than the clean snapshot.

| Control | Clean `HEAD` `torch_rs` median | Candidate `torch_rs` median | Candidate change |
| --- | ---: | ---: | ---: |
| `same_contiguous_prime_control` | 13.177 us | 12.433 us | -5.6% |
| `same_contiguous_bandwidth_control` | 694.995 us | 667.246 us | -4.0% |
| `same_noncontiguous_transpose_control` | 366.288 us | 244.369 us | -33.3% |
