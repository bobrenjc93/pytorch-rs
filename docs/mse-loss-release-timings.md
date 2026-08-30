# `torch.nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`080389a4da20a5048ef0961c90a96dd33231b9f4`

Command shape: release `maturin develop --release --locked` build from the
current worktree, installed into the worktree-local `.venv`. The timing driver
ran after imports, input construction, and a full bitwise PyTorch 2.13.0 parity
check for every workload. The timed loop constructed a fresh
`mse_loss(reduction="none")` output from exact CPU `float32` tensors and
consumed output pointer, shape, stride, and element-count metadata. This keeps
the benchmark focused on eager MSE output materialization instead of measuring
this project's slower reduction implementation as part of the checksum.

The focused native and reference MSE tests were run before timing:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
```

Result: 25 tests passed.

Additional checks run for this change:

```bash
CARGO_HOME="$PWD/.cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo fmt --check

CARGO_HOME="$PWD/.cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo clippy --all-targets --locked -- -D warnings

PYO3_PYTHON="$PWD/.venv/bin/python" \
  CARGO_HOME="$PWD/.cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo clippy --all-targets --features python-bindings --locked -- -D warnings

CARGO_HOME="$PWD/.cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo test --all-targets --locked

env -u PYTHONPATH \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  PYTHONHOME="/home/bobren/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu" \
  LD_LIBRARY_PATH="/home/bobren/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH-}" \
  CARGO_HOME="$PWD/.cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo test --all-targets --features python-bindings --locked
```

Results: non-Python Rust tests passed 94 library tests, 79 autograd tests, and
100 tensor-baseline tests. Python-bindings Rust tests passed 105 library tests,
79 autograd tests, and 100 tensor-baseline tests.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.10.19 from worktree-local `.venv`
- NumPy: 2.2.6
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.10/site-packages/torch`
- `torch_rs`: 0.1.0 from `python/torch_rs`, native extension
  `python/torch_rs/torch_rs.abi3.so`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1

Broadcast size-mismatch warnings were ignored inside the timed loops for both
implementations; the focused reference tests above cover warning parity. Times
below are median microseconds per call after 9 warmup blocks and 51 measured
blocks. MAD is median absolute deviation in microseconds, and variance is
sample variance of the per-call sample timings in microseconds squared.
`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. The capped ratio clamps each per-cell ratio to `[0.10x, 10.00x]`
before geometric aggregation.

The same-shape and noncontiguous rows are regression controls. The new scalar
broadcast helper is only reachable when exactly one operand is rank-0 and the
other operand has row-major contiguous storage, so those controls remain on
their existing same-shape and fallback paths.

Geometric mean `torch_rs / PyTorch` slowdown for scalar-broadcast cells:

- Uncapped: 1.19x
- Capped to `[0.10x, 10.00x]` per cell: 1.19x

Geometric mean `torch_rs / PyTorch` slowdown for the same-shape and
noncontiguous controls:

- Uncapped: 1.70x
- Capped to `[0.10x, 10.00x]` per cell: 1.70x

Geometric mean `torch_rs / PyTorch` slowdown for all cells in the table:

- Uncapped: 1.74x
- Capped to `[0.10x, 10.00x]` per cell: 1.36x

## Native Materialization

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Capped ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_prime_heldout` | `(257, 263)` / `(257, 263)`, contiguous | `(257, 263)`, stride `(263, 1)` | 64 | 12.628 us +/- 0.055, var 0.023 | 14.762 us +/- 0.055, var 0.055 | 0.86x | 0.86x |
| `same_contiguous_bandwidth_heldout` | `(1536, 1536)` / `(1536, 1536)`, contiguous | `(1536, 1536)`, stride `(1536, 1)` | 4 | 867.399 us +/- 23.784, var 953.686 | 621.990 us +/- 10.145, var 134.958 | 1.39x | 1.39x |
| `same_noncontiguous_transpose` | transposed `(512, 1024)` / `(512, 1024)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 8 | 364.336 us +/- 3.112, var 723.020 | 88.098 us +/- 0.980, var 4.193 | 4.14x | 4.14x |
| `broadcast_scalar_input` | `()` / `(512, 1024)` | `(512, 1024)`, stride `(1024, 1)` | 8 | 77.013 us +/- 0.629, var 10.885 | 59.633 us +/- 0.538, var 3.966 | 1.29x | 1.29x |
| `broadcast_scalar_target` | `(512, 1024)` / `()` | `(512, 1024)`, stride `(1024, 1)` | 8 | 71.032 us +/- 0.583, var 1.084 | 62.058 us +/- 0.430, var 1.988 | 1.14x | 1.14x |
| `broadcast_scalar_input_heldout` | `()` / `(769, 773)` | `(769, 773)`, stride `(773, 1)` | 8 | 78.904 us +/- 0.625, var 0.742 | 68.705 us +/- 0.786, var 0.881 | 1.15x | 1.15x |
| `broadcast_scalar_target_heldout` | `(769, 773)` / `()` | `(769, 773)`, stride `(773, 1)` | 8 | 85.092 us +/- 2.023, var 16.339 | 70.886 us +/- 0.925, var 8.228 | 1.20x | 1.20x |
| `broadcast_vector_target` | `(512, 1024)` / `(1024,)` | `(512, 1024)`, stride `(1024, 1)` | 8 | 56.797 us +/- 0.436, var 4.776 | 59.363 us +/- 0.349, var 0.997 | 0.96x | 0.96x |
| `broadcast_column_target` | `(512, 1024)` / `(512, 1)` | `(512, 1024)`, stride `(1024, 1)` | 8 | 55.381 us +/- 0.488, var 0.880 | 64.837 us +/- 0.535, var 1.647 | 0.85x | 0.85x |
| `broadcast_noncontig_vector` | transposed `(512, 1024)` / `(1024,)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 8578.356 us +/- 28.188, var 168215.324 | 58.163 us +/- 0.265, var 0.492 | 147.49x | 10.00x |
| `broadcast_empty_scalar` | transposed empty `(3, 0, 2)` / `()` | `(3, 0, 2)`, stride `(2, 2, 1)` | 5000 | 2.520 us +/- 0.013, var 0.006 | 6.828 us +/- 0.027, var 0.016 | 0.37x | 0.37x |
