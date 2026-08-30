# `torch.nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: `1f062c540a16b0d52491ff3b9f74b659ca3094db` plus the
uncommitted same-shape contiguous MSE candidate changes in this worktree.

Command shape: release `maturin develop --release --locked --offline` build
from the current worktree, installed into the worktree-local `.venv`. The timing
driver ran after imports and input construction, with 9 warmup blocks and 51
measured blocks per implementation. Every measured call used exact CPU
`float32` tensors, called `torch.nn.functional.mse_loss(input, target,
reduction="none")`, and immediately consumed the full output with
`output.sum().item()`; empty outputs contributed their rank to the checksum.
Before timing, every `torch_rs` output was bitwise-checked against the
equivalent PyTorch 2.13.0 result, including shape, stride, and values.

The focused native and reference MSE tests were run before timing:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
```

Result: 25 tests passed.

Additional checks run on the final candidate:

```bash
CARGO_HOME="$PWD/target/cargo-home" \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo fmt --check

VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  CARGO_HOME="$PWD/target/cargo-home" \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo clippy --all-targets --features python-bindings --offline -- -D warnings

VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  CARGO_HOME="$PWD/target/cargo-home" \
  PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  cargo test --all-targets --features python-bindings --offline
```

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.12
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch`
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
below are median microseconds per call including full-output native reduction.
MAD is median absolute deviation in microseconds, and variance is sample
variance of the per-call sample timings in microseconds squared.
`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. The capped ratio clamps each per-cell ratio to `[0.10x, 10.00x]` before
geometric aggregation. The transpose and broadcast rows are retained as current
regression guard cells; they continue to use the existing generic MSE paths.
Those guard cells were also rerun with the previous report's
`np.asarray(output).sum(dtype=np.float64)` materialization, where the largest
`torch_rs` median movement was the transpose cell at +1.59%; all broadcast
cells were faster than the prior report, so none exceeded the 5% regression
budget.

Geometric mean `torch_rs / PyTorch` slowdown:

- Uncapped: 3.42x
- Capped to `[0.10x, 10.00x]` per cell: 2.70x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Capped ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_scalar` | `()` / `()`, contiguous | `()`, stride `()` | 682 / 71 | 2.023 us +/- 0.007, var 0.001 | 6.648 us +/- 0.096, var 0.316 | 0.30x | 0.30x |
| `same_contiguous_empty` | `(0, 1024)` / `(0, 1024)`, contiguous | `(0, 1024)`, stride `(1024, 1)` | 1605 / 792 | 2.077 us +/- 0.011, var 0.003 | 4.731 us +/- 0.018, var 0.016 | 0.44x | 0.44x |
| `same_contiguous_prime_small` | `(257, 263)` / `(257, 263)`, contiguous | `(257, 263)`, stride `(263, 1)` | 104 / 110 | 69.763 us +/- 0.101, var 0.086 | 20.857 us +/- 0.081, var 0.269 | 3.34x | 3.34x |
| `same_contiguous_edges` | `(2, 4)` / `(2, 4)`, contiguous signed-zero/NaN/infinity values | `(2, 4)`, stride `(4, 1)` | 1116 / 375 | 1.922 us +/- 0.009, var 0.003 | 6.882 us +/- 0.036, var 0.033 | 0.28x | 0.28x |
| `same_contiguous_bandwidth_heldout` | `(768, 1536)` / `(768, 1536)`, contiguous | `(768, 1536)`, stride `(1536, 1)` | 6 / 19 | 1198.823 us +/- 6.965, var 555.310 | 258.384 us +/- 0.793, var 5.124 | 4.64x | 4.64x |
| `same_noncontiguous_transpose_current` | transposed `(512, 1024)` / `(512, 1024)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 6 / 34 | 1399.633 us +/- 3.450, var 85.554 | 107.091 us +/- 0.242, var 1.844 | 13.07x | 10.00x |
| `broadcast_scalar_input_current` | `()` / `(512, 1024)` | `(512, 1024)`, stride `(1024, 1)` | 17 / 38 | 483.733 us +/- 1.574, var 22.249 | 83.966 us +/- 0.196, var 0.119 | 5.76x | 5.76x |
| `broadcast_scalar_target_current` | `(512, 1024)` / `()` | `(512, 1024)`, stride `(1024, 1)` | 17 / 40 | 483.375 us +/- 1.266, var 3.946 | 87.107 us +/- 0.359, var 0.687 | 5.55x | 5.55x |
| `broadcast_vector_target_current` | `(512, 1024)` / `(1024,)` | `(512, 1024)`, stride `(1024, 1)` | 17 / 35 | 489.580 us +/- 1.968, var 10.282 | 88.277 us +/- 0.229, var 0.363 | 5.55x | 5.55x |
| `broadcast_column_target_current` | `(512, 1024)` / `(512, 1)` | `(512, 1024)`, stride `(1024, 1)` | 17 / 36 | 485.464 us +/- 1.459, var 5.456 | 87.757 us +/- 0.296, var 2.410 | 5.53x | 5.53x |
| `broadcast_noncontig_vector_current` | transposed `(512, 1024)` / `(1024,)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 1 / 22 | 8779.749 us +/- 221.985, var 241171.740 | 88.117 us +/- 0.352, var 0.760 | 99.64x | 10.00x |
