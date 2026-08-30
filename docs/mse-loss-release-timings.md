# `torch.nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`e3af6e66c9ec0b6c0d1f70455ec7b4d8dfe94bd7`

Command shape: release `maturin develop --release --locked` build from the
current worktree, installed into the worktree-local `.venv`. The timing driver
ran after imports and input construction, with 9 warmup blocks and 51 measured
blocks per implementation. Each measured call constructed a fresh
`mse_loss(reduction="none")` output from exact CPU `float32` tensors. The MSE
call table observes the returned tensor metadata and data pointer after every
call; CPU eager execution materializes the full output before the call returns.
The full-checksum guard table additionally consumes each non-empty output with
`output.sum().item()`; empty outputs contribute their rank to the checksum.
Before timing every workload, the `torch_rs` output was bitwise-checked against
the equivalent PyTorch 2.13.0 result, including shape and stride metadata.

The focused native and reference MSE tests were run before timing:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
```

Result: 25 tests passed.

Additional checks run for this change:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --all-targets --features python-bindings -- -D warnings
cargo test --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --all-targets --features python-bindings
cargo test --doc
```

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core; CPU flags observed for this run included `avx2`, `avx512f`,
  `fma`, and `sse4_2`
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.12
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from the worktree-local `.venv`; CUDA runtime 13.0,
  `torch.cuda.is_available()` reported `False` because `CUDA_VISIBLE_DEVICES=`
  was set for the CPU timing run
- `torch_rs`: 0.1.0 from `python/torch_rs`, native extension
  `python/torch_rs/torch_rs.abi3.so`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1

Broadcast size-mismatch warnings were ignored inside the timed loops for both
implementations; the focused reference tests above cover warning parity. Times
below are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of the per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. The capped ratio clamps each per-cell ratio to
`[0.10x, 10.00x]` before geometric aggregation.

Geometric mean `torch_rs / PyTorch` slowdown for the held-out scalar-broadcast
MSE-call cells:

- Uncapped: 0.74x
- Capped to `[0.10x, 10.00x]` per cell: 0.74x

Geometric mean `torch_rs / PyTorch` slowdown for the MSE-call controls:

- Uncapped: 1.50x
- Capped to `[0.10x, 10.00x]` per cell: 1.50x

## MSE Call

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Capped ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `broadcast_scalar_input_prime_heldout` | `()` stride `()` / `(257, 263)` stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 16 | 9.221 us +/- 0.026, var 0.027 | 14.891 us +/- 0.081, var 0.086 | 0.62x | 0.62x |
| `broadcast_scalar_target_prime_heldout` | `(257, 263)` stride `(263, 1)` / `()` stride `()` | `(257, 263)`, stride `(263, 1)` | 16 | 8.852 us +/- 0.034, var 0.101 | 15.050 us +/- 0.058, var 0.272 | 0.59x | 0.59x |
| `broadcast_scalar_input_bandwidth_heldout` | `()` stride `()` / `(1536, 1536)` stride `(1536, 1)` | `(1536, 1536)`, stride `(1536, 1)` | 2 | 232.231 us +/- 1.678, var 8.366 | 252.316 us +/- 3.852, var 185.490 | 0.92x | 0.92x |
| `broadcast_scalar_target_bandwidth_heldout` | `(1536, 1536)` stride `(1536, 1)` / `()` stride `()` | `(1536, 1536)`, stride `(1536, 1)` | 2 | 230.869 us +/- 2.258, var 20.367 | 252.536 us +/- 2.649, var 18.861 | 0.91x | 0.91x |
| `broadcast_empty_scalar_heldout` | `(5, 0, 257)` stride `(257, 257, 1)` / `()` stride `()` | `(5, 0, 257)`, stride `(257, 257, 1)` | 5000 | 3.132 us +/- 0.012, var 0.001 | 7.175 us +/- 0.029, var 0.007 | 0.44x | 0.44x |
| `same_contiguous_prime_control` | `(257, 263)` stride `(263, 1)` / `(257, 263)` stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 16 | 13.681 us +/- 0.091, var 3.933 | 15.751 us +/- 0.078, var 0.319 | 0.87x | 0.87x |
| `same_contiguous_bandwidth_control` | `(1536, 1536)` stride `(1536, 1)` / `(1536, 1536)` stride `(1536, 1)` | `(1536, 1536)`, stride `(1536, 1)` | 2 | 630.608 us +/- 24.227, var 2343.317 | 512.720 us +/- 21.713, var 1789.669 | 1.23x | 1.23x |
| `same_noncontiguous_transpose_control` | `(512, 1024)` stride `(1, 512)` / `(512, 1024)` stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 245.681 us +/- 1.923, var 23.737 | 77.998 us +/- 0.306, var 7.500 | 3.15x | 3.15x |

## Full-Output Checksum Guard

This guard preserves the prior report's full-output consumption pattern. Its
large non-empty cells include `torch_rs.sum()` time, which is intentionally
reported separately from the MSE-call timings above.

Geometric mean `torch_rs / PyTorch` slowdown for the held-out scalar-broadcast
checksum cells:

- Uncapped: 4.43x
- Capped to `[0.10x, 10.00x]` per cell: 4.43x

Geometric mean `torch_rs / PyTorch` slowdown for the checksum controls:

- Uncapped: 5.60x
- Capped to `[0.10x, 10.00x]` per cell: 5.22x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Capped ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `broadcast_scalar_input_prime_heldout` | `()` stride `()` / `(257, 263)` stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 16 | 63.649 us +/- 0.096, var 0.174 | 18.722 us +/- 0.105, var 0.214 | 3.40x | 3.40x |
| `broadcast_scalar_target_prime_heldout` | `(257, 263)` stride `(263, 1)` / `()` stride `()` | `(257, 263)`, stride `(263, 1)` | 16 | 63.650 us +/- 0.183, var 0.106 | 19.508 us +/- 0.129, var 0.264 | 3.26x | 3.26x |
| `broadcast_scalar_input_bandwidth_heldout` | `()` stride `()` / `(1536, 1536)` stride `(1536, 1)` | `(1536, 1536)`, stride `(1536, 1)` | 2 | 2217.470 us +/- 26.801, var 6412.733 | 382.733 us +/- 22.274, var 1215.485 | 5.79x | 5.79x |
| `broadcast_scalar_target_bandwidth_heldout` | `(1536, 1536)` stride `(1536, 1)` / `()` stride `()` | `(1536, 1536)`, stride `(1536, 1)` | 2 | 2222.573 us +/- 21.222, var 1895.537 | 372.498 us +/- 7.597, var 876.391 | 5.97x | 5.97x |
| `broadcast_empty_scalar_heldout` | `(5, 0, 257)` stride `(257, 257, 1)` / `()` stride `()` | `(5, 0, 257)`, stride `(257, 257, 1)` | 5000 | 2.905 us +/- 0.011, var 0.046 | 6.894 us +/- 0.113, var 0.059 | 0.42x | 0.42x |
| `same_contiguous_prime_control` | `(257, 263)` stride `(263, 1)` / `(257, 263)` stride `(263, 1)` | `(257, 263)`, stride `(263, 1)` | 16 | 69.936 us +/- 0.496, var 0.962 | 20.251 us +/- 0.087, var 0.688 | 3.45x | 3.45x |
| `same_contiguous_bandwidth_control` | `(1536, 1536)` stride `(1536, 1)` / `(1536, 1536)` stride `(1536, 1)` | `(1536, 1536)`, stride `(1536, 1)` | 2 | 2728.909 us +/- 37.391, var 5350.825 | 663.727 us +/- 20.290, var 1077.439 | 4.11x | 4.11x |
| `same_noncontiguous_transpose_control` | `(512, 1024)` stride `(1, 512)` / `(512, 1024)` stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 1270.100 us +/- 16.639, var 2328.326 | 102.796 us +/- 0.356, var 12.119 | 12.36x | 10.00x |

Compared with the 2026-08-30 prior full-checksum report, the unchanged control
cells stayed inside the 5% regression guard:

| Control | Prior `torch_rs` median | Current `torch_rs` median | Change |
| --- | ---: | ---: | ---: |
| `same_contiguous_prime_control` | 67.753 us | 69.936 us | +3.2% |
| `same_contiguous_bandwidth_control` | 2626.825 us | 2728.909 us | +3.9% |
| `same_noncontiguous_transpose_control` | 1255.974 us | 1270.100 us | +1.1% |
