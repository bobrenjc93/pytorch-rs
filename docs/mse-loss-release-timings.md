# `torch.nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`28be6899a04fc71637fbbfa1c7368dcb2d2e2e69`

Command shape: release `maturin develop --release --locked` build from the
current worktree, installed into the worktree-local `.venv`. The timing driver
ran after imports and input construction, with 9 warmup blocks and 51 measured
blocks per implementation. Each measured call constructed a fresh
`mse_loss(reduction="none")` output from exact CPU `float32` tensors and
immediately consumed the full output with `output.sum().item()`; empty outputs
contributed their rank to the checksum. Before timing every workload, the
`torch_rs` output was bitwise-checked against the equivalent PyTorch 2.13.0
result, including shape and stride metadata.

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
cargo fmt --check
cargo clippy --all-targets -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --all-targets --features python-bindings -- -D warnings
cargo test --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --all-targets --features python-bindings
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
below are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of the per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. The capped ratio clamps each per-cell ratio to
`[0.10x, 10.00x]` before geometric aggregation.

Geometric mean `torch_rs / PyTorch` slowdown for the held-out same-shape
contiguous cells:

- Uncapped: 0.67x
- Capped to `[0.10x, 10.00x]` per cell: 0.68x

Geometric mean `torch_rs / PyTorch` slowdown for all cells in the native
checksum table:

- Uncapped: 2.74x
- Capped to `[0.10x, 10.00x]` per cell: 2.17x

## Native Checksum

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Capped ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_scalar_heldout` | `()` / `()`, contiguous | `()`, stride `()` | 5000 | 0.597 us +/- 0.004, var 0.000 | 4.989 us +/- 0.024, var 0.006 | 0.12x | 0.12x |
| `same_contiguous_empty_heldout` | `(0, 1024)` / `(0, 1024)`, contiguous | `(0, 1024)`, stride `(1024, 1)` | 5000 | 0.306 us +/- 0.002, var 0.000 | 3.114 us +/- 0.016, var 0.038 | 0.10x | 0.10x |
| `same_contiguous_prime_heldout` | `(257, 263)` / `(257, 263)`, contiguous | `(257, 263)`, stride `(263, 1)` | 16 | 67.753 us +/- 0.200, var 0.293 | 17.969 us +/- 0.122, var 1.949 | 3.77x | 3.77x |
| `same_contiguous_bandwidth_heldout` | `(1536, 1536)` / `(1536, 1536)`, contiguous | `(1536, 1536)`, stride `(1536, 1)` | 2 | 2626.825 us +/- 36.425, var 2973.074 | 564.473 us +/- 13.535, var 3225.375 | 4.65x | 4.65x |
| `same_noncontiguous_transpose` | transposed `(512, 1024)` / `(512, 1024)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 1255.974 us +/- 3.596, var 56.628 | 104.353 us +/- 1.688, var 9.363 | 12.04x | 10.00x |
| `broadcast_scalar_input` | `()` / `(512, 1024)` | `(512, 1024)`, stride `(1024, 1)` | 2 | 477.807 us +/- 1.733, var 13.392 | 85.394 us +/- 0.435, var 7.676 | 5.60x | 5.60x |
| `broadcast_scalar_target` | `(512, 1024)` / `()` | `(512, 1024)`, stride `(1024, 1)` | 2 | 479.159 us +/- 1.752, var 16.074 | 79.159 us +/- 0.300, var 1.886 | 6.05x | 6.05x |
| `broadcast_vector_target` | `(512, 1024)` / `(1024,)` | `(512, 1024)`, stride `(1024, 1)` | 2 | 483.265 us +/- 0.916, var 3.525 | 88.784 us +/- 0.431, var 2.753 | 5.44x | 5.44x |
| `broadcast_column_target` | `(512, 1024)` / `(512, 1)` | `(512, 1024)`, stride `(1024, 1)` | 2 | 481.462 us +/- 0.616, var 4.172 | 81.994 us +/- 0.331, var 5.746 | 5.87x | 5.87x |
| `broadcast_noncontig_vector` | transposed `(512, 1024)` / `(1024,)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 9400.924 us +/- 57.457, var 37066.076 | 84.558 us +/- 0.761, var 8.767 | 111.18x | 10.00x |
| `broadcast_empty_scalar` | transposed empty `(3, 0, 2)` / `()` | `(3, 0, 2)`, stride `(2, 2, 1)` | 5000 | 1.096 us +/- 0.005, var 0.000 | 5.066 us +/- 0.058, var 0.229 | 0.22x | 0.22x |

## Prior Materialized-Checksum Guard

The 2026-08-29 report consumed each output with
`np.asarray(output).sum(dtype=np.float64)`. That path is dominated by
`torch_rs`'s Python-list based `Tensor.__array__` conversion for non-empty
outputs, so it is reported separately from the native checksum timings above.
To guard against regressions in the already-supported broadcast and transpose
cells, the same full-NumPy-consumption timing was rerun after an extra allocator
warmup pass. Every measured `torch_rs` median below is within 5% of the
2026-08-29 `torch_rs` median for the same cell.

| Workload | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Change vs 2026-08-29 `torch_rs` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `same_noncontiguous_transpose` | 1 | 20846.387 us +/- 271.911, var 2236323.724 | 249.286 us +/- 3.084, var 39.447 | 83.62x | +0.3% |
| `broadcast_scalar_input` | 1 | 19899.205 us +/- 164.098, var 1053329.956 | 223.938 us +/- 1.742, var 174.712 | 88.86x | -0.5% |
| `broadcast_scalar_target` | 1 | 19695.176 us +/- 120.722, var 986424.253 | 228.115 us +/- 2.634, var 68.939 | 86.34x | -2.0% |
| `broadcast_vector_target` | 1 | 19871.613 us +/- 198.140, var 994311.411 | 224.730 us +/- 1.231, var 22.774 | 88.42x | -2.3% |
| `broadcast_column_target` | 1 | 19660.785 us +/- 121.174, var 1170532.562 | 224.789 us +/- 1.862, var 52.642 | 87.46x | -2.7% |
| `broadcast_noncontig_vector` | 1 | 28236.236 us +/- 394.897, var 1425982.293 | 236.758 us +/- 9.615, var 171.838 | 119.26x | -1.8% |
| `broadcast_empty_scalar` | 2000 | 2.642 us +/- 0.024, var 0.074 | 6.501 us +/- 0.033, var 0.812 | 0.41x | -1.9% |
