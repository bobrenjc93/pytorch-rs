# `torch.nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-29

Revision under test: `bc8e0ad9c1a4d0c95bb4032c202e181d9bccbd36`

Command shape: release `maturin develop --release --locked` build from the
current worktree, installed into the worktree-local `.venv`. The timing driver
ran after imports and input construction, with 9 warmup blocks and 51 measured
blocks per implementation. Each measured call used exact CPU `float32` tensors
and immediately consumed the full `mse_loss(reduction="none")` output with
`np.asarray(output).sum(dtype=np.float64)`; empty outputs contributed their rank
to the checksum. Before timing, every `torch_rs` output was bitwise-checked
against the equivalent PyTorch 2.13.0 result.

The focused native and reference MSE tests were run before timing:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
```

Result: 24 tests passed.

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
below are median microseconds per call including output materialization. MAD is
median absolute deviation in microseconds, and variance is sample variance of
the per-call sample timings in microseconds squared. `torch_rs / PyTorch` is a
slowdown ratio, so lower is better and 1.00x is parity. The capped ratio clamps
each per-cell ratio to `[0.10x, 10.00x]` before geometric aggregation.

Geometric mean `torch_rs / PyTorch` slowdown:

- Uncapped: 21.00x
- Capped to `[0.10x, 10.00x]` per cell: 4.50x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Capped ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_small` | `(128, 256)` / `(128, 256)`, contiguous | `(128, 256)`, stride `(256, 1)` | 12 / 656 | 1212.229 us +/- 6.352, var 114.263 | 22.613 us +/- 0.093, var 0.983 | 53.61x | 10.00x |
| `same_contiguous_bandwidth` | `(1024, 1024)` / `(1024, 1024)`, contiguous | `(1024, 1024)`, stride `(1024, 1)` | 1 / 30 | 46610.299 us +/- 1387.237, var 9794707.112 | 523.300 us +/- 5.426, var 204.706 | 89.07x | 10.00x |
| `same_noncontiguous_transpose` | transposed `(512, 1024)` / `(512, 1024)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 1 / 51 | 20779.657 us +/- 95.965, var 308352.965 | 258.692 us +/- 0.942, var 6.275 | 80.33x | 10.00x |
| `same_channels_last` | channels-last `(8, 16, 32, 32)` / `(8, 16, 32, 32)`, input stride `(16384, 1, 512, 16)` | `(8, 16, 32, 32)`, stride `(16384, 1, 512, 16)` | 2 / 140 | 5282.089 us +/- 74.853, var 8590.594 | 72.587 us +/- 0.872, var 1.448 | 72.77x | 10.00x |
| `same_empty_contiguous` | `(0, 1024)` / `(0, 1024)`, contiguous | `(0, 1024)`, stride `(1024, 1)` | 2000 / 2000 | 1.851 us +/- 0.009, var 0.001 | 4.512 us +/- 0.019, var 0.001 | 0.41x | 0.41x |
| `same_empty_strided` | transposed empty `(8, 0, 16)` / `(8, 0, 16)`, input stride `(1, 8, 8)` | `(8, 0, 16)`, stride `(16, 16, 1)` | 2000 / 2000 | 1.857 us +/- 0.012, var 0.003 | 4.554 us +/- 0.025, var 0.003 | 0.41x | 0.41x |
| `broadcast_scalar_input` | `()` / `(512, 1024)` | `(512, 1024)`, stride `(1024, 1)` | 1 / 54 | 20008.461 us +/- 180.412, var 146673.585 | 241.752 us +/- 3.387, var 19.386 | 82.76x | 10.00x |
| `broadcast_scalar_target` | `(512, 1024)` / `()` | `(512, 1024)`, stride `(1024, 1)` | 1 / 48 | 20087.751 us +/- 217.969, var 170582.571 | 247.483 us +/- 3.238, var 35.407 | 81.17x | 10.00x |
| `broadcast_vector_target` | `(512, 1024)` / `(1024,)` | `(512, 1024)`, stride `(1024, 1)` | 1 / 54 | 20347.124 us +/- 207.343, var 281786.665 | 242.340 us +/- 3.130, var 17.633 | 83.96x | 10.00x |
| `broadcast_column_target` | `(512, 1024)` / `(512, 1)` | `(512, 1024)`, stride `(1024, 1)` | 1 / 34 | 20209.856 us +/- 134.183, var 176240.393 | 359.463 us +/- 1.762, var 10.175 | 56.22x | 10.00x |
| `broadcast_noncontig_vector` | transposed `(512, 1024)` / `(1024,)`, input stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 1 / 54 | 28754.511 us +/- 136.746, var 301509.698 | 237.677 us +/- 2.219, var 15.731 | 120.98x | 10.00x |
| `broadcast_empty_scalar` | transposed empty `(8, 0, 16)` / `()` | `(8, 0, 16)`, stride `(16, 16, 1)` | 2000 / 2000 | 2.694 us +/- 0.009, var 0.001 | 6.558 us +/- 0.018, var 0.004 | 0.41x | 0.41x |
