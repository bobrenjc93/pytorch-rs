# `torch.nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`71c534a66a36f11d6f1636be668aa562563b0206`

Command shape: worktree-local `uv venv --clear --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, then a
release wheel build and install through `./scripts/test-python.sh`. The timing
driver ran against that installed wheel after imports and input construction,
with 9 warmup blocks and 51 measured blocks per implementation. Inputs were CPU
`float32` tensors. Broadcast size-mismatch warning parity was checked before
timing, then `UserWarning` was ignored symmetrically for both implementations
inside the timing run.

The primary timings below measure eager `mse_loss(reduction="none")`
construction and consume the last output after each measured block as a
dead-code and deferred-work guard. The full-output checksum table consumes every
fresh result with `output.sum().item()` inside the timed loop; those numbers are
kept as a conservative end-to-end guard and are dominated by the current
`torch_rs.sum` implementation for non-empty outputs.

Checks run before timing:

```bash
/home/bobren/.cargo/bin/cargo fmt --check
/home/bobren/.cargo/bin/cargo clippy --all-targets -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" \
  /home/bobren/.cargo/bin/cargo clippy --all-targets --features python-bindings -- -D warnings
/home/bobren/.cargo/bin/cargo test --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" \
  /home/bobren/.cargo/bin/cargo test --all-targets --features python-bindings
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_mse_loss \
  tests.test_nn_functional_mse_loss_reference
PATH="/home/bobren/.cargo/bin:$PATH" \
  UV_CACHE_DIR="$PWD/.uv-cache" \
  ./scripts/test-python.sh
```

Results: the focused MSE Python tests passed 25 tests. The wheel-installed full
Python suite passed 4181 tests with 3 skips.

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
- Dependency installation: locked `uv sync` completed in 16.65s
- Build time: first release extension build completed in 31.51s; the later
  wheel-based test build reused cached artifacts and reported 0.01s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## MSE Construction

Geometric mean `torch_rs / PyTorch` slowdown for the scalar-broadcast held-out
cells:

- Uncapped: 0.69x
- Capped to `[0.10x, 10.00x]` per cell: 0.69x

Geometric mean `torch_rs / PyTorch` slowdown for all construction cells:

- Uncapped: 0.89x
- Capped to `[0.10x, 10.00x]` per cell: 0.89x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `scalar_input_2d_heldout` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 2 | 46.685 us +/- 1.142, var 15.725 | 52.399 us +/- 0.446, var 5.195 | 0.89x |
| `scalar_target_2d_heldout` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 2 | 45.603 us +/- 0.721, var 1.213 | 53.806 us +/- 0.316, var 2.457 | 0.85x |
| `scalar_input_3d_heldout` | `()` / `(17, 257, 263)` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 113.952 us +/- 2.394, var 23428.370 | 114.332 us +/- 1.894, var 7629.680 | 1.00x |
| `scalar_target_3d_heldout` | `(17, 257, 263)` / `()` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 113.341 us +/- 4.446, var 552.479 | 114.824 us +/- 1.042, var 205.496 | 0.99x |
| `scalar_target_empty_contiguous` | `(0, 4096)` / `()` | `(0, 4096)`, stride `(4096, 1)` | 5000 | 1.026 us +/- 0.005, var 0.000 | 4.840 us +/- 0.017, var 0.004 | 0.21x |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 16 | 11.647 us +/- 0.048, var 0.026 | 13.042 us +/- 0.113, var 0.558 | 0.89x |
| `same_noncontiguous_transpose_control` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 244.610 us +/- 3.099, var 45.961 | 80.431 us +/- 0.721, var 3.602 | 3.04x |

## Full-Output Checksum Guard

The same held-out cells were also timed while consuming every output with
`output.sum().item()` inside the measured loop. Relative to the previous
2026-08-30 report, the same-shape controls did not regress by more than 5%:
`same_contiguous_prime_control` changed from 67.753 us to 65.968 us (-2.6%),
and `same_noncontiguous_transpose_control` changed from 1255.974 us to
1284.220 us (+2.2%).

Geometric mean `torch_rs / PyTorch` slowdown for the scalar-broadcast held-out
cells:

- Uncapped: 3.34x
- Capped to `[0.10x, 10.00x]` per cell: 3.34x

Geometric mean `torch_rs / PyTorch` slowdown for all full-output checksum cells:

- Uncapped: 4.09x
- Capped to `[0.10x, 10.00x]` per cell: 3.96x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `scalar_input_2d_heldout` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 2 | 448.793 us +/- 1.317, var 15.159 | 74.152 us +/- 0.516, var 16.501 | 6.05x |
| `scalar_target_2d_heldout` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 2 | 448.082 us +/- 1.673, var 6.384 | 75.159 us +/- 0.195, var 3.789 | 5.96x |
| `scalar_input_3d_heldout` | `()` / `(17, 257, 263)` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 1042.686 us +/- 2.304, var 211.364 | 162.585 us +/- 1.684, var 11.779 | 6.41x |
| `scalar_target_3d_heldout` | `(17, 257, 263)` / `()` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 1042.635 us +/- 1.993, var 134.014 | 163.297 us +/- 1.973, var 7.323 | 6.39x |
| `scalar_target_empty_contiguous` | `(0, 4096)` / `()` | `(0, 4096)`, stride `(4096, 1)` | 5000 | 1.528 us +/- 0.006, var 0.000 | 5.259 us +/- 0.033, var 0.021 | 0.29x |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 16 | 65.968 us +/- 0.117, var 0.142 | 18.678 us +/- 0.087, var 0.159 | 3.53x |
| `same_noncontiguous_transpose_control` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 1284.220 us +/- 2.644, var 272.436 | 101.313 us +/- 1.432, var 13.647 | 12.68x |
