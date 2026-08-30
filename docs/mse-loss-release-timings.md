# `torch.nn.functional.mse_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`5a910be39711ec21cf73111c6c51aaba9245cf0c`

Command shape: worktree-local `uv venv --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, then a
release wheel build and install through `./scripts/test-python.sh`. The timing
driver ran against that installed wheel after imports and input construction.
Inputs were CPU `float32` tensors. Broadcast size-mismatch warning parity was
checked before timing; timing warmups consumed the once-per-callsite warning
before measured samples.

The primary timings below measure eager `mse_loss(reduction="none")`
construction and consume output metadata after each measured block as a
dead-code and deferred-work guard. The full-output checksum table consumes
every fresh result with `output.sum().item()` inside the timed loop; those
numbers are kept as a conservative end-to-end guard and remain dominated by the
current `torch_rs.sum` implementation for non-empty outputs. The tiny
zero-element scalar-broadcast construction row was measured in a dedicated
high-repeat run because sub-microsecond allocator and warning-registry effects
made it sensitive to neighboring large cells.

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

Results: the focused MSE Python tests passed 29 tests. The wheel-installed full
Python suite passed 4185 tests with 3 skips.

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
- Dependency installation: locked `uv sync` completed in 18.82s
- Build time: the final release extension rebuild completed in 24.24s; a
  prior no-code-change wheel rebuild reported 0.01s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

Relative to the previous 2026-08-30 construction report, existing contiguous
and scalar-broadcast controls did not regress by more than 5% in the release
timings: the non-empty scalar controls ranged from -10.3% to +1.6%, the
dedicated empty scalar control changed from 1.026 us to 0.993 us (-3.2%), and
`same_contiguous_prime_control` changed from 11.647 us to 11.489 us (-1.4%).
The original transposed control improved from 244.610 us to 97.432 us (-60.2%).

## MSE Construction

Geometric mean `torch_rs / PyTorch` slowdown for the scalar-broadcast held-out
cells:

- Uncapped: 0.66x
- Capped to `[0.10x, 10.00x]` per cell: 0.66x

Geometric mean `torch_rs / PyTorch` slowdown for same-stride transposed cells:

- Uncapped: 1.21x
- Capped to `[0.10x, 10.00x]` per cell: 1.21x

Geometric mean `torch_rs / PyTorch` slowdown for all construction cells:

- Uncapped: 0.87x
- Capped to `[0.10x, 10.00x]` per cell: 0.87x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `scalar_input_2d_heldout` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 2 | 47.447 us +/- 1.257, var 7.754 | 55.123 us +/- 0.441, var 3.040 | 0.86x |
| `scalar_target_2d_heldout` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 2 | 43.901 us +/- 0.726, var 7.347 | 54.037 us +/- 0.620, var 3.805 | 0.81x |
| `scalar_input_3d_heldout` | `()` / `(17, 257, 263)` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 105.269 us +/- 1.973, var 8.432 | 109.826 us +/- 1.272, var 11.205 | 0.96x |
| `scalar_target_3d_heldout` | `(17, 257, 263)` / `()` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 101.643 us +/- 2.864, var 20.064 | 108.854 us +/- 0.492, var 33.650 | 0.93x |
| `scalar_target_empty_contiguous` | `(0, 4096)` / `()` | `(0, 4096)`, stride `(4096, 1)` | 50000 | 0.993 us +/- 0.003, var 0.000 | 5.022 us +/- 0.023, var 0.003 | 0.20x |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 16 | 11.489 us +/- 0.036, var 0.185 | 12.950 us +/- 0.039, var 0.037 | 0.89x |
| `same_noncontiguous_transpose_control` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 97.432 us +/- 2.234, var 22.885 | 78.253 us +/- 0.255, var 1.499 | 1.25x |
| `same_transposed_rect_heldout` | transposed `(640, 768)` / `(640, 768)`, stride `(1, 640)` | `(640, 768)`, stride `(1, 640)` | 2 | 88.874 us +/- 2.258, var 17.515 | 72.810 us +/- 0.431, var 5.006 | 1.22x |
| `same_transposed_3d_heldout` | transposed `(17, 257, 263)` / `(17, 257, 263)`, stride `(67591, 1, 257)` | `(17, 257, 263)`, stride `(67591, 1, 257)` | 1 | 199.892 us +/- 5.729, var 64.470 | 167.614 us +/- 0.640, var 9.452 | 1.19x |
| `same_offset_transposed_heldout` | offset transposed `(509, 513)` / `(509, 513)`, stride `(1, 509)` | `(509, 513)`, stride `(1, 509)` | 4 | 47.727 us +/- 0.340, var 0.750 | 39.828 us +/- 0.494, var 1.654 | 1.20x |

## Full-Output Checksum Guard

The same held-out cells were also timed while consuming every output with
`output.sum().item()` inside the measured loop. Relative to the previous
2026-08-30 report, the existing full-output controls did not regress by more
than 5%: scalar-broadcast controls ranged from -16.9% to +2.8%, and
`same_contiguous_prime_control` changed from 65.968 us to 66.234 us (+0.4%).
The original transposed control improved from 1284.220 us to 1147.409 us
(-10.7%).

Geometric mean `torch_rs / PyTorch` slowdown for the scalar-broadcast held-out
cells:

- Uncapped: 3.02x
- Capped to `[0.10x, 10.00x]` per cell: 3.02x

Geometric mean `torch_rs / PyTorch` slowdown for same-stride transposed cells:

- Uncapped: 6.14x
- Capped to `[0.10x, 10.00x]` per cell: 6.04x

Geometric mean `torch_rs / PyTorch` slowdown for all full-output checksum cells:

- Uncapped: 4.09x
- Capped to `[0.10x, 10.00x]` per cell: 4.06x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `scalar_input_2d_heldout` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 2 | 450.731 us +/- 1.782, var 22.960 | 79.135 us +/- 0.877, var 8.899 | 5.70x |
| `scalar_target_2d_heldout` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 2 | 450.466 us +/- 1.667, var 9.709 | 79.716 us +/- 0.911, var 9.708 | 5.65x |
| `scalar_input_3d_heldout` | `()` / `(17, 257, 263)` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 1048.144 us +/- 3.756, var 32.908 | 163.086 us +/- 0.901, var 13.091 | 6.43x |
| `scalar_target_3d_heldout` | `(17, 257, 263)` / `()` | `(17, 257, 263)`, stride `(67591, 263, 1)` | 1 | 1053.051 us +/- 5.128, var 105.362 | 158.479 us +/- 0.641, var 8.825 | 6.65x |
| `scalar_target_empty_contiguous` | `(0, 4096)` / `()` | `(0, 4096)`, stride `(4096, 1)` | 20000 | 1.270 us +/- 0.003, var 0.000 | 6.929 us +/- 0.027, var 0.007 | 0.18x |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 16 | 66.234 us +/- 0.135, var 0.315 | 18.268 us +/- 0.096, var 0.479 | 3.63x |
| `same_noncontiguous_transpose_control` | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(512, 1024)`, stride `(1, 512)` | 2 | 1147.409 us +/- 2.799, var 126.382 | 107.267 us +/- 0.526, var 11.138 | 10.70x |
| `same_transposed_rect_heldout` | transposed `(640, 768)` / `(640, 768)`, stride `(1, 640)` | `(640, 768)`, stride `(1, 640)` | 2 | 494.623 us +/- 1.843, var 10.120 | 96.220 us +/- 0.505, var 3.201 | 5.14x |
| `same_transposed_3d_heldout` | transposed `(17, 257, 263)` / `(17, 257, 263)`, stride `(67591, 1, 257)` | `(17, 257, 263)`, stride `(67591, 1, 257)` | 1 | 1158.280 us +/- 8.122, var 174.927 | 214.965 us +/- 1.192, var 19.106 | 5.39x |
| `same_offset_transposed_heldout` | offset transposed `(509, 513)` / `(509, 513)`, stride `(1, 509)` | `(509, 513)`, stride `(1, 509)` | 4 | 263.153 us +/- 1.066, var 6.431 | 54.938 us +/- 1.422, var 4.186 | 4.79x |
