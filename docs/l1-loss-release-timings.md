# `torch.nn.functional.l1_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`2f88968c63d43c35fa08662aa68a0eedeeb1131a`

Baseline control: a release wheel built from `git archive HEAD` into
`target/l1-benchmark-base-src.1788110366`, before this worktree's L1 fast-path
changes. Candidate and baseline wheels were built inside this worktree with the
locked Cargo dependency graph.

Command shape: worktree-local `uv venv --clear --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, release
wheel builds with `maturin build --release --locked`, then `uv pip install
--force-reinstall --no-deps` before each timing process. The timing driver ran
after imports and input construction, with 9 warmup blocks and 51 measured
blocks per implementation. Inputs were CPU `float32` tensors. Broadcast
size-mismatch warning parity was checked before timing, then `UserWarning` was
ignored symmetrically for both implementations inside the measured region.

The construction timings measure eager `l1_loss(reduction="none")`
construction and consume the last output after each measured block as a
dead-code and deferred-work guard. The checksum timings consume every fresh
result with `output.sum().item()` inside the timed loop.

Checks run before timing:

```bash
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_HOME="$PWD/.cargo-home" \
  CARGO_NET_OFFLINE=true cargo fmt --check
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_HOME="$PWD/.cargo-home" \
  CARGO_NET_OFFLINE=true cargo clippy --all-targets -- -D warnings
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_HOME="$PWD/.cargo-home" \
  CARGO_NET_OFFLINE=true cargo test --all-targets
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_HOME="$PWD/.cargo-home" \
  CARGO_NET_OFFLINE=true PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --all-targets --features python-bindings -- -D warnings
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_HOME="$PWD/.cargo-home" \
  CARGO_NET_OFFLINE=true PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --all-targets --features python-bindings
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
PATH="/home/bobren/.cargo/bin:$PATH" CARGO_HOME="$PWD/.cargo-home" \
  CARGO_NET_OFFLINE=true UV_CACHE_DIR="$PWD/.uv-cache" \
  ./scripts/test-python.sh
```

Results: the focused L1 Python tests passed 17 tests. The wheel-installed full
Python suite passed 4196 tests with 3 skips.

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
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 30 ms, prepared in
  16.08 s, and installed in 2.92 s
- Build time: baseline release wheel 31.90 s; candidate release wheel 25.30 s
  through `./scripts/test-python.sh`, with the later benchmark wheel rebuild
  reusing cached artifacts in 0.27 s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `candidate / PyTorch` and `candidate / base` are slowdown
ratios, so lower is better and 1.00x is parity. Capped geomeans clamp each
per-cell `candidate / PyTorch` ratio to `[0.10x, 10.00x]`.

## L1 Construction

Same-shape row-major contiguous cells improved by a 0.73x geometric mean
candidate/base ratio. Their geometric mean `candidate / PyTorch` slowdown was
0.18x uncapped and 0.23x capped. Existing broadcast and non-contiguous controls
did not regress by more than 5%; worst construction movement was
`broadcast_vector_640x768`, from 190.498 us to 193.879 us (+1.8%).

| Workload | Category | Input / target | Output stride | Repeats | Base `torch_rs` median +/- MAD, variance | Candidate `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | Candidate / base | Candidate / PyTorch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_scalar` | same-shape contiguous | `()` / `()` | `()` | 20000 | 0.440 us +/- 0.002, var 0.000 | 0.382 us +/- 0.003, var 0.000 | 3.806 us +/- 0.033, var 0.003 | 0.87x | 0.10x |
| `same_empty_0x4096` | same-shape contiguous | `(0, 4096)` / `(0, 4096)` | `(4096, 1)` | 10000 | 0.259 us +/- 0.001, var 0.000 | 0.191 us +/- 0.001, var 0.000 | 3.597 us +/- 0.012, var 0.005 | 0.74x | 0.05x |
| `same_small_7x11` | same-shape contiguous | `(7, 11)` / `(7, 11)` | `(11, 1)` | 4000 | 0.315 us +/- 0.001, var 0.000 | 0.231 us +/- 0.002, var 0.000 | 4.121 us +/- 0.062, var 0.013 | 0.73x | 0.06x |
| `same_prime_257x263` | same-shape contiguous | `(257, 263)` / `(257, 263)` | `(263, 1)` | 32 | 17.059 us +/- 0.061, var 0.105 | 11.225 us +/- 0.055, var 0.236 | 22.120 us +/- 0.125, var 0.236 | 0.66x | 0.51x |
| `same_bandwidth_1024x1024` | same-shape contiguous | `(1024, 1024)` / `(1024, 1024)` | `(1024, 1)` | 2 | 574.869 us +/- 11.267, var 917.436 | 386.695 us +/- 10.255, var 988.450 | 293.689 us +/- 9.865, var 2227.573 | 0.67x | 1.32x |
| `broadcast_vector_640x768` | broadcast control | `(640, 768)` / `(768,)` | `(768, 1)` | 2 | 190.498 us +/- 8.839, var 210.783 | 193.879 us +/- 6.264, var 101.745 | 97.908 us +/- 0.641, var 695.051 | 1.02x | 1.98x |
| `broadcast_scalar_target_640x768` | broadcast control | `(640, 768)` / `()` | `(768, 1)` | 2 | 169.376 us +/- 9.950, var 239.861 | 170.708 us +/- 7.457, var 173.542 | 92.219 us +/- 0.436, var 637.654 | 1.01x | 1.85x |
| `noncontiguous_transpose_512x1024` | non-contiguous control | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(1, 512)` | 2 | 515.454 us +/- 8.849, var 4512.148 | 395.123 us +/- 11.367, var 366.192 | 123.697 us +/- 1.617, var 645.476 | 0.77x | 3.19x |
| `noncontiguous_channels_last_8x15x31x33` | non-contiguous control | channels-last `(8, 15, 31, 33)` / `(8, 15, 31, 33)` | `(15345, 1, 495, 15)` | 4 | 98.984 us +/- 1.042, var 5.365 | 72.139 us +/- 0.879, var 51.277 | 34.367 us +/- 0.205, var 34.739 | 0.73x | 2.10x |

## Full-Output Checksum Guard

Same-shape row-major contiguous checksum cells improved by a 0.87x geometric
mean candidate/base ratio. Their geometric mean `candidate / PyTorch` slowdown
was 0.32x uncapped and 0.37x capped. Existing broadcast and non-contiguous
controls did not regress by more than 5%; worst checksum movement was
`broadcast_vector_640x768`, from 520.467 us to 523.893 us (+0.7%).

| Workload | Category | Input / target | Output stride | Repeats | Base `torch_rs` median +/- MAD, variance | Candidate `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | Candidate / base | Candidate / PyTorch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_scalar` | same-shape contiguous | `()` / `()` | `()` | 20000 | 0.624 us +/- 0.002, var 0.000 | 0.583 us +/- 0.004, var 0.000 | 5.874 us +/- 0.050, var 0.007 | 0.93x | 0.10x |
| `same_empty_0x4096` | same-shape contiguous | `(0, 4096)` / `(0, 4096)` | `(4096, 1)` | 10000 | 0.308 us +/- 0.002, var 0.000 | 0.244 us +/- 0.001, var 0.000 | 3.715 us +/- 0.038, var 0.033 | 0.79x | 0.07x |
| `same_small_7x11` | same-shape contiguous | `(7, 11)` / `(7, 11)` | `(11, 1)` | 4000 | 0.595 us +/- 0.005, var 0.000 | 0.493 us +/- 0.002, var 0.000 | 6.268 us +/- 0.077, var 0.044 | 0.83x | 0.08x |
| `same_prime_257x263` | same-shape contiguous | `(257, 263)` / `(257, 263)` | `(263, 1)` | 32 | 73.318 us +/- 0.375, var 1.026 | 67.583 us +/- 0.751, var 0.992 | 28.060 us +/- 0.106, var 0.594 | 0.92x | 2.41x |
| `same_bandwidth_1024x1024` | same-shape contiguous | `(1024, 1024)` / `(1024, 1024)` | `(1024, 1)` | 2 | 1215.519 us +/- 32.164, var 3004.395 | 1054.715 us +/- 8.237, var 636.034 | 377.361 us +/- 15.438, var 2303.813 | 0.87x | 2.79x |
| `broadcast_vector_640x768` | broadcast control | `(640, 768)` / `(768,)` | `(768, 1)` | 2 | 520.467 us +/- 1.567, var 219.841 | 523.893 us +/- 3.276, var 144.861 | 123.287 us +/- 1.427, var 913.905 | 1.01x | 4.25x |
| `broadcast_scalar_target_640x768` | broadcast control | `(640, 768)` / `()` | `(768, 1)` | 2 | 506.496 us +/- 5.799, var 544.838 | 502.144 us +/- 1.923, var 450.184 | 119.816 us +/- 0.836, var 716.971 | 0.99x | 4.19x |
| `noncontiguous_transpose_512x1024` | non-contiguous control | transposed `(512, 1024)` / `(512, 1024)`, stride `(1, 512)` | `(1, 512)` | 2 | 1442.377 us +/- 4.862, var 345.581 | 1362.120 us +/- 5.409, var 564.205 | 147.804 us +/- 1.272, var 912.567 | 0.94x | 9.22x |
| `noncontiguous_channels_last_8x15x31x33` | non-contiguous control | channels-last `(8, 15, 31, 33)` / `(8, 15, 31, 33)` | `(15345, 1, 495, 15)` | 4 | 198.172 us +/- 0.463, var 13.341 | 172.007 us +/- 0.564, var 27.287 | 43.316 us +/- 0.178, var 82.805 | 0.87x | 3.97x |
