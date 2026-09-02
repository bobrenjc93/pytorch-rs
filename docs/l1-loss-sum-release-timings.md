# `torch.nn.functional.l1_loss(reduction="sum")` Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot at
`b964254d1af2963644194d9f756391b64c4c1a26`.

The benchmark used CPU `float32` inputs created outside the timed region and
measured the installed release wheel against PyTorch 2.13. The timing driver was
the ignored one-off `target/l1_loss_sum_release_timings.py` and emitted JSON
under `target/l1-loss-sum-release-timings-pass*.json`. It ran two pinned process
passes, first `torch_rs` then PyTorch and then the reverse order, with 15 warmup
blocks and 81 measured blocks per implementation. Broadcast size-mismatch
warning parity was checked before timing, then `UserWarning` was ignored
symmetrically for both implementations inside timed loops.

Exact command sequence:

```bash
env UV_CACHE_DIR="$PWD/.uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" \
  uv venv --python 3.12 .venv
env UV_CACHE_DIR="$PWD/.uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
env PATH="$HOME/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true \
  UV_CACHE_DIR="$PWD/.uv-cache" TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
env PATH="$HOME/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true cargo fmt --check
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
env PATH="$HOME/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true \
  UV_CACHE_DIR="$PWD/.uv-cache" TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked \
  --out target/l1-loss-sum-wheels
env UV_CACHE_DIR="$PWD/.uv-cache" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps \
  target/l1-loss-sum-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/l1_loss_sum_release_timings.py \
  --warmups 15 --samples 81 --order torch_rs,pytorch \
  > target/l1-loss-sum-release-timings-pass1.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/l1_loss_sum_release_timings.py \
  --warmups 15 --samples 81 --order pytorch,torch_rs \
  > target/l1-loss-sum-release-timings-pass2.json
env PATH="$HOME/.cargo/bin:$PATH" CARGO_NET_OFFLINE=true cargo test --all-targets
```

Checks passed: the focused L1 Python implementation and PyTorch 2.13
differential tests passed 31 tests after the release wheel install;
`cargo fmt --check` passed; `cargo test --all-targets` passed 298 Rust tests.
Before timing each workload, the driver checked output shape, stride, storage
offset, contiguity, dtype, device, `requires_grad`, leaf status, warning parity,
fresh output storage, and scalar values against PyTorch with `rtol=1e-4`,
`atol=1e-4`, and `equal_nan=True`-equivalent finite handling. The tolerance
allows equivalent CPU float32 full reductions to differ by accumulation order.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from
  `.venv/lib/python3.12/site-packages/torch`; `torch.version.cuda` reported
  13.0
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release wheel from `maturin build --release --locked`, Cargo
  `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run, and
  `torch.cuda.is_available()` reported `False` in the timing process
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`;
  `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` in the
  driver; `torch_rs.get_num_threads()` and
  `torch_rs.get_num_interop_threads()` both reported 1
- Dependency installation: locked `uv sync` resolved in 29 ms, prepared
  packages in 16.12s, and installed in 2.72s
- Build time: the first release extension build completed in 35.43s; the
  non-editable release wheel build reused those artifacts and completed in
  0.26s wall time; wheel reinstall completed in 0.19s

Times are median microseconds per `l1_loss(..., reduction="sum")` call. MAD is
median absolute deviation in microseconds, and variance is sample variance of
per-call sample timings in microseconds squared. Reported medians are the
medians of the two per-process medians; MAD and variance are the medians of
the per-process MAD and variance values. `torch_rs / PyTorch` is a slowdown
ratio, so lower is better and 1.00x is parity. Capped geomeans clamp each
per-cell ratio to `[0.10x, 10.00x]`.

The materialized checksum is the sum of the last scalar output read with
`.item()` after each of the 15 warmup and 81 measured blocks in a process pass;
both process passes produced the same checksum values listed below.

Geometric mean `torch_rs / PyTorch` slowdown for supported CPU float32
`reduction="sum"` cells:

- Uncapped: 0.93x
- Capped to `[0.10x, 10.00x]` per cell: 1.09x

| Workload | Category | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksum, `torch_rs` / PyTorch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `scalar` | scalar | `()`, stride `()`, offset 0 / `()`, stride `()`, offset 0 | `()`, stride `()`, offset 0 | 10000 | 0.304 us +/- 0.002, var 0.000 | 5.410 us +/- 0.071, var 0.039 | 0.06x | 252 (`0x437c0000`) / 252 (`0x437c0000`) |
| `empty` | empty | `(0, 4096)`, stride `(4096, 1)`, offset 0 / `(0, 4096)`, stride `(4096, 1)`, offset 0 | `()`, stride `()`, offset 0 | 10000 | 0.313 us +/- 0.002, var 0.000 | 5.307 us +/- 0.063, var 0.046 | 0.06x | 0 (`0x00000000`) / 0 (`0x00000000`) |
| `same_contiguous_257x263` | contiguous | `(257, 263)`, stride `(263, 1)`, offset 0 / `(257, 263)`, stride `(263, 1)`, offset 0 | `()`, stride `()`, offset 0 | 32 | 67.546 us +/- 0.246, var 0.276 | 26.925 us +/- 0.160, var 20.424 | 2.51x | 7320584.25 (`0x4adf6810`) / 7320597 (`0x4adf682a`) |
| `channels_last_8x15x31x33` | channels-last | `(8, 15, 31, 33)`, stride `(15345, 1, 495, 15)`, offset 0, channels_last=True / `(8, 15, 31, 33)`, stride `(15345, 1, 495, 15)`, offset 0, channels_last=True | `()`, stride `()`, offset 0 | 8 | 123.420 us +/- 0.429, var 1.477 | 41.826 us +/- 0.196, var 1.520 | 2.95x | 13273572 (`0x4b4a89e4`) / 13273609.5 (`0x4b4a8a0a`) |
| `broadcast_256x384_by_384` | broadcast | `(256, 384)`, stride `(384, 1)`, offset 0 / `(384,)`, stride `(1,)`, offset 0 | `()`, stride `()`, offset 0 | 16 | 107.499 us +/- 0.517, var 1.460 | 35.033 us +/- 0.449, var 0.514 | 3.07x | 10386113.2 (`0x4b1e7ac1`) / 10386136.5 (`0x4b1e7ad8`) |
| `offset_96x80` | offset | `(96, 80)`, stride `(80, 1)`, offset 7680 / `(96, 80)`, stride `(80, 1)`, offset 0 | `()`, stride `()`, offset 0 | 128 | 8.034 us +/- 0.029, var 0.069 | 7.707 us +/- 0.034, var 0.021 | 1.04x | 825574.688 (`0x49498e6b`) / 825574.125 (`0x49498e62`) |
| `rank2_transposed_512x1024` | rank-2 transposed | `(512, 1024)`, stride `(1, 512)`, offset 0 / `(512, 1024)`, stride `(1, 512)`, offset 0 | `()`, stride `()`, offset 0 | 4 | 1145.354 us +/- 5.607, var 223.563 | 150.093 us +/- 1.430, var 10.127 | 7.63x | 56811204 (`0x4c58b7b1`) / 56812560 (`0x4c58b904`) |

## Correctness Values

These are the pre-timing scalar outputs from the same seeded inputs. The
broadcast row emitted the same PyTorch 2.13 size-mismatch warning in both
implementations before timing.

| Workload | `torch_rs` value, bits | PyTorch value, bits | Absolute / relative error |
| --- | ---: | ---: | ---: |
| `scalar` | 2.625 (`0x40280000`) | 2.625 (`0x40280000`) | 0 / 0 |
| `empty` | 0 (`0x00000000`) | 0 (`0x00000000`) | 0 / 0 |
| `same_contiguous_257x263` | 76256.085938 (`0x4794f00b`) | 76256.218750 (`0x4794f01c`) | 0.132812 / 1.74e-06 |
| `channels_last_8x15x31x33` | 138266.375000 (`0x48070698`) | 138266.765625 (`0x480706b1`) | 0.390625 / 2.83e-06 |
| `broadcast_256x384_by_384` | 108188.679688 (`0x47d34e57`) | 108188.921875 (`0x47d34e76`) | 0.242188 / 2.24e-06 |
| `offset_96x80` | 8599.736328 (`0x46065ef2`) | 8599.730469 (`0x46065eec`) | 0.005859 / 6.81e-07 |
| `rank2_transposed_512x1024` | 591783.375000 (`0x49107a76`) | 591797.500000 (`0x49107b58`) | 14.125000 / 2.39e-05 |

## Zero-Credit Unsupported Cells

The driver also probed unsupported L1 cells that an external evaluator should
score as zero performance credit if they are included in a broader denominator.
They are not included in the supported-cell geomean above.

| Cell | Probe | Result | Performance credit |
| --- | --- | --- | ---: |
| `l1_mean_reduction` | `l1_loss(input, target, reduction="mean")` | `NotImplementedError: torch_rs.nn.functional.l1_loss only supports reduction='none' or reduction='sum'` | 0 |
| `l1_sum_weight_argument` | `l1_loss(input, target, reduction="sum", weight=target)` | `NotImplementedError: torch_rs.nn.functional.l1_loss only supports weight=None` | 0 |
| `l1_sum_active_autograd` | `l1_loss(input_requires_grad, target, reduction="sum")` with grad mode enabled | `RuntimeError: l1_loss(): autograd recording is not supported` | 0 |
