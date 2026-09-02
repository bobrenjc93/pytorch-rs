# `torch.nn.functional.l1_loss(reduction="sum")` Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot based on
`123ad635c137da09765334b22b8dc29345dd4fa0`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/l1-loss-sum-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`, `CONDA_SHLVL=0`), so
setup used a worktree-local `.venv`. Cargo registry data was copied read-only
from the existing user cache into `target/cargo-home`, then Cargo ran offline
so build artifacts and dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  /usr/bin/time -f 'uv venv wall %e s' \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  /usr/bin/time -f 'uv sync wall %e s' \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
mkdir -p target/l1-loss-sum-wheels
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  /usr/bin/time -f 'maturin build wall %e s' \
  .venv/bin/maturin build --release --locked --offline \
  --out target/l1-loss-sum-wheels
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  /usr/bin/time -f 'uv pip install wall %e s' \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps target/l1-loss-sum-wheels/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets absolute_difference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/l1_loss_sum_release_timings.py \
  > target/l1-loss-sum-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  L1_LOSS_SUM_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/l1_loss_sum_release_timings.py \
  > target/l1-loss-sum-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets absolute_difference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 31 tests. `cargo fmt --check` passed. The focused Rust
`absolute_difference` filter passed 8 tests. The README/docs smoke test passed
7 tests, and `git diff --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0, from
  `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: `uv venv --clear` completed in 0.26s; locked
  `uv sync` resolved in 26 ms, prepared packages in 17.32s, and installed in
  1.70s
- Build time: successful offline release wheel build completed in 37.79s; the
  release wheel reinstall resolved in 2 ms, prepared in 45 ms, and installed in
  17 ms

Inputs were created outside the timed region with NumPy seed `20260902`.
Every supported cell used CPU `float32`, matching shape, stride, storage
offset, contiguity, dtype, device, `requires_grad`, leaf status, and warning
behavior against PyTorch 2.13 before timing. Output metadata was checked the
same way, and the scalar output value was bit-compared with PyTorch. The input
patterns use small integer-valued `float32` data so the full sums remain exactly
representable. Broadcast size-mismatch warnings were checked before timing and
then ignored symmetrically inside the timed loops.

Each implementation ran in two pinned process passes. The first pass measured
`torch_rs` before PyTorch, and the second pass reversed that order. Each pass
used 15 untimed warmup blocks and 81 measured blocks. A block repeated the
operation according to the table's `Repeats` column; times below are median
microseconds per operation. Reported medians are the medians of the two
per-process medians. MAD and variance are the medians of the per-process MAD
and sample variance values.

After every warmup and measured block, the driver consumed the last scalar
output as a BLAKE2b checksum over tensor metadata and logical bytes. The
checksum column shows the final rolling sink as `torch_rs`/PyTorch; both
process passes produced the same sink pair for every supported cell.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.96x uncapped, 1.13x capped
- Contiguous cells: 2.58x uncapped, 2.58x capped
- Scalar cell: 0.05x uncapped, 0.10x capped
- Empty cell: 0.05x uncapped, 0.10x capped
- Channels-last cell: 2.91x uncapped, 2.91x capped
- Broadcast cell: 4.27x uncapped, 4.27x capped
- Offset cell: 1.04x uncapped, 1.04x capped
- Rank-2 transposed cell: 2.99x uncapped, 2.99x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 1.74x.

| Workload | Category | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `l1_sum_scalar` | scalar | `(), stride (), offset 0` / `(), stride (), offset 0` | `(), stride (), offset 0, requires_grad=False` | 10000 | 0.286 us +/- 0.002 us, var 0.003 | 5.481 us +/- 0.071 us, var 0.088 | 0.05x | `b4839d057680575c`/`b4839d057680575c` |
| `l1_sum_empty_transposed_3x0x2` | empty | `(3, 0, 2), stride (1, 3, 3), offset 0` / `(3, 0, 2), stride (1, 3, 3), offset 0` | `(), stride (), offset 0, requires_grad=False` | 5000 | 0.298 us +/- 0.003 us, var 0.000 | 5.485 us +/- 0.050 us, var 0.042 | 0.05x | `e081af3ec94883c6`/`e081af3ec94883c6` |
| `l1_sum_contiguous_257x263` | contiguous | `(257, 263), stride (263, 1), offset 0` / `(257, 263), stride (263, 1), offset 0` | `(), stride (), offset 0, requires_grad=False` | 32 | 68.375 us +/- 0.538 us, var 2.071 | 26.885 us +/- 0.249 us, var 0.590 | 2.54x | `b02bfc8ca46cd1e0`/`b02bfc8ca46cd1e0` |
| `l1_sum_contiguous_1024x1024` | contiguous | `(1024, 1024), stride (1024, 1), offset 0` / `(1024, 1024), stride (1024, 1), offset 0` | `(), stride (), offset 0, requires_grad=False` | 2 | 1061.748 us +/- 11.019 us, var 907.351 | 405.090 us +/- 14.146 us, var 929.354 | 2.62x | `f58ccacf8a33503e`/`f58ccacf8a33503e` |
| `l1_sum_channels_last_8x15x31x33` | channels-last | `(8, 15, 31, 33), stride (15345, 1, 495, 15), offset 0, channels_last=True` / `(8, 15, 31, 33), stride (15345, 1, 495, 15), offset 0, channels_last=True` | `(), stride (), offset 0, requires_grad=False` | 16 | 124.163 us +/- 0.959 us, var 3.001 | 42.688 us +/- 0.561 us, var 2.802 | 2.91x | `1f6e712ecdd63f36`/`1f6e712ecdd63f36` |
| `l1_sum_broadcast_640x768_by_768` | broadcast | `(640, 768), stride (768, 1), offset 0` / `(768,), stride (1,), offset 0` | `(), stride (), offset 0, requires_grad=False` | 8 | 530.159 us +/- 3.627 us, var 32.323 | 124.100 us +/- 1.107 us, var 13.796 | 4.27x | `3f46ef7caa414d61`/`3f46ef7caa414d61` |
| `l1_sum_offset_96x80` | offset | `(96, 80), stride (80, 1), offset 7680` / `(96, 80), stride (80, 1), offset 0` | `(), stride (), offset 0, requires_grad=False` | 64 | 8.077 us +/- 0.121 us, var 0.038 | 7.741 us +/- 0.141 us, var 0.162 | 1.04x | `a05fe9b32544f147`/`a05fe9b32544f147` |
| `l1_sum_transposed_256x512` | rank-2 transposed | `(256, 512), stride (1, 256), offset 0` / `(256, 512), stride (1, 256), offset 0` | `(), stride (), offset 0, requires_grad=False` | 4 | 133.507 us +/- 2.279 us, var 16.695 | 44.580 us +/- 0.368 us, var 6.027 | 2.99x | `835f45772bc41fe1`/`835f45772bc41fe1` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | Category | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `l1_sum_requires_grad_contiguous` | autograd | `RuntimeError: l1_loss(): autograd recording is not supported` | supported scalar CPU `float32`, `requires_grad=True`, checksum `db73cfd6677dcacf` | zero |
| `l1_sum_float64_contiguous` | dtype | unsupported: no exact native CPU `float64` Tensor type | supported scalar CPU `float64`, checksum `18ae988bad0fc210` | zero |
