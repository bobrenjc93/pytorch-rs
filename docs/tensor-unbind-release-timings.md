# Arbitrary-Dimension Unbind Release Timings

Date: 2026-09-01

Candidate provenance: composite source snapshot based on
`8688a7089ca578bb52615be1c5df1eacb4d17359`, plus the review-time worktree
changes that add this timing evidence and the rank-2 strided-matmul
materialization path.

The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/arbitrary-dimension-unbind-release-timings*.json`.
No Conda environment was active in the shell (`CONDA_PREFIX=`), so setup used a
worktree-local `.venv`. Cargo registry data was copied read-only from the
existing user cache into `target/cargo-home`, then Cargo ran offline so build
artifacts and dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
wheel_dir="$(mktemp -d "$PWD/target/review-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/review-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/review-wheel-dir.txt)"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python \
  target/arbitrary_dimension_unbind_release_timings.py \
  > target/arbitrary-dimension-unbind-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  UNBIND_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python \
  target/arbitrary_dimension_unbind_release_timings.py \
  > target/arbitrary-dimension-unbind-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_unbind tests.test_unbind_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets unbind
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 27 tests. `cargo fmt --check` passed. The filtered native Rust unbind
test passed. The README/docs smoke test passed 7 tests, and `git diff --check`
passed.

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
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` with a warm worktree-local cache
  resolved 36 packages in 39 ms. The release wheel reinstall resolved in 3 ms,
  prepared in 0.17 ms, and installed in 37 ms.
- Build time: successful offline release extension build completed in 35.46s

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each forward cell, the driver checked tuple type, output count,
values, shape, stride, storage offset, dtype, device, `requires_grad`, leaf
status, `output_nr`, `data_ptr()` aliasing against `select`, and `is_set_to`
against PyTorch 2.13. Before timing each backward cell, it checked the same
view metadata, the scalar full-`sum` loss, and the original leaf gradient. Each
backward timed invocation used a pre-created fresh leaf/view input so the
timed region did not include input construction and did not reuse a freed
graph.

After every warmup and measured block, the driver consumed the last forward
output tuple, or the last backward output tuple plus scalar loss plus leaf
gradient, as a 64-bit BLAKE2b rolling checksum over tensor metadata and
logical bytes. The checksum column shows final rolling sinks from pass 1 and
pass 2 as `torch_rs:pass1,pass2; PyTorch:pass1,pass2`.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.24x uncapped, 0.24x capped
- Dimension-0 regression cells: 0.28x uncapped, 0.28x capped
- Arbitrary-dimension cells: 0.27x uncapped, 0.27x capped
- Trailing-dimension cells: 0.30x uncapped, 0.30x capped
- Offset cells: 0.29x uncapped, 0.29x capped
- Noncontiguous cells: 0.28x uncapped, 0.28x capped
- Negative-dimension cells: 0.30x uncapped, 0.30x capped
- Empty retained-dimension cells: 0.31x uncapped, 0.31x capped
- Empty unbound-dimension cells: 0.38x uncapped, 0.38x capped
- Backward-through-full-`sum` cells: 0.11x uncapped, 0.11x capped
- `Tensor.unbind` cells: 0.24x uncapped, 0.24x capped
- `torch.unbind` cells: 0.25x uncapped, 0.25x capped

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `dim0_offset_noncontiguous_regression` | dimension-0 regression | `Tensor.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24; dim=0` | 3 views of `(2, 4), stride (12, 1), requires_grad=False` | 5000 | 0.611 us +/- 0.005 us, var 0.000 | 2.251 us +/- 0.010 us, var 0.003 | 0.27x | `torch_rs:11600952581116631772,11600952581116631772; PyTorch:11600952581116631772,11600952581116631772` |
| `dim0_offset_noncontiguous_regression` | dimension-0 regression | `torch.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24; dim=0` | 3 views of `(2, 4), stride (12, 1), requires_grad=False` | 5000 | 0.672 us +/- 0.008 us, var 0.002 | 2.323 us +/- 0.013 us, var 0.002 | 0.29x | `torch_rs:11600952581116631772,11600952581116631772; PyTorch:11600952581116631772,11600952581116631772` |
| `dim1_contiguous_2x3x4` | arbitrary dimension | `Tensor.unbind` | `(2, 3, 4), stride (12, 4, 1), offset 0; dim=1` | 3 views of `(2, 4), stride (12, 1), requires_grad=False` | 5000 | 0.608 us +/- 0.006 us, var 0.001 | 2.258 us +/- 0.018 us, var 0.005 | 0.27x | `torch_rs:6890014648969093239,6890014648969093239; PyTorch:6890014648969093239,6890014648969093239` |
| `dim1_contiguous_2x3x4` | arbitrary dimension | `torch.unbind` | `(2, 3, 4), stride (12, 4, 1), offset 0; dim=1` | 3 views of `(2, 4), stride (12, 1), requires_grad=False` | 5000 | 0.636 us +/- 0.005 us, var 0.001 | 2.312 us +/- 0.010 us, var 0.002 | 0.27x | `torch_rs:6890014648969093239,6890014648969093239; PyTorch:6890014648969093239,6890014648969093239` |
| `dim2_contiguous_2x3x4` | trailing dimension | `Tensor.unbind` | `(2, 3, 4), stride (12, 4, 1), offset 0; dim=2` | 4 views of `(2, 3), stride (12, 4), requires_grad=False` | 5000 | 0.776 us +/- 0.005 us, var 0.000 | 2.662 us +/- 0.029 us, var 0.134 | 0.29x | `torch_rs:410385904219058273,410385904219058273; PyTorch:410385904219058273,410385904219058273` |
| `dim2_contiguous_2x3x4` | trailing dimension | `torch.unbind` | `(2, 3, 4), stride (12, 4, 1), offset 0; dim=2` | 4 views of `(2, 3), stride (12, 4), requires_grad=False` | 5000 | 0.840 us +/- 0.007 us, var 0.002 | 2.688 us +/- 0.029 us, var 0.217 | 0.31x | `torch_rs:410385904219058273,410385904219058273; PyTorch:410385904219058273,410385904219058273` |
| `dim1_offset_3x4x5` | offset | `Tensor.unbind` | `(3, 4, 5), stride (20, 5, 1), offset 60; dim=1` | 4 views of `(3, 5), stride (20, 1), requires_grad=False` | 5000 | 0.746 us +/- 0.009 us, var 0.002 | 2.688 us +/- 0.026 us, var 0.009 | 0.28x | `torch_rs:2693826119427001056,2693826119427001056; PyTorch:2693826119427001056,2693826119427001056` |
| `dim1_offset_3x4x5` | offset | `torch.unbind` | `(3, 4, 5), stride (20, 5, 1), offset 60; dim=1` | 4 views of `(3, 5), stride (20, 1), requires_grad=False` | 5000 | 0.816 us +/- 0.012 us, var 0.002 | 2.746 us +/- 0.024 us, var 0.005 | 0.30x | `torch_rs:2693826119427001056,2693826119427001056; PyTorch:2693826119427001056,2693826119427001056` |
| `dim1_noncontiguous_3x2x4` | noncontiguous | `Tensor.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24; dim=1` | 2 views of `(3, 4), stride (4, 1), requires_grad=False` | 5000 | 0.486 us +/- 0.006 us, var 0.001 | 1.757 us +/- 0.015 us, var 0.006 | 0.28x | `torch_rs:16298652815885394807,16298652815885394807; PyTorch:16298652815885394807,16298652815885394807` |
| `dim1_noncontiguous_3x2x4` | noncontiguous | `torch.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24; dim=1` | 2 views of `(3, 4), stride (4, 1), requires_grad=False` | 5000 | 0.513 us +/- 0.006 us, var 0.001 | 1.834 us +/- 0.009 us, var 0.008 | 0.28x | `torch_rs:16298652815885394807,16298652815885394807; PyTorch:16298652815885394807,16298652815885394807` |
| `negative_dim_noncontiguous_trailing` | negative dimension | `Tensor.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24; dim=-1` | 4 views of `(3, 2), stride (4, 12), requires_grad=False` | 5000 | 0.781 us +/- 0.011 us, var 0.001 | 2.676 us +/- 0.036 us, var 0.004 | 0.29x | `torch_rs:14392105484550759809,14392105484550759809; PyTorch:14392105484550759809,14392105484550759809` |
| `negative_dim_noncontiguous_trailing` | negative dimension | `torch.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24; dim=-1` | 4 views of `(3, 2), stride (4, 12), requires_grad=False` | 5000 | 0.868 us +/- 0.015 us, var 0.003 | 2.747 us +/- 0.042 us, var 0.008 | 0.32x | `torch_rs:14392105484550759809,14392105484550759809; PyTorch:14392105484550759809,14392105484550759809` |
| `dim1_empty_retained_2x3x0x4` | empty retained dimension | `Tensor.unbind` | `(2, 3, 0, 4), stride (12, 4, 4, 1), offset 0; dim=1` | 3 empty views of `(2, 0, 4), stride (12, 4, 1)` | 10000 | 0.596 us +/- 0.015 us, var 0.001 | 1.932 us +/- 0.009 us, var 0.001 | 0.31x | `torch_rs:14307478042116917350,14307478042116917350; PyTorch:14307478042116917350,14307478042116917350` |
| `dim1_empty_retained_2x3x0x4` | empty retained dimension | `torch.unbind` | `(2, 3, 0, 4), stride (12, 4, 4, 1), offset 0; dim=1` | 3 empty views of `(2, 0, 4), stride (12, 4, 1)` | 10000 | 0.657 us +/- 0.005 us, var 0.001 | 2.053 us +/- 0.008 us, var 0.001 | 0.32x | `torch_rs:14307478042116917350,14307478042116917350; PyTorch:14307478042116917350,14307478042116917350` |
| `dim1_empty_unbound_2x0x3` | empty unbound dimension | `Tensor.unbind` | `(2, 0, 3), stride (3, 3, 1), offset 0; dim=1` | empty tuple | 20000 | 0.187 us +/- 0.002 us, var 0.000 | 0.564 us +/- 0.004 us, var 0.000 | 0.33x | `torch_rs:10341777315501865172,10341777315501865172; PyTorch:10341777315501865172,10341777315501865172` |
| `dim1_empty_unbound_2x0x3` | empty unbound dimension | `torch.unbind` | `(2, 0, 3), stride (3, 3, 1), offset 0; dim=1` | empty tuple | 20000 | 0.247 us +/- 0.001 us, var 0.000 | 0.577 us +/- 0.004 us, var 0.004 | 0.43x | `torch_rs:10341777315501865172,10341777315501865172; PyTorch:10341777315501865172,10341777315501865172` |
| `dim0_backward_full_sum_3x2x4` | backward full sum | `Tensor.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24, requires_grad=True; dim=0` | 3 views, scalar loss, leaf grad | 200 | 8.260 us +/- 0.094 us, var 0.201 | 75.640 us +/- 0.762 us, var 16.069 | 0.11x | `torch_rs:9387072831650498470,9387072831650498470; PyTorch:9387072831650498470,9387072831650498470` |
| `dim0_backward_full_sum_3x2x4` | backward full sum | `torch.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24, requires_grad=True; dim=0` | 3 views, scalar loss, leaf grad | 200 | 8.338 us +/- 0.075 us, var 0.046 | 75.826 us +/- 1.232 us, var 4.094 | 0.11x | `torch_rs:9387072831650498470,9387072831650498470; PyTorch:9387072831650498470,9387072831650498470` |
| `dim1_backward_full_sum_3x2x4` | backward full sum | `Tensor.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24, requires_grad=True; dim=1` | 2 views, scalar loss, leaf grad | 200 | 6.885 us +/- 0.062 us, var 0.018 | 64.938 us +/- 0.988 us, var 2.856 | 0.11x | `torch_rs:17792571628892854354,17792571628892854354; PyTorch:17792571628892854354,17792571628892854354` |
| `dim1_backward_full_sum_3x2x4` | backward full sum | `torch.unbind` | `(3, 2, 4), stride (4, 12, 1), offset 24, requires_grad=True; dim=1` | 2 views, scalar loss, leaf grad | 200 | 6.939 us +/- 0.062 us, var 0.019 | 65.319 us +/- 0.551 us, var 1.450 | 0.11x | `torch_rs:17792571628892854354,17792571628892854354; PyTorch:17792571628892854354,17792571628892854354` |

## Zero-Credit Unsupported Cells

No unsupported unbind cells were included in this release-timing evidence set.
The unsupported boundaries remain dtype/device expansion keywords, tensor
subclasses without a handling override, and active mode execution beyond the
ordinary `__torch_function__` dispatch covered by the correctness suite.
