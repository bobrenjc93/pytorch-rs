# `Tensor.reciprocal` and `torch.reciprocal` Release Timings

Date: 2026-09-02

Candidate provenance: source snapshot based on
`715994b76541eab2af761075d079f30e31913695`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-reciprocal-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`), so setup used a
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
mkdir -p target/cargo-home/registry && \
  cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/ && \
  wheel_dir="$(mktemp -d "$PWD/target/tensor-reciprocal-wheels.XXXXXX")" && \
  printf '%s\n' "$wheel_dir" > target/tensor-reciprocal-wheel-dir.txt && \
  env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
    CARGO_HOME="$PWD/target/cargo-home" \
    CARGO_TARGET_DIR="$PWD/target" \
    TMPDIR="$PWD/target" \
    VIRTUAL_ENV="$PWD/.venv" \
    PYO3_PYTHON="$PWD/.venv/bin/python" \
    .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/tensor-reciprocal-wheel-dir.txt)" && \
  env UV_CACHE_DIR="$PWD/target/uv-cache" \
    UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
    uv pip install --python "$PWD/.venv/bin/python" \
    --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_reciprocal tests.test_reciprocal_reference \
  tests.test_top_level_reciprocal tests.test_top_level_reciprocal_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_reciprocal_release_timings.py \
  > target/tensor-reciprocal-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  RECIPROCAL_TIMING_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/tensor_reciprocal_release_timings.py \
  > target/tensor-reciprocal-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets reciprocal
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_reciprocal tests.test_reciprocal_reference \
  tests.test_top_level_reciprocal tests.test_top_level_reciprocal_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 28 tests. The focused Rust `reciprocal` filter passed 8 tests, `cargo
fmt --check` passed, the README/docs smoke test passed, and `git diff --check`
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
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 29 ms, prepared
  packages in 16.12s, and installed in 994 ms
- Build time: successful offline release extension build completed in 35.49s;
  the release wheel reinstall resolved in 2 ms, prepared in 43 ms, and
  installed in 14 ms

Inputs were created outside the timed region with NumPy seed `20260902`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported forward cell, the driver bit-compared `torch_rs`
output values with PyTorch and checked shape, stride, storage offset,
contiguity, dtype, device, `requires_grad`, and leaf status. Before timing each
backward cell, it bit-compared the forward output, checked the scalar
full-`sum` loss metadata and value with `rtol=1e-5`, `atol=1e-4`, and
`equal_nan=True`, then bit-compared the leaf gradient and checked its metadata.
Backward timings used pre-created fresh leaf/view inputs for every measured
invocation so the timed region did not include input construction and did not
reuse a freed graph.

After every warmup and measured block, the driver consumed the last forward
output, or the last backward scalar loss plus leaf gradient, as a 64-bit
BLAKE2b rolling checksum over tensor metadata and logical bytes. The checksum
column shows the final rolling sink from one pass as `torch_rs`/PyTorch; both
process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.61x uncapped, 0.65x capped
- Forward cells: 0.77x uncapped, 0.77x capped
- Backward-through-full-`sum` cells: 0.49x uncapped, 0.55x capped
- `Tensor.reciprocal` forward cells: 0.73x uncapped, 0.73x capped
- `torch.reciprocal` forward cells: 0.81x uncapped, 0.81x capped
- `Tensor.reciprocal().sum().backward()` cells: 0.47x uncapped, 0.54x capped
- `torch.reciprocal(input).sum().backward()` cells: 0.51x uncapped, 0.57x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 0.84x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar` | scalar | `Tensor.reciprocal` | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.166 us +/- 0.001 us, var 0.000 | 1.001 us +/- 0.013 us, var 0.003 | 0.17x | `4916277017294898581`/`4916277017294898581` |
| `empty` | empty | `Tensor.reciprocal` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (3, 0, 2), stride (2, 2, 1), offset 0, requires_grad=False | 5000 | 0.178 us +/- 0.001 us, var 0.000 | 0.890 us +/- 0.005 us, var 0.000 | 0.20x | `11447192971687011400`/`11447192971687011400` |
| `contiguous_257x263` | contiguous | `Tensor.reciprocal` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 14.774 us +/- 0.227 us, var 0.129 | 8.981 us +/- 0.110 us, var 0.197 | 1.65x | `4508128906113893534`/`4508128906113893534` |
| `offset_transposed_521x509` | offset | `Tensor.reciprocal` | (521, 509), stride (1, 521), offset 265189, requires_grad=False | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 65.985 us +/- 2.025 us, var 12.385 | 30.585 us +/- 0.415 us, var 1.998 | 2.16x | `8466222436342175445`/`8466222436342175445` |
| `noncontig_transpose_512x1024` | noncontiguous | `Tensor.reciprocal` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 119.223 us +/- 3.151 us, var 54.925 | 67.072 us +/- 3.713 us, var 180.128 | 1.78x | `1694720749299286662`/`1694720749299286662` |
| `scalar` | scalar backward | `Tensor.reciprocal().sum().backward()` | (), stride (), offset 0, requires_grad=True | loss (); grad (), stride (), offset 0, requires_grad=False | 100 | 1.594 us +/- 0.017 us, var 0.008 | 25.234 us +/- 0.235 us, var 3025.206 | 0.06x | `1721096168380643457`/`1721096168380643457` |
| `empty` | empty backward | `Tensor.reciprocal().sum().backward()` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=True | loss (); grad (2, 0, 3), stride (3, 3, 1), offset 0, requires_grad=False | 100 | 2.308 us +/- 0.055 us, var 0.008 | 29.695 us +/- 0.327 us, var 1.170 | 0.08x | `13100444987921828214`/`13100444987921828214` |
| `contiguous_127x131` | contiguous backward | `Tensor.reciprocal().sum().backward()` | (127, 131), stride (131, 1), offset 0, requires_grad=True | loss (); grad (127, 131), stride (131, 1), offset 0, requires_grad=False | 10 | 41.617 us +/- 1.390 us, var 105.052 | 65.043 us +/- 3.787 us, var 127.256 | 0.64x | `18413004760398800853`/`6227659679809171796` |
| `offset_transposed_127x131` | offset backward | `Tensor.reciprocal().sum().backward()` | (127, 131), stride (1, 127), offset 16637, requires_grad=True | loss (); grad (3, 131, 127), stride (16637, 127, 1), offset 0, requires_grad=False | 10 | 249.932 us +/- 4.909 us, var 481.204 | 142.371 us +/- 8.942 us, var 614.483 | 1.76x | `9975472551233173590`/`11195252099507233775` |
| `noncontig_transpose_128x256` | noncontiguous backward | `Tensor.reciprocal().sum().backward()` | (128, 256), stride (1, 128), offset 0, requires_grad=True | loss (); grad (256, 128), stride (128, 1), offset 0, requires_grad=False | 5 | 431.231 us +/- 9.069 us, var 334.053 | 107.696 us +/- 2.876 us, var 52.220 | 4.00x | `5728927617571034017`/`1333014126399323896` |
| `scalar` | scalar | `torch.reciprocal` | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.232 us +/- 0.002 us, var 0.000 | 1.055 us +/- 0.006 us, var 0.001 | 0.22x | `4916277017294898581`/`4916277017294898581` |
| `empty` | empty | `torch.reciprocal` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (3, 0, 2), stride (2, 2, 1), offset 0, requires_grad=False | 5000 | 0.248 us +/- 0.002 us, var 0.000 | 0.920 us +/- 0.004 us, var 0.000 | 0.27x | `11447192971687011400`/`11447192971687011400` |
| `contiguous_257x263` | contiguous | `torch.reciprocal` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 14.751 us +/- 0.226 us, var 0.094 | 8.465 us +/- 0.052 us, var 0.073 | 1.74x | `4508128906113893534`/`4508128906113893534` |
| `offset_transposed_521x509` | offset | `torch.reciprocal` | (521, 509), stride (1, 521), offset 265189, requires_grad=False | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 59.256 us +/- 1.407 us, var 6.092 | 33.169 us +/- 0.863 us, var 3.618 | 1.79x | `8466222436342175445`/`8466222436342175445` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.reciprocal` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 121.480 us +/- 4.202 us, var 122.238 | 62.972 us +/- 2.291 us, var 28.705 | 1.93x | `1694720749299286662`/`1694720749299286662` |
| `scalar` | scalar backward | `torch.reciprocal(input).sum().backward()` | (), stride (), offset 0, requires_grad=True | loss (); grad (), stride (), offset 0, requires_grad=False | 100 | 1.705 us +/- 0.028 us, var 0.018 | 24.925 us +/- 0.275 us, var 4703.060 | 0.07x | `1721096168380643457`/`1721096168380643457` |
| `empty` | empty backward | `torch.reciprocal(input).sum().backward()` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=True | loss (); grad (2, 0, 3), stride (3, 3, 1), offset 0, requires_grad=False | 100 | 2.464 us +/- 0.078 us, var 0.018 | 29.684 us +/- 0.335 us, var 1.170 | 0.08x | `13100444987921828214`/`13100444987921828214` |
| `contiguous_127x131` | contiguous backward | `torch.reciprocal(input).sum().backward()` | (127, 131), stride (131, 1), offset 0, requires_grad=True | loss (); grad (127, 131), stride (131, 1), offset 0, requires_grad=False | 10 | 30.212 us +/- 0.796 us, var 1.392 | 44.294 us +/- 0.999 us, var 6.175 | 0.68x | `18413004760398800853`/`6227659679809171796` |
| `offset_transposed_127x131` | offset backward | `torch.reciprocal(input).sum().backward()` | (127, 131), stride (1, 127), offset 16637, requires_grad=True | loss (); grad (3, 131, 127), stride (16637, 127, 1), offset 0, requires_grad=False | 10 | 236.637 us +/- 12.430 us, var 570.732 | 110.015 us +/- 3.908 us, var 178.980 | 2.15x | `9975472551233173590`/`11195252099507233775` |
| `noncontig_transpose_128x256` | noncontiguous backward | `torch.reciprocal(input).sum().backward()` | (128, 256), stride (1, 128), offset 0, requires_grad=True | loss (); grad (256, 128), stride (128, 1), offset 0, requires_grad=False | 5 | 437.049 us +/- 2.976 us, var 198.739 | 110.219 us +/- 2.436 us, var 58.789 | 3.97x | `5728927617571034017`/`1333014126399323896` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `tensor_reciprocal_in_place_reciprocal_` | `AttributeError: 'torch_rs.Tensor' object has no attribute 'reciprocal_'` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
| `top_level_torch_reciprocal_out_tensor` | `RuntimeError: reciprocal(): the 'out' argument is not supported` | supported `(1,), stride (1,), offset 0, requires_grad=False` | zero |
