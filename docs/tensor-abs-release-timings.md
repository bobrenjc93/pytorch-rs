# `Tensor.abs` and `torch.abs` Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`30e67e57d37d0aff48c06019da39146da23c76ab`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-abs-release-timings*.json`. No Conda
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
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
wheel_dir="$(mktemp -d "$PWD/target/tensor-abs-wheels.XXXXXX")"
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_abs tests.test_abs_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_abs_release_timings.py \
  > target/tensor-abs-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_abs_release_timings.py \
  > target/tensor-abs-release-timings-pass2.json
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
  cargo test --locked --offline --all-targets abs
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_abs tests.test_abs_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 25 tests. The focused Rust `abs` filter passed 10 tests, `cargo fmt
--check` passed, the README/docs smoke test passed, and `git diff --check`
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
- Dependency installation: locked `uv sync` resolved in 28 ms, prepared
  packages in 40.02s, and installed in 931 ms
- Build time: successful offline release extension build completed in 33.90s;
  the release wheel reinstall resolved in 1 ms, prepared in 43 ms, and
  installed in 15 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
Each pass used 15 untimed warmup blocks and 81 measured blocks. A block
repeated the operation according to the table's `Repeats` column; times below
are median microseconds per operation. Reported medians are medians of the two
per-process medians. MAD and variance are the medians of the per-process MAD
and sample variance values.

Before timing each supported forward cell, the driver bit-compared `torch_rs`
output values with PyTorch and checked shape, stride, storage offset,
contiguity, dtype, device, `requires_grad`, and leaf status. Before timing each
backward cell, it checked the scalar full-`sum` loss metadata and value with
`rtol=1e-5`, `atol=1e-4`, and `equal_nan=True`, then bit-compared the leaf
gradient and checked the same metadata fields. Backward timings used
pre-created fresh leaf/view inputs for every measured invocation so the timed
region did not include input construction and did not reuse a freed graph.
After every warmup and measured block, the driver consumed the last forward
output or the last backward scalar loss plus leaf gradient as a byte-level
checksum. The checksum column shows the final rolling sink from one pass as
`torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.46x uncapped, 0.50x capped
- Forward cells: 0.52x uncapped, 0.52x capped
- Backward-through-full-`sum` cells: 0.42x uncapped, 0.47x capped
- `Tensor.abs` forward cells: 0.48x uncapped, 0.48x capped
- `torch.abs` forward cells: 0.56x uncapped, 0.56x capped
- `Tensor.abs().sum().backward()` cells: 0.41x uncapped, 0.47x capped
- `torch.abs(input).sum().backward()` cells: 0.42x uncapped, 0.48x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 0.65x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar` | scalar | `Tensor.abs` | (), stride (), offset 0, requires_grad=False | (), stride (), requires_grad=False | 10000 | 0.160 us +/- 0.002 us, var 0.000 | 1.434 us +/- 0.031 us, var 0.004 | 0.11x | `2833669408475005987`/`2833669408475005987` |
| `empty` | empty | `Tensor.abs` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (3, 0, 2), stride (2, 2, 1), requires_grad=False | 5000 | 0.184 us +/- 0.005 us, var 0.000 | 1.275 us +/- 0.013 us, var 0.003 | 0.14x | `8789815901978462371`/`8789815901978462371` |
| `contiguous_257x263` | contiguous | `Tensor.abs` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (257, 263), stride (263, 1), requires_grad=False | 32 | 560.925 us +/- 4.831 us, var 1380.546 | 489.544 us +/- 8.321 us, var 193.605 | 1.15x | `7072671920403779939`/`7072671920403779939` |
| `offset_transposed_521x509` | offset | `Tensor.abs` | (521, 509), stride (1, 521), offset 265189, requires_grad=False | (521, 509), stride (1, 521), requires_grad=False | 5 | 14200.802 us +/- 119.495 us, var 91632.711 | 12115.455 us +/- 94.330 us, var 332633.243 | 1.17x | `5365401783112995107`/`5365401783112995107` |
| `noncontig_transpose_512x1024` | noncontiguous | `Tensor.abs` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (512, 1024), stride (1, 512), requires_grad=False | 5 | 27425.274 us +/- 198.443 us, var 249495.105 | 23477.977 us +/- 123.336 us, var 1853472.695 | 1.17x | `3709683643347319971`/`3709683643347319971` |
| `scalar` | scalar | `torch.abs` | (), stride (), offset 0, requires_grad=False | (), stride (), requires_grad=False | 10000 | 0.222 us +/- 0.001 us, var 0.000 | 1.300 us +/- 0.011 us, var 0.007 | 0.17x | `2833669408475005987`/`2833669408475005987` |
| `empty` | empty | `torch.abs` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (3, 0, 2), stride (2, 2, 1), requires_grad=False | 5000 | 0.238 us +/- 0.002 us, var 0.000 | 1.157 us +/- 0.010 us, var 0.007 | 0.21x | `8789815901978462371`/`8789815901978462371` |
| `contiguous_257x263` | contiguous | `torch.abs` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (257, 263), stride (263, 1), requires_grad=False | 32 | 560.588 us +/- 5.114 us, var 111.947 | 485.201 us +/- 5.319 us, var 152.154 | 1.16x | `7072671920403779939`/`7072671920403779939` |
| `offset_transposed_521x509` | offset | `torch.abs` | (521, 509), stride (1, 521), offset 265189, requires_grad=False | (521, 509), stride (1, 521), requires_grad=False | 5 | 14198.734 us +/- 81.267 us, var 108064.280 | 12048.553 us +/- 52.600 us, var 101794.043 | 1.18x | `5365401783112995107`/`5365401783112995107` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.abs` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (512, 1024), stride (1, 512), requires_grad=False | 5 | 27606.322 us +/- 376.771 us, var 2799079.297 | 23867.684 us +/- 405.663 us, var 1408128.070 | 1.16x | `3709683643347319971`/`3709683643347319971` |
| `scalar` | scalar backward | `Tensor.abs().sum().backward()` | (), stride (), offset 0, requires_grad=True | loss (); grad (), stride () | 100 | 1.592 us +/- 0.019 us, var 0.028 | 24.523 us +/- 0.228 us, var 0.868 | 0.06x | `4076808430936957091`/`4076808430936957091` |
| `empty` | empty backward | `Tensor.abs().sum().backward()` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=True | loss (); grad (2, 0, 3), stride (3, 3, 1) | 100 | 2.240 us +/- 0.067 us, var 0.070 | 29.113 us +/- 0.306 us, var 1.269 | 0.08x | `6244622332573450659`/`6244622332573450659` |
| `contiguous_127x131` | contiguous backward | `Tensor.abs().sum().backward()` | (127, 131), stride (131, 1), offset 0, requires_grad=True | loss (); grad (127, 131), stride (131, 1) | 10 | 514.733 us +/- 6.386 us, var 342.556 | 447.267 us +/- 5.563 us, var 396.759 | 1.15x | `2366273887588498051`/`542543805154234979` |
| `offset_transposed_127x131` | offset backward | `Tensor.abs().sum().backward()` | (127, 131), stride (1, 127), offset 16637, requires_grad=True | loss (); grad (3, 131, 127), stride (16637, 127, 1) | 10 | 1676.602 us +/- 20.134 us, var 933.365 | 1212.597 us +/- 15.693 us, var 6015.742 | 1.38x | `14593145935623951011`/`12788056754867498499` |
| `noncontig_transpose_128x256` | noncontiguous backward | `Tensor.abs().sum().backward()` | (128, 256), stride (1, 128), offset 0, requires_grad=True | loss (); grad (256, 128), stride (128, 1) | 5 | 2306.828 us +/- 15.212 us, var 6293.129 | 1547.350 us +/- 16.194 us, var 2065.521 | 1.49x | `17941825962735956835`/`4704681492796864419` |
| `scalar` | scalar backward | `torch.abs(input).sum().backward()` | (), stride (), offset 0, requires_grad=True | loss (); grad (), stride () | 100 | 1.622 us +/- 0.030 us, var 0.042 | 24.543 us +/- 0.332 us, var 1.526 | 0.07x | `4076808430936957091`/`4076808430936957091` |
| `empty` | empty backward | `torch.abs(input).sum().backward()` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=True | loss (); grad (2, 0, 3), stride (3, 3, 1) | 100 | 2.287 us +/- 0.072 us, var 0.048 | 28.336 us +/- 0.365 us, var 12.395 | 0.08x | `6244622332573450659`/`6244622332573450659` |
| `contiguous_127x131` | contiguous backward | `torch.abs(input).sum().backward()` | (127, 131), stride (131, 1), offset 0, requires_grad=True | loss (); grad (127, 131), stride (131, 1) | 10 | 505.764 us +/- 5.105 us, var 135.797 | 433.683 us +/- 6.054 us, var 231.257 | 1.17x | `2366273887588498051`/`542543805154234979` |
| `offset_transposed_127x131` | offset backward | `torch.abs(input).sum().backward()` | (127, 131), stride (1, 127), offset 16637, requires_grad=True | loss (); grad (3, 131, 127), stride (16637, 127, 1) | 10 | 1684.243 us +/- 22.740 us, var 3727.575 | 1197.318 us +/- 13.187 us, var 617.604 | 1.41x | `14593145935623951011`/`12788056754867498499` |
| `noncontig_transpose_128x256` | noncontiguous backward | `torch.abs(input).sum().backward()` | (128, 256), stride (1, 128), offset 0, requires_grad=True | loss (); grad (256, 128), stride (128, 1) | 5 | 2311.758 us +/- 16.859 us, var 2434.524 | 1540.033 us +/- 10.752 us, var 7348.794 | 1.50x | `17941825962735956835`/`4704681492796864419` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `tensor_abs_in_place_abs_` | `AttributeError: 'torch_rs.Tensor' object has no attribute 'abs_'` | supported `tensor([2., 3.])` | zero |
| `top_level_torch_abs_out_tensor` | `RuntimeError: abs(): the 'out' argument is not supported` | supported `tensor([2., 3.])` | zero |
