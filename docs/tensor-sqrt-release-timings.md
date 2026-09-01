# `Tensor.sqrt` and `torch.sqrt` Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`4c091843e5569f1c1ba8ce8e67cd02be20766b92`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-sqrt-release-timings*.json`. No Conda
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
wheel_dir="$(mktemp -d "$PWD/target/tensor-sqrt-wheels.XXXXXX")"
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
  .venv/bin/python -m unittest tests.test_sqrt tests.test_sqrt_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_sqrt_release_timings.py \
  > target/tensor-sqrt-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_sqrt_release_timings.py \
  > target/tensor-sqrt-release-timings-pass2.json
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
  cargo test --locked --offline --all-targets sqrt
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_sqrt tests.test_sqrt_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 14 tests. The focused Rust `sqrt` filter passed 3 tests, `cargo fmt
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
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  packages in 16.10s, and installed in 1.36s
- Build time: successful offline release extension build completed in 36.19s;
  the release wheel reinstall resolved in 2 ms, prepared in 38 ms, and
  installed in 12 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
Each pass used 15 untimed warmup blocks and 81 measured blocks. A block
repeated the operation according to the table's `Repeats` column; times below
are median microseconds per operation. Reported medians are medians of the two
per-process medians. MAD and variance are the medians of the per-process MAD
and sample variance values.

Before timing each supported forward cell, the driver compared `torch_rs`
output values with PyTorch using `rtol=1e-5`, `atol=1e-6`, and
`equal_nan=True`, checked signed-zero bits where zeros were present, and
checked shape, stride, storage offset, contiguity, dtype, device,
`requires_grad`, and leaf status. Before timing each backward cell, it checked
the scalar full-`sum` loss metadata and value with the same tolerances, then
checked the leaf gradient values and metadata. Backward timings used fresh
leaf/view inputs for every measured invocation so the timed region did not
include input construction and did not reuse a freed graph. After every warmup
and measured block, the driver consumed the last forward output or the last
backward scalar loss plus leaf gradient as a byte-level rolling checksum. The
checksum column shows the final rolling sink from one pass as `torch_rs`/
PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.74x uncapped, 0.83x capped
- Forward cells: 1.06x uncapped, 1.06x capped
- Backward-through-full-`sum` cells: 0.52x uncapped, 0.66x capped
- `Tensor.sqrt` forward cells: 1.02x uncapped, 1.02x capped
- `torch.sqrt` forward cells: 1.10x uncapped, 1.10x capped
- `Tensor.sqrt().sum().backward()` cells: 0.52x uncapped, 0.66x capped
- `torch.sqrt(input).sum().backward()` cells: 0.52x uncapped, 0.66x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 1.04x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar` | scalar | `Tensor.sqrt` | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.192 us +/- 0.023 us, var 0.001 | 1.198 us +/- 0.103 us, var 0.059 | 0.16x | `14547830640957252357`/`14547830640957252357` |
| `empty` | empty | `Tensor.sqrt` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (3, 0, 2), stride (2, 2, 1), offset 0, requires_grad=False | 5000 | 0.232 us +/- 0.009 us, var 0.001 | 0.922 us +/- 0.033 us, var 0.025 | 0.25x | `3006588711413039173`/`3006588711413039173` |
| `contiguous_257x263` | contiguous | `Tensor.sqrt` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 93.919 us +/- 0.390 us, var 0.715 | 34.369 us +/- 1.053 us, var 11.001 | 2.73x | `7505626677791183013`/`2995096984328018885` |
| `offset_transposed_521x509` | offset | `Tensor.sqrt` | (521, 509), stride (1, 521), offset 265189, requires_grad=False | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 367.643 us +/- 1.557 us, var 14.992 | 119.108 us +/- 2.007 us, var 69.573 | 3.09x | `10013884849469089061`/`2518652306359623333` |
| `noncontig_transpose_512x1024` | noncontiguous | `Tensor.sqrt` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 730.050 us +/- 3.093 us, var 36.272 | 229.556 us +/- 2.727 us, var 198.919 | 3.18x | `14826007983310759205`/`9562068787381281349` |
| `scalar` | scalar backward | `Tensor.sqrt().sum().backward()` | (), stride (), offset 0, requires_grad=True | loss (); grad (), stride (), offset 0 | 100 | 1.404 us +/- 0.091 us, var 0.030 | 26.989 us +/- 0.300 us, var 1.046 | 0.05x | `234391188702739493`/`234391188702739493` |
| `empty` | empty backward | `Tensor.sqrt().sum().backward()` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=True | loss (); grad (2, 0, 3), stride (3, 3, 1), offset 0 | 100 | 1.830 us +/- 0.016 us, var 0.005 | 31.744 us +/- 0.972 us, var 48.825 | 0.06x | `2966908844158765861`/`2966908844158765861` |
| `contiguous_127x131` | contiguous backward | `Tensor.sqrt().sum().backward()` | (127, 131), stride (131, 1), offset 0, requires_grad=True | loss (); grad (127, 131), stride (131, 1), offset 0 | 10 | 50.901 us +/- 0.773 us, var 1.332 | 58.590 us +/- 2.723 us, var 50.422 | 0.87x | `16464595072181516869`/`4908686610900008197` |
| `offset_transposed_127x131` | offset backward | `Tensor.sqrt().sum().backward()` | (127, 131), stride (1, 127), offset 16637, requires_grad=True | loss (); grad (3, 131, 127), stride (16637, 127, 1), offset 0 | 10 | 240.856 us +/- 4.575 us, var 426.695 | 93.441 us +/- 6.502 us, var 147.646 | 2.58x | `1566598548195367589`/`298177036387046405` |
| `noncontig_transpose_128x256` | noncontiguous backward | `Tensor.sqrt().sum().backward()` | (128, 256), stride (1, 128), offset 0, requires_grad=True | loss (); grad (256, 128), stride (128, 1), offset 0 | 5 | 463.341 us +/- 10.299 us, var 650.376 | 85.289 us +/- 3.217 us, var 123.557 | 5.43x | `15833103653478580485`/`3173209404822589509` |
| `scalar` | scalar | `torch.sqrt` | (), stride (), offset 0, requires_grad=False | (), stride (), offset 0, requires_grad=False | 10000 | 0.229 us +/- 0.002 us, var 0.000 | 1.154 us +/- 0.012 us, var 0.012 | 0.20x | `14547830640957252357`/`14547830640957252357` |
| `empty` | empty | `torch.sqrt` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=False | (3, 0, 2), stride (2, 2, 1), offset 0, requires_grad=False | 5000 | 0.254 us +/- 0.005 us, var 0.005 | 0.943 us +/- 0.010 us, var 0.007 | 0.27x | `3006588711413039173`/`3006588711413039173` |
| `contiguous_257x263` | contiguous | `torch.sqrt` | (257, 263), stride (263, 1), offset 0, requires_grad=False | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 93.630 us +/- 0.252 us, var 0.551 | 30.613 us +/- 0.212 us, var 0.748 | 3.06x | `7505626677791183013`/`2995096984328018885` |
| `offset_transposed_521x509` | offset | `torch.sqrt` | (521, 509), stride (1, 521), offset 265189, requires_grad=False | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 368.725 us +/- 1.396 us, var 7.296 | 119.275 us +/- 2.092 us, var 23.279 | 3.09x | `10013884849469089061`/`2518652306359623333` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.sqrt` | (512, 1024), stride (1, 512), offset 0, requires_grad=False | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 729.237 us +/- 2.968 us, var 25.046 | 230.360 us +/- 1.836 us, var 66.371 | 3.17x | `14826007983310759205`/`9562068787381281349` |
| `scalar` | scalar backward | `torch.sqrt(input).sum().backward()` | (), stride (), offset 0, requires_grad=True | loss (); grad (), stride (), offset 0 | 100 | 1.404 us +/- 0.010 us, var 0.005 | 26.901 us +/- 0.262 us, var 2.166 | 0.05x | `234391188702739493`/`234391188702739493` |
| `empty` | empty backward | `torch.sqrt(input).sum().backward()` | (3, 0, 2), stride (1, 3, 3), offset 0, requires_grad=True | loss (); grad (2, 0, 3), stride (3, 3, 1), offset 0 | 100 | 1.893 us +/- 0.028 us, var 0.013 | 31.506 us +/- 0.443 us, var 2.309 | 0.06x | `2966908844158765861`/`2966908844158765861` |
| `contiguous_127x131` | contiguous backward | `torch.sqrt(input).sum().backward()` | (127, 131), stride (131, 1), offset 0, requires_grad=True | loss (); grad (127, 131), stride (131, 1), offset 0 | 10 | 51.277 us +/- 0.788 us, var 1.246 | 60.230 us +/- 3.384 us, var 53.559 | 0.85x | `16464595072181516869`/`4908686610900008197` |
| `offset_transposed_127x131` | offset backward | `torch.sqrt(input).sum().backward()` | (127, 131), stride (1, 127), offset 16637, requires_grad=True | loss (); grad (3, 131, 127), stride (16637, 127, 1), offset 0 | 10 | 244.312 us +/- 3.937 us, var 347.206 | 98.505 us +/- 5.061 us, var 60.317 | 2.48x | `1566598548195367589`/`298177036387046405` |
| `noncontig_transpose_128x256` | noncontiguous backward | `torch.sqrt(input).sum().backward()` | (128, 256), stride (1, 128), offset 0, requires_grad=True | loss (); grad (256, 128), stride (128, 1), offset 0 | 5 | 527.798 us +/- 11.082 us, var 1586.100 | 92.434 us +/- 5.289 us, var 185.344 | 5.71x | `15833103653478580485`/`3173209404822589509` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `tensor_sqrt_in_place_sqrt_` | `AttributeError: 'torch_rs.Tensor' object has no attribute 'sqrt_'` | supported `tensor([2.])` | zero |
| `top_level_torch_sqrt_out_tensor` | `RuntimeError: sqrt(): the 'out' argument is not supported` | supported `tensor([2.])` | zero |
