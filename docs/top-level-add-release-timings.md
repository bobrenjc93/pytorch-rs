# Top-Level `torch.add` Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`5cc5ddb9fdd0180c281385edc5cbf1adbe72f5d9`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/top-level-add-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`, `VIRTUAL_ENV=`), so setup
used a worktree-local `.venv`. Cargo registry data was copied read-only from
the existing user cache into `target/cargo-home`, then Cargo ran offline so
build artifacts and dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home
cp -a /home/bobren/.cargo/registry target/cargo-home/
wheel_dir="$(mktemp -d "$PWD/target/top-level-add-wheels.XXXXXX")"
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
  .venv/bin/python -m unittest \
  tests.test_top_level_add tests.test_top_level_add_reference
env PATH="$PWD/.venv/bin:/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/top_level_add_release_timings.py \
  > target/top-level-add-release-timings.json
env PATH="$PWD/.venv/bin:/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/top_level_add_release_timings.py \
  > target/top-level-add-release-timings-pass2.json
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
  cargo test --locked --offline --all-targets add
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets binary_arithmetic
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets scalar_arithmetic
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_tensor_add tests.test_tensor_add_reference \
  tests.test_top_level_add tests.test_top_level_add_reference
git diff --check
git diff --check --no-index /dev/null docs/top-level-add-release-timings.md; \
  status=$?; if [ "$status" -eq 1 ]; then exit 0; else exit "$status"; fi
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 19 tests. The focused Rust `add` filter passed 1 test, the Rust binary
arithmetic filter passed 4 tests, the Rust scalar arithmetic filter passed 1
test, `cargo fmt --check` passed, and whitespace checks passed.

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
  packages in 15.17s, and installed in 1.10s
- Build time: successful offline release extension build completed in 36.53s;
  the release wheel reinstall resolved in 1 ms, prepared in 52 ms, and
  installed in 15 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
Each pass used 15 untimed warmup blocks and 81 measured blocks. A block
repeated the operation according to the table's `Repeats` column; times below
are median microseconds per operation. Reported medians are medians of the two
per-process medians. MAD and variance are the medians of the per-process MAD
and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch, and checked shape, stride, storage offset, contiguity,
dtype, device, `requires_grad`, and leaf status. The backward cell timed
`torch.add(...).sum().backward()` using pre-created fresh leaf tensors for
every measured invocation so the timed region did not include input
construction and did not reuse a freed graph; it checked both leaf gradients
against PyTorch. After every warmup and measured block, the driver consumed
the last output as a byte-level checksum; the backward cell consumed both leaf
gradients. The checksum column shows the final rolling sink from one pass as
`torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.97x uncapped, 0.97x capped
- Small contiguous cells: 0.25x uncapped, 0.25x capped
- Tensor/tensor contiguous cells: 0.87x uncapped, 0.87x capped
- Broadcasting cells: 1.05x uncapped, 1.05x capped
- Empty cells: 0.30x uncapped, 0.30x capped
- Noncontiguous transpose cells: 4.53x uncapped, 4.53x capped
- Offset transpose cells: 5.12x uncapped, 5.12x capped
- `no_grad` cells: 0.86x uncapped, 0.86x capped
- Autograd forward+backward cells: 0.56x uncapped, 0.56x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 2.38x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `small_contiguous_4x4` | small contiguous | `torch.add(left, other)` | left/right (4, 4), stride (4, 1) | (4, 4), stride (4, 1), requires_grad=False | 5000 | 0.304 us +/- 0.003, var 0.000 | 1.215 us +/- 0.011, var 0.003 | 0.25x | `18338139204163168436/18338139204163168436` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.add(left, other)` | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), requires_grad=False | 32 | 9.351 us +/- 0.144, var 0.117 | 10.698 us +/- 0.169, var 0.074 | 0.87x | `6969356787147152873/6969356787147152873` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.add(left, other)` | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), requires_grad=False | 16 | 53.948 us +/- 1.118, var 45.839 | 51.435 us +/- 0.585, var 2.810 | 1.05x | `7295163087248881325/7295163087248881325` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.add(left, other)` | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), requires_grad=False | 2000 | 0.371 us +/- 0.004, var 0.000 | 1.216 us +/- 0.007, var 0.005 | 0.30x | `14047266399210317956/14047266399210317956` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.add(left, other)` | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), requires_grad=False | 5 | 402.102 us +/- 18.990, var 4699.698 | 88.701 us +/- 3.265, var 46.159 | 4.53x | `17600738153719287395/17600738153719287395` |
| `offset_transposed_521x509` | offset | `torch.add(left, other)` | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), requires_grad=False | 5 | 194.998 us +/- 2.598, var 198.041 | 38.095 us +/- 0.670, var 3.732 | 5.12x | `5992407195856939839/5992407195856939839` |
| `no_grad_requires_grad_257x263` | no_grad | `torch.add(left, other)` | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), requires_grad=False | 32 | 9.233 us +/- 0.158, var 0.182 | 10.720 us +/- 0.142, var 0.228 | 0.86x | `6969356787147152873/6969356787147152873` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.add(left, other)` | left/right leaves (32, 33), requires_grad=True; timed torch.add(...).sum().backward() | scalar loss plus leaf gradients | 5 | 17.630 us +/- 0.262, var 0.785 | 31.444 us +/- 0.485, var 4.565 | 0.56x | `5596488364558270575/5596488364558270575` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `top_level_torch_add_tensor_scalar` | `NotImplementedError: add(): only exact native CPU float32 Tensor/Tensor operands are supported` | supported `tensor([3.])` | zero |
| `top_level_torch_add_scalar_tensor` | `NotImplementedError: add(): only exact native CPU float32 Tensor/Tensor operands are supported` | supported `tensor([3.])` | zero |
| `top_level_torch_add_scalar_scalar` | `NotImplementedError: add(): only exact native CPU float32 Tensor/Tensor operands are supported` | supported `tensor(5.)` | zero |
| `top_level_torch_add_nondefault_alpha_2` | `NotImplementedError: add(): alpha values other than 1 are not supported` | supported `tensor([7.])` | zero |
| `top_level_torch_add_out` | `RuntimeError: add(): the 'out' argument is not supported` | supported `tensor([4.])` | zero |
