# Rank-2 `@`, `Tensor.matmul`, and `torch.matmul` Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`d2e5eac41179053c8e99814067615979e6bb4820`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/rank2-matmul-release-timings*.json`. No Conda
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
wheel_dir="$(mktemp -d "$PWD/target/rank2-matmul-wheels.XXXXXX")" && \
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir" && \
  printf '%s\n' "$wheel_dir" > target/rank2-matmul-wheel-dir.txt
wheel_dir="$(cat target/rank2-matmul-wheel-dir.txt)"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_matmul tests.test_matmul_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/rank2_matmul_release_timings.py \
  > target/rank2-matmul-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/rank2_matmul_release_timings.py \
  > target/rank2-matmul-release-timings-pass2.json
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
  cargo test --locked --offline --all-targets matmul
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_matmul tests.test_matmul_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 19 tests. The focused Rust `matmul` filter passed 9 tests across the
library and tensor baseline targets. `cargo fmt --check` passed. The README/docs
smoke test and `git diff --check` passed.

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
- Dependency installation: locked `uv sync` resolved in 31 ms, prepared
  packages in 16.14s, and installed in 1.09s
- Build time: successful offline release extension build completed in 34.77s;
  the release wheel reinstall resolved in 2 ms, prepared in 44 ms, and
  installed in 53 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
Each pass used 15 untimed warmup blocks and 81 measured blocks. A block
repeated the operation according to the table's `Repeats` column; times below
are median microseconds per operation. Reported medians are medians of the two
per-process medians. MAD and variance are the medians of the per-process MAD
and sample variance values.

Before timing each supported cell, the driver compared `torch_rs` output values
with PyTorch using `rtol=2e-6`, `atol=1e-5`, and `equal_nan=True`, and checked
shape, stride, storage offset, contiguity, dtype, device, `requires_grad`, and
leaf status. After every warmup and measured block, the driver materialized the
last output as a byte-level checksum. The checksum column shows the final
rolling sink from one pass as `torch_rs`/PyTorch; both process passes produced
the same sink pairs for every supported cell.

The current rank-2 matmul surface does not support autograd recording:
`requires_grad=True` inputs produce a leaf result with `requires_grad=False`.
Backward-through-full-`sum` cells are therefore reported as zero-credit
unsupported cells rather than removed from the denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.73x uncapped, 2.41x capped
- `left @ right` cells: 2.57x uncapped, 2.27x capped
- `left.matmul(other=right)` cells: 2.72x uncapped, 2.40x capped
- `torch.matmul(input=left, other=right)` cells: 2.91x uncapped, 2.58x capped
- Square cells: 3.34x uncapped, 3.34x capped
- Rectangular cells: 4.24x uncapped, 4.24x capped
- Skinny cells: 3.71x uncapped, 3.71x capped
- Empty-dimension cells: 0.44x uncapped, 0.44x capped
- Offset cells: 3.72x uncapped, 3.72x capped
- Noncontiguous cells: 26.72x uncapped, 10.00x capped
- `no_grad` cells: 3.05x uncapped, 3.05x capped

Including the unsupported backward cells below as zero-credit denominator
entries with a 10.00x capped penalty gives a combined capped aggregate of
2.83x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `square_128` | square | `left @ right` | left/right (128, 128), stride (128, 1) | (128, 128), stride (128, 1), requires_grad=False | 5 | 148.774 us +/- 1.313 us, var 141.895 | 44.399 us +/- 0.210 us, var 3.567 | 3.35x | `14442705102491945123`/`2260766533322199203` |
| `square_128` | square | `left.matmul(other=right)` | left/right (128, 128), stride (128, 1) | (128, 128), stride (128, 1), requires_grad=False | 5 | 149.760 us +/- 1.425 us, var 54.626 | 44.850 us +/- 0.322 us, var 2.955 | 3.34x | `14442705102491945123`/`2260766533322199203` |
| `square_128` | square | `torch.matmul(input=left, other=right)` | left/right (128, 128), stride (128, 1) | (128, 128), stride (128, 1), requires_grad=False | 5 | 150.295 us +/- 1.171 us, var 9.573 | 44.987 us +/- 0.432 us, var 3.920 | 3.34x | `14442705102491945123`/`2260766533322199203` |
| `rectangular_prime_127x257_by_257x131` | rectangular | `left @ right` | left (127, 257), right (257, 131), row-major contiguous | (127, 131), stride (131, 1), requires_grad=False | 3 | 462.182 us +/- 3.569 us, var 536.625 | 108.342 us +/- 0.568 us, var 12.223 | 4.27x | `15166792651644264867`/`13813085691924600483` |
| `rectangular_prime_127x257_by_257x131` | rectangular | `left.matmul(other=right)` | left (127, 257), right (257, 131), row-major contiguous | (127, 131), stride (131, 1), requires_grad=False | 3 | 457.418 us +/- 3.600 us, var 109.662 | 108.604 us +/- 0.699 us, var 5.102 | 4.21x | `15166792651644264867`/`13813085691924600483` |
| `rectangular_prime_127x257_by_257x131` | rectangular | `torch.matmul(input=left, other=right)` | left (127, 257), right (257, 131), row-major contiguous | (127, 131), stride (131, 1), requires_grad=False | 3 | 461.587 us +/- 4.074 us, var 320.106 | 108.975 us +/- 1.053 us, var 15.782 | 4.24x | `15166792651644264867`/`13813085691924600483` |
| `skinny_4096x8_by_8x16` | skinny | `left @ right` | left (4096, 8), right (8, 16), row-major contiguous | (4096, 16), stride (16, 1), requires_grad=False | 10 | 74.843 us +/- 0.489 us, var 3.467 | 19.919 us +/- 0.227 us, var 0.715 | 3.76x | `8061016654608921635`/`12967001385132842147` |
| `skinny_4096x8_by_8x16` | skinny | `left.matmul(other=right)` | left (4096, 8), right (8, 16), row-major contiguous | (4096, 16), stride (16, 1), requires_grad=False | 10 | 75.361 us +/- 0.584 us, var 2.112 | 20.617 us +/- 0.573 us, var 2.123 | 3.66x | `8061016654608921635`/`12967001385132842147` |
| `skinny_4096x8_by_8x16` | skinny | `torch.matmul(input=left, other=right)` | left (4096, 8), right (8, 16), row-major contiguous | (4096, 16), stride (16, 1), requires_grad=False | 10 | 75.101 us +/- 0.574 us, var 2.784 | 20.174 us +/- 0.231 us, var 3.453 | 3.72x | `8061016654608921635`/`12967001385132842147` |
| `empty_rows_0x257_by_257x131` | empty | `left @ right` | left (0, 257), right (257, 131) | (0, 131), stride (131, 1), requires_grad=False | 5000 | 0.196 us +/- 0.002 us, var 0.000 | 0.914 us +/- 0.006 us, var 0.000 | 0.21x | `6546202739528872547`/`6546202739528872547` |
| `empty_rows_0x257_by_257x131` | empty | `left.matmul(other=right)` | left (0, 257), right (257, 131) | (0, 131), stride (131, 1), requires_grad=False | 5000 | 0.378 us +/- 0.003 us, var 0.000 | 0.982 us +/- 0.008 us, var 0.005 | 0.39x | `6546202739528872547`/`6546202739528872547` |
| `empty_rows_0x257_by_257x131` | empty | `torch.matmul(input=left, other=right)` | left (0, 257), right (257, 131) | (0, 131), stride (131, 1), requires_grad=False | 5000 | 0.488 us +/- 0.004 us, var 0.000 | 0.921 us +/- 0.005 us, var 0.000 | 0.53x | `6546202739528872547`/`6546202739528872547` |
| `empty_inner_257x0_by_0x131` | empty | `left @ right` | left (257, 0), right (0, 131) | (257, 131), stride (131, 1), requires_grad=False | 50 | 1.491 us +/- 0.009 us, var 0.008 | 2.735 us +/- 0.025 us, var 0.048 | 0.55x | `14140264245699171171`/`14140264245699171171` |
| `empty_inner_257x0_by_0x131` | empty | `left.matmul(other=right)` | left (257, 0), right (0, 131) | (257, 131), stride (131, 1), requires_grad=False | 50 | 1.708 us +/- 0.020 us, var 0.123 | 3.437 us +/- 0.027 us, var 0.037 | 0.50x | `14140264245699171171`/`14140264245699171171` |
| `empty_inner_257x0_by_0x131` | empty | `torch.matmul(input=left, other=right)` | left (257, 0), right (0, 131) | (257, 131), stride (131, 1), requires_grad=False | 50 | 2.072 us +/- 0.038 us, var 0.017 | 3.375 us +/- 0.018 us, var 0.019 | 0.61x | `14140264245699171171`/`14140264245699171171` |
| `offset_contiguous_129x131_by_131x127` | offset | `left @ right` | left tensor((3, 129, 131))[1], right tensor((2, 131, 127))[1] | (129, 127), stride (127, 1), requires_grad=False | 5 | 227.013 us +/- 3.481 us, var 268.980 | 60.486 us +/- 0.259 us, var 4.075 | 3.75x | `12535240843143135395`/`10535414583130479203` |
| `offset_contiguous_129x131_by_131x127` | offset | `left.matmul(other=right)` | left tensor((3, 129, 131))[1], right tensor((2, 131, 127))[1] | (129, 127), stride (127, 1), requires_grad=False | 5 | 226.903 us +/- 2.523 us, var 80.152 | 60.625 us +/- 0.385 us, var 5.444 | 3.74x | `12535240843143135395`/`10535414583130479203` |
| `offset_contiguous_129x131_by_131x127` | offset | `torch.matmul(input=left, other=right)` | left tensor((3, 129, 131))[1], right tensor((2, 131, 127))[1] | (129, 127), stride (127, 1), requires_grad=False | 5 | 228.776 us +/- 3.823 us, var 39.698 | 62.343 us +/- 1.857 us, var 36.019 | 3.67x | `12535240843143135395`/`10535414583130479203` |
| `noncontig_transpose_128x257_by_257x129` | noncontiguous | `left @ right` | left tensor((257, 128)).T, right tensor((129, 257)).T | (128, 129), stride (129, 1), requires_grad=False | 2 | 2705.131 us +/- 11.598 us, var 6526.054 | 100.655 us +/- 0.401 us, var 11.674 | 26.88x | `4559525647265685155`/`6666729941669748867` |
| `noncontig_transpose_128x257_by_257x129` | noncontiguous | `left.matmul(other=right)` | left tensor((257, 128)).T, right tensor((129, 257)).T | (128, 129), stride (129, 1), requires_grad=False | 2 | 2700.432 us +/- 22.667 us, var 47351.572 | 101.463 us +/- 0.859 us, var 14.678 | 26.61x | `4559525647265685155`/`6666729941669748867` |
| `noncontig_transpose_128x257_by_257x129` | noncontiguous | `torch.matmul(input=left, other=right)` | left tensor((257, 128)).T, right tensor((129, 257)).T | (128, 129), stride (129, 1), requires_grad=False | 2 | 2697.287 us +/- 16.904 us, var 3497.859 | 101.173 us +/- 0.543 us, var 11.357 | 26.66x | `4559525647265685155`/`6666729941669748867` |
| `no_grad_127x131_by_131x64` | no_grad | `left @ right` | left/right leaves with requires_grad=True inside no_grad | (127, 64), stride (64, 1), requires_grad=False | 5 | 79.277 us +/- 1.331 us, var 8.842 | 26.287 us +/- 0.702 us, var 1.954 | 3.02x | `15645642304887614115`/`10712686822361836547` |
| `no_grad_127x131_by_131x64` | no_grad | `left.matmul(other=right)` | left/right leaves with requires_grad=True inside no_grad | (127, 64), stride (64, 1), requires_grad=False | 5 | 79.587 us +/- 2.027 us, var 17.977 | 26.359 us +/- 0.189 us, var 2.835 | 3.02x | `15645642304887614115`/`10712686822361836547` |
| `no_grad_127x131_by_131x64` | no_grad | `torch.matmul(input=left, other=right)` | left/right leaves with requires_grad=True inside no_grad | (127, 64), stride (64, 1), requires_grad=False | 5 | 80.203 us +/- 2.297 us, var 11.079 | 25.828 us +/- 0.164 us, var 3.009 | 3.11x | `15645642304887614115`/`10712686822361836547` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | API | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `backward_full_sum_17x19_by_19x13` | `left @ right` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported loss shape (), left grad (17, 19), right grad (19, 13), checksums 3388331765814850405/2204697730407513040 | zero |
| `backward_full_sum_17x19_by_19x13` | `left.matmul(other=right)` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported loss shape (), left grad (17, 19), right grad (19, 13), checksums 3388331765814850405/2204697730407513040 | zero |
| `backward_full_sum_17x19_by_19x13` | `torch.matmul(input=left, other=right)` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported loss shape (), left grad (17, 19), right grad (19, 13), checksums 3388331765814850405/2204697730407513040 | zero |
