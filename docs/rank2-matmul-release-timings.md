# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`66fc44f0b1d0302ef6e2ac83ce5a3b8bd112da7d`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/rank2-matmul-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`), so setup used a
worktree-local `.venv`. Cargo registry data was copied read-only from the
existing user cache into `target/cargo-home`, then Cargo ran offline so build
artifacts and dependency state stayed inside this worktree.

```bash
/usr/bin/time -p env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
/usr/bin/time -p env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
wheel_dir="$(mktemp -d "$PWD/target/rank2-matmul-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/rank2-matmul-wheel-dir.txt
/usr/bin/time -p env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/rank2-matmul-wheel-dir.txt)"
/usr/bin/time -p env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_matmul tests.test_matmul_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets matmul
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
  RANK2_MATMUL_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/rank2_matmul_release_timings.py \
  > target/rank2-matmul-release-timings-pass2.json
.venv/bin/python target/rank2_matmul_release_timings.py --summarize \
  target/rank2-matmul-release-timings.json \
  target/rank2-matmul-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_matmul tests.test_matmul_reference
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
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential
tests passed 19 tests. The focused Rust `matmul` filter passed 9 tests across
the library and integration test targets. `cargo fmt --check`, the README/docs
smoke test, and `git diff --check` passed.

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
  packages in 16.81s, and installed in 1.12s; the virtualenv was recreated in
  0.20s
- Build time: successful offline release extension build completed in 35.30s;
  the release wheel reinstall resolved in 3 ms, prepared in 38 ms, and
  installed in 13 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, rank-2 shapes, layouts,
grad mode, and thread settings. Every timing cell ran in two pinned process
passes. The first pass measured `torch_rs` before PyTorch; the second pass
reversed that order. Each pass used 15 untimed warmup blocks and 81 measured
blocks. A block repeated the operation according to the table's `Repeats`
column; times below are median microseconds per operation. Reported medians
are medians of the two per-process medians. MAD and variance are the medians
of the per-process MAD and sample variance values.

Before timing each supported cell, the driver checked `torch_rs` output against
PyTorch output for shape, stride, storage offset, contiguity, dtype, device,
`requires_grad`, leaf status, and values with `rtol=2e-5`, `atol=2e-5`, and
`equal_nan=True`. No CUDA tensors or non-float32 dtypes were used. After every
warmup and measured block, the driver materialized the last output as a
byte-level BLAKE2b rolling checksum over output metadata and logical float32
bytes. The checksum column shows the final rolling sink from one pass as
`torch_rs`/PyTorch; both process passes produced the same sink pairs. Nonempty
matmul cells can have different checksum values because the correctness gate is
PyTorch-compatible float32 tolerance, not bitwise equality.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 1.91x uncapped, 1.70x capped
- `@` operator cells: 1.82x uncapped, 1.63x capped
- `Tensor.matmul` cells: 1.89x uncapped, 1.69x capped
- `torch.matmul` cells: 2.02x uncapped, 1.80x capped
- Square cells: 2.19x uncapped, 2.19x capped
- Rectangular cells: 3.12x uncapped, 3.12x capped
- Skinny cells: 1.35x uncapped, 1.35x capped
- Empty-row cells: 0.42x uncapped, 0.42x capped
- Empty-inner cells: 0.41x uncapped, 0.41x capped
- Offset cells: 2.70x uncapped, 2.70x capped
- Noncontiguous transpose cells: 27.49x uncapped, 10.00x capped
- `no_grad` cells: 1.28x uncapped, 1.28x capped

Including the unsupported backward cells below as zero-credit denominator
entries with a 10.00x capped penalty gives a combined capped aggregate of
2.03x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `empty_inner_64x0_by_0x32` | empty inner | `left @ other` | left `(64, 0)`, stride `(1, 1)`; right `(0, 32)`, stride `(32, 1)` | `(64, 32)`, stride `(32, 1)`, offset `0`, requires_grad=`False` | 400 | 0.536 us +/- 0.007, var 0.001 | 1.512 us +/- 0.013, var 0.002 | 0.35x | `2545020339041362411`/`2545020339041362411` |
| `empty_inner_64x0_by_0x32` | empty inner | `left.matmul(other)` | left `(64, 0)`, stride `(1, 1)`; right `(0, 32)`, stride `(32, 1)` | `(64, 32)`, stride `(32, 1)`, offset `0`, requires_grad=`False` | 400 | 0.595 us +/- 0.004, var 0.001 | 1.477 us +/- 0.016, var 0.001 | 0.40x | `2545020339041362411`/`2545020339041362411` |
| `empty_inner_64x0_by_0x32` | empty inner | `torch.matmul(left, other)` | left `(64, 0)`, stride `(1, 1)`; right `(0, 32)`, stride `(32, 1)` | `(64, 32)`, stride `(32, 1)`, offset `0`, requires_grad=`False` | 400 | 0.676 us +/- 0.005, var 0.001 | 1.375 us +/- 0.024, var 0.002 | 0.49x | `2545020339041362411`/`2545020339041362411` |
| `empty_rows_0x257_by_257x64` | empty rows | `left @ other` | left `(0, 257)`, stride `(257, 1)`; right `(257, 64)`, stride `(64, 1)` | `(0, 64)`, stride `(64, 1)`, offset `0`, requires_grad=`False` | 2000 | 0.404 us +/- 0.003, var 0.001 | 1.153 us +/- 0.007, var 0.000 | 0.35x | `12446462772835730445`/`12446462772835730445` |
| `empty_rows_0x257_by_257x64` | empty rows | `left.matmul(other)` | left `(0, 257)`, stride `(257, 1)`; right `(257, 64)`, stride `(64, 1)` | `(0, 64)`, stride `(64, 1)`, offset `0`, requires_grad=`False` | 2000 | 0.462 us +/- 0.002, var 0.000 | 1.156 us +/- 0.006, var 0.000 | 0.40x | `12446462772835730445`/`12446462772835730445` |
| `empty_rows_0x257_by_257x64` | empty rows | `torch.matmul(left, other)` | left `(0, 257)`, stride `(257, 1)`; right `(257, 64)`, stride `(64, 1)` | `(0, 64)`, stride `(64, 1)`, offset `0`, requires_grad=`False` | 2000 | 0.552 us +/- 0.003, var 0.000 | 1.054 us +/- 0.005, var 0.000 | 0.52x | `12446462772835730445`/`12446462772835730445` |
| `no_grad_requires_grad_32x33_by_33x31` | no_grad | `left @ other` | left `(32, 33)`, stride `(33, 1)`, `requires_grad=True`; right `(33, 31)`, stride `(31, 1)`, `requires_grad=True`; operation inside `no_grad` | `(32, 31)`, stride `(31, 1)`, offset `0`, requires_grad=`False` | 50 | 6.365 us +/- 0.057, var 0.066 | 5.058 us +/- 0.024, var 0.033 | 1.26x | `15916834861566738946`/`2072294723053516900` |
| `no_grad_requires_grad_32x33_by_33x31` | no_grad | `left.matmul(other)` | left `(32, 33)`, stride `(33, 1)`, `requires_grad=True`; right `(33, 31)`, stride `(31, 1)`, `requires_grad=True`; operation inside `no_grad` | `(32, 31)`, stride `(31, 1)`, offset `0`, requires_grad=`False` | 50 | 6.411 us +/- 0.075, var 0.176 | 5.027 us +/- 0.031, var 0.042 | 1.28x | `15916834861566738946`/`2072294723053516900` |
| `no_grad_requires_grad_32x33_by_33x31` | no_grad | `torch.matmul(left, other)` | left `(32, 33)`, stride `(33, 1)`, `requires_grad=True`; right `(33, 31)`, stride `(31, 1)`, `requires_grad=True`; operation inside `no_grad` | `(32, 31)`, stride `(31, 1)`, offset `0`, requires_grad=`False` | 50 | 6.454 us +/- 0.095, var 0.050 | 4.941 us +/- 0.038, var 0.044 | 1.31x | `15916834861566738946`/`2072294723053516900` |
| `noncontiguous_transpose_80x96_by_96x72` | noncontiguous | `left @ other` | left `(80, 96)`, stride `(1, 80)`; right `(96, 72)`, stride `(1, 96)` | `(80, 72)`, stride `(72, 1)`, offset `0`, requires_grad=`False` | 8 | 396.5 us +/- 4.702, var 471.343 | 14.46 us +/- 0.051, var 0.457 | 27.42x | `8177957980774383573`/`16075817447928244127` |
| `noncontiguous_transpose_80x96_by_96x72` | noncontiguous | `left.matmul(other)` | left `(80, 96)`, stride `(1, 80)`; right `(96, 72)`, stride `(1, 96)` | `(80, 72)`, stride `(72, 1)`, offset `0`, requires_grad=`False` | 8 | 395.3 us +/- 2.109, var 288.963 | 14.42 us +/- 0.044, var 0.618 | 27.41x | `8177957980774383573`/`16075817447928244127` |
| `noncontiguous_transpose_80x96_by_96x72` | noncontiguous | `torch.matmul(left, other)` | left `(80, 96)`, stride `(1, 80)`; right `(96, 72)`, stride `(1, 96)` | `(80, 72)`, stride `(72, 1)`, offset `0`, requires_grad=`False` | 8 | 395.2 us +/- 2.519, var 1084.026 | 14.29 us +/- 0.046, var 0.299 | 27.65x | `8177957980774383573`/`16075817447928244127` |
| `offset_contiguous_72x64_by_64x80` | offset | `left @ other` | left `(72, 64)`, stride `(64, 1)`, input offset `4608`; right `(64, 80)`, stride `(80, 1)`, input offset `5120` | `(72, 80)`, stride `(80, 1)`, offset `0`, requires_grad=`False` | 10 | 26.48 us +/- 0.044, var 1.123 | 9.909 us +/- 0.034, var 0.219 | 2.67x | `15232414875145337912`/`9711080311053434357` |
| `offset_contiguous_72x64_by_64x80` | offset | `left.matmul(other)` | left `(72, 64)`, stride `(64, 1)`, input offset `4608`; right `(64, 80)`, stride `(80, 1)`, input offset `5120` | `(72, 80)`, stride `(80, 1)`, offset `0`, requires_grad=`False` | 10 | 26.70 us +/- 0.072, var 0.595 | 9.855 us +/- 0.032, var 0.159 | 2.71x | `15232414875145337912`/`9711080311053434357` |
| `offset_contiguous_72x64_by_64x80` | offset | `torch.matmul(left, other)` | left `(72, 64)`, stride `(64, 1)`, input offset `4608`; right `(64, 80)`, stride `(80, 1)`, input offset `5120` | `(72, 80)`, stride `(80, 1)`, offset `0`, requires_grad=`False` | 10 | 26.67 us +/- 0.067, var 0.748 | 9.769 us +/- 0.042, var 0.218 | 2.73x | `15232414875145337912`/`9711080311053434357` |
| `rectangular_64x192_by_192x48` | rectangular | `left @ other` | left `(64, 192)`, stride `(192, 1)`; right `(192, 48)`, stride `(48, 1)` | `(64, 48)`, stride `(48, 1)`, offset `0`, requires_grad=`False` | 10 | 45.25 us +/- 0.800, var 3.089 | 14.60 us +/- 0.070, var 0.421 | 3.10x | `10541693758228010420`/`16220627843161036500` |
| `rectangular_64x192_by_192x48` | rectangular | `left.matmul(other)` | left `(64, 192)`, stride `(192, 1)`; right `(192, 48)`, stride `(48, 1)` | `(64, 48)`, stride `(48, 1)`, offset `0`, requires_grad=`False` | 10 | 45.12 us +/- 0.666, var 3.405 | 14.50 us +/- 0.041, var 0.491 | 3.11x | `10541693758228010420`/`16220627843161036500` |
| `rectangular_64x192_by_192x48` | rectangular | `torch.matmul(left, other)` | left `(64, 192)`, stride `(192, 1)`; right `(192, 48)`, stride `(48, 1)` | `(64, 48)`, stride `(48, 1)`, offset `0`, requires_grad=`False` | 10 | 45.79 us +/- 1.046, var 1.721 | 14.47 us +/- 0.061, var 0.380 | 3.16x | `10541693758228010420`/`16220627843161036500` |
| `skinny_257x3_by_3x257` | skinny | `left @ other` | left `(257, 3)`, stride `(3, 1)`; right `(3, 257)`, stride `(257, 1)` | `(257, 257)`, stride `(257, 1)`, offset `0`, requires_grad=`False` | 20 | 23.50 us +/- 0.488, var 1.044 | 17.37 us +/- 0.130, var 1.269 | 1.35x | `300947814828308306`/`12355792441354865403` |
| `skinny_257x3_by_3x257` | skinny | `left.matmul(other)` | left `(257, 3)`, stride `(3, 1)`; right `(3, 257)`, stride `(257, 1)` | `(257, 257)`, stride `(257, 1)`, offset `0`, requires_grad=`False` | 20 | 23.34 us +/- 0.301, var 0.362 | 17.46 us +/- 0.258, var 0.389 | 1.34x | `300947814828308306`/`12355792441354865403` |
| `skinny_257x3_by_3x257` | skinny | `torch.matmul(left, other)` | left `(257, 3)`, stride `(3, 1)`; right `(3, 257)`, stride `(257, 1)` | `(257, 257)`, stride `(257, 1)`, offset `0`, requires_grad=`False` | 20 | 23.70 us +/- 0.496, var 0.521 | 17.25 us +/- 0.182, var 0.278 | 1.37x | `300947814828308306`/`12355792441354865403` |
| `square_32x32_by_32x32` | square | `left @ other` | left/right `(32, 32)`, stride `(32, 1)` | `(32, 32)`, stride `(32, 1)`, offset `0`, requires_grad=`False` | 80 | 4.042 us +/- 0.026, var 0.022 | 2.760 us +/- 0.023, var 0.085 | 1.46x | `8377037977828245610`/`8270611424449925180` |
| `square_32x32_by_32x32` | square | `left.matmul(other)` | left/right `(32, 32)`, stride `(32, 1)` | `(32, 32)`, stride `(32, 1)`, offset `0`, requires_grad=`False` | 80 | 4.182 us +/- 0.080, var 0.033 | 2.716 us +/- 0.008, var 0.009 | 1.54x | `8377037977828245610`/`8270611424449925180` |
| `square_32x32_by_32x32` | square | `torch.matmul(left, other)` | left/right `(32, 32)`, stride `(32, 1)` | `(32, 32)`, stride `(32, 1)`, offset `0`, requires_grad=`False` | 80 | 4.196 us +/- 0.036, var 0.020 | 2.617 us +/- 0.011, var 0.009 | 1.60x | `8377037977828245610`/`8270611424449925180` |
| `square_96x96_by_96x96` | square | `left @ other` | left/right `(96, 96)`, stride `(96, 1)` | `(96, 96)`, stride `(96, 1)`, offset `0`, requires_grad=`False` | 8 | 64.06 us +/- 1.251, var 3.593 | 20.55 us +/- 0.194, var 1.653 | 3.12x | `10398773188212464315`/`6357555168851627360` |
| `square_96x96_by_96x96` | square | `left.matmul(other)` | left/right `(96, 96)`, stride `(96, 1)` | `(96, 96)`, stride `(96, 1)`, offset `0`, requires_grad=`False` | 8 | 63.96 us +/- 1.289, var 2.475 | 20.38 us +/- 0.156, var 1.504 | 3.14x | `10398773188212464315`/`6357555168851627360` |
| `square_96x96_by_96x96` | square | `torch.matmul(left, other)` | left/right `(96, 96)`, stride `(96, 1)` | `(96, 96)`, stride `(96, 1)`, offset `0`, requires_grad=`False` | 8 | 64.01 us +/- 1.238, var 1.726 | 20.50 us +/- 0.162, var 0.692 | 3.12x | `10398773188212464315`/`6357555168851627360` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch behavior. They are preserved as zero-credit cells instead of being
removed from the evidence set. Current rank-2 matmul accepts operands with
`requires_grad=True`, but the result is a non-grad leaf, so full-`sum`
backward is unsupported.

| Workload | API | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `backward_full_sum_2x2_left_@_other` | `left @ other` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported; left.grad shape `(2, 2)`, right.grad shape `(2, 2)` | zero |
| `backward_full_sum_2x2_left.matmul(other)` | `left.matmul(other)` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported; left.grad shape `(2, 2)`, right.grad shape `(2, 2)` | zero |
| `backward_full_sum_2x2_torch.matmul(left,_other)` | `torch.matmul(left, other)` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported; left.grad shape `(2, 2)`, right.grad shape `(2, 2)` | zero |
