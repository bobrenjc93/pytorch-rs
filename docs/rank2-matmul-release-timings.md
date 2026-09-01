# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`4c091843e5569f1c1ba8ce8e67cd02be20766b92`. This branch adds timing evidence
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
wheel_dir="$(mktemp -d "$PWD/target/rank2-matmul-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/rank2-matmul-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
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
git diff --check
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
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 19 tests. The focused Rust `matmul` filter passed 6 unit tests and 3
integration tests, `cargo fmt --check` passed, and `git diff --check` passed.

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
  packages in 16.07s, and installed in 920 ms
- Build time: successful offline release extension build completed in 34.38s;
  the release wheel reinstall resolved in 1 ms, prepared in 37 ms, and
  installed in 17 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
The first pass measured `torch_rs` before PyTorch; the second pass reversed
that order. Each pass used 15 untimed warmup blocks and 81 measured blocks.
A block repeated the operation according to the table's `Repeats` column;
times below are median microseconds per operation. Reported medians are
medians of the two per-process medians. MAD and variance are the medians of the
per-process MAD and sample variance values.

Before timing each supported cell, the driver compared `torch_rs` output values
with PyTorch using `rtol=5e-5`, `atol=5e-5`, and `equal_nan=True`, checked NaN
masks and sign bits for non-NaN values, and checked shape, stride, storage
offset, contiguity, dtype, device, `requires_grad`, and leaf status. The
current matmul surface does not support active-autograd
backward-through-full-`sum`: tensors requiring grad are accepted under
`torch.no_grad()`, while ordinary grad-enabled matmul returns an untracked leaf
and cannot be backpropagated equivalently to PyTorch. Those backward cells are
listed as zero-credit unsupported cells below.

After every warmup and measured block, the driver consumed the last output as a
64-bit BLAKE2b rolling checksum over tensor metadata and logical bytes. The
checksum column shows the final rolling sink from both passes as
`torch_rs`/PyTorch; matching checksums across passes show stable materialized
outputs for each implementation. `torch_rs` and PyTorch checksums can differ
for non-empty matmul cells because their legal float32 accumulation orders do
not produce bit-identical output bytes, but every timed cell passed the numeric
and metadata gates before sampling.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 2.95x uncapped, 2.67x capped
- `@` operator cells: 2.75x uncapped, 2.49x capped
- `Tensor.matmul` cells: 2.93x uncapped, 2.66x capped
- `torch.matmul` cells: 3.18x uncapped, 2.88x capped
- Square cells: 4.54x uncapped, 4.54x capped
- Rectangular cells: 6.58x uncapped, 6.58x capped
- Skinny cells: 6.61x uncapped, 6.61x capped
- Empty inner-dimension cells: 0.24x uncapped, 0.24x capped
- Empty output-dimension cells: 0.30x uncapped, 0.30x capped
- Offset cells: 4.61x uncapped, 4.61x capped
- Noncontiguous cells: 22.20x uncapped, 10.00x capped
- `no_grad` cells: 3.81x uncapped, 3.81x capped
- Backward-through-full-`sum` cells: no supported cells in this revision

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 3.48x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `square_128x128_by_128x128` | square | `left @ other` | left/right (128, 128), stride (128, 1); grad disabled by inputs | (128, 128), stride (128, 1), offset 0, requires_grad=False, is_leaf=True | 5 | 291.782 us +/- 1.757 us, var 54.998 | 64.479 us +/- 0.929 us, var 26.651 | 4.53x | pass1 `879202295713601824`/`1842448931789410336`; pass2 `879202295713601824`/`1842448931789410336` |
| `square_128x128_by_128x128` | square | `left.matmul(other)` | left/right (128, 128), stride (128, 1); grad disabled by inputs | (128, 128), stride (128, 1), offset 0, requires_grad=False, is_leaf=True | 5 | 292.403 us +/- 2.065 us, var 41.936 | 64.177 us +/- 0.522 us, var 4.637 | 4.56x | pass1 `879202295713601824`/`1842448931789410336`; pass2 `879202295713601824`/`1842448931789410336` |
| `square_128x128_by_128x128` | square | `torch.matmul(left, other)` | left/right (128, 128), stride (128, 1); grad disabled by inputs | (128, 128), stride (128, 1), offset 0, requires_grad=False, is_leaf=True | 5 | 294.726 us +/- 3.286 us, var 190.265 | 65.049 us +/- 1.304 us, var 17.119 | 4.53x | pass1 `879202295713601824`/`1842448931789410336`; pass2 `879202295713601824`/`1842448931789410336` |
| `rectangular_257x131_by_131x263` | rectangular | `left @ other` | left (257, 131), stride (131, 1); right (131, 263), stride (263, 1); grad disabled by inputs | (257, 263), stride (263, 1), offset 0, requires_grad=False, is_leaf=True | 1 | 3856.604 us +/- 25.068 us, var 16723.385 | 566.677 us +/- 9.875 us, var 1805.158 | 6.81x | pass1 `7382256426239526976`/`5924235438047838240`; pass2 `7382256426239526976`/`5924235438047838240` |
| `rectangular_257x131_by_131x263` | rectangular | `left.matmul(other)` | left (257, 131), stride (131, 1); right (131, 263), stride (263, 1); grad disabled by inputs | (257, 263), stride (263, 1), offset 0, requires_grad=False, is_leaf=True | 1 | 3857.871 us +/- 23.621 us, var 35568.122 | 589.762 us +/- 21.538 us, var 1268.783 | 6.54x | pass1 `7382256426239526976`/`5924235438047838240`; pass2 `7382256426239526976`/`5924235438047838240` |
| `rectangular_257x131_by_131x263` | rectangular | `torch.matmul(left, other)` | left (257, 131), stride (131, 1); right (131, 263), stride (263, 1); grad disabled by inputs | (257, 263), stride (263, 1), offset 0, requires_grad=False, is_leaf=True | 1 | 3862.793 us +/- 20.366 us, var 17042.903 | 602.846 us +/- 11.182 us, var 379.848 | 6.41x | pass1 `7382256426239526976`/`5924235438047838240`; pass2 `7382256426239526976`/`5924235438047838240` |
| `skinny_1024x8_by_8x64` | skinny | `left @ other` | left (1024, 8), stride (8, 1); right (8, 64), stride (64, 1); grad disabled by inputs | (1024, 64), stride (64, 1), offset 0, requires_grad=False, is_leaf=True | 8 | 400.383 us +/- 1.912 us, var 73.491 | 60.613 us +/- 0.948 us, var 3.694 | 6.61x | pass1 `5476710130286159136`/`10598860312489436704`; pass2 `5476710130286159136`/`10598860312489436704` |
| `skinny_1024x8_by_8x64` | skinny | `left.matmul(other)` | left (1024, 8), stride (8, 1); right (8, 64), stride (64, 1); grad disabled by inputs | (1024, 64), stride (64, 1), offset 0, requires_grad=False, is_leaf=True | 8 | 400.063 us +/- 2.268 us, var 19.315 | 60.710 us +/- 0.980 us, var 6.891 | 6.59x | pass1 `5476710130286159136`/`10598860312489436704`; pass2 `5476710130286159136`/`10598860312489436704` |
| `skinny_1024x8_by_8x64` | skinny | `torch.matmul(left, other)` | left (1024, 8), stride (8, 1); right (8, 64), stride (64, 1); grad disabled by inputs | (1024, 64), stride (64, 1), offset 0, requires_grad=False, is_leaf=True | 8 | 402.344 us +/- 2.879 us, var 73.434 | 60.679 us +/- 1.032 us, var 7.995 | 6.63x | pass1 `5476710130286159136`/`10598860312489436704`; pass2 `5476710130286159136`/`10598860312489436704` |
| `empty_inner_13x0_by_0x17` | empty inner dimension | `left @ other` | left ones((13, 0)); right zeros((0, 17)); grad disabled by inputs | (13, 17), stride (17, 1), offset 0, requires_grad=False, is_leaf=True | 1000 | 0.226 us +/- 0.004 us, var 0.000 | 1.232 us +/- 0.037 us, var 0.028 | 0.18x | pass1 `880690194175310880`/`880690194175310880`; pass2 `880690194175310880`/`880690194175310880` |
| `empty_inner_13x0_by_0x17` | empty inner dimension | `left.matmul(other)` | left ones((13, 0)); right zeros((0, 17)); grad disabled by inputs | (13, 17), stride (17, 1), offset 0, requires_grad=False, is_leaf=True | 1000 | 0.286 us +/- 0.005 us, var 0.000 | 1.176 us +/- 0.015 us, var 0.003 | 0.24x | pass1 `880690194175310880`/`880690194175310880`; pass2 `880690194175310880`/`880690194175310880` |
| `empty_inner_13x0_by_0x17` | empty inner dimension | `torch.matmul(left, other)` | left ones((13, 0)); right zeros((0, 17)); grad disabled by inputs | (13, 17), stride (17, 1), offset 0, requires_grad=False, is_leaf=True | 1000 | 0.357 us +/- 0.005 us, var 0.000 | 1.091 us +/- 0.027 us, var 0.006 | 0.33x | pass1 `880690194175310880`/`880690194175310880`; pass2 `880690194175310880`/`880690194175310880` |
| `empty_rows_0x37_by_37x23` | empty output dimension | `left @ other` | left zeros((0, 37)); right ones((37, 23)); grad disabled by inputs | (0, 23), stride (23, 1), offset 0, requires_grad=False, is_leaf=True | 2000 | 0.209 us +/- 0.004 us, var 0.000 | 0.907 us +/- 0.007 us, var 0.001 | 0.23x | pass1 `18435809664101201184`/`18435809664101201184`; pass2 `18435809664101201184`/`18435809664101201184` |
| `empty_rows_0x37_by_37x23` | empty output dimension | `left.matmul(other)` | left zeros((0, 37)); right ones((37, 23)); grad disabled by inputs | (0, 23), stride (23, 1), offset 0, requires_grad=False, is_leaf=True | 2000 | 0.263 us +/- 0.006 us, var 0.000 | 0.903 us +/- 0.005 us, var 0.000 | 0.29x | pass1 `18435809664101201184`/`18435809664101201184`; pass2 `18435809664101201184`/`18435809664101201184` |
| `empty_rows_0x37_by_37x23` | empty output dimension | `torch.matmul(left, other)` | left zeros((0, 37)); right ones((37, 23)); grad disabled by inputs | (0, 23), stride (23, 1), offset 0, requires_grad=False, is_leaf=True | 2000 | 0.331 us +/- 0.003 us, var 0.000 | 0.791 us +/- 0.005 us, var 0.000 | 0.42x | pass1 `18435809664101201184`/`18435809664101201184`; pass2 `18435809664101201184`/`18435809664101201184` |
| `offset_97x131_by_131x89` | offset | `left @ other` | left tensor((3, 97, 131))[1], offset 12707; right tensor((2, 131, 89))[1], offset 11659; grad disabled by inputs | (97, 89), stride (89, 1), offset 0, requires_grad=False, is_leaf=True | 5 | 196.900 us +/- 1.314 us, var 16.319 | 42.718 us +/- 0.293 us, var 3.649 | 4.61x | pass1 `9556352583582127136`/`257495262900597536`; pass2 `9556352583582127136`/`257495262900597536` |
| `offset_97x131_by_131x89` | offset | `left.matmul(other)` | left tensor((3, 97, 131))[1], offset 12707; right tensor((2, 131, 89))[1], offset 11659; grad disabled by inputs | (97, 89), stride (89, 1), offset 0, requires_grad=False, is_leaf=True | 5 | 197.305 us +/- 1.572 us, var 12.860 | 42.867 us +/- 0.490 us, var 2.887 | 4.60x | pass1 `9556352583582127136`/`257495262900597536`; pass2 `9556352583582127136`/`257495262900597536` |
| `offset_97x131_by_131x89` | offset | `torch.matmul(left, other)` | left tensor((3, 97, 131))[1], offset 12707; right tensor((2, 131, 89))[1], offset 11659; grad disabled by inputs | (97, 89), stride (89, 1), offset 0, requires_grad=False, is_leaf=True | 5 | 197.040 us +/- 1.002 us, var 11.216 | 42.745 us +/- 0.480 us, var 5.846 | 4.61x | pass1 `9556352583582127136`/`257495262900597536`; pass2 `9556352583582127136`/`257495262900597536` |
| `noncontig_transpose_129x257_by_257x65` | noncontiguous | `left @ other` | left tensor((257, 129)).transpose(0, 1), stride (1, 129); right tensor((65, 257)).transpose(0, 1), stride (1, 257); grad disabled by inputs | (129, 65), stride (65, 1), offset 0, requires_grad=False, is_leaf=True | 3 | 1690.133 us +/- 8.840 us, var 2940.522 | 76.245 us +/- 0.703 us, var 13.973 | 22.17x | pass1 `4544370897586302528`/`16909366630615619712`; pass2 `4544370897586302528`/`16909366630615619712` |
| `noncontig_transpose_129x257_by_257x65` | noncontiguous | `left.matmul(other)` | left tensor((257, 129)).transpose(0, 1), stride (1, 129); right tensor((65, 257)).transpose(0, 1), stride (1, 257); grad disabled by inputs | (129, 65), stride (65, 1), offset 0, requires_grad=False, is_leaf=True | 3 | 1677.472 us +/- 17.092 us, var 798.433 | 75.903 us +/- 0.455 us, var 4.839 | 22.10x | pass1 `4544370897586302528`/`16909366630615619712`; pass2 `4544370897586302528`/`16909366630615619712` |
| `noncontig_transpose_129x257_by_257x65` | noncontiguous | `torch.matmul(left, other)` | left tensor((257, 129)).transpose(0, 1), stride (1, 129); right tensor((65, 257)).transpose(0, 1), stride (1, 257); grad disabled by inputs | (129, 65), stride (65, 1), offset 0, requires_grad=False, is_leaf=True | 3 | 1693.082 us +/- 12.345 us, var 1264.246 | 75.788 us +/- 0.452 us, var 4.862 | 22.34x | pass1 `4544370897586302528`/`16909366630615619712`; pass2 `4544370897586302528`/`16909366630615619712` |
| `no_grad_requires_grad_64x65_by_65x61` | no_grad | `left @ other` | left/right leaves requiring grad, shapes (64, 65) and (65, 61); operation inside no_grad | (64, 61), stride (61, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 49.798 us +/- 0.919 us, var 3.556 | 13.434 us +/- 0.365 us, var 0.573 | 3.71x | pass1 `8018036478015674944`/`2351523269521802784`; pass2 `8018036478015674944`/`2351523269521802784` |
| `no_grad_requires_grad_64x65_by_65x61` | no_grad | `left.matmul(other)` | left/right leaves requiring grad, shapes (64, 65) and (65, 61); operation inside no_grad | (64, 61), stride (61, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 49.770 us +/- 0.664 us, var 1.368 | 12.910 us +/- 0.075 us, var 0.457 | 3.85x | pass1 `8018036478015674944`/`2351523269521802784`; pass2 `8018036478015674944`/`2351523269521802784` |
| `no_grad_requires_grad_64x65_by_65x61` | no_grad | `torch.matmul(left, other)` | left/right leaves requiring grad, shapes (64, 65) and (65, 61); operation inside no_grad | (64, 61), stride (61, 1), offset 0, requires_grad=False, is_leaf=True | 10 | 49.501 us +/- 0.711 us, var 2.507 | 12.813 us +/- 0.088 us, var 1.706 | 3.86x | pass1 `8018036478015674944`/`2351523269521802784`; pass2 `8018036478015674944`/`2351523269521802784` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | API | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `operator_backward_full_sum_16x17_by_17x19` | `left @ other` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported output `(16, 19)`, loss `()`, left grad `(16, 17)`, right grad `(17, 19)` | zero |
| `tensor_matmul_backward_full_sum_16x17_by_17x19` | `left.matmul(other)` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported output `(16, 19)`, loss `()`, left grad `(16, 17)`, right grad `(17, 19)` | zero |
| `top_level_torch_matmul_backward_full_sum_16x17_by_17x19` | `torch.matmul(left, other)` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported output `(16, 19)`, loss `()`, left grad `(16, 17)`, right grad `(17, 19)` | zero |
| `top_level_torch_matmul_out` | `torch.matmul(left, other, out=out)` | `TypeError: matmul() got an unexpected keyword argument 'out'` | supported output `(16, 19)`, aliases_out=True | zero |
| `operator_vector_vector_rank1` | `left @ other` | `RuntimeError: matmul currently requires two rank-2 tensors, got [3] and [3]` | supported output `()` | zero |
| `top_level_torch_matmul_batched_rank3` | `torch.matmul(left, other)` | `RuntimeError: matmul currently requires two rank-2 tensors, got [2, 3, 4] and [2, 4, 5]` | supported output `(2, 3, 5)` | zero |
