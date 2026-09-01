# `/`, `Tensor.div`, and `Tensor.divide` Release Timings

Date: 2026-08-31

Candidate provenance: source snapshot based on
`762cccf8dd4ff82a207e4f839c28af0df1bcbe52`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-div-release-timings*.json`.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo vendor --locked target/vendor > target/cargo-vendor-config.toml
mkdir -p target/cargo-home
printf '[source.crates-io]\nreplace-with = "vendored-sources"\n\n[source.vendored-sources]\ndirectory = "%s/target/vendor"\n' \
  "$PWD" > target/cargo-home/config.toml
wheel_dir="$(mktemp -d "$PWD/target/tensor-div-wheels.XXXXXX")"
printf '%s\n' "$wheel_dir" > target/tensor-div-wheel-dir.txt
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  /usr/bin/time -f 'build elapsed %e s' \
  .venv/bin/maturin build --release --locked --out "$wheel_dir"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_tensor_div tests.test_tensor_div_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_div_release_timings.py \
  > target/tensor-div-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_div_release_timings.py \
  > target/tensor-div-release-timings-pass2.json
.venv/bin/python target/tensor_div_release_timings.py --combine \
  target/tensor-div-release-timings.json \
  target/tensor-div-release-timings-pass2.json \
  > target/tensor-div-release-timings-summary.json
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
  cargo test div --all-targets
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_tensor_div tests.test_tensor_div_reference
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 9 tests. The focused Rust division filter passed 7 tests across unit
and integration targets, the README/docs smoke test passed 7 tests, and `cargo
fmt --check` plus `git diff --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing runs
  and PyTorch `torch.cuda.is_available()` reported `False`
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 32 ms, prepared
  packages in 16.08s, and installed in 4.44s
- Build time: first successful release extension wheel build completed in
  33.75s

Inputs were created outside the timed region with NumPy seed `20260831`.
Each implementation used the same CPU `float32` values, shapes, layouts, and
thread settings. Every timing cell ran in two pinned process passes. Each pass
used 15 untimed warmup blocks and 81 measured blocks. A block repeated the
operation according to the table's `Repeats` column; times below are median
microseconds per operation. Reported medians are medians of the two
per-process medians. MAD and variance are the medians of the per-process MAD
and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch, and checked shape, stride, storage offset, contiguity,
dtype, device, and `requires_grad`. After every warmup and measured block, it
materialized the last output as NumPy bytes and consumed a 64-bit checksum as a
dead-code and deferred-work guard. The checksum column shows the final rolling
sink from one pass as `torch_rs`/PyTorch; both process passes produced matching
sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 1.65x uncapped, 1.65x capped
- `/` operator cells, including reflected scalar: 1.64x uncapped, 1.64x capped
- `Tensor.div` cells: 1.65x uncapped, 1.65x capped
- `Tensor.divide` cells: 1.66x uncapped, 1.66x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 2.54x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `same_contiguous_257x263_slash` | tensor/tensor contiguous | `left / other` | left/right (257, 263), stride (263, 1) | `(257, 263), stride (263, 1)` | 32 | 14.991 us +/- 0.264, var 0.435 | 10.477 us +/- 0.143, var 0.265 | 1.43x | `12503820734280513539`/`12503820734280513539` |
| `same_contiguous_257x263_div` | tensor/tensor contiguous | `left.div(other)` | left/right (257, 263), stride (263, 1) | `(257, 263), stride (263, 1)` | 32 | 14.964 us +/- 0.347, var 0.171 | 11.172 us +/- 0.120, var 0.156 | 1.34x | `12503820734280513539`/`12503820734280513539` |
| `same_contiguous_257x263_divide` | tensor/tensor contiguous | `left.divide(other)` | left/right (257, 263), stride (263, 1) | `(257, 263), stride (263, 1)` | 32 | 15.079 us +/- 0.285, var 0.321 | 10.317 us +/- 0.133, var 0.801 | 1.46x | `12503820734280513539`/`12503820734280513539` |
| `tensor_scalar_640x768_slash` | tensor/scalar contiguous | `left / other` | left (640, 768), stride (768, 1); scalar -2.25 | `(640, 768), stride (768, 1)` | 10 | 105.343 us +/- 1.087, var 20.094 | 58.294 us +/- 0.956, var 14.276 | 1.81x | `16842452311130121731`/`16842452311130121731` |
| `tensor_scalar_640x768_div` | tensor/scalar contiguous | `left.div(other)` | left (640, 768), stride (768, 1); scalar -2.25 | `(640, 768), stride (768, 1)` | 10 | 112.707 us +/- 3.741, var 32.451 | 59.486 us +/- 1.003, var 4.292 | 1.89x | `16842452311130121731`/`16842452311130121731` |
| `tensor_scalar_640x768_divide` | tensor/scalar contiguous | `left.divide(other)` | left (640, 768), stride (768, 1); scalar -2.25 | `(640, 768), stride (768, 1)` | 10 | 109.097 us +/- 1.722, var 40.108 | 57.982 us +/- 0.832, var 8.011 | 1.88x | `16842452311130121731`/`16842452311130121731` |
| `reflected_scalar_640x768_slash` | reflected scalar | `other / left` | scalar 3.5; left (640, 768), stride (768, 1) | `(640, 768), stride (768, 1)` | 10 | 110.927 us +/- 2.482, var 32.428 | 108.415 us +/- 3.023, var 137.212 | 1.02x | `12404491596035863427`/`12404491596035863427` |
| `vector_broadcast_640x768_by_768_slash` | broadcasting | `left / other` | left (640, 768), stride (768, 1); right (768,), stride (1,) | `(640, 768), stride (768, 1)` | 16 | 107.284 us +/- 1.085, var 9.506 | 56.700 us +/- 0.728, var 30.052 | 1.89x | `6666703060830335683`/`6666703060830335683` |
| `vector_broadcast_640x768_by_768_div` | broadcasting | `left.div(other)` | left (640, 768), stride (768, 1); right (768,), stride (1,) | `(640, 768), stride (768, 1)` | 16 | 107.166 us +/- 1.587, var 13.415 | 57.095 us +/- 0.951, var 5.330 | 1.88x | `6666703060830335683`/`6666703060830335683` |
| `vector_broadcast_640x768_by_768_divide` | broadcasting | `left.divide(other)` | left (640, 768), stride (768, 1); right (768,), stride (1,) | `(640, 768), stride (768, 1)` | 16 | 107.551 us +/- 0.949, var 14.994 | 61.895 us +/- 1.926, var 27.084 | 1.74x | `6666703060830335683`/`6666703060830335683` |
| `empty_strided_broadcast_3x0x2_slash` | empty | `left / other` | left zeros((2, 0, 3)).transpose(0, 2); right (1, 1, 2) | `(3, 0, 2), stride (1, 3, 0)` | 2000 | 0.258 us +/- 0.011, var 0.001 | 1.259 us +/- 0.050, var 0.044 | 0.21x | `6647038707075381891`/`6647038707075381891` |
| `empty_strided_broadcast_3x0x2_div` | empty | `left.div(other)` | left zeros((2, 0, 3)).transpose(0, 2); right (1, 1, 2) | `(3, 0, 2), stride (1, 3, 0)` | 2000 | 0.335 us +/- 0.005, var 0.000 | 1.217 us +/- 0.012, var 0.003 | 0.28x | `6647038707075381891`/`6647038707075381891` |
| `empty_strided_broadcast_3x0x2_divide` | empty | `left.divide(other)` | left zeros((2, 0, 3)).transpose(0, 2); right (1, 1, 2) | `(3, 0, 2), stride (1, 3, 0)` | 2000 | 0.332 us +/- 0.005, var 0.000 | 1.269 us +/- 0.023, var 0.018 | 0.26x | `6647038707075381891`/`6647038707075381891` |
| `noncontig_transpose_512x1024_slash` | noncontiguous | `left / other` | left/right tensor((1024, 512)).transpose(0, 1) | `(512, 1024), stride (1, 512)` | 5 | 527.295 us +/- 27.760, var 4774.166 | 95.137 us +/- 10.920, var 285.439 | 5.54x | `7666416238271708291`/`7666416238271708291` |
| `noncontig_transpose_512x1024_div` | noncontiguous | `left.div(other)` | left/right tensor((1024, 512)).transpose(0, 1) | `(512, 1024), stride (1, 512)` | 5 | 324.128 us +/- 9.045, var 622.643 | 89.669 us +/- 6.089, var 284.119 | 3.61x | `7666416238271708291`/`7666416238271708291` |
| `noncontig_transpose_512x1024_divide` | noncontiguous | `left.divide(other)` | left/right tensor((1024, 512)).transpose(0, 1) | `(512, 1024), stride (1, 512)` | 5 | 330.261 us +/- 19.209, var 3759.544 | 83.457 us +/- 5.237, var 112.926 | 3.96x | `7666416238271708291`/`7666416238271708291` |
| `offset_transposed_521x509_slash` | offset | `left / other` | left/right tensor((3, 509, 521))[1].transpose(0, 1), input offset 265189 | `(521, 509), stride (1, 521)` | 5 | 203.180 us +/- 6.518, var 592.002 | 36.677 us +/- 0.779, var 18.669 | 5.54x | `16253689815069309379`/`16253689815069309379` |
| `offset_transposed_521x509_div` | offset | `left.div(other)` | left/right tensor((3, 509, 521))[1].transpose(0, 1), input offset 265189 | `(521, 509), stride (1, 521)` | 5 | 162.614 us +/- 6.344, var 181.571 | 38.036 us +/- 0.918, var 5.221 | 4.28x | `16253689815069309379`/`16253689815069309379` |
| `offset_transposed_521x509_divide` | offset | `left.divide(other)` | left/right tensor((3, 509, 521))[1].transpose(0, 1), input offset 265189 | `(521, 509), stride (1, 521)` | 5 | 162.513 us +/- 6.583, var 368.748 | 38.634 us +/- 1.567, var 19.339 | 4.21x | `16253689815069309379`/`16253689815069309379` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `top_level_torch_div_tensor_tensor` | `AttributeError: module 'torch_rs' has no attribute 'div'` | supported tensor/tensor true-division result | zero |
| `top_level_torch_divide_tensor_tensor` | `AttributeError: module 'torch_rs' has no attribute 'divide'` | supported tensor/tensor true-division result | zero |
| `tensor_div_rounding_mode_floor` | `NotImplementedError: div(): non-None rounding_mode is not supported` | supported rounded division result | zero |
| `tensor_divide_rounding_mode_trunc` | `NotImplementedError: divide(): non-None rounding_mode is not supported` | supported rounded division result | zero |
| `tensor_div_in_place_div_` | `AttributeError: 'torch_rs.Tensor' object has no attribute 'div_'` | supported in-place result | zero |
| `active_autograd_tensor_div` | `RuntimeError: div(): autograd recording is not supported` | supported differentiable result | zero |
