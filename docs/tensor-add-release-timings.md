# `+` and `Tensor.add` Release Timings

Date: 2026-08-31

Candidate provenance: source snapshot based on
`d8f3bd62ca6766ccf4b6b702cc007ba7e0ad76c3`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-add-release-timings*.json`.

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
wheel_dir="$(mktemp -d "$PWD/target/tensor-add-wheels.XXXXXX")"
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --out "$wheel_dir"
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_tensor_add tests.test_tensor_add_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_add_release_timings.py \
  > target/tensor-add-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_add_release_timings.py \
  > target/tensor-add-release-timings-pass2.json
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
  cargo test add --all-targets
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test binary_arithmetic --all-targets
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test scalar_arithmetic --all-targets
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_tensor_add tests.test_tensor_add_reference
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 11 tests. The focused Rust add filter passed 1 test, the Rust binary
arithmetic filter passed 4 tests, the Rust scalar arithmetic filter passed 1
test, the README/docs smoke test passed 7 tests, and `cargo fmt --check`
passed.

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
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 28 ms, prepared
  packages in 16.20s, and installed in 1.39s
- Build time: first successful release extension build completed in 33.02s;
  the final wheel build reused existing release artifacts and completed in
  0.01s

Inputs were created outside the timed region with NumPy seed `20260831`.
Each implementation used the same CPU `float32` values, shapes, layouts, and
thread settings. Every timing cell ran in two pinned process passes. Each pass
used 15 untimed warmup blocks and 81 measured blocks. A block repeated the
operation according to the table's `Repeats` column; times below are median
microseconds per operation. Reported medians are medians of the two per-process
medians. MAD and variance are the medians of the per-process MAD and sample
variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch, and checked shape, stride, storage offset, contiguity,
dtype, device, and `requires_grad`. For autograd cells it also checked
backward leaf gradients. After every warmup and measured block, it consumed the
last output as a `uint32` checksum; forward+backward cells consumed both leaf
gradients. The checksum column shows the final rolling sink from one pass as
`torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.98x uncapped, 0.98x capped
- `+` operator cells: 1.00x uncapped, 1.00x capped
- `Tensor.add` cells: 0.95x uncapped, 0.95x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 1.49x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `same_contiguous_257x263` | tensor/tensor contiguous | `left + other` | left/right `(257, 263)`, stride `(263, 1)` | `(257, 263), stride (263, 1)` | 32 | 9.269 us +/- 0.135, var 0.226 | 11.120 us +/- 0.115, var 0.354 | 0.83x | `6100739642681038400`/`6100739642681038400` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `left.add(other)` | left/right `(257, 263)`, stride `(263, 1)` | `(257, 263), stride (263, 1)` | 32 | 9.339 us +/- 0.179, var 0.238 | 10.915 us +/- 0.108, var 0.409 | 0.86x | `6100739642681038400`/`6100739642681038400` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `left + other` | left `(640, 768)`, stride `(768, 1)`; scalar `-2.25` | `(640, 768), stride (768, 1)` | 10 | 144.252 us +/- 13.125, var 554.954 | 100.874 us +/- 2.817, var 297.735 | 1.43x | `3860113463754325184`/`3860113463754325184` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `left.add(other)` | left `(640, 768)`, stride `(768, 1)`; scalar `-2.25` | `(640, 768), stride (768, 1)` | 10 | 135.926 us +/- 7.959, var 1074.335 | 98.123 us +/- 1.604, var 262.640 | 1.39x | `3860113463754325184`/`3860113463754325184` |
| `vector_broadcast_640x768_by_768` | broadcasting | `left + other` | left `(640, 768)`, stride `(768, 1)`; right `(768,)`, stride `(1,)` | `(640, 768), stride (768, 1)` | 16 | 79.467 us +/- 2.949, var 133.624 | 48.429 us +/- 1.016, var 12.735 | 1.64x | `3929297721838628096`/`3929297721838628096` |
| `vector_broadcast_640x768_by_768` | broadcasting | `left.add(other)` | left `(640, 768)`, stride `(768, 1)`; right `(768,)`, stride `(1,)` | `(640, 768), stride (768, 1)` | 16 | 54.289 us +/- 1.556, var 43.277 | 47.022 us +/- 0.365, var 8.484 | 1.15x | `3929297721838628096`/`3929297721838628096` |
| `empty_strided_broadcast_3x0x2` | empty | `left + other` | left `zeros((2, 0, 3)).transpose(0, 2)` -> `(3, 0, 2)`; right `(1, 1, 2)` | `(3, 0, 2), stride (1, 3, 0)` | 2000 | 0.327 us +/- 0.005, var 0.000 | 1.327 us +/- 0.013, var 0.000 | 0.25x | `731871699860958400`/`731871699860958400` |
| `empty_strided_broadcast_3x0x2` | empty | `left.add(other)` | left `zeros((2, 0, 3)).transpose(0, 2)` -> `(3, 0, 2)`; right `(1, 1, 2)` | `(3, 0, 2), stride (1, 3, 0)` | 2000 | 0.389 us +/- 0.003, var 0.000 | 1.301 us +/- 0.010, var 0.000 | 0.30x | `731871699860958400`/`731871699860958400` |
| `noncontig_transpose_512x1024` | noncontiguous | `left + other` | left/right `tensor((1024, 512)).transpose(0, 1)` -> `(512, 1024)`, stride `(1, 512)` | `(512, 1024), stride (1024, 1)` | 5 | 366.639 us +/- 12.397, var 2601.424 | 267.319 us +/- 25.962, var 4712.468 | 1.37x | `13134066074063467840`/`13134066074063467840` |
| `noncontig_transpose_512x1024` | noncontiguous | `left.add(other)` | left/right `tensor((1024, 512)).transpose(0, 1)` -> `(512, 1024)`, stride `(1, 512)` | `(512, 1024), stride (1024, 1)` | 5 | 288.637 us +/- 6.796, var 137.153 | 251.083 us +/- 9.253, var 2383.906 | 1.15x | `13134066074063467840`/`13134066074063467840` |
| `offset_transposed_521x509` | offset | `left + other` | left/right `tensor((3, 509, 521))[1].transpose(0, 1)` -> `(521, 509)`, stride `(1, 521)`, input offset `265189` | `(521, 509), stride (509, 1)` | 5 | 135.645 us +/- 3.842, var 61.842 | 38.258 us +/- 0.704, var 19.988 | 3.55x | `16461805920662915840`/`16461805920662915840` |
| `offset_transposed_521x509` | offset | `left.add(other)` | left/right `tensor((3, 509, 521))[1].transpose(0, 1)` -> `(521, 509)`, stride `(1, 521)`, input offset `265189` | `(521, 509), stride (509, 1)` | 5 | 140.315 us +/- 5.763, var 514.089 | 41.122 us +/- 1.143, var 50.183 | 3.41x | `16461805920662915840`/`16461805920662915840` |
| `autograd_forward_127x131` | autograd forward | `left + other` | left/right leaves `(127, 131)`, `requires_grad=True`; forward construction only | `(127, 131), stride (131, 1), requires_grad=True` | 20 | 2.960 us +/- 0.098, var 0.384 | 3.392 us +/- 0.022, var 0.299 | 0.87x | `1089297279446765568`/`1089297279446765568` |
| `autograd_forward_127x131` | autograd forward | `left.add(other)` | left/right leaves `(127, 131)`, `requires_grad=True`; forward construction only | `(127, 131), stride (131, 1), requires_grad=True` | 20 | 3.037 us +/- 0.062, var 0.196 | 3.877 us +/- 0.033, var 0.276 | 0.78x | `1089297279446765568`/`1089297279446765568` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `left + other` | left/right leaves `(32, 33)`, `requires_grad=True`; timed `op(...).sum().backward()` | scalar loss plus leaf gradients | 5 | 14.893 us +/- 0.089, var 1.819 | 25.216 us +/- 0.491, var 48.324 | 0.59x | `8437791370385291648`/`8437791370385291648` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `left.add(other)` | left/right leaves `(32, 33)`, `requires_grad=True`; timed `op(...).sum().backward()` | scalar loss plus leaf gradients | 5 | 14.975 us +/- 0.081, var 1.554 | 25.176 us +/- 0.213, var 44.627 | 0.59x | `8437791370385291648`/`8437791370385291648` |
| `no_grad_requires_grad_257x263` | no_grad | `left + other` | left/right leaves `(257, 263)`, `requires_grad=True`; operation inside `no_grad` | `(257, 263), stride (263, 1), requires_grad=False` | 32 | 9.135 us +/- 0.224, var 0.145 | 10.621 us +/- 0.050, var 0.164 | 0.86x | `9694657342471783296`/`9694657342471783296` |
| `no_grad_requires_grad_257x263` | no_grad | `left.add(other)` | left/right leaves `(257, 263)`, `requires_grad=True`; operation inside `no_grad` | `(257, 263), stride (263, 1), requires_grad=False` | 32 | 9.321 us +/- 0.153, var 0.529 | 10.772 us +/- 0.143, var 0.384 | 0.87x | `9694657342471783296`/`9694657342471783296` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `top_level_torch_add_tensor_tensor` | `AttributeError: module 'torch_rs' has no attribute 'add'` | supported tensor/tensor result | zero |
| `top_level_torch_add_tensor_scalar` | `AttributeError: module 'torch_rs' has no attribute 'add'` | supported tensor/scalar result | zero |
| `tensor_add_nondefault_alpha_2` | `NotImplementedError: add(): alpha values other than 1 are not supported` | supported scaled-add result | zero |
| `tensor_add_in_place_add_` | `AttributeError: 'torch_rs.Tensor' object has no attribute 'add_'` | supported in-place result | zero |
