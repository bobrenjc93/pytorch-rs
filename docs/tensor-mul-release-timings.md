# `*`, `Tensor.mul`/`Tensor.multiply`, and `torch.mul`/`torch.multiply` Release Timings

Date: 2026-08-31

Candidate provenance: source snapshot based on
`01469d79fb5d9eea1ca438b9f7a49efd6c6281ef`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/tensor-mul-release-timings*.json`. No Conda
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
mkdir -p target/cargo-home
cp -a /home/bobren/.cargo/registry target/cargo-home/
wheel_dir="$(mktemp -d "$PWD/target/tensor-mul-wheels.XXXXXX")"
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
  tests.test_mul tests.test_mul_reference \
  tests.test_multiply tests.test_multiply_reference \
  tests.test_top_level_mul tests.test_top_level_mul_reference \
  tests.test_top_level_multiply tests.test_top_level_multiply_reference
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_mul_release_timings.py \
  > target/tensor-mul-release-timings.json
env PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/tensor_mul_release_timings.py \
  > target/tensor-mul-release-timings-pass2.json
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
  cargo test --locked --offline --all-targets multiply
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
  tests.test_mul tests.test_mul_reference \
  tests.test_multiply tests.test_multiply_reference \
  tests.test_top_level_mul tests.test_top_level_mul_reference \
  tests.test_top_level_multiply tests.test_top_level_multiply_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_readme_quickstart
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 38 tests with 1 skip. The focused Rust `multiply` filter passed 6 tests,
the Rust binary arithmetic filter passed 4 tests, the Rust scalar arithmetic
filter passed 1 test, `cargo fmt --check` passed, the README/docs smoke test
passed, and `git diff --check` passed.

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
  packages in 16.53s, and installed in 915 ms
- Build time: successful offline release extension build completed in 34.84s;
  the release wheel reinstall resolved in 2 ms, prepared in 39 ms, and
  installed in 19 ms

Inputs were created outside the timed region with NumPy seed `20260831`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
Each pass used 15 untimed warmup blocks and 81 measured blocks. A block
repeated the operation according to the table's `Repeats` column; times below
are median microseconds per operation. Reported medians are medians of the two
per-process medians. MAD and variance are the medians of the per-process MAD
and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch, and checked shape, stride, storage offset, contiguity,
dtype, device, and `requires_grad`. For autograd forward+backward cells it
also checked both leaf gradients after `op(...).sum().backward()`. The backward
timings used pre-created fresh leaf tensors for every measured invocation so
the timed region did not include input construction and did not reuse a freed
graph. After every warmup and measured block, the driver materialized the last
output as a byte-level checksum; forward+backward cells consumed both leaf
gradients. The checksum column shows the final rolling sink from one pass as
`torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 1.05x uncapped, 1.05x capped
- `*` operator cells: 1.02x uncapped, 1.02x capped
- `Tensor.mul` cells: 1.04x uncapped, 1.04x capped
- `Tensor.multiply` cells: 1.03x uncapped, 1.03x capped
- `torch.mul` cells: 1.08x uncapped, 1.08x capped
- `torch.multiply` cells: 1.09x uncapped, 1.09x capped
- Tensor/tensor contiguous cells: 0.87x uncapped, 0.87x capped
- Tensor/scalar contiguous cells: 1.09x uncapped, 1.09x capped
- Reflected scalar contiguous cells: 1.09x uncapped, 1.09x capped
- Broadcasting cells: 1.12x uncapped, 1.12x capped
- Empty cells: 0.29x uncapped, 0.29x capped
- Noncontiguous transpose cells: 3.60x uncapped, 3.60x capped
- Offset transpose cells: 3.75x uncapped, 3.75x capped
- Autograd forward cells: 0.68x uncapped, 0.68x capped
- Autograd forward+backward cells: 0.70x uncapped, 0.70x capped
- `no_grad` cells: 0.80x uncapped, 0.80x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 1.35x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `same_contiguous_257x263` | tensor/tensor contiguous | `left * other` | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), requires_grad=False | 32 | 9.394 us +/- 0.095, var 0.162 | 10.840 us +/- 0.134, var 0.188 | 0.87x | `11952351347695027075`/`11952351347695027075` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `left.mul(other)` | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), requires_grad=False | 32 | 9.352 us +/- 0.163, var 0.182 | 10.938 us +/- 0.122, var 0.106 | 0.85x | `11952351347695027075`/`11952351347695027075` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `left.multiply(other)` | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), requires_grad=False | 32 | 9.262 us +/- 0.264, var 1.001 | 10.703 us +/- 0.073, var 0.170 | 0.87x | `11952351347695027075`/`11952351347695027075` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.mul(left, other)` | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), requires_grad=False | 32 | 9.358 us +/- 0.137, var 0.339 | 10.923 us +/- 0.282, var 0.270 | 0.86x | `11952351347695027075`/`11952351347695027075` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.multiply(left, other)` | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), requires_grad=False | 32 | 9.520 us +/- 0.166, var 0.725 | 10.779 us +/- 0.178, var 0.099 | 0.88x | `11952351347695027075`/`11952351347695027075` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `left * other` | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), requires_grad=False | 10 | 53.147 us +/- 1.241, var 60.251 | 49.080 us +/- 1.542, var 18.319 | 1.08x | `13505710309104674147`/`13505710309104674147` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `left.mul(other)` | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), requires_grad=False | 10 | 56.755 us +/- 1.967, var 58.001 | 51.291 us +/- 2.172, var 51.436 | 1.11x | `13505710309104674147`/`13505710309104674147` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `left.multiply(other)` | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), requires_grad=False | 10 | 53.245 us +/- 1.127, var 15.266 | 52.314 us +/- 1.701, var 10.421 | 1.02x | `13505710309104674147`/`13505710309104674147` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.mul(left, other)` | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), requires_grad=False | 10 | 56.883 us +/- 1.577, var 35.628 | 51.363 us +/- 1.245, var 62.974 | 1.11x | `13505710309104674147`/`13505710309104674147` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.multiply(left, other)` | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), requires_grad=False | 10 | 55.058 us +/- 1.247, var 12.970 | 48.223 us +/- 0.925, var 8.287 | 1.14x | `13505710309104674147`/`13505710309104674147` |
| `reflected_scalar_640x768` | reflected scalar contiguous | `left * other` | scalar -2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), requires_grad=False | 10 | 53.512 us +/- 1.299, var 65.084 | 52.853 us +/- 1.366, var 8.965 | 1.01x | `716965539951150211`/`716965539951150211` |
| `reflected_scalar_640x768` | reflected scalar contiguous | `torch.mul(left, other)` | scalar -2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), requires_grad=False | 10 | 56.786 us +/- 1.867, var 65.029 | 49.909 us +/- 1.027, var 7.491 | 1.14x | `716965539951150211`/`716965539951150211` |
| `reflected_scalar_640x768` | reflected scalar contiguous | `torch.multiply(left, other)` | scalar -2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), requires_grad=False | 10 | 56.314 us +/- 1.626, var 54.995 | 49.533 us +/- 0.995, var 32.499 | 1.14x | `716965539951150211`/`716965539951150211` |
| `vector_broadcast_640x768_by_768` | broadcasting | `left * other` | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), requires_grad=False | 16 | 54.029 us +/- 1.267, var 41.613 | 48.022 us +/- 0.762, var 15.740 | 1.13x | `6619359080709093795`/`6619359080709093795` |
| `vector_broadcast_640x768_by_768` | broadcasting | `left.mul(other)` | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), requires_grad=False | 16 | 52.614 us +/- 1.137, var 33.614 | 47.345 us +/- 0.688, var 3.009 | 1.11x | `6619359080709093795`/`6619359080709093795` |
| `vector_broadcast_640x768_by_768` | broadcasting | `left.multiply(other)` | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), requires_grad=False | 16 | 52.401 us +/- 1.021, var 25.866 | 46.765 us +/- 0.767, var 13.082 | 1.12x | `6619359080709093795`/`6619359080709093795` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.mul(left, other)` | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), requires_grad=False | 16 | 52.829 us +/- 1.167, var 35.366 | 48.268 us +/- 1.113, var 47.122 | 1.09x | `6619359080709093795`/`6619359080709093795` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.multiply(left, other)` | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), requires_grad=False | 16 | 54.376 us +/- 0.815, var 2.078 | 47.794 us +/- 1.300, var 21.829 | 1.14x | `6619359080709093795`/`6619359080709093795` |
| `empty_strided_broadcast_3x0x2` | empty | `left * other` | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), requires_grad=False | 2000 | 0.292 us +/- 0.004, var 0.000 | 1.268 us +/- 0.008, var 0.001 | 0.23x | `14688941508328670819`/`14688941508328670819` |
| `empty_strided_broadcast_3x0x2` | empty | `left.mul(other)` | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), requires_grad=False | 2000 | 0.360 us +/- 0.012, var 0.000 | 1.283 us +/- 0.006, var 0.002 | 0.28x | `14688941508328670819`/`14688941508328670819` |
| `empty_strided_broadcast_3x0x2` | empty | `left.multiply(other)` | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), requires_grad=False | 2000 | 0.358 us +/- 0.002, var 0.000 | 1.330 us +/- 0.007, var 0.005 | 0.27x | `14688941508328670819`/`14688941508328670819` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.mul(left, other)` | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), requires_grad=False | 2000 | 0.442 us +/- 0.003, var 0.000 | 1.299 us +/- 0.008, var 0.002 | 0.34x | `14688941508328670819`/`14688941508328670819` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.multiply(left, other)` | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), requires_grad=False | 2000 | 0.452 us +/- 0.002, var 0.000 | 1.362 us +/- 0.007, var 0.002 | 0.33x | `14688941508328670819`/`14688941508328670819` |
| `noncontig_transpose_512x1024` | noncontiguous | `left * other` | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), requires_grad=False | 5 | 279.129 us +/- 5.534, var 169.895 | 87.974 us +/- 9.651, var 359.782 | 3.17x | `14757972718812675939`/`14757972718812675939` |
| `noncontig_transpose_512x1024` | noncontiguous | `left.mul(other)` | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), requires_grad=False | 5 | 289.412 us +/- 5.394, var 352.919 | 77.071 us +/- 3.229, var 97.588 | 3.76x | `14757972718812675939`/`14757972718812675939` |
| `noncontig_transpose_512x1024` | noncontiguous | `left.multiply(other)` | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), requires_grad=False | 5 | 288.909 us +/- 6.511, var 274.047 | 77.757 us +/- 2.840, var 123.199 | 3.72x | `14757972718812675939`/`14757972718812675939` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.mul(left, other)` | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), requires_grad=False | 5 | 291.823 us +/- 7.244, var 348.643 | 79.363 us +/- 4.345, var 664.472 | 3.68x | `14757972718812675939`/`14757972718812675939` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.multiply(left, other)` | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), requires_grad=False | 5 | 292.638 us +/- 13.158, var 1744.752 | 78.507 us +/- 4.159, var 303.137 | 3.73x | `14757972718812675939`/`14757972718812675939` |
| `offset_transposed_521x509` | offset | `left * other` | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), requires_grad=False | 5 | 153.615 us +/- 14.055, var 510.944 | 36.484 us +/- 0.541, var 2.119 | 4.21x | `3988052893130270819`/`3988052893130270819` |
| `offset_transposed_521x509` | offset | `left.mul(other)` | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), requires_grad=False | 5 | 135.738 us +/- 3.895, var 120.025 | 38.207 us +/- 0.903, var 4.758 | 3.55x | `3988052893130270819`/`3988052893130270819` |
| `offset_transposed_521x509` | offset | `left.multiply(other)` | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), requires_grad=False | 5 | 137.442 us +/- 4.364, var 86.149 | 36.997 us +/- 0.622, var 2.230 | 3.71x | `3988052893130270819`/`3988052893130270819` |
| `offset_transposed_521x509` | offset | `torch.mul(left, other)` | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), requires_grad=False | 5 | 138.139 us +/- 3.308, var 44.471 | 37.923 us +/- 0.754, var 4.299 | 3.64x | `3988052893130270819`/`3988052893130270819` |
| `offset_transposed_521x509` | offset | `torch.multiply(left, other)` | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), requires_grad=False | 5 | 137.086 us +/- 4.974, var 196.210 | 37.221 us +/- 0.909, var 4.361 | 3.68x | `3988052893130270819`/`3988052893130270819` |
| `autograd_forward_127x131` | autograd forward | `left * other` | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), requires_grad=True | 20 | 2.384 us +/- 0.018, var 0.040 | 3.541 us +/- 0.014, var 0.097 | 0.67x | `16979819663235646051`/`16979819663235646051` |
| `autograd_forward_127x131` | autograd forward | `left.mul(other)` | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), requires_grad=True | 20 | 2.408 us +/- 0.017, var 0.037 | 3.536 us +/- 0.028, var 0.292 | 0.68x | `16979819663235646051`/`16979819663235646051` |
| `autograd_forward_127x131` | autograd forward | `left.multiply(other)` | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), requires_grad=True | 20 | 2.428 us +/- 0.018, var 0.125 | 3.643 us +/- 0.016, var 0.092 | 0.67x | `16979819663235646051`/`16979819663235646051` |
| `autograd_forward_127x131` | autograd forward | `torch.mul(left, other)` | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), requires_grad=True | 20 | 2.566 us +/- 0.016, var 0.048 | 3.611 us +/- 0.016, var 0.184 | 0.71x | `16979819663235646051`/`16979819663235646051` |
| `autograd_forward_127x131` | autograd forward | `torch.multiply(left, other)` | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), requires_grad=True | 20 | 2.553 us +/- 0.021, var 0.110 | 3.701 us +/- 0.016, var 0.079 | 0.69x | `16979819663235646051`/`16979819663235646051` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `left * other` | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | scalar loss plus leaf gradients | 5 | 21.592 us +/- 0.196, var 1.769 | 31.155 us +/- 0.531, var 11.729 | 0.69x | `14722111757845455203`/`14722111757845455203` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `left.mul(other)` | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | scalar loss plus leaf gradients | 5 | 21.653 us +/- 0.203, var 3.412 | 31.160 us +/- 0.516, var 42.958 | 0.69x | `14722111757845455203`/`14722111757845455203` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `left.multiply(other)` | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | scalar loss plus leaf gradients | 5 | 21.680 us +/- 0.202, var 5.578 | 31.214 us +/- 0.402, var 12.391 | 0.69x | `14722111757845455203`/`14722111757845455203` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.mul(left, other)` | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | scalar loss plus leaf gradients | 5 | 22.087 us +/- 0.306, var 8.292 | 31.429 us +/- 0.549, var 7.497 | 0.70x | `14722111757845455203`/`14722111757845455203` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.multiply(left, other)` | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | scalar loss plus leaf gradients | 5 | 21.838 us +/- 0.222, var 4.507 | 31.228 us +/- 0.631, var 21.552 | 0.70x | `14722111757845455203`/`14722111757845455203` |
| `no_grad_requires_grad_257x263` | no_grad | `left * other` | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), requires_grad=False | 32 | 10.061 us +/- 0.138, var 0.470 | 13.002 us +/- 0.172, var 0.558 | 0.77x | `8361314265077325155`/`8361314265077325155` |
| `no_grad_requires_grad_257x263` | no_grad | `left.mul(other)` | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), requires_grad=False | 32 | 10.215 us +/- 0.165, var 0.240 | 12.848 us +/- 0.249, var 0.520 | 0.80x | `8361314265077325155`/`8361314265077325155` |
| `no_grad_requires_grad_257x263` | no_grad | `left.multiply(other)` | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), requires_grad=False | 32 | 10.210 us +/- 0.190, var 0.476 | 12.818 us +/- 0.214, var 0.114 | 0.80x | `8361314265077325155`/`8361314265077325155` |
| `no_grad_requires_grad_257x263` | no_grad | `torch.mul(left, other)` | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), requires_grad=False | 32 | 10.564 us +/- 0.274, var 0.268 | 12.728 us +/- 0.238, var 0.224 | 0.83x | `8361314265077325155`/`8361314265077325155` |
| `no_grad_requires_grad_257x263` | no_grad | `torch.multiply(left, other)` | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), requires_grad=False | 32 | 10.455 us +/- 0.228, var 0.534 | 12.929 us +/- 0.311, var 0.506 | 0.81x | `8361314265077325155`/`8361314265077325155` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `tensor_mul_in_place_mul_` | `AttributeError: 'torch_rs.Tensor' object has no attribute 'mul_'` | supported `tensor([6.])` | zero |
| `tensor_multiply_in_place_multiply_` | `AttributeError: 'torch_rs.Tensor' object has no attribute 'multiply_'` | supported `tensor([6.])` | zero |
| `top_level_torch_mul_out` | `TypeError: mul() got an unexpected keyword argument 'out'` | supported `tensor([6.])` | zero |
| `top_level_torch_multiply_out` | `TypeError: multiply() got an unexpected keyword argument 'out'` | supported `tensor([6.])` | zero |
| `top_level_torch_mul_scalar_scalar` | `TypeError: mul(): scalar-scalar multiplication is not supported; at least one operand must be Tensor` | supported `6` | zero |
| `top_level_torch_multiply_scalar_scalar` | `TypeError: multiply(): scalar-scalar multiplication is not supported; at least one operand must be Tensor` | supported `6` | zero |
