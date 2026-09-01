# Rank-2 Matmul Release Timings

Date: 2026-09-01

Candidate provenance: source snapshot based on
`fffeb1a3287a290703070f1dd5f767d544f38f01`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/rank2-matmul-release-timings*.json`. No Conda
environment was active in the shell (`CONDA_PREFIX=`, `CONDA_SHLVL=0`), so
setup used a worktree-local `.venv`. Cargo registry data was copied read-only
from the existing user cache into `target/cargo-home/registry`, then Cargo ran
offline so build artifacts and dependency state stayed inside this worktree.

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
library and tensor-baseline test targets, `cargo fmt --check` passed, the
README/docs smoke test passed 7 tests, and `git diff --check` passed.

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
  packages in 16.12s, and installed in 1.02s
- Build time: successful offline release extension build completed in 35.71s;
  the release wheel reinstall resolved in 2 ms, prepared in 44 ms, and
  installed in 15 ms

Inputs were created outside the timed region with NumPy seed `20260901`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, and thread settings. Every timing cell ran in two pinned process passes.
Each pass used 15 untimed warmup blocks and 81 measured blocks. A block
repeated the operation according to the table's `Repeats` column; times below
are median microseconds per operation. Reported medians are medians of the two
per-process medians. MAD and variance are the medians of the per-process MAD
and sample variance values.

Before timing each supported cell, the driver compared `torch_rs` outputs with
PyTorch using `rtol=1e-5`, `atol=1e-5`, and `equal_nan=True`, checked signed
zero bits where zeros were present, and checked shape, stride, storage offset,
contiguity, dtype, device, `requires_grad`, and leaf status. After every
warmup and measured block, the driver consumed the last output as a byte-level
rolling checksum. The checksum column shows the final rolling sink from one
pass as `torch_rs`/PyTorch; both process passes produced the same sink pairs.

No backward-through-full-`sum` rank-2 matmul cells are supported in this
revision. The three API-specific backward rows are therefore recorded as
zero-credit unsupported cells below: PyTorch 2.13 records the graph and
materializes the scalar loss plus both leaf gradients, while `torch_rs` returns
a non-grad matmul leaf and rejects `.sum().backward()`.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 1.88x uncapped, 1.66x capped
- `@` cells: 1.70x uncapped, 1.50x capped
- `Tensor.matmul` cells: 1.87x uncapped, 1.65x capped
- `torch.matmul` cells: 2.11x uncapped, 1.86x capped
- Square contiguous cells: 3.36x uncapped, 3.36x capped
- Rectangular contiguous cells: 3.11x uncapped, 3.11x capped
- Skinny contiguous cells: 2.99x uncapped, 2.99x capped
- Empty rows cells: 0.33x uncapped, 0.33x capped
- Empty columns cells: 0.35x uncapped, 0.35x capped
- Empty inner cells: 0.33x uncapped, 0.33x capped
- Offset contiguous cells: 2.85x uncapped, 2.85x capped
- Noncontiguous transpose cells: 30.97x uncapped, 10.00x capped
- `no_grad` cells: 2.86x uncapped, 2.86x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 2.40x.

| Workload | Category | API | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `square_contiguous_128` | square contiguous | `@` | left (128, 128), stride (128, 1); right (128, 128), stride (128, 1) | (128, 128), stride (128, 1), offset 0, requires_grad=False | 3 | 155.260 us +/- 5.906 us, var 662.051 | 45.117 us +/- 0.556 us, var 20.057 | 3.44x | `2234437898128481312`/`15814464564571506688` |
| `square_contiguous_128` | square contiguous | `Tensor.matmul` | left (128, 128), stride (128, 1); right (128, 128), stride (128, 1) | (128, 128), stride (128, 1), offset 0, requires_grad=False | 3 | 150.081 us +/- 2.769 us, var 30.590 | 45.011 us +/- 0.342 us, var 6.098 | 3.33x | `2234437898128481312`/`15814464564571506688` |
| `square_contiguous_128` | square contiguous | `torch.matmul` | left (128, 128), stride (128, 1); right (128, 128), stride (128, 1) | (128, 128), stride (128, 1), offset 0, requires_grad=False | 3 | 150.154 us +/- 3.293 us, var 83.048 | 45.485 us +/- 0.702 us, var 5.855 | 3.30x | `2234437898128481312`/`15814464564571506688` |
| `rectangular_contiguous_127x257_by_257x63` | rectangular contiguous | `@` | left (127, 257), stride (257, 1); right (257, 63), stride (63, 1) | (127, 63), stride (63, 1), offset 0, requires_grad=False | 3 | 240.579 us +/- 4.951 us, var 91.172 | 76.319 us +/- 1.132 us, var 41.923 | 3.15x | `12796901252065200192`/`18165165521131479104` |
| `rectangular_contiguous_127x257_by_257x63` | rectangular contiguous | `Tensor.matmul` | left (127, 257), stride (257, 1); right (257, 63), stride (63, 1) | (127, 63), stride (63, 1), offset 0, requires_grad=False | 3 | 240.567 us +/- 4.355 us, var 424.734 | 80.761 us +/- 3.103 us, var 38.177 | 2.98x | `12796901252065200192`/`18165165521131479104` |
| `rectangular_contiguous_127x257_by_257x63` | rectangular contiguous | `torch.matmul` | left (127, 257), stride (257, 1); right (257, 63), stride (63, 1) | (127, 63), stride (63, 1), offset 0, requires_grad=False | 3 | 234.912 us +/- 3.379 us, var 224.182 | 73.394 us +/- 0.534 us, var 11.448 | 3.20x | `12796901252065200192`/`18165165521131479104` |
| `skinny_contiguous_1024x8_by_8x16` | skinny contiguous | `@` | left (1024, 8), stride (8, 1); right (8, 16), stride (16, 1) | (1024, 16), stride (16, 1), offset 0, requires_grad=False | 8 | 18.607 us +/- 0.069 us, var 1.247 | 6.336 us +/- 0.069 us, var 0.770 | 2.94x | `13749277909466167872`/`17684335751541538944` |
| `skinny_contiguous_1024x8_by_8x16` | skinny contiguous | `Tensor.matmul` | left (1024, 8), stride (8, 1); right (8, 16), stride (16, 1) | (1024, 16), stride (16, 1), offset 0, requires_grad=False | 8 | 18.854 us +/- 0.136 us, var 0.980 | 6.360 us +/- 0.061 us, var 0.241 | 2.96x | `13749277909466167872`/`17684335751541538944` |
| `skinny_contiguous_1024x8_by_8x16` | skinny contiguous | `torch.matmul` | left (1024, 8), stride (8, 1); right (8, 16), stride (16, 1) | (1024, 16), stride (16, 1), offset 0, requires_grad=False | 8 | 18.894 us +/- 0.100 us, var 1.868 | 6.139 us +/- 0.061 us, var 0.230 | 3.08x | `13749277909466167872`/`17684335751541538944` |
| `empty_rows_0x64_by_64x32` | empty rows | `@` | left (0, 64), stride (64, 1); right (64, 32), stride (32, 1) | (0, 32), stride (32, 1), offset 0, requires_grad=False | 1000 | 0.257 us +/- 0.002 us, var 0.000 | 1.126 us +/- 0.065 us, var 0.013 | 0.23x | `16434192061009871104`/`16434192061009871104` |
| `empty_rows_0x64_by_64x32` | empty rows | `Tensor.matmul` | left (0, 64), stride (64, 1); right (64, 32), stride (32, 1) | (0, 32), stride (32, 1), offset 0, requires_grad=False | 1000 | 0.386 us +/- 0.025 us, var 0.003 | 1.136 us +/- 0.043 us, var 0.008 | 0.34x | `16434192061009871104`/`16434192061009871104` |
| `empty_rows_0x64_by_64x32` | empty rows | `torch.matmul` | left (0, 64), stride (64, 1); right (64, 32), stride (32, 1) | (0, 32), stride (32, 1), offset 0, requires_grad=False | 1000 | 0.487 us +/- 0.027 us, var 0.002 | 1.022 us +/- 0.107 us, var 0.025 | 0.48x | `16434192061009871104`/`16434192061009871104` |
| `empty_columns_32x64_by_64x0` | empty columns | `@` | left (32, 64), stride (64, 1); right (64, 0), stride (1, 1) | (32, 0), stride (1, 1), offset 0, requires_grad=False | 1000 | 0.258 us +/- 0.002 us, var 0.000 | 0.994 us +/- 0.006 us, var 0.000 | 0.26x | `8740957323260166656`/`8740957323260166656` |
| `empty_columns_32x64_by_64x0` | empty columns | `Tensor.matmul` | left (32, 64), stride (64, 1); right (64, 0), stride (1, 1) | (32, 0), stride (1, 1), offset 0, requires_grad=False | 1000 | 0.349 us +/- 0.006 us, var 0.000 | 0.988 us +/- 0.005 us, var 0.000 | 0.35x | `8740957323260166656`/`8740957323260166656` |
| `empty_columns_32x64_by_64x0` | empty columns | `torch.matmul` | left (32, 64), stride (64, 1); right (64, 0), stride (1, 1) | (32, 0), stride (1, 1), offset 0, requires_grad=False | 1000 | 0.408 us +/- 0.007 us, var 0.001 | 0.878 us +/- 0.004 us, var 0.001 | 0.46x | `8740957323260166656`/`8740957323260166656` |
| `empty_inner_32x0_by_0x16` | empty inner | `@` | left (32, 0), stride (1, 1); right (0, 16), stride (16, 1) | (32, 16), stride (16, 1), offset 0, requires_grad=False | 200 | 0.315 us +/- 0.008 us, var 0.001 | 1.231 us +/- 0.009 us, var 0.002 | 0.26x | `88303876093528192`/`88303876093528192` |
| `empty_inner_32x0_by_0x16` | empty inner | `Tensor.matmul` | left (32, 0), stride (1, 1); right (0, 16), stride (16, 1) | (32, 16), stride (16, 1), offset 0, requires_grad=False | 200 | 0.383 us +/- 0.015 us, var 0.003 | 1.241 us +/- 0.041 us, var 0.005 | 0.31x | `88303876093528192`/`88303876093528192` |
| `empty_inner_32x0_by_0x16` | empty inner | `torch.matmul` | left (32, 0), stride (1, 1); right (0, 16), stride (16, 1) | (32, 16), stride (16, 1), offset 0, requires_grad=False | 200 | 0.492 us +/- 0.033 us, var 0.002 | 1.119 us +/- 0.006 us, var 0.002 | 0.44x | `88303876093528192`/`88303876093528192` |
| `offset_contiguous_96x64_by_64x80` | offset contiguous | `@` | left base[1] -> (96, 64), stride (64, 1), offset 6144; right base[1] -> (64, 80), stride (80, 1), offset 5120 | (96, 80), stride (80, 1), offset 0, requires_grad=False | 4 | 35.096 us +/- 0.099 us, var 10.466 | 12.482 us +/- 0.129 us, var 1.789 | 2.81x | `12347065076992943904`/`14727830144762775616` |
| `offset_contiguous_96x64_by_64x80` | offset contiguous | `Tensor.matmul` | left base[1] -> (96, 64), stride (64, 1), offset 6144; right base[1] -> (64, 80), stride (80, 1), offset 5120 | (96, 80), stride (80, 1), offset 0, requires_grad=False | 4 | 35.249 us +/- 0.075 us, var 2.098 | 12.371 us +/- 0.066 us, var 0.879 | 2.85x | `12347065076992943904`/`14727830144762775616` |
| `offset_contiguous_96x64_by_64x80` | offset contiguous | `torch.matmul` | left base[1] -> (96, 64), stride (64, 1), offset 6144; right base[1] -> (64, 80), stride (80, 1), offset 5120 | (96, 80), stride (80, 1), offset 0, requires_grad=False | 4 | 35.335 us +/- 0.128 us, var 5.877 | 12.210 us +/- 0.055 us, var 2.549 | 2.89x | `12347065076992943904`/`14727830144762775616` |
| `noncontiguous_transpose_96x128_by_128x64` | noncontiguous transpose | `@` | left tensor((128, 96)).transpose(0, 1) -> (96, 128), stride (1, 96); right tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128) | (96, 64), stride (64, 1), offset 0, requires_grad=False | 3 | 586.480 us +/- 5.587 us, var 11810.625 | 19.082 us +/- 0.138 us, var 5.237 | 30.73x | `13867806792300972096`/`6072559669262435360` |
| `noncontiguous_transpose_96x128_by_128x64` | noncontiguous transpose | `Tensor.matmul` | left tensor((128, 96)).transpose(0, 1) -> (96, 128), stride (1, 96); right tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128) | (96, 64), stride (64, 1), offset 0, requires_grad=False | 3 | 590.394 us +/- 5.420 us, var 8509.398 | 18.955 us +/- 0.097 us, var 1.011 | 31.15x | `13867806792300972096`/`6072559669262435360` |
| `noncontiguous_transpose_96x128_by_128x64` | noncontiguous transpose | `torch.matmul` | left tensor((128, 96)).transpose(0, 1) -> (96, 128), stride (1, 96); right tensor((64, 128)).transpose(0, 1) -> (128, 64), stride (1, 128) | (96, 64), stride (64, 1), offset 0, requires_grad=False | 3 | 591.174 us +/- 4.126 us, var 398.575 | 19.050 us +/- 0.117 us, var 3.534 | 31.03x | `13867806792300972096`/`6072559669262435360` |
| `no_grad_requires_grad_96x64_by_64x32` | no_grad | `@` | left (96, 64), stride (64, 1), requires_grad=True; right (64, 32), stride (32, 1), requires_grad=True; operation inside `no_grad` | (96, 32), stride (32, 1), offset 0, requires_grad=False | 5 | 22.286 us +/- 0.059 us, var 2.078 | 7.860 us +/- 0.043 us, var 0.335 | 2.84x | `4909791260149529120`/`15655191744390754080` |
| `no_grad_requires_grad_96x64_by_64x32` | no_grad | `Tensor.matmul` | left (96, 64), stride (64, 1), requires_grad=True; right (64, 32), stride (32, 1), requires_grad=True; operation inside `no_grad` | (96, 32), stride (32, 1), offset 0, requires_grad=False | 5 | 22.418 us +/- 0.060 us, var 2.253 | 7.887 us +/- 0.066 us, var 0.779 | 2.84x | `4909791260149529120`/`15655191744390754080` |
| `no_grad_requires_grad_96x64_by_64x32` | no_grad | `torch.matmul` | left (96, 64), stride (64, 1), requires_grad=True; right (64, 32), stride (32, 1), requires_grad=True; operation inside `no_grad` | (96, 32), stride (32, 1), offset 0, requires_grad=False | 5 | 22.470 us +/- 0.058 us, var 1.771 | 7.734 us +/- 0.087 us, var 0.487 | 2.91x | `4909791260149529120`/`15655191744390754080` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `operator_backward_sum_16x17_by_17x9` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported scalar loss plus both leaf gradients; checksums `14143793691661545073`, `8237576367647143277`, `6012079219004156328` | zero |
| `method_backward_sum_16x17_by_17x9` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported scalar loss plus both leaf gradients; checksums `14143793691661545073`, `8237576367647143277`, `6012079219004156328` | zero |
| `torch_backward_sum_16x17_by_17x9` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | supported scalar loss plus both leaf gradients; checksums `14143793691661545073`, `8237576367647143277`, `6012079219004156328` | zero |
| `vector_dot_257` | `RuntimeError: matmul currently requires two rank-2 tensors, got [257] and [257]` | supported scalar dot product; checksum `9908422579666769089` | zero |
| `matrix_vector_64x32_by_32` | `RuntimeError: matmul currently requires two rank-2 tensors, got [64, 32] and [32]` | supported matrix-vector result; checksum `18151391646007639559` | zero |
| `batched_rank3_2x16x17_by_2x17x8` | `RuntimeError: matmul currently requires two rank-2 tensors, got [2, 16, 17] and [2, 17, 8]` | supported batched matrix result; checksum `17090840966776604060` | zero |
| `top_level_out_keyword_16x17_by_17x9` | `TypeError: matmul() got an unexpected keyword argument 'out'` | supported `out=` result; checksum `4370535962129004602` | zero |
