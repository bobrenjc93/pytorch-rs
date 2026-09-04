# `torch.sub` and `torch.subtract` Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on
`d6a6f67baca678926b9619112223de692df1a935`. This branch adds timing evidence
only; it does not change the runtime implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The timing driver was a one-off file under ignored `target/` storage and
emitted JSON under `target/top-level-subtract-release-timings*.json`. A Conda
environment was active in the shell (`CONDA_PREFIX=/home/bobren/local/a/pytorch-env`),
but the reference tests require PyTorch 2.13.0, so setup used a worktree-local
`.venv` and did not install packages into the Conda environment. Cargo registry
data was copied read-only from the existing user cache into `target/cargo-home`,
then Cargo ran offline so build artifacts and dependency state stayed inside
this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry && \
  cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/ && \
  wheel_dir="$(mktemp -d "$PWD/target/top-level-subtract-wheels.XXXXXX")" && \
  printf '%s\n' "$wheel_dir" > target/top-level-subtract-wheel-dir.txt && \
  env CARGO_HOME="$PWD/target/cargo-home" \
    CARGO_TARGET_DIR="$PWD/target" \
    TMPDIR="$PWD/target" \
    VIRTUAL_ENV="$PWD/.venv" \
    PYO3_PYTHON="$PWD/.venv/bin/python" \
    .venv/bin/maturin build --release --locked --offline --out "$wheel_dir"
wheel_dir="$(cat target/top-level-subtract-wheel-dir.txt)" && \
  env UV_CACHE_DIR="$PWD/target/uv-cache" \
    UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
    uv pip install --python "$PWD/.venv/bin/python" \
    --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_tensor_sub tests.test_tensor_sub_reference \
  tests.test_top_level_sub tests.test_top_level_sub_reference
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python target/top_level_subtract_release_timings.py \
  > target/top-level-subtract-release-timings.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  SUBTRACT_TIMING_IMPL_ORDER=pytorch,torch_rs \
  taskset -c 24 .venv/bin/python target/top_level_subtract_release_timings.py \
  > target/top-level-subtract-release-timings-pass2.json
```

Checks run for this evidence:

```bash
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets sub
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_tensor_sub tests.test_tensor_sub_reference \
  tests.test_top_level_sub tests.test_top_level_sub_reference
git diff --check
```

Results: the focused Python implementation and PyTorch 2.13 differential tests
passed 21 tests. The focused Rust `sub` filter passed 6 tests, `cargo fmt
--check` passed, and `git diff --check` passed.

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
  packages in 16.04s, and installed in 1.23s
- Build time: successful offline release extension build completed in 38.12s;
  the release wheel reinstall resolved in 3 ms, prepared in 47 ms, and
  installed in 24 ms

Inputs were created outside the timed region with NumPy seed `20260903`.
Each implementation used the same CPU `float32` values, shapes, layouts, grad
mode, default `alpha=1`, and thread settings. Every timing cell ran in two
pinned process passes. The first pass measured `torch_rs` before PyTorch; the
second pass reversed that order. Each pass used 15 untimed warmup blocks and
81 measured blocks. A block repeated the operation according to the table's
`Repeats` column; times below are median microseconds per operation. Reported
medians are medians of the two per-process medians. MAD and variance are the
medians of the per-process MAD and sample variance values.

Before timing each supported cell, the driver bit-compared `torch_rs` output
values with PyTorch and checked shape, stride, storage offset, contiguity,
dtype, device, `requires_grad`, and leaf status. Backward cells timed
`op(...).sum().backward()` with pre-created fresh leaf tensors and checked the
subtraction output plus leaf gradients. The `no_grad` cells used pre-created
leaf tensors with `requires_grad=True` and timed the top-level operation inside
the `no_grad` context; outputs were required to be fresh leaf tensors with
`requires_grad=False`. After every warmup and measured block, the driver
materialized the last output tensor, or the last output plus leaf gradients for
backward cells, as a 64-bit BLAKE2b rolling checksum over output metadata and
logical bytes. The checksum column shows the final rolling sink from one pass
as `torch_rs`/PyTorch; both process passes produced the same sink pairs.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Supported Timed Cells

Geometric mean `torch_rs / PyTorch` slowdown for the supported timed cells:

- All supported cells: 0.72x uncapped, 0.72x capped
- `torch.sub` cells: 0.72x uncapped, 0.72x capped
- `torch.subtract` cells: 0.73x uncapped, 0.73x capped
- Tensor/tensor cells: 0.91x uncapped, 0.91x capped
- Tensor/scalar cells: 0.64x uncapped, 0.64x capped
- Scalar/tensor cells: 0.63x uncapped, 0.63x capped
- Scalar cells: 0.34x uncapped, 0.34x capped
- Empty cells: 0.40x uncapped, 0.40x capped
- Broadcasting cells: 0.99x uncapped, 0.99x capped
- Offset cells: 1.85x uncapped, 1.85x capped
- Noncontiguous cells: 1.75x uncapped, 1.75x capped
- Autograd forward cells: 0.64x uncapped, 0.64x capped
- Autograd forward+backward cells: 0.26x uncapped, 0.26x capped
- `no_grad` cells: 0.82x uncapped, 0.82x capped

Including the unsupported cells below as zero-credit denominator entries with a
10.00x capped penalty gives a combined capped aggregate of 1.29x.

| Workload | Category | API | Operand path | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_tensor_tensor` | scalar | `torch.sub` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | (), stride (), offset 0, requires_grad=False | 10000 | 0.270 us +/- 0.005 us, var 0.001 | 1.195 us +/- 0.006 us, var 0.001 | 0.23x | `17411068900020221376`/`17411068900020221376` |
| `scalar_tensor_scalar` | scalar | `torch.sub` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -1.25 | (), stride (), offset 0, requires_grad=False | 10000 | 1.103 us +/- 0.006 us, var 0.003 | 2.610 us +/- 0.011 us, var 0.005 | 0.42x | `17411068900020221376`/`17411068900020221376` |
| `scalar_scalar_tensor` | scalar | `torch.sub` | scalar/tensor | scalar 3.5; right scalar tensor, shape (), stride () | (), stride (), offset 0, requires_grad=False | 10000 | 1.125 us +/- 0.008 us, var 0.000 | 2.615 us +/- 0.020 us, var 0.013 | 0.43x | `17411068900020221376`/`17411068900020221376` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.sub` | tensor/tensor | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 9.951 us +/- 0.308 us, var 0.647 | 10.692 us +/- 0.169 us, var 0.195 | 0.93x | `4551877856694431456`/`4551877856694431456` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.sub` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 79.584 us +/- 11.300 us, var 459.471 | 50.010 us +/- 1.516 us, var 10.459 | 1.59x | `6366111633317229440`/`6366111633317229440` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.sub` | scalar/tensor | scalar 2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 55.560 us +/- 2.152 us, var 16.196 | 54.748 us +/- 3.283 us, var 786.901 | 1.01x | `8932958842433612512`/`8932958842433612512` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.sub` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 16 | 55.664 us +/- 2.862 us, var 195.522 | 67.816 us +/- 14.839 us, var 887.872 | 0.82x | `5279161873209427840`/`5279161873209427840` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.sub` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 0.378 us +/- 0.004 us, var 0.000 | 1.198 us +/- 0.010 us, var 0.001 | 0.32x | `17785396177732869856`/`17785396177732869856` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.sub` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 1.5 | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 1.207 us +/- 0.023 us, var 0.007 | 2.629 us +/- 0.019 us, var 0.083 | 0.46x | `17785396177732869856`/`17785396177732869856` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.sub` | scalar/tensor | scalar 1.5; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 1.175 us +/- 0.009 us, var 0.000 | 2.644 us +/- 0.018 us, var 0.077 | 0.44x | `17785396177732869856`/`17785396177732869856` |
| `offset_transposed_521x509` | offset | `torch.sub` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 192.485 us +/- 3.570 us, var 275.056 | 41.393 us +/- 1.287 us, var 4.265 | 4.65x | `14680168967699366656`/`14680168967699366656` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.sub` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -1.75 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 32.937 us +/- 1.678 us, var 10.144 | 30.681 us +/- 1.378 us, var 23.810 | 1.07x | `7054971806291306208`/`7054971806291306208` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.sub` | scalar/tensor | scalar -1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 33.771 us +/- 1.606 us, var 7.192 | 29.407 us +/- 0.824 us, var 5.589 | 1.15x | `18078639128036371168`/`18078639128036371168` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.sub` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 390.891 us +/- 4.094 us, var 254.873 | 95.294 us +/- 5.675 us, var 74.428 | 4.10x | `9562468068292614368`/`9562468068292614368` |
| `noncontig_tensor_scalar_512x1024` | noncontiguous tensor/scalar | `torch.sub` | tensor/scalar | left tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512); scalar 0.75 | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 73.621 us +/- 2.603 us, var 156.519 | 67.615 us +/- 5.747 us, var 430.951 | 1.09x | `16320383031002564800`/`16320383031002564800` |
| `noncontig_scalar_tensor_512x1024` | noncontiguous scalar/tensor | `torch.sub` | scalar/tensor | scalar 0.75; right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 68.527 us +/- 2.135 us, var 40.396 | 70.881 us +/- 6.973 us, var 106.578 | 0.97x | `9356231027853232352`/`9356231027853232352` |
| `autograd_forward_127x131` | autograd forward | `torch.sub` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.489 us +/- 0.021 us, var 0.031 | 3.316 us +/- 0.020 us, var 0.096 | 0.75x | `12348544507361971648`/`12348544507361971648` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.sub` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.724 us +/- 0.035 us, var 0.046 | 4.529 us +/- 0.059 us, var 0.157 | 0.60x | `38191912406754048`/`38191912406754048` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.692 us +/- 0.029 us, var 0.047 | 4.433 us +/- 0.033 us, var 0.052 | 0.61x | `5001989801803119328`/`5001989801803119328` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.sub` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients | 5 | 17.603 us +/- 0.364 us, var 1.712 | 33.270 us +/- 0.548 us, var 7.054 | 0.53x | `1809845486482294880`/`1809845486482294880` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.sub` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar -2.25; timed op(...).sum().backward() | subtraction output plus leaf gradients | 5 | 5.611 us +/- 0.147 us, var 0.185 | 30.364 us +/- 0.366 us, var 7.263 | 0.18x | `883381000662101850`/`883381000662101850` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients | 5 | 5.816 us +/- 0.189 us, var 1.006 | 30.975 us +/- 0.396 us, var 11.257 | 0.19x | `5531273182039811123`/`5531273182039811123` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.sub` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 9.442 us +/- 0.192 us, var 0.145 | 10.624 us +/- 0.217 us, var 0.279 | 0.89x | `6954214707903426272`/`6954214707903426272` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.sub` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar -2.25; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 7.118 us +/- 0.105 us, var 0.122 | 9.309 us +/- 0.702 us, var 1.443 | 0.76x | `2996224970376515008`/`2996224970376515008` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 7.107 us +/- 0.133 us, var 0.208 | 8.627 us +/- 0.076 us, var 0.197 | 0.82x | `7811019950661285984`/`7811019950661285984` |
| `scalar_tensor_tensor` | scalar | `torch.subtract` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | (), stride (), offset 0, requires_grad=False | 10000 | 0.270 us +/- 0.002 us, var 0.000 | 1.236 us +/- 0.007 us, var 0.020 | 0.22x | `17411068900020221376`/`17411068900020221376` |
| `scalar_tensor_scalar` | scalar | `torch.subtract` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -1.25 | (), stride (), offset 0, requires_grad=False | 10000 | 1.094 us +/- 0.011 us, var 0.002 | 2.643 us +/- 0.025 us, var 0.167 | 0.41x | `17411068900020221376`/`17411068900020221376` |
| `scalar_scalar_tensor` | scalar | `torch.subtract` | scalar/tensor | scalar 3.5; right scalar tensor, shape (), stride () | (), stride (), offset 0, requires_grad=False | 10000 | 1.097 us +/- 0.026 us, var 0.002 | 2.635 us +/- 0.017 us, var 0.154 | 0.42x | `17411068900020221376`/`17411068900020221376` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.subtract` | tensor/tensor | left/right (257, 263), stride (263, 1) | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 9.303 us +/- 0.206 us, var 0.259 | 10.738 us +/- 0.218 us, var 0.176 | 0.87x | `4551877856694431456`/`4551877856694431456` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.subtract` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 65.576 us +/- 6.648 us, var 129.791 | 55.306 us +/- 3.582 us, var 105.642 | 1.19x | `6366111633317229440`/`6366111633317229440` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.subtract` | scalar/tensor | scalar 2.25; right (640, 768), stride (768, 1) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 10 | 69.038 us +/- 3.843 us, var 259.707 | 49.922 us +/- 1.841 us, var 20.197 | 1.38x | `8932958842433612512`/`8932958842433612512` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.subtract` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | (640, 768), stride (768, 1), offset 0, requires_grad=False | 16 | 57.634 us +/- 2.005 us, var 36.157 | 47.847 us +/- 0.777 us, var 26.659 | 1.20x | `5279161873209427840`/`5279161873209427840` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.subtract` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 0.377 us +/- 0.003 us, var 0.000 | 1.208 us +/- 0.005 us, var 0.000 | 0.31x | `17785396177732869856`/`17785396177732869856` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.subtract` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 1.5 | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 1.184 us +/- 0.013 us, var 0.002 | 2.706 us +/- 0.015 us, var 0.238 | 0.44x | `17785396177732869856`/`17785396177732869856` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.subtract` | scalar/tensor | scalar 1.5; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | (3, 0, 2), stride (1, 3, 0), offset 0, requires_grad=False | 5000 | 1.186 us +/- 0.021 us, var 0.002 | 2.712 us +/- 0.016 us, var 0.234 | 0.44x | `17785396177732869856`/`17785396177732869856` |
| `offset_transposed_521x509` | offset | `torch.subtract` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 197.663 us +/- 4.069 us, var 219.749 | 39.902 us +/- 1.962 us, var 27.569 | 4.95x | `14680168967699366656`/`14680168967699366656` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.subtract` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -1.75 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 37.353 us +/- 2.138 us, var 15.148 | 30.424 us +/- 1.110 us, var 14.718 | 1.23x | `7054971806291306208`/`7054971806291306208` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.subtract` | scalar/tensor | scalar -1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | (521, 509), stride (1, 521), offset 0, requires_grad=False | 5 | 33.799 us +/- 1.558 us, var 10.617 | 29.543 us +/- 0.667 us, var 3.363 | 1.14x | `18078639128036371168`/`18078639128036371168` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.subtract` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 390.567 us +/- 5.878 us, var 438.445 | 88.236 us +/- 4.941 us, var 105.128 | 4.43x | `9562468068292614368`/`9562468068292614368` |
| `noncontig_tensor_scalar_512x1024` | noncontiguous tensor/scalar | `torch.subtract` | tensor/scalar | left tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512); scalar 0.75 | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 81.591 us +/- 4.418 us, var 245.131 | 64.926 us +/- 5.097 us, var 107.091 | 1.26x | `16320383031002564800`/`16320383031002564800` |
| `noncontig_scalar_tensor_512x1024` | noncontiguous scalar/tensor | `torch.subtract` | scalar/tensor | scalar 0.75; right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | (512, 1024), stride (1, 512), offset 0, requires_grad=False | 5 | 92.856 us +/- 19.193 us, var 599.510 | 77.449 us +/- 6.705 us, var 171.558 | 1.20x | `9356231027853232352`/`9356231027853232352` |
| `autograd_forward_127x131` | autograd forward | `torch.subtract` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.513 us +/- 0.033 us, var 0.058 | 3.529 us +/- 0.023 us, var 0.060 | 0.71x | `12348544507361971648`/`12348544507361971648` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.723 us +/- 0.022 us, var 0.052 | 4.656 us +/- 0.038 us, var 0.076 | 0.58x | `38191912406754048`/`38191912406754048` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (127, 131), requires_grad=True; forward construction only | (127, 131), stride (131, 1), offset 0, requires_grad=True | 20 | 2.761 us +/- 0.042 us, var 0.287 | 4.659 us +/- 0.038 us, var 0.058 | 0.59x | `5001989801803119328`/`5001989801803119328` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.subtract` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients | 5 | 17.691 us +/- 0.319 us, var 0.925 | 33.784 us +/- 0.800 us, var 9.492 | 0.52x | `1809845486482294880`/`1809845486482294880` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar -2.25; timed op(...).sum().backward() | subtraction output plus leaf gradients | 5 | 5.636 us +/- 0.177 us, var 0.598 | 30.758 us +/- 0.499 us, var 15.340 | 0.18x | `883381000662101850`/`883381000662101850` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients | 5 | 5.693 us +/- 0.162 us, var 2.134 | 31.090 us +/- 0.411 us, var 15.741 | 0.18x | `5531273182039811123`/`5531273182039811123` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.subtract` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 9.646 us +/- 0.186 us, var 0.157 | 10.823 us +/- 0.151 us, var 0.092 | 0.89x | `6954214707903426272`/`6954214707903426272` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar -2.25; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 7.095 us +/- 0.180 us, var 0.113 | 9.221 us +/- 0.200 us, var 0.258 | 0.77x | `2996224970376515008`/`2996224970376515008` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (257, 263), requires_grad=True; operation inside no_grad | (257, 263), stride (263, 1), offset 0, requires_grad=False | 32 | 7.112 us +/- 0.209 us, var 0.315 | 9.124 us +/- 0.269 us, var 0.270 | 0.78x | `7811019950661285984`/`7811019950661285984` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent
PyTorch operation. They are preserved as zero-credit cells instead of being
removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `top_level_torch_sub_out_tensor_tensor` | `RuntimeError: sub(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_sub_out_tensor_scalar` | `RuntimeError: sub(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_sub_out_scalar_tensor` | `RuntimeError: sub(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_sub_nondefault_alpha_tensor_tensor` | `NotImplementedError: sub(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_sub_nondefault_alpha_tensor_scalar` | `NotImplementedError: sub(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_sub_nondefault_alpha_scalar_tensor` | `NotImplementedError: sub(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_sub_scalar_scalar` | `TypeError: sub(): scalar-scalar subtraction is not supported; at least one operand must be Tensor` | supported (), stride (), offset 0, requires_grad=False | zero |
| `top_level_torch_subtract_out_tensor_tensor` | `RuntimeError: subtract(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_subtract_out_tensor_scalar` | `RuntimeError: subtract(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_subtract_out_scalar_tensor` | `RuntimeError: subtract(): the 'out' argument is not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_subtract_nondefault_alpha_tensor_tensor` | `NotImplementedError: subtract(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_subtract_nondefault_alpha_tensor_scalar` | `NotImplementedError: subtract(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_subtract_nondefault_alpha_scalar_tensor` | `NotImplementedError: subtract(): alpha values other than 1 are not supported` | supported (1,), stride (1,), offset 0, requires_grad=False | zero |
| `top_level_torch_subtract_scalar_scalar` | `TypeError: subtract(): scalar-scalar subtraction is not supported; at least one operand must be Tensor` | supported (), stride (), offset 0, requires_grad=False | zero |
