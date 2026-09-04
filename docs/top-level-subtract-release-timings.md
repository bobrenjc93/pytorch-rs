# `torch.sub` and `torch.subtract` Release Timings

Date: 2026-09-04

Candidate provenance: source snapshot based on
`0bb1eea16e64b4ec1451eacfe294be3571581537`. This branch adds a CPU float32
no-grad same-shape tensor/tensor fast path for matching dense transposed and
offset subtraction layouts, then refreshes the checked-in raw JSON and markdown
report from `scripts/benchmark_top_level_subtract.py`.

Exact setup, build, check, and timing commands were run from the repository
root. The active Conda environment in the shell provided PyTorch 2.14, while
the reference differential tests and benchmark require PyTorch 2.13.0, so setup
used a worktree-local `.venv` and did not install packages into the Conda
environment. Cargo registry data was copied read-only from the existing user
cache into `target/cargo-home` for the locked offline Cargo verification.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --python "$PWD/.venv/bin/python" \
  --no-install-project --group dev --group reference
env -u CONDA_PREFIX \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  .venv/bin/python scripts/benchmark_top_level_subtract.py \
  --cpu 24 --threads 1 \
  --output docs/benchmark-data/top-level-subtract-release-timings.json
.venv/bin/python scripts/benchmark_top_level_subtract.py \
  --render-markdown-summary \
  docs/benchmark-data/top-level-subtract-release-timings.json \
  > target/top-level-subtract-summary.md
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/. target/cargo-home/registry/
```

Checks run for this evidence:

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/benchmark_top_level_subtract.py --validate-artifact
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  .venv/bin/python -m unittest \
  tests.test_tensor_sub tests.test_tensor_sub_reference \
  tests.test_top_level_sub tests.test_top_level_sub_reference \
  tests.test_top_level_subtract_benchmark_artifact \
  tests.test_readme_quickstart
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets sub
git diff --check
```

Results: the checked-in driver produced `docs/benchmark-data/top-level-subtract-release-timings.json`
in 179.96 seconds with two implementation orders, 15 untimed warmup blocks, and
81 measured blocks per implementation pass. The generated markdown below is
validated byte-for-byte against that artifact by
`scripts/benchmark_top_level_subtract.py --validate-artifact` and
`tests.test_top_level_subtract_benchmark_artifact`.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0, imported from
  `.venv/lib/python3.12/site-packages/torch`; `CUDA_VISIBLE_DEVICES=` made CUDA
  unavailable for timing
- `torch_rs`: 0.1.0 from the editable release extension at
  `python/torch_rs`
- Benchmark driver: `scripts/benchmark_top_level_subtract.py`, SHA-256
  `4250435fc11ae8d572d4db879486ec5bb38d862eb249406e51f9ab64cf5107eb`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32
- CPU affinity: selected CPU 24, pinned affinity `[24]`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 29 ms, prepared packages
  in 16.41s, and installed in 954 ms
- Build time: successful editable release extension build completed in 33.55s

Inputs are created outside timed regions from deterministic CPU `float32`
values with fixed seeds. Every supported cell first compares `torch_rs` against
PyTorch for shape, stride, storage offset, contiguity, dtype, device,
`requires_grad`, leaf status, and exact logical value bits. Backward cells
materialize the subtraction output plus leaf gradients. No-grad cells use
pre-created `requires_grad=True` leaves and require fresh `requires_grad=False`
leaf outputs. Every warmup and measured block materializes its final output
bundle as a 64-bit BLAKE2b checksum over output metadata and logical bytes; the
artifact validates stable equal checksum sets for `torch_rs` and PyTorch.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/top-level-subtract-release-timings.json`
- Benchmark: `top_level_subtract_cpu_benchmark_v1`
- Timed supported cells: 50 (2 APIs x 25 workload shapes and modes)
- Zero-credit unsupported cells: 14
- Implementation orders: torch_rs then pytorch, pytorch then torch_rs; each implementation appears once before and once after the other implementation
- Warmup and sampling: 15 untimed warmup blocks and 81 measured blocks per implementation pass
- CPU affinity: selected CPU 24, pinned affinity [24]; threads=1
- All supported cells: 0.73x uncapped, 0.73x capped
- `torch.sub` cells: 0.73x uncapped, 0.73x capped
- `torch.subtract` cells: 0.73x uncapped, 0.73x capped
- Tensor/tensor cells: 0.86x uncapped, 0.86x capped
- Tensor/scalar cells: 0.67x uncapped, 0.67x capped
- Scalar/tensor cells: 0.67x uncapped, 0.67x capped
- Scalar cells: 0.49x uncapped, 0.49x capped
- Empty cells: 0.52x uncapped, 0.52x capped
- Broadcasting cells: 1.57x uncapped, 1.57x capped
- Offset cells: 1.27x uncapped, 1.27x capped
- Noncontiguous cells: 1.56x uncapped, 1.56x capped
- Autograd forward cells: 0.66x uncapped, 0.66x capped
- Autograd forward+backward cells: 0.22x uncapped, 0.22x capped
- `no_grad` cells: 0.76x uncapped, 0.76x capped

Including the unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 1.29x.

## Supported Timed Cells

| Workload | Category | API | Operand path | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_tensor_tensor` | scalar | `torch.sub` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 0.868 us +/- 0.043 us, var 0.006 | 1.977 us +/- 0.086 us, var 0.015 | 0.44x | `1208035594771451726`/`1208035594771451726` |
| `scalar_tensor_scalar` | scalar | `torch.sub` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -1.25 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.856 us +/- 0.075 us, var 0.025 | 3.595 us +/- 0.110 us, var 0.037 | 0.52x | `5109002241948048247`/`5109002241948048247` |
| `scalar_scalar_tensor` | scalar | `torch.sub` | scalar/tensor | scalar 3.5; right scalar tensor, shape (), stride () | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.857 us +/- 0.077 us, var 0.015 | 3.674 us +/- 0.093 us, var 0.022 | 0.51x | `1208035594771451726`/`1208035594771451726` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.sub` | tensor/tensor | left/right (257, 263), stride (263, 1) | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 11.189 us +/- 0.396 us, var 0.371 | 12.686 us +/- 0.513 us, var 0.637 | 0.88x | `1293673457235018937`/`1293673457235018937` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.sub` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 73.975 us +/- 6.352 us, var 183.828 | 66.592 us +/- 16.040 us, var 2072.367 | 1.11x | `4282783516486587185`/`4282783516486587185` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.sub` | scalar/tensor | scalar 2.25; right (640, 768), stride (768, 1) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 86.429 us +/- 27.265 us, var 1949.372 | 71.030 us +/- 17.558 us, var 2359.625 | 1.22x | `9701670657857802040`/`9701670657857802040` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.sub` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 16 | 75.589 us +/- 21.581 us, var 848.636 | 56.576 us +/- 7.295 us, var 386.082 | 1.34x | `14706663387298716370`/`14706663387298716370` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.sub` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.006 us +/- 0.087 us, var 0.039 | 1.957 us +/- 0.094 us, var 0.020 | 0.51x | `4775394982383302379`/`4775394982383302379` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.sub` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 1.5 | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.804 us +/- 0.090 us, var 0.022 | 3.135 us +/- 0.015 us, var 0.036 | 0.58x | `4775394982383302379`/`4775394982383302379` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.sub` | scalar/tensor | scalar 1.5; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.656 us +/- 0.012 us, var 0.028 | 3.165 us +/- 0.014 us, var 0.039 | 0.52x | `4775394982383302379`/`4775394982383302379` |
| `offset_transposed_521x509` | offset | `torch.sub` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 64.682 us +/- 3.737 us, var 43.333 | 37.854 us +/- 0.921 us, var 26.135 | 1.71x | `11124672438608992435`/`11124672438608992435` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.sub` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -1.75 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 36.832 us +/- 2.927 us, var 83.483 | 30.862 us +/- 1.018 us, var 8.328 | 1.19x | `6629913853578705580`/`6629913853578705580` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.sub` | scalar/tensor | scalar -1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 38.608 us +/- 3.369 us, var 51.224 | 45.042 us +/- 8.058 us, var 263.622 | 0.86x | `17487024768410659571`/`17487024768410659571` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.sub` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 200.654 us +/- 74.906 us, var 16689.518 | 109.410 us +/- 7.480 us, var 1426.313 | 1.83x | `9213456276073332991`/`9213456276073332991` |
| `noncontig_tensor_scalar_512x1024` | noncontiguous tensor/scalar | `torch.sub` | tensor/scalar | left tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512); scalar 0.75 | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 104.655 us +/- 28.919 us, var 8832.885 | 63.229 us +/- 3.560 us, var 803.261 | 1.66x | `12611605271690085440`/`12611605271690085440` |
| `noncontig_scalar_tensor_512x1024` | noncontiguous scalar/tensor | `torch.sub` | scalar/tensor | scalar 0.75; right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 122.349 us +/- 48.044 us, var 12167.529 | 62.333 us +/- 4.271 us, var 726.778 | 1.96x | `2707089764599809596`/`2707089764599809596` |
| `autograd_forward_127x131` | autograd forward | `torch.sub` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.002 us +/- 0.035 us, var 0.082 | 3.894 us +/- 0.029 us, var 0.045 | 0.77x | `8379595895519062337`/`8379595895519062337` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.sub` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.181 us +/- 0.023 us, var 0.049 | 5.127 us +/- 0.044 us, var 0.119 | 0.62x | `1776482405669258380`/`1776482405669258380` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.181 us +/- 0.025 us, var 0.041 | 5.136 us +/- 0.037 us, var 0.067 | 0.62x | `16658581833958532288`/`16658581833958532288` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.sub` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 18.220 us +/- 0.143 us, var 0.728 | 43.613 us +/- 0.713 us, var 12.787 | 0.42x | `5161853694224322493`/`5161853694224322493` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.sub` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar -2.25; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.461 us +/- 0.129 us, var 0.307 | 40.318 us +/- 0.620 us, var 5.682 | 0.16x | `18194694021996292960`/`18194694021996292960` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.499 us +/- 0.131 us, var 0.314 | 40.846 us +/- 0.611 us, var 9.314 | 0.16x | `13826423854746859558`/`13826423854746859558` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.sub` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 11.151 us +/- 0.249 us, var 0.262 | 13.202 us +/- 0.240 us, var 0.129 | 0.84x | `12593093982454061264`/`12593093982454061264` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.sub` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar -2.25; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 8.232 us +/- 0.163 us, var 0.206 | 11.214 us +/- 0.274 us, var 0.133 | 0.73x | `15633502805010059547`/`15633502805010059547` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 8.096 us +/- 0.122 us, var 0.444 | 10.926 us +/- 0.201 us, var 0.115 | 0.74x | `12427253315330196583`/`12427253315330196583` |
| `scalar_tensor_tensor` | scalar | `torch.subtract` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 0.746 us +/- 0.006 us, var 0.015 | 1.732 us +/- 0.008 us, var 0.019 | 0.43x | `1208035594771451726`/`1208035594771451726` |
| `scalar_tensor_scalar` | scalar | `torch.subtract` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -1.25 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.577 us +/- 0.015 us, var 0.068 | 3.120 us +/- 0.015 us, var 0.005 | 0.51x | `5109002241948048247`/`5109002241948048247` |
| `scalar_scalar_tensor` | scalar | `torch.subtract` | scalar/tensor | scalar 3.5; right scalar tensor, shape (), stride () | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.814 us +/- 0.012 us, var 0.007 | 3.158 us +/- 0.039 us, var 0.056 | 0.57x | `1208035594771451726`/`1208035594771451726` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.subtract` | tensor/tensor | left/right (257, 263), stride (263, 1) | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 11.581 us +/- 0.363 us, var 0.331 | 13.933 us +/- 0.311 us, var 0.386 | 0.83x | `1293673457235018937`/`1293673457235018937` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.subtract` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 66.718 us +/- 7.850 us, var 164.644 | 51.689 us +/- 1.544 us, var 64.327 | 1.29x | `4282783516486587185`/`4282783516486587185` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.subtract` | scalar/tensor | scalar 2.25; right (640, 768), stride (768, 1) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 88.019 us +/- 16.408 us, var 587.804 | 59.157 us +/- 3.460 us, var 115.806 | 1.49x | `9701670657857802040`/`9701670657857802040` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.subtract` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 16 | 92.704 us +/- 8.484 us, var 451.307 | 50.101 us +/- 1.302 us, var 26.449 | 1.85x | `14706663387298716370`/`14706663387298716370` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.subtract` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 0.875 us +/- 0.007 us, var 0.001 | 1.759 us +/- 0.009 us, var 0.000 | 0.50x | `4775394982383302379`/`4775394982383302379` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.subtract` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 1.5 | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.654 us +/- 0.011 us, var 0.001 | 3.240 us +/- 0.024 us, var 0.068 | 0.51x | `4775394982383302379`/`4775394982383302379` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.subtract` | scalar/tensor | scalar 1.5; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.650 us +/- 0.013 us, var 0.001 | 3.224 us +/- 0.016 us, var 0.055 | 0.51x | `4775394982383302379`/`4775394982383302379` |
| `offset_transposed_521x509` | offset | `torch.subtract` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 66.767 us +/- 2.954 us, var 77.878 | 40.246 us +/- 1.926 us, var 574.184 | 1.66x | `11124672438608992435`/`11124672438608992435` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.subtract` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -1.75 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 37.825 us +/- 2.142 us, var 14.227 | 31.603 us +/- 0.779 us, var 6.836 | 1.20x | `6629913853578705580`/`6629913853578705580` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.subtract` | scalar/tensor | scalar -1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 38.701 us +/- 2.272 us, var 11.057 | 32.201 us +/- 0.996 us, var 6.033 | 1.20x | `17487024768410659571`/`17487024768410659571` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.subtract` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 152.035 us +/- 18.898 us, var 5904.243 | 81.632 us +/- 3.493 us, var 733.792 | 1.86x | `9213456276073332991`/`9213456276073332991` |
| `noncontig_tensor_scalar_512x1024` | noncontiguous tensor/scalar | `torch.subtract` | tensor/scalar | left tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512); scalar 0.75 | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 89.729 us +/- 6.550 us, var 287.483 | 73.263 us +/- 6.044 us, var 242.673 | 1.22x | `12611605271690085440`/`12611605271690085440` |
| `noncontig_scalar_tensor_512x1024` | noncontiguous scalar/tensor | `torch.subtract` | scalar/tensor | scalar 0.75; right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 77.374 us +/- 3.231 us, var 157.364 | 72.880 us +/- 7.420 us, var 4182.349 | 1.06x | `2707089764599809596`/`2707089764599809596` |
| `autograd_forward_127x131` | autograd forward | `torch.subtract` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.018 us +/- 0.025 us, var 0.039 | 4.002 us +/- 0.047 us, var 0.061 | 0.75x | `8379595895519062337`/`8379595895519062337` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.236 us +/- 0.037 us, var 0.060 | 5.310 us +/- 0.088 us, var 0.090 | 0.61x | `1776482405669258380`/`1776482405669258380` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.205 us +/- 0.034 us, var 0.215 | 5.220 us +/- 0.106 us, var 0.357 | 0.61x | `16658581833958532288`/`16658581833958532288` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.subtract` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 18.442 us +/- 0.242 us, var 2.526 | 44.294 us +/- 1.018 us, var 10.003 | 0.42x | `5161853694224322493`/`5161853694224322493` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar -2.25; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.592 us +/- 0.133 us, var 0.225 | 41.503 us +/- 0.862 us, var 4.838 | 0.16x | `18194694021996292960`/`18194694021996292960` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.667 us +/- 0.150 us, var 0.346 | 42.138 us +/- 0.928 us, var 11.877 | 0.16x | `13826423854746859558`/`13826423854746859558` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.subtract` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 11.495 us +/- 0.265 us, var 0.185 | 13.473 us +/- 0.256 us, var 0.410 | 0.85x | `12593093982454061264`/`12593093982454061264` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar -2.25; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 8.532 us +/- 0.287 us, var 0.526 | 12.118 us +/- 0.311 us, var 1.484 | 0.70x | `15633502805010059547`/`15633502805010059547` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 8.464 us +/- 0.180 us, var 0.237 | 11.744 us +/- 0.251 us, var 0.463 | 0.72x | `12427253315330196583`/`12427253315330196583` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent PyTorch operation. They are preserved as zero-credit cells instead of being removed from the evidence set.

| Workload | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- |
| `top_level_torch_sub_out_tensor_tensor` | `RuntimeError: sub(): the 'out' argument is not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_sub_out_tensor_scalar` | `RuntimeError: sub(): the 'out' argument is not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_sub_out_scalar_tensor` | `RuntimeError: sub(): the 'out' argument is not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_sub_nondefault_alpha_tensor_tensor` | `NotImplementedError: sub(): alpha values other than 1 are not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_sub_nondefault_alpha_tensor_scalar` | `NotImplementedError: sub(): alpha values other than 1 are not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_sub_nondefault_alpha_scalar_tensor` | `NotImplementedError: sub(): alpha values other than 1 are not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_sub_scalar_scalar` | `TypeError: sub(): scalar-scalar subtraction is not supported; at least one operand must be Tensor` | `supported (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_subtract_out_tensor_tensor` | `RuntimeError: subtract(): the 'out' argument is not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_subtract_out_tensor_scalar` | `RuntimeError: subtract(): the 'out' argument is not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_subtract_out_scalar_tensor` | `RuntimeError: subtract(): the 'out' argument is not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_subtract_nondefault_alpha_tensor_tensor` | `NotImplementedError: subtract(): alpha values other than 1 are not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_subtract_nondefault_alpha_tensor_scalar` | `NotImplementedError: subtract(): alpha values other than 1 are not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_subtract_nondefault_alpha_scalar_tensor` | `NotImplementedError: subtract(): alpha values other than 1 are not supported` | `supported (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_subtract_scalar_scalar` | `TypeError: subtract(): scalar-scalar subtraction is not supported; at least one operand must be Tensor` | `supported (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
