# `torch.sub` and `torch.subtract` Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on
`de221b72e5876af80a4fb497844fee7079721de5`. This branch replaces the PR
#1809 one-off ignored `target/` timing driver with the checked-in
`scripts/benchmark_top_level_subtract.py` driver and checked-in raw JSON at
`docs/benchmark-data/top-level-subtract-release-timings.json`; it does not
change the runtime subtraction implementation.

Exact setup, build, check, and timing commands were run from the repository
root. The active Conda environment in the shell provided PyTorch 2.14, while
the reference differential tests and benchmark require PyTorch 2.13.0, so setup
used a worktree-local `.venv` and did not install packages into the Conda
environment. Cargo registry data was copied read-only from the existing user
cache into `target/cargo-home`, then Cargo ran offline so build artifacts and
dependency state stayed inside this worktree.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --python "$PWD/.venv/bin/python" \
  --no-install-project --group dev --group reference
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
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  .venv/bin/python scripts/benchmark_top_level_subtract.py \
  --cpu 24 --threads 1 \
  --output docs/benchmark-data/top-level-subtract-release-timings.json
.venv/bin/python scripts/benchmark_top_level_subtract.py \
  --render-markdown-summary \
  docs/benchmark-data/top-level-subtract-release-timings.json \
  > target/top-level-subtract-summary.md
```

Checks run for this evidence:

```bash
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
in 175.47 seconds with two implementation orders, 15 untimed warmup blocks, and
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
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
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
- Dependency installation: locked `uv sync` resolved in 33 ms, prepared packages
  in 15.75s, and installed in 1.23s
- Build time: successful offline release extension build completed in 37.79s;
  the release wheel reinstall resolved in 2 ms, prepared in 42 ms, and installed
  in 18 ms

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
- All supported cells: 0.78x uncapped, 0.78x capped
- `torch.sub` cells: 0.79x uncapped, 0.79x capped
- `torch.subtract` cells: 0.76x uncapped, 0.76x capped
- Tensor/tensor cells: 1.02x uncapped, 1.02x capped
- Tensor/scalar cells: 0.68x uncapped, 0.68x capped
- Scalar/tensor cells: 0.65x uncapped, 0.65x capped
- Scalar cells: 0.48x uncapped, 0.48x capped
- Empty cells: 0.54x uncapped, 0.54x capped
- Broadcasting cells: 0.96x uncapped, 0.96x capped
- Offset cells: 1.91x uncapped, 1.91x capped
- Noncontiguous cells: 1.87x uncapped, 1.87x capped
- Autograd forward cells: 0.66x uncapped, 0.66x capped
- Autograd forward+backward cells: 0.21x uncapped, 0.21x capped
- `no_grad` cells: 0.80x uncapped, 0.80x capped

Including the unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 1.36x.

## Supported Timed Cells

| Workload | Category | API | Operand path | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_tensor_tensor` | scalar | `torch.sub` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 0.721 us +/- 0.008 us, var 0.001 | 1.690 us +/- 0.008 us, var 0.001 | 0.43x | `1208035594771451726`/`1208035594771451726` |
| `scalar_tensor_scalar` | scalar | `torch.sub` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -1.25 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.600 us +/- 0.007 us, var 0.002 | 3.130 us +/- 0.014 us, var 0.014 | 0.51x | `5109002241948048247`/`5109002241948048247` |
| `scalar_scalar_tensor` | scalar | `torch.sub` | scalar/tensor | scalar 3.5; right scalar tensor, shape (), stride () | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.580 us +/- 0.008 us, var 0.001 | 3.160 us +/- 0.024 us, var 0.011 | 0.50x | `1208035594771451726`/`1208035594771451726` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.sub` | tensor/tensor | left/right (257, 263), stride (263, 1) | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 10.589 us +/- 0.214 us, var 0.248 | 11.246 us +/- 0.244 us, var 0.275 | 0.94x | `1293673457235018937`/`1293673457235018937` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.sub` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 86.177 us +/- 24.654 us, var 2921.974 | 59.429 us +/- 9.795 us, var 1680.346 | 1.45x | `4282783516486587185`/`4282783516486587185` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.sub` | scalar/tensor | scalar 2.25; right (640, 768), stride (768, 1) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 96.444 us +/- 35.076 us, var 1914.451 | 61.905 us +/- 11.231 us, var 2528.624 | 1.56x | `9701670657857802040`/`9701670657857802040` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.sub` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 16 | 53.707 us +/- 0.580 us, var 2.330 | 61.792 us +/- 13.139 us, var 786.800 | 0.87x | `14706663387298716370`/`14706663387298716370` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.sub` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.093 us +/- 0.113 us, var 0.034 | 1.784 us +/- 0.077 us, var 0.111 | 0.61x | `4775394982383302379`/`4775394982383302379` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.sub` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 1.5 | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.680 us +/- 0.031 us, var 0.120 | 3.119 us +/- 0.019 us, var 0.013 | 0.54x | `4775394982383302379`/`4775394982383302379` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.sub` | scalar/tensor | scalar 1.5; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.680 us +/- 0.014 us, var 0.046 | 3.127 us +/- 0.015 us, var 0.006 | 0.54x | `4775394982383302379`/`4775394982383302379` |
| `offset_transposed_521x509` | offset | `torch.sub` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 194.609 us +/- 2.982 us, var 85.088 | 38.841 us +/- 1.079 us, var 691.488 | 5.01x | `11124672438608992435`/`11124672438608992435` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.sub` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -1.75 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 36.189 us +/- 1.709 us, var 8.471 | 30.235 us +/- 0.532 us, var 10.924 | 1.20x | `6629913853578705580`/`6629913853578705580` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.sub` | scalar/tensor | scalar -1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 37.091 us +/- 1.672 us, var 16.728 | 32.542 us +/- 1.850 us, var 28.596 | 1.14x | `17487024768410659571`/`17487024768410659571` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.sub` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 396.399 us +/- 4.690 us, var 759.458 | 95.612 us +/- 6.704 us, var 1160.051 | 4.15x | `9213456276073332991`/`9213456276073332991` |
| `noncontig_tensor_scalar_512x1024` | noncontiguous tensor/scalar | `torch.sub` | tensor/scalar | left tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512); scalar 0.75 | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 100.980 us +/- 14.141 us, var 319.373 | 67.297 us +/- 4.160 us, var 295.660 | 1.50x | `12611605271690085440`/`12611605271690085440` |
| `noncontig_scalar_tensor_512x1024` | noncontiguous scalar/tensor | `torch.sub` | scalar/tensor | scalar 0.75; right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 79.364 us +/- 3.681 us, var 149.851 | 71.619 us +/- 5.530 us, var 196.517 | 1.11x | `2707089764599809596`/`2707089764599809596` |
| `autograd_forward_127x131` | autograd forward | `torch.sub` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.038 us +/- 0.027 us, var 0.065 | 3.963 us +/- 0.048 us, var 0.119 | 0.77x | `8379595895519062337`/`8379595895519062337` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.sub` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.268 us +/- 0.051 us, var 0.146 | 5.229 us +/- 0.073 us, var 0.318 | 0.63x | `1776482405669258380`/`1776482405669258380` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.262 us +/- 0.062 us, var 0.173 | 5.295 us +/- 0.093 us, var 0.147 | 0.62x | `16658581833958532288`/`16658581833958532288` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.sub` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 19.366 us +/- 0.265 us, var 3.641 | 47.299 us +/- 1.872 us, var 11.335 | 0.41x | `5161853694224322493`/`5161853694224322493` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.sub` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar -2.25; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.723 us +/- 0.127 us, var 0.362 | 41.338 us +/- 0.937 us, var 5.496 | 0.16x | `18194694021996292960`/`18194694021996292960` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.759 us +/- 0.141 us, var 0.402 | 44.199 us +/- 1.933 us, var 35.701 | 0.15x | `13826423854746859558`/`13826423854746859558` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.sub` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 11.000 us +/- 0.282 us, var 0.496 | 13.112 us +/- 0.290 us, var 0.302 | 0.84x | `12593093982454061264`/`12593093982454061264` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.sub` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar -2.25; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 13.811 us +/- 0.320 us, var 3.101 | 15.154 us +/- 0.588 us, var 2.174 | 0.91x | `15633502805010059547`/`15633502805010059547` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 8.430 us +/- 0.333 us, var 3.553 | 11.267 us +/- 0.160 us, var 0.132 | 0.75x | `12427253315330196583`/`12427253315330196583` |
| `scalar_tensor_tensor` | scalar | `torch.subtract` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 0.758 us +/- 0.021 us, var 0.002 | 1.724 us +/- 0.009 us, var 0.002 | 0.44x | `1208035594771451726`/`1208035594771451726` |
| `scalar_tensor_scalar` | scalar | `torch.subtract` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -1.25 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.610 us +/- 0.014 us, var 0.002 | 3.135 us +/- 0.024 us, var 0.023 | 0.51x | `5109002241948048247`/`5109002241948048247` |
| `scalar_scalar_tensor` | scalar | `torch.subtract` | scalar/tensor | scalar 3.5; right scalar tensor, shape (), stride () | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.614 us +/- 0.020 us, var 0.002 | 3.150 us +/- 0.015 us, var 0.011 | 0.51x | `1208035594771451726`/`1208035594771451726` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.subtract` | tensor/tensor | left/right (257, 263), stride (263, 1) | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 10.621 us +/- 0.461 us, var 0.689 | 11.304 us +/- 0.214 us, var 0.277 | 0.94x | `1293673457235018937`/`1293673457235018937` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.subtract` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 60.914 us +/- 4.750 us, var 113.183 | 54.994 us +/- 2.264 us, var 28.786 | 1.11x | `4282783516486587185`/`4282783516486587185` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.subtract` | scalar/tensor | scalar 2.25; right (640, 768), stride (768, 1) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 60.445 us +/- 2.544 us, var 33.016 | 51.939 us +/- 1.581 us, var 8.432 | 1.16x | `9701670657857802040`/`9701670657857802040` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.subtract` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 16 | 56.836 us +/- 1.983 us, var 87.795 | 53.063 us +/- 2.085 us, var 205.138 | 1.07x | `14706663387298716370`/`14706663387298716370` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.subtract` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 0.865 us +/- 0.008 us, var 0.001 | 1.716 us +/- 0.012 us, var 0.002 | 0.50x | `4775394982383302379`/`4775394982383302379` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.subtract` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 1.5 | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.688 us +/- 0.011 us, var 0.001 | 3.197 us +/- 0.017 us, var 0.123 | 0.53x | `4775394982383302379`/`4775394982383302379` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.subtract` | scalar/tensor | scalar 1.5; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.699 us +/- 0.015 us, var 0.001 | 3.219 us +/- 0.028 us, var 0.047 | 0.53x | `4775394982383302379`/`4775394982383302379` |
| `offset_transposed_521x509` | offset | `torch.subtract` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 197.381 us +/- 3.732 us, var 255.879 | 40.475 us +/- 1.721 us, var 15.845 | 4.88x | `11124672438608992435`/`11124672438608992435` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.subtract` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -1.75 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 40.171 us +/- 1.777 us, var 8.040 | 32.047 us +/- 1.071 us, var 6.077 | 1.25x | `6629913853578705580`/`6629913853578705580` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.subtract` | scalar/tensor | scalar -1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 38.906 us +/- 1.695 us, var 15.699 | 33.612 us +/- 1.675 us, var 16.201 | 1.16x | `17487024768410659571`/`17487024768410659571` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.subtract` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 400.170 us +/- 4.408 us, var 76.805 | 91.647 us +/- 6.273 us, var 160.443 | 4.37x | `9213456276073332991`/`9213456276073332991` |
| `noncontig_tensor_scalar_512x1024` | noncontiguous tensor/scalar | `torch.subtract` | tensor/scalar | left tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512); scalar 0.75 | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 111.023 us +/- 12.422 us, var 794.165 | 94.135 us +/- 10.422 us, var 389.571 | 1.18x | `12611605271690085440`/`12611605271690085440` |
| `noncontig_scalar_tensor_512x1024` | noncontiguous scalar/tensor | `torch.subtract` | scalar/tensor | scalar 0.75; right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 80.697 us +/- 3.795 us, var 159.399 | 67.613 us +/- 3.628 us, var 43.122 | 1.19x | `2707089764599809596`/`2707089764599809596` |
| `autograd_forward_127x131` | autograd forward | `torch.subtract` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.063 us +/- 0.029 us, var 0.052 | 4.049 us +/- 0.037 us, var 0.129 | 0.76x | `8379595895519062337`/`8379595895519062337` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.324 us +/- 0.041 us, var 0.049 | 5.356 us +/- 0.089 us, var 0.210 | 0.62x | `1776482405669258380`/`1776482405669258380` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.264 us +/- 0.033 us, var 0.128 | 5.298 us +/- 0.112 us, var 0.135 | 0.62x | `16658581833958532288`/`16658581833958532288` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.subtract` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 21.598 us +/- 0.642 us, var 4.904 | 53.178 us +/- 3.186 us, var 30.646 | 0.41x | `5161853694224322493`/`5161853694224322493` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar -2.25; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 7.703 us +/- 0.164 us, var 1.080 | 52.808 us +/- 3.770 us, var 34.742 | 0.15x | `18194694021996292960`/`18194694021996292960` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.510 us +/- 0.172 us, var 0.377 | 42.816 us +/- 1.752 us, var 35.862 | 0.15x | `13826423854746859558`/`13826423854746859558` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.subtract` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 12.524 us +/- 0.633 us, var 1.131 | 16.152 us +/- 0.379 us, var 1.313 | 0.78x | `12593093982454061264`/`12593093982454061264` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar -2.25; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 11.759 us +/- 0.719 us, var 1.671 | 14.924 us +/- 0.381 us, var 0.369 | 0.79x | `15633502805010059547`/`15633502805010059547` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 8.390 us +/- 0.151 us, var 0.122 | 11.127 us +/- 0.194 us, var 0.144 | 0.75x | `12427253315330196583`/`12427253315330196583` |

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
