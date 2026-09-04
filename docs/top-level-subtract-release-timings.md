# `torch.sub` and `torch.subtract` Release Timings

Date: 2026-09-04

Candidate provenance: current branch head
`a32b87cb92a715c35e40c429ab0e32abf342c364`. The refreshed raw JSON records a
clean git status at benchmark start for that source state. This branch adds a
CPU float32 no-grad same-shape tensor/tensor fast path for matching dense
transposed and offset subtraction layouts, preserves the active-autograd
fallback, and keeps PyTorch-compatible subtract NaN semantics on fallback paths.

Exact build, check, and timing commands were run from the repository root. The
benchmark used the worktree-local `.venv` with pinned PyTorch 2.13.0 and did not
install packages outside the worktree. `CUDA_VISIBLE_DEVICES=` kept this
CPU-only benchmark from selecting the host GPUs.

```bash
env -u CONDA_PREFIX \
  TMPDIR="$PWD/target" \
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
cargo fmt --check
cargo test --locked --all-targets
git diff --check
```

Results: the checked-in driver produced `docs/benchmark-data/top-level-subtract-release-timings.json`
in 166.03 seconds with two implementation orders, 15 untimed warmup blocks, and
81 measured blocks per implementation pass. The generated markdown below is
validated byte-for-byte against that artifact by
`scripts/benchmark_top_level_subtract.py --validate-artifact` and
`tests.test_top_level_subtract_benchmark_artifact`. The targeted same-shape
transposed tensor/tensor cells measured 1.46x and 1.49x for offset-transposed
`torch.sub`/`torch.subtract`, and 1.78x and 2.18x for noncontiguous transposed
`torch.sub`/`torch.subtract`.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.12
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
- Build: successful editable release extension rebuild before timing; Cargo
  reported the release profile finished in 0.01s

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
- All supported cells: 0.72x uncapped, 0.72x capped
- `torch.sub` cells: 0.71x uncapped, 0.71x capped
- `torch.subtract` cells: 0.72x uncapped, 0.72x capped
- Tensor/tensor cells: 0.94x uncapped, 0.94x capped
- Tensor/scalar cells: 0.62x uncapped, 0.62x capped
- Scalar/tensor cells: 0.62x uncapped, 0.62x capped
- Scalar cells: 0.48x uncapped, 0.48x capped
- Empty cells: 0.46x uncapped, 0.46x capped
- Broadcasting cells: 1.67x uncapped, 1.67x capped
- Offset cells: 1.28x uncapped, 1.28x capped
- Noncontiguous cells: 1.48x uncapped, 1.48x capped
- Autograd forward cells: 0.68x uncapped, 0.68x capped
- Autograd forward+backward cells: 0.22x uncapped, 0.22x capped
- `no_grad` cells: 0.78x uncapped, 0.78x capped

Including the unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 1.28x.

## Supported Timed Cells

| Workload | Category | API | Operand path | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_tensor_tensor` | scalar | `torch.sub` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 0.910 us +/- 0.005 us, var 0.004 | 1.577 us +/- 0.016 us, var 0.051 | 0.58x | `1208035594771451726`/`1208035594771451726` |
| `scalar_tensor_scalar` | scalar | `torch.sub` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -1.25 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.294 us +/- 0.007 us, var 0.000 | 2.964 us +/- 0.016 us, var 0.003 | 0.44x | `5109002241948048247`/`5109002241948048247` |
| `scalar_scalar_tensor` | scalar | `torch.sub` | scalar/tensor | scalar 3.5; right scalar tensor, shape (), stride () | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.301 us +/- 0.010 us, var 0.023 | 3.002 us +/- 0.017 us, var 0.002 | 0.43x | `1208035594771451726`/`1208035594771451726` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.sub` | tensor/tensor | left/right (257, 263), stride (263, 1) | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 12.112 us +/- 0.448 us, var 3.515 | 11.054 us +/- 0.214 us, var 0.842 | 1.10x | `1293673457235018937`/`1293673457235018937` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.sub` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 65.356 us +/- 6.172 us, var 79.665 | 56.043 us +/- 6.920 us, var 3760.739 | 1.17x | `4282783516486587185`/`4282783516486587185` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.sub` | scalar/tensor | scalar 2.25; right (640, 768), stride (768, 1) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 62.787 us +/- 8.620 us, var 1923.187 | 61.946 us +/- 14.130 us, var 4458.230 | 1.01x | `9701670657857802040`/`9701670657857802040` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.sub` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 16 | 86.792 us +/- 6.760 us, var 755.978 | 47.406 us +/- 0.449 us, var 1012.248 | 1.83x | `14706663387298716370`/`14706663387298716370` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.sub` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 0.757 us +/- 0.005 us, var 0.001 | 1.580 us +/- 0.011 us, var 0.001 | 0.48x | `4775394982383302379`/`4775394982383302379` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.sub` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 1.5 | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.364 us +/- 0.009 us, var 0.000 | 3.029 us +/- 0.019 us, var 0.049 | 0.45x | `4775394982383302379`/`4775394982383302379` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.sub` | scalar/tensor | scalar 1.5; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.370 us +/- 0.008 us, var 0.000 | 3.035 us +/- 0.013 us, var 0.046 | 0.45x | `4775394982383302379`/`4775394982383302379` |
| `offset_transposed_521x509` | offset | `torch.sub` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 56.600 us +/- 1.900 us, var 17.742 | 38.697 us +/- 1.294 us, var 586.621 | 1.46x | `11124672438608992435`/`11124672438608992435` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.sub` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -1.75 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 38.280 us +/- 1.824 us, var 12.029 | 30.985 us +/- 0.810 us, var 4.096 | 1.24x | `6629913853578705580`/`6629913853578705580` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.sub` | scalar/tensor | scalar -1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 41.104 us +/- 1.716 us, var 11.819 | 33.885 us +/- 2.181 us, var 27.437 | 1.21x | `17487024768410659571`/`17487024768410659571` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.sub` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 183.742 us +/- 63.699 us, var 80045.795 | 103.318 us +/- 5.997 us, var 2044.241 | 1.78x | `9213456276073332991`/`9213456276073332991` |
| `noncontig_tensor_scalar_512x1024` | noncontiguous tensor/scalar | `torch.sub` | tensor/scalar | left tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512); scalar 0.75 | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 87.076 us +/- 8.987 us, var 134.198 | 71.574 us +/- 5.652 us, var 2260.792 | 1.22x | `12611605271690085440`/`12611605271690085440` |
| `noncontig_scalar_tensor_512x1024` | noncontiguous scalar/tensor | `torch.sub` | scalar/tensor | scalar 0.75; right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 75.653 us +/- 2.378 us, var 44.854 | 68.558 us +/- 4.824 us, var 69.858 | 1.10x | `2707089764599809596`/`2707089764599809596` |
| `autograd_forward_127x131` | autograd forward | `torch.sub` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.646 us +/- 0.037 us, var 0.078 | 3.832 us +/- 0.046 us, var 0.066 | 0.95x | `8379595895519062337`/`8379595895519062337` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.sub` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 2.941 us +/- 0.026 us, var 0.061 | 5.122 us +/- 0.062 us, var 0.215 | 0.57x | `1776482405669258380`/`1776482405669258380` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 2.897 us +/- 0.029 us, var 0.038 | 5.139 us +/- 0.084 us, var 0.180 | 0.56x | `16658581833958532288`/`16658581833958532288` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.sub` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 18.306 us +/- 0.215 us, var 1.309 | 41.826 us +/- 0.789 us, var 4.865 | 0.44x | `5161853694224322493`/`5161853694224322493` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.sub` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar -2.25; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.249 us +/- 0.135 us, var 0.425 | 38.946 us +/- 0.664 us, var 4.262 | 0.16x | `18194694021996292960`/`18194694021996292960` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.236 us +/- 0.194 us, var 0.858 | 39.110 us +/- 0.635 us, var 4.506 | 0.16x | `13826423854746859558`/`13826423854746859558` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.sub` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 11.954 us +/- 0.284 us, var 0.206 | 12.827 us +/- 0.397 us, var 0.296 | 0.93x | `12593093982454061264`/`12593093982454061264` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.sub` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar -2.25; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 7.640 us +/- 0.068 us, var 0.182 | 10.847 us +/- 0.303 us, var 0.220 | 0.70x | `15633502805010059547`/`15633502805010059547` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.sub` | scalar/tensor | scalar 2.25; right leaf (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 7.545 us +/- 0.148 us, var 0.197 | 10.725 us +/- 0.182 us, var 0.166 | 0.70x | `12427253315330196583`/`12427253315330196583` |
| `scalar_tensor_tensor` | scalar | `torch.subtract` | tensor/tensor | left/right scalar tensors, shape (), stride (), offset 0 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 0.928 us +/- 0.011 us, var 0.002 | 1.595 us +/- 0.008 us, var 0.001 | 0.58x | `1208035594771451726`/`1208035594771451726` |
| `scalar_tensor_scalar` | scalar | `torch.subtract` | tensor/scalar | left scalar tensor, shape (), stride (); scalar -1.25 | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.289 us +/- 0.007 us, var 0.000 | 2.991 us +/- 0.021 us, var 0.006 | 0.43x | `5109002241948048247`/`5109002241948048247` |
| `scalar_scalar_tensor` | scalar | `torch.subtract` | scalar/tensor | scalar 3.5; right scalar tensor, shape (), stride () | subtraction output; (), stride (), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.289 us +/- 0.006 us, var 0.002 | 2.985 us +/- 0.014 us, var 0.002 | 0.43x | `1208035594771451726`/`1208035594771451726` |
| `same_contiguous_257x263` | tensor/tensor contiguous | `torch.subtract` | tensor/tensor | left/right (257, 263), stride (263, 1) | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 11.450 us +/- 0.302 us, var 0.495 | 11.317 us +/- 0.166 us, var 0.141 | 1.01x | `1293673457235018937`/`1293673457235018937` |
| `tensor_scalar_640x768` | tensor/scalar contiguous | `torch.subtract` | tensor/scalar | left (640, 768), stride (768, 1); scalar -2.25 | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 58.053 us +/- 1.824 us, var 32.918 | 53.961 us +/- 1.743 us, var 73.439 | 1.08x | `4282783516486587185`/`4282783516486587185` |
| `scalar_tensor_640x768` | scalar/tensor contiguous | `torch.subtract` | scalar/tensor | scalar 2.25; right (640, 768), stride (768, 1) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10 | 59.518 us +/- 2.088 us, var 40.351 | 50.751 us +/- 1.419 us, var 24.253 | 1.17x | `9701670657857802040`/`9701670657857802040` |
| `vector_broadcast_640x768_by_768` | broadcasting | `torch.subtract` | tensor/tensor | left (640, 768), stride (768, 1); right (768,), stride (1,) | subtraction output; (640, 768), stride (768, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 16 | 81.076 us +/- 1.676 us, var 33.145 | 53.232 us +/- 2.149 us, var 38.532 | 1.52x | `14706663387298716370`/`14706663387298716370` |
| `empty_strided_broadcast_3x0x2` | empty | `torch.subtract` | tensor/tensor | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); right (1, 1, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 0.747 us +/- 0.006 us, var 0.001 | 1.616 us +/- 0.014 us, var 0.002 | 0.46x | `4775394982383302379`/`4775394982383302379` |
| `empty_tensor_scalar_3x0x2` | empty tensor/scalar | `torch.subtract` | tensor/scalar | left zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2); scalar 1.5 | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.368 us +/- 0.013 us, var 0.004 | 3.077 us +/- 0.021 us, var 0.106 | 0.44x | `4775394982383302379`/`4775394982383302379` |
| `empty_scalar_tensor_3x0x2` | empty scalar/tensor | `torch.subtract` | scalar/tensor | scalar 1.5; right zeros((2, 0, 3)).transpose(0, 2) -> (3, 0, 2) | subtraction output; (3, 0, 2), stride (1, 3, 0), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.382 us +/- 0.028 us, var 0.004 | 3.088 us +/- 0.018 us, var 0.060 | 0.45x | `4775394982383302379`/`4775394982383302379` |
| `offset_transposed_521x509` | offset | `torch.subtract` | tensor/tensor | left/right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 56.473 us +/- 2.735 us, var 23.553 | 37.992 us +/- 1.113 us, var 162.031 | 1.49x | `11124672438608992435`/`11124672438608992435` |
| `offset_tensor_scalar_521x509` | offset tensor/scalar | `torch.subtract` | tensor/scalar | left tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189; scalar -1.75 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 37.593 us +/- 1.601 us, var 10.609 | 32.002 us +/- 0.959 us, var 7.280 | 1.17x | `6629913853578705580`/`6629913853578705580` |
| `offset_scalar_tensor_521x509` | offset scalar/tensor | `torch.subtract` | scalar/tensor | scalar -1.75; right tensor((3, 509, 521))[1].transpose(0, 1) -> (521, 509), stride (1, 521), input offset 265189 | subtraction output; (521, 509), stride (1, 521), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 36.005 us +/- 1.467 us, var 7.691 | 30.665 us +/- 0.894 us, var 4.794 | 1.17x | `17487024768410659571`/`17487024768410659571` |
| `noncontig_transpose_512x1024` | noncontiguous | `torch.subtract` | tensor/tensor | left/right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 187.868 us +/- 74.271 us, var 15183.614 | 86.010 us +/- 4.789 us, var 317.676 | 2.18x | `9213456276073332991`/`9213456276073332991` |
| `noncontig_tensor_scalar_512x1024` | noncontiguous tensor/scalar | `torch.subtract` | tensor/scalar | left tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512); scalar 0.75 | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 76.507 us +/- 2.538 us, var 72.734 | 60.387 us +/- 3.452 us, var 466.426 | 1.27x | `12611605271690085440`/`12611605271690085440` |
| `noncontig_scalar_tensor_512x1024` | noncontiguous scalar/tensor | `torch.subtract` | scalar/tensor | scalar 0.75; right tensor((1024, 512)).transpose(0, 1) -> (512, 1024), stride (1, 512) | subtraction output; (512, 1024), stride (1, 512), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5 | 100.228 us +/- 24.203 us, var 14667.814 | 62.407 us +/- 3.646 us, var 446.842 | 1.61x | `2707089764599809596`/`2707089764599809596` |
| `autograd_forward_127x131` | autograd forward | `torch.subtract` | tensor/tensor | left/right leaves (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 3.717 us +/- 0.072 us, var 0.083 | 3.885 us +/- 0.036 us, var 0.146 | 0.96x | `8379595895519062337`/`8379595895519062337` |
| `autograd_forward_tensor_scalar_127x131` | autograd forward tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (127, 131), requires_grad=True; scalar -2.25; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 2.963 us +/- 0.031 us, var 0.050 | 5.185 us +/- 0.112 us, var 0.314 | 0.57x | `1776482405669258380`/`1776482405669258380` |
| `autograd_forward_scalar_tensor_127x131` | autograd forward scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (127, 131), requires_grad=True; forward construction only | subtraction output; (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 2.999 us +/- 0.038 us, var 0.038 | 5.148 us +/- 0.076 us, var 0.198 | 0.58x | `16658581833958532288`/`16658581833958532288` |
| `autograd_forward_backward_32x33` | autograd forward+backward | `torch.subtract` | tensor/tensor | left/right leaves (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 18.614 us +/- 0.316 us, var 1.474 | 43.027 us +/- 1.119 us, var 10.917 | 0.43x | `5161853694224322493`/`5161853694224322493` |
| `autograd_forward_backward_tensor_scalar_32x33` | autograd forward+backward tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (32, 33), requires_grad=True; scalar -2.25; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.450 us +/- 0.246 us, var 1.744 | 39.256 us +/- 0.812 us, var 5.482 | 0.16x | `18194694021996292960`/`18194694021996292960` |
| `autograd_forward_backward_scalar_tensor_32x33` | autograd forward+backward scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (32, 33), requires_grad=True; timed op(...).sum().backward() | subtraction output plus leaf gradients; (32, 33), stride (33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 6.363 us +/- 0.174 us, var 0.271 | 40.610 us +/- 1.217 us, var 6.383 | 0.16x | `13826423854746859558`/`13826423854746859558` |
| `no_grad_tensor_tensor_257x263` | no_grad tensor/tensor | `torch.subtract` | tensor/tensor | left/right leaves (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 12.264 us +/- 0.441 us, var 0.538 | 12.541 us +/- 0.328 us, var 0.211 | 0.98x | `12593093982454061264`/`12593093982454061264` |
| `no_grad_tensor_scalar_257x263` | no_grad tensor/scalar | `torch.subtract` | tensor/scalar | left leaf (257, 263), requires_grad=True; scalar -2.25; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 7.955 us +/- 0.199 us, var 1.275 | 11.210 us +/- 0.289 us, var 5.463 | 0.71x | `15633502805010059547`/`15633502805010059547` |
| `no_grad_scalar_tensor_257x263` | no_grad scalar/tensor | `torch.subtract` | scalar/tensor | scalar 2.25; right leaf (257, 263), requires_grad=True; operation inside no_grad | subtraction output; (257, 263), stride (263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 32 | 7.617 us +/- 0.136 us, var 0.517 | 11.026 us +/- 0.274 us, var 0.548 | 0.69x | `12427253315330196583`/`12427253315330196583` |

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
