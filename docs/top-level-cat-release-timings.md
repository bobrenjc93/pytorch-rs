# `torch.cat`, `torch.concat`, and `torch.concatenate` 1-D Release Timings

Date: 2026-09-07

Candidate provenance: current worktree snapshot for the 1-D `torch.cat`
benchmark evidence branch. This branch adds timing evidence and validation only;
it does not change the runtime implementation. The raw JSON records git
provenance, including the worktree diff visible to the timing driver.

Exact setup, build, check, and timing commands were run from the repository
root. The benchmark used the worktree-local `.venv` with pinned PyTorch 2.13.0
and did not install packages outside the worktree. `CUDA_VISIBLE_DEVICES=` kept
this CPU-only benchmark from selecting the host GPUs.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
env -u CONDA_PREFIX \
  TMPDIR="$PWD/target" \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  .venv/bin/python scripts/benchmark_top_level_cat.py \
  --cpu 24 --threads 1 \
  --output docs/benchmark-data/top-level-cat-release-timings.json
.venv/bin/python scripts/benchmark_top_level_cat.py \
  --render-markdown-summary \
  docs/benchmark-data/top-level-cat-release-timings.json \
  > target/top-level-cat-summary.md
```

Checks run for this evidence:

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/benchmark_top_level_cat.py --validate-artifact
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  .venv/bin/python -m unittest \
  tests.test_top_level_cat tests.test_top_level_cat_reference \
  tests.test_top_level_cat_benchmark_artifact \
  tests.test_readme_quickstart
cargo fmt --check
cargo test --locked --all-targets
git diff --check
```

Results: the checked-in driver produced
`docs/benchmark-data/top-level-cat-release-timings.json` in 51.09 seconds with two
implementation orders, 15 untimed warmup blocks, and 81 measured blocks per
implementation pass. The generated markdown below is validated byte-for-byte
against that artifact by `scripts/benchmark_top_level_cat.py
--validate-artifact` and `tests.test_top_level_cat_benchmark_artifact`.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta from the worktree-local `.venv`
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch`;
  `CUDA_VISIBLE_DEVICES=` made CUDA unavailable for timing
- `torch_rs`: 0.1.0 from the editable release extension at `python/torch_rs`
- Benchmark driver: `scripts/benchmark_top_level_cat.py`, SHA-256
  `b644f360182e3be8532cc0fa962c53fbb184fdb4bfd688310bdf2b3702094cdb`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32
- CPU affinity: selected CPU 24
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 26 ms, prepared
  packages in 15.32s, and installed in 1.85s
- Build: successful editable release extension rebuild before timing; Cargo
  reported the release profile finished in 39.37s

Inputs are created outside timed regions from deterministic CPU `float32`
values with fixed NumPy seeds recorded per workload in the JSON artifact. Every
supported cell first compares `torch_rs` against PyTorch for shape, stride,
storage offset, contiguity, dtype, device, `requires_grad`, leaf status, and
exact logical value bits. Every warmup and measured block materializes its
final output as a 64-bit BLAKE2b checksum over output metadata and logical
bytes. The artifact validates stable equal checksum sets for `torch_rs` and
PyTorch.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/top-level-cat-release-timings.json`
- Benchmark: `top_level_cat_cpu_1d_benchmark_v1`
- Timed supported cells: 27 (3 APIs x 9 workload shapes and modes)
- Zero-credit unsupported cells: 12
- Implementation orders: torch_rs then pytorch, pytorch then torch_rs; each implementation appears once before and once after the other implementation
- Warmup and sampling: 15 untimed warmup blocks and 81 measured blocks per implementation pass
- CPU affinity: selected CPU 24, pinned affinity [24]; threads=1
- All supported cells: 2.99x uncapped, 2.69x capped
- `torch.cat` cells: 3.01x uncapped, 2.70x capped
- `torch.concat` cells: 2.96x uncapped, 2.67x capped
- `torch.concatenate` cells: 3.00x uncapped, 2.70x capped
- Singleton cells: 14.91x uncapped, 10.00x capped
- Multi-input cells: 2.55x uncapped, 2.55x capped
- Empty-operand cells: 1.24x uncapped, 1.24x capped
- Offset cells: 14.00x uncapped, 10.00x capped
- Noncontiguous cells: 12.23x uncapped, 10.00x capped
- Axis-keyword cells: 0.63x uncapped, 0.63x capped
- `no_grad` cells: 1.19x uncapped, 1.19x capped

Including the unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 4.03x.

## Supported Timed Cells

| Workload | Category | API | Call form | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `singleton_contiguous_8192` | singleton | `torch.cat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 37.564 us +/- 0.202 us, var 0.947 | 2.507 us +/- 0.067 us, var 0.014 | 14.98x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.cat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.872 us +/- 0.105 us, var 0.106 | 2.092 us +/- 0.014 us, var 0.001 | 2.33x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.cat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 8.057 us +/- 0.047 us, var 0.033 | 3.439 us +/- 0.027 us, var 0.120 | 2.34x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.cat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.027 us +/- 0.006 us, var 0.005 | 1.562 us +/- 0.009 us, var 0.003 | 0.66x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.cat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 37.623 us +/- 0.140 us, var 0.124 | 2.630 us +/- 0.036 us, var 0.007 | 14.30x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.cat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 90.031 us +/- 0.365 us, var 7.387 | 7.439 us +/- 0.058 us, var 0.065 | 12.10x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.cat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.745 us +/- 0.075 us, var 0.049 | 2.020 us +/- 0.015 us, var 0.001 | 2.84x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.cat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.241 us +/- 0.013 us, var 0.003 | 1.951 us +/- 0.014 us, var 0.002 | 0.64x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 3.773 us +/- 0.055 us, var 0.006 | 3.160 us +/- 0.034 us, var 0.003 | 1.19x | `4113829215800195259`/`4113829215800195259` |
| `singleton_contiguous_8192` | singleton | `torch.concat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 37.541 us +/- 0.159 us, var 0.172 | 2.526 us +/- 0.034 us, var 0.014 | 14.86x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.876 us +/- 0.044 us, var 0.031 | 2.152 us +/- 0.013 us, var 0.011 | 2.27x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 8.330 us +/- 0.138 us, var 0.364 | 3.514 us +/- 0.030 us, var 0.088 | 2.37x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.019 us +/- 0.005 us, var 0.003 | 1.584 us +/- 0.009 us, var 0.002 | 0.64x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 37.789 us +/- 0.196 us, var 2.048 | 2.724 us +/- 0.047 us, var 0.024 | 13.87x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 91.121 us +/- 0.509 us, var 1.421 | 7.498 us +/- 0.047 us, var 0.045 | 12.15x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.712 us +/- 0.033 us, var 0.032 | 2.054 us +/- 0.012 us, var 0.002 | 2.78x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.234 us +/- 0.008 us, var 0.001 | 1.984 us +/- 0.017 us, var 0.001 | 0.62x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 3.704 us +/- 0.024 us, var 0.019 | 3.185 us +/- 0.026 us, var 0.003 | 1.16x | `4113829215800195259`/`4113829215800195259` |
| `singleton_contiguous_8192` | singleton | `torch.concatenate` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 37.667 us +/- 0.123 us, var 0.402 | 2.529 us +/- 0.019 us, var 0.003 | 14.89x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concatenate` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.858 us +/- 0.029 us, var 0.008 | 2.145 us +/- 0.022 us, var 0.008 | 2.26x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concatenate` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 8.680 us +/- 0.219 us, var 0.257 | 3.576 us +/- 0.017 us, var 0.002 | 2.43x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concatenate` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.024 us +/- 0.006 us, var 0.002 | 1.588 us +/- 0.007 us, var 0.000 | 0.65x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concatenate` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 37.712 us +/- 0.257 us, var 2.687 | 2.727 us +/- 0.046 us, var 0.005 | 13.83x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concatenate` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 94.278 us +/- 0.709 us, var 2.395 | 7.589 us +/- 0.088 us, var 0.185 | 12.42x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concatenate` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.805 us +/- 0.041 us, var 0.024 | 2.016 us +/- 0.013 us, var 0.003 | 2.88x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concatenate` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.237 us +/- 0.006 us, var 0.000 | 1.976 us +/- 0.011 us, var 0.001 | 0.63x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 3.789 us +/- 0.096 us, var 0.017 | 3.125 us +/- 0.018 us, var 0.004 | 1.21x | `4113829215800195259`/`4113829215800195259` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent PyTorch operation. They are preserved as zero-credit cells instead of being removed from the evidence set.

| Workload | Category | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `top_level_torch_cat_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_cat_active_autograd_1d` | active autograd | `RuntimeError: cat(): autograd recording is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False` | zero |
| `top_level_torch_cat_rank2_dim0` | general-dimensional cat | `NotImplementedError: cat(): only exact native CPU float32 1-D Tensor inputs are supported` | `supported (3, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_cat_rank2_dim1` | general-dimensional cat | `NotImplementedError: cat(): only exact native CPU float32 1-D Tensor inputs are supported` | `supported (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_active_autograd_1d` | active autograd | `RuntimeError: cat(): autograd recording is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False` | zero |
| `top_level_torch_concat_rank2_dim0` | general-dimensional cat | `NotImplementedError: cat(): only exact native CPU float32 1-D Tensor inputs are supported` | `supported (3, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_rank2_dim1` | general-dimensional cat | `NotImplementedError: cat(): only exact native CPU float32 1-D Tensor inputs are supported` | `supported (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_active_autograd_1d` | active autograd | `RuntimeError: cat(): autograd recording is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False` | zero |
| `top_level_torch_concatenate_rank2_dim0` | general-dimensional cat | `NotImplementedError: cat(): only exact native CPU float32 1-D Tensor inputs are supported` | `supported (3, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_rank2_dim1` | general-dimensional cat | `NotImplementedError: cat(): only exact native CPU float32 1-D Tensor inputs are supported` | `supported (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
