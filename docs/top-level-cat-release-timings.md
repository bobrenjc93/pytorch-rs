# `torch.cat`, `torch.concat`, and `torch.concatenate` 1-D Release Timings

Date: 2026-09-08

Candidate provenance: current worktree snapshot for the 1-D `torch.cat`
benchmark evidence branch. This evidence update does not change the runtime
implementation. The raw JSON records git provenance, including the worktree diff
visible to the timing driver.

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
`docs/benchmark-data/top-level-cat-release-timings.json` in 42.37 seconds with two
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
  `27eddce29145009dcd2d86387f7e6e362bc40fac09928d5669c58c8ba4c2fc1e`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32
- CPU affinity: selected CPU 24
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 29 ms, prepared
  packages in 16.56s, and installed in 1.32s
- Build: successful editable release extension rebuild before timing; Cargo
  reported the release profile finished in 41.61s

Inputs are created outside timed regions from deterministic CPU `float32`
values with fixed NumPy seeds recorded per workload in the JSON artifact. Every
supported cell first compares `torch_rs` against PyTorch for shape, stride,
storage offset, contiguity, dtype, device, `requires_grad`, leaf status, and
exact logical value bits. Every warmup and measured block materializes its
final output bundle as a 64-bit BLAKE2b checksum over output metadata and
logical bytes. Backward cells include the concatenation output and leaf
gradients in that materialized bundle. The artifact validates stable equal
checksum sets for `torch_rs` and PyTorch.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/top-level-cat-release-timings.json`
- Benchmark: `top_level_cat_cpu_1d_benchmark_v2`
- Timed supported cells: 33 (3 APIs x 11 workload shapes and modes)
- Zero-credit unsupported cells: 9
- Implementation orders: torch_rs then pytorch, pytorch then torch_rs; each implementation appears once before and once after the other implementation
- Warmup and sampling: 15 untimed warmup blocks and 81 measured blocks per implementation pass
- CPU affinity: selected CPU 24, pinned affinity [24]; threads=1
- All supported cells: 0.60x uncapped, 0.60x capped
- `torch.cat` cells: 0.62x uncapped, 0.62x capped
- `torch.concat` cells: 0.60x uncapped, 0.60x capped
- `torch.concatenate` cells: 0.60x uncapped, 0.60x capped
- Singleton cells: 0.76x uncapped, 0.76x capped
- Multi-input cells: 0.61x uncapped, 0.61x capped
- Empty-operand cells: 0.51x uncapped, 0.51x capped
- Offset cells: 0.69x uncapped, 0.69x capped
- Noncontiguous cells: 4.32x uncapped, 4.32x capped
- Axis-keyword cells: 0.55x uncapped, 0.55x capped
- Autograd-forward cells: 0.63x uncapped, 0.63x capped
- Autograd-forward+backward cells: 0.12x uncapped, 0.12x capped
- `no_grad` cells: 0.46x uncapped, 0.46x capped

Including the unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 1.10x.

## Supported Timed Cells

| Workload | Category | API | Call form | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `singleton_contiguous_8192` | singleton | `torch.cat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.827 us +/- 0.036 us, var 0.006 | 2.454 us +/- 0.039 us, var 0.007 | 0.74x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.cat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.310 us +/- 0.013 us, var 0.002 | 2.128 us +/- 0.031 us, var 0.061 | 0.62x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.cat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.326 us +/- 0.029 us, var 0.004 | 3.493 us +/- 0.030 us, var 0.009 | 0.38x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.cat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.095 us +/- 0.006 us, var 0.001 | 1.554 us +/- 0.012 us, var 0.002 | 0.71x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.cat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.918 us +/- 0.020 us, var 0.011 | 2.617 us +/- 0.030 us, var 0.007 | 0.73x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.cat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 32.967 us +/- 0.193 us, var 0.612 | 7.479 us +/- 0.060 us, var 1.155 | 4.41x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.cat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.342 us +/- 0.019 us, var 0.006 | 2.025 us +/- 0.010 us, var 0.001 | 0.66x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.cat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.087 us +/- 0.007 us, var 0.000 | 1.946 us +/- 0.010 us, var 0.003 | 0.56x | `13790662490979461913`/`13790662490979461913` |
| `autograd_forward_repeated_257_263_257` | autograd forward | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263, left repeated; forward construction only | concatenation output; (777,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 1024 | 1.642 us +/- 0.020 us, var 0.001 | 2.539 us +/- 0.022 us, var 0.003 | 0.65x | `2989753682302512681`/`2989753682302512681` |
| `autograd_forward_backward_repeated_32_33_32` | autograd forward+backward | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 32 and 33, left repeated; timed op(...).sum().backward() | concatenation output plus leaf gradients; (97,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 128 | 4.486 us +/- 0.059 us, var 0.041 | 38.767 us +/- 0.540 us, var 1.126 | 0.12x | `10440134875662465031`/`10440134875662465031` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.451 us +/- 0.012 us, var 0.003 | 3.105 us +/- 0.026 us, var 0.008 | 0.47x | `4113829215800195259`/`4113829215800195259` |
| `singleton_contiguous_8192` | singleton | `torch.concat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 2.058 us +/- 0.216 us, var 0.057 | 2.533 us +/- 0.032 us, var 0.018 | 0.81x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.269 us +/- 0.019 us, var 0.001 | 2.185 us +/- 0.019 us, var 0.025 | 0.58x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.355 us +/- 0.029 us, var 0.044 | 3.671 us +/- 0.032 us, var 0.099 | 0.37x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.103 us +/- 0.010 us, var 0.001 | 1.604 us +/- 0.019 us, var 0.004 | 0.69x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.869 us +/- 0.059 us, var 0.069 | 2.815 us +/- 0.087 us, var 0.161 | 0.66x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 32.987 us +/- 0.466 us, var 0.799 | 7.742 us +/- 0.123 us, var 0.292 | 4.26x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.254 us +/- 0.013 us, var 0.001 | 2.106 us +/- 0.025 us, var 0.007 | 0.60x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.111 us +/- 0.012 us, var 0.000 | 2.014 us +/- 0.016 us, var 0.001 | 0.55x | `13790662490979461913`/`13790662490979461913` |
| `autograd_forward_repeated_257_263_257` | autograd forward | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263, left repeated; forward construction only | concatenation output; (777,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 1024 | 1.670 us +/- 0.018 us, var 0.002 | 2.709 us +/- 0.036 us, var 0.003 | 0.62x | `2989753682302512681`/`2989753682302512681` |
| `autograd_forward_backward_repeated_32_33_32` | autograd forward+backward | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 32 and 33, left repeated; timed op(...).sum().backward() | concatenation output plus leaf gradients; (97,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 128 | 4.509 us +/- 0.070 us, var 0.079 | 39.517 us +/- 0.733 us, var 8.435 | 0.11x | `10440134875662465031`/`10440134875662465031` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.456 us +/- 0.012 us, var 0.001 | 3.186 us +/- 0.023 us, var 0.019 | 0.46x | `4113829215800195259`/`4113829215800195259` |
| `singleton_contiguous_8192` | singleton | `torch.concatenate` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.795 us +/- 0.030 us, var 0.002 | 2.503 us +/- 0.025 us, var 0.004 | 0.72x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concatenate` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.253 us +/- 0.012 us, var 0.010 | 2.182 us +/- 0.018 us, var 0.028 | 0.57x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concatenate` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.330 us +/- 0.015 us, var 0.001 | 3.645 us +/- 0.021 us, var 0.007 | 0.36x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concatenate` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.082 us +/- 0.007 us, var 0.000 | 1.590 us +/- 0.007 us, var 0.000 | 0.68x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concatenate` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.851 us +/- 0.014 us, var 0.009 | 2.697 us +/- 0.040 us, var 0.053 | 0.69x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concatenate` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 32.860 us +/- 0.273 us, var 2.157 | 7.630 us +/- 0.057 us, var 0.064 | 4.31x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concatenate` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.282 us +/- 0.046 us, var 0.069 | 2.088 us +/- 0.014 us, var 0.000 | 0.61x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concatenate` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.094 us +/- 0.006 us, var 0.001 | 1.981 us +/- 0.012 us, var 0.007 | 0.55x | `13790662490979461913`/`13790662490979461913` |
| `autograd_forward_repeated_257_263_257` | autograd forward | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263, left repeated; forward construction only | concatenation output; (777,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 1024 | 1.649 us +/- 0.018 us, var 0.001 | 2.691 us +/- 0.020 us, var 0.004 | 0.61x | `2989753682302512681`/`2989753682302512681` |
| `autograd_forward_backward_repeated_32_33_32` | autograd forward+backward | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 32 and 33, left repeated; timed op(...).sum().backward() | concatenation output plus leaf gradients; (97,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 128 | 4.498 us +/- 0.060 us, var 0.046 | 38.942 us +/- 0.574 us, var 140.558 | 0.12x | `10440134875662465031`/`10440134875662465031` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.454 us +/- 0.012 us, var 0.002 | 3.140 us +/- 0.015 us, var 0.006 | 0.46x | `4113829215800195259`/`4113829215800195259` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent PyTorch operation. They are preserved as zero-credit cells instead of being removed from the evidence set.

| Workload | Category | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `top_level_torch_cat_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_cat_rank3_dim0` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 3, 4), stride (12, 4, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_cat_rank3_dim2` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 5), stride (15, 5, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_rank3_dim0` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 3, 4), stride (12, 4, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_rank3_dim2` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 5), stride (15, 5, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_rank3_dim0` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 3, 4), stride (12, 4, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_rank3_dim2` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 5), stride (15, 5, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
