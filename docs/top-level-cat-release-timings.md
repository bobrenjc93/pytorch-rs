# `torch.cat`, `torch.concat`, and `torch.concatenate` 1-D Release Timings

Date: 2026-09-07

Candidate provenance: current composite worktree snapshot after integrating the
1-D `torch.cat` benchmark evidence with rank-2 and active-autograd `cat`
support. The raw JSON records git provenance, including the worktree diff
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
`docs/benchmark-data/top-level-cat-release-timings.json` in 53.40 seconds with two
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
  `fd4d39df5f15b86e48f01f5577d1eef8d61b34e593a4d4676b82692065d1c916`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32
- CPU affinity: selected CPU 24
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 27 ms, prepared
  packages in 15.51s, and installed in 2.29s
- Build: successful editable release extension rebuild before timing; Cargo
  reported the release profile finished in 38.75s

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
- Benchmark: `top_level_cat_cpu_1d_benchmark_v2`
- Timed supported cells: 30 (3 APIs x 10 workload shapes and modes)
- Zero-credit unsupported cells: 9
- Implementation orders: torch_rs then pytorch, pytorch then torch_rs; each implementation appears once before and once after the other implementation
- Warmup and sampling: 15 untimed warmup blocks and 81 measured blocks per implementation pass
- CPU affinity: selected CPU 24, pinned affinity [24]; threads=1
- All supported cells: 2.74x uncapped, 2.52x capped
- `torch.cat` cells: 2.76x uncapped, 2.54x capped
- `torch.concat` cells: 2.73x uncapped, 2.51x capped
- `torch.concatenate` cells: 2.73x uncapped, 2.51x capped
- Singleton cells: 14.18x uncapped, 10.00x capped
- Multi-input cells: 2.42x uncapped, 2.42x capped
- Empty-operand cells: 1.24x uncapped, 1.24x capped
- Offset cells: 13.03x uncapped, 10.00x capped
- Noncontiguous cells: 12.41x uncapped, 10.00x capped
- Axis-keyword cells: 0.64x uncapped, 0.64x capped
- `no_grad` cells: 1.17x uncapped, 1.17x capped
- Active-autograd cells: 1.54x uncapped, 1.54x capped

Including the unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 3.47x.

## Supported Timed Cells

| Workload | Category | API | Call form | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `singleton_contiguous_8192` | singleton | `torch.cat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 35.100 us +/- 0.379 us, var 1.365 | 2.440 us +/- 0.032 us, var 0.007 | 14.39x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.cat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.525 us +/- 0.028 us, var 0.003 | 2.105 us +/- 0.018 us, var 0.005 | 2.15x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.cat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 7.632 us +/- 0.040 us, var 0.015 | 3.426 us +/- 0.024 us, var 0.005 | 2.23x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.cat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.087 us +/- 0.009 us, var 0.000 | 1.522 us +/- 0.009 us, var 0.005 | 0.71x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.cat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 33.333 us +/- 0.294 us, var 0.601 | 2.614 us +/- 0.051 us, var 0.034 | 12.75x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.cat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 92.325 us +/- 0.563 us, var 3.221 | 7.394 us +/- 0.050 us, var 0.037 | 12.49x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.cat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.467 us +/- 0.029 us, var 0.009 | 1.975 us +/- 0.014 us, var 0.001 | 2.77x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.cat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.259 us +/- 0.012 us, var 0.001 | 1.915 us +/- 0.014 us, var 0.004 | 0.66x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 3.607 us +/- 0.029 us, var 0.006 | 3.129 us +/- 0.045 us, var 0.047 | 1.15x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_grad_inputs_257_263` | active autograd | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 in grad mode | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 1024 | 3.655 us +/- 0.035 us, var 0.006 | 2.324 us +/- 0.030 us, var 0.019 | 1.57x | `14264808484827237231`/`14264808484827237231` |
| `singleton_contiguous_8192` | singleton | `torch.concat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 35.297 us +/- 0.193 us, var 0.349 | 2.516 us +/- 0.024 us, var 0.002 | 14.03x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.568 us +/- 0.035 us, var 0.054 | 2.104 us +/- 0.011 us, var 0.003 | 2.17x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 7.619 us +/- 0.040 us, var 0.091 | 3.501 us +/- 0.025 us, var 0.007 | 2.18x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.078 us +/- 0.007 us, var 0.000 | 1.553 us +/- 0.009 us, var 0.002 | 0.69x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 35.172 us +/- 0.123 us, var 0.137 | 2.668 us +/- 0.026 us, var 0.007 | 13.18x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 91.939 us +/- 0.268 us, var 2.309 | 7.438 us +/- 0.033 us, var 0.032 | 12.36x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.497 us +/- 0.024 us, var 0.013 | 2.042 us +/- 0.012 us, var 0.001 | 2.69x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.260 us +/- 0.008 us, var 0.000 | 1.975 us +/- 0.013 us, var 0.004 | 0.64x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 3.655 us +/- 0.022 us, var 0.007 | 3.105 us +/- 0.021 us, var 0.009 | 1.18x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_grad_inputs_257_263` | active autograd | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 in grad mode | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 1024 | 3.646 us +/- 0.019 us, var 0.006 | 2.406 us +/- 0.033 us, var 0.021 | 1.52x | `14264808484827237231`/`14264808484827237231` |
| `singleton_contiguous_8192` | singleton | `torch.concatenate` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 35.151 us +/- 0.149 us, var 0.886 | 2.488 us +/- 0.023 us, var 0.010 | 14.13x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concatenate` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.564 us +/- 0.030 us, var 0.007 | 2.131 us +/- 0.013 us, var 0.002 | 2.14x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concatenate` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 7.464 us +/- 0.030 us, var 0.044 | 3.468 us +/- 0.020 us, var 0.009 | 2.15x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concatenate` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.076 us +/- 0.006 us, var 0.004 | 1.546 us +/- 0.007 us, var 0.000 | 0.70x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concatenate` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 35.384 us +/- 0.159 us, var 0.488 | 2.687 us +/- 0.054 us, var 0.008 | 13.17x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concatenate` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 92.115 us +/- 0.326 us, var 1.592 | 7.444 us +/- 0.040 us, var 0.077 | 12.37x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concatenate` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.516 us +/- 0.044 us, var 0.041 | 2.025 us +/- 0.020 us, var 0.034 | 2.72x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concatenate` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.254 us +/- 0.005 us, var 0.000 | 1.985 us +/- 0.012 us, var 0.004 | 0.63x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 3.679 us +/- 0.032 us, var 0.010 | 3.113 us +/- 0.018 us, var 0.004 | 1.18x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_grad_inputs_257_263` | active autograd | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 in grad mode | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 1024 | 3.642 us +/- 0.021 us, var 0.051 | 2.381 us +/- 0.035 us, var 0.008 | 1.53x | `14264808484827237231`/`14264808484827237231` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent PyTorch operation. They are preserved as zero-credit cells instead of being removed from the evidence set.

| Workload | Category | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `top_level_torch_cat_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_cat_rank3_dim0` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 2, 3), stride (6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_cat_rank3_dim2` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 3), stride (9, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_rank3_dim0` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 2, 3), stride (6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_rank3_dim2` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 3), stride (9, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_rank3_dim0` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 2, 3), stride (6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_rank3_dim2` | higher-rank cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 3), stride (9, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
