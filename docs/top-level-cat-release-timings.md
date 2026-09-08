# `torch.cat`, `torch.concat`, and `torch.concatenate` 1-D Release Timings

Date: 2026-09-07

Candidate provenance: current composite worktree head
`67cf36b1eeed9c2711643f6174e6eabc2047f535`. The benchmark evidence was
refreshed after integration so active-autograd 1-D `cat` is timed and only true
remaining gaps are retained as zero-credit unsupported cells. The raw JSON
records clean git provenance, including empty `status_short` and `diff_stat`
fields captured before the artifact write.

Exact build, timing, summary, and check commands were run from the repository
root. The benchmark used the worktree-local `.venv` with pinned PyTorch 2.13.0
and did not install packages outside the worktree. The benchmark driver set
`CUDA_VISIBLE_DEVICES=` so this CPU-only benchmark did not select the host GPUs.

```bash
env -u CONDA_PREFIX \
  TMPDIR="$PWD/target" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
env PYTHONNOUSERSITE=1 \
  TMPDIR="$PWD/target" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/python scripts/benchmark_top_level_cat.py \
  --cpu 24 --threads 1 \
  --output docs/benchmark-data/top-level-cat-release-timings.json
env PYTHONNOUSERSITE=1 \
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
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --all-targets
git diff --check
```

Results: the checked-in driver produced
`docs/benchmark-data/top-level-cat-release-timings.json` in 53.89 seconds with two
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
  `980f4d6686c86be82b7fb684753ed1f233f69f4017cec9546c6b6b058ee6a8b8`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32
- CPU affinity: selected CPU 24, pinned affinity `[24]`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: reused the locked worktree-local `.venv`; no package
  installation outside the worktree was performed for this refresh
- Build: successful editable release extension rebuild before timing; Cargo
  reported the release profile finished in 0.01s

Inputs are created outside timed regions from deterministic CPU `float32`
values with fixed NumPy seeds recorded per workload in the JSON artifact. Every
supported cell first compares `torch_rs` against PyTorch for shape, stride,
storage offset, contiguity, dtype, device, `requires_grad`, leaf status, and
exact logical value bits. Active-autograd cells time forward graph construction
only; no-grad cells use pre-created `requires_grad=True` leaves and require
fresh `requires_grad=False` leaf outputs. Every warmup and measured block
materializes its final output as a 64-bit BLAKE2b checksum over output metadata
and logical bytes. The artifact validates stable equal checksum sets for
`torch_rs` and PyTorch.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/top-level-cat-release-timings.json`
- Benchmark: `top_level_cat_cpu_1d_benchmark_v1`
- Timed supported cells: 30 (3 APIs x 10 workload shapes and modes)
- Zero-credit unsupported cells: 9
- Implementation orders: torch_rs then pytorch, pytorch then torch_rs; each implementation appears once before and once after the other implementation
- Warmup and sampling: 15 untimed warmup blocks and 81 measured blocks per implementation pass
- CPU affinity: selected CPU 24, pinned affinity [24]; threads=1
- All supported cells: 2.92x uncapped, 2.65x capped
- `torch.cat` cells: 2.92x uncapped, 2.68x capped
- `torch.concat` cells: 2.89x uncapped, 2.62x capped
- `torch.concatenate` cells: 2.95x uncapped, 2.66x capped
- Singleton cells: 15.53x uncapped, 10.00x capped
- Multi-input cells: 2.63x uncapped, 2.63x capped
- Empty-operand cells: 1.32x uncapped, 1.32x capped
- Offset cells: 14.70x uncapped, 10.00x capped
- Noncontiguous cells: 11.28x uncapped, 9.83x capped
- Axis-keyword cells: 0.69x uncapped, 0.69x capped
- Active-autograd cells: 1.57x uncapped, 1.57x capped
- `no_grad` cells: 1.36x uncapped, 1.36x capped

Including the unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 3.60x.

## Supported Timed Cells

| Workload | Category | API | Call form | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `singleton_contiguous_8192` | singleton | `torch.cat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 38.261 us +/- 0.249 us, var 0.770 | 2.403 us +/- 0.031 us, var 0.003 | 15.92x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.cat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.932 us +/- 0.036 us, var 0.008 | 2.054 us +/- 0.015 us, var 0.002 | 2.40x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.cat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 8.290 us +/- 0.087 us, var 0.106 | 3.446 us +/- 0.027 us, var 0.006 | 2.41x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.cat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.149 us +/- 0.007 us, var 0.001 | 1.509 us +/- 0.008 us, var 0.001 | 0.76x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.cat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 38.313 us +/- 0.440 us, var 0.896 | 2.559 us +/- 0.028 us, var 0.013 | 14.97x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.cat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 87.927 us +/- 0.707 us, var 165.090 | 9.262 us +/- 1.135 us, var 3.817 | 9.49x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.cat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.891 us +/- 0.115 us, var 0.041 | 1.995 us +/- 0.015 us, var 0.003 | 2.95x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.cat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.330 us +/- 0.009 us, var 0.001 | 1.893 us +/- 0.014 us, var 0.001 | 0.70x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.201 us +/- 0.052 us, var 0.015 | 3.044 us +/- 0.024 us, var 0.088 | 1.38x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 3.889 us +/- 0.023 us, var 0.004 | 2.451 us +/- 0.019 us, var 0.002 | 1.59x | `14264808484827237231`/`14264808484827237231` |
| `singleton_contiguous_8192` | singleton | `torch.concat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 38.113 us +/- 0.240 us, var 1.686 | 2.481 us +/- 0.026 us, var 0.004 | 15.36x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.942 us +/- 0.040 us, var 0.021 | 2.120 us +/- 0.015 us, var 0.005 | 2.33x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 8.267 us +/- 0.063 us, var 0.043 | 3.638 us +/- 0.025 us, var 0.022 | 2.27x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.137 us +/- 0.008 us, var 0.000 | 1.553 us +/- 0.013 us, var 0.001 | 0.73x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 38.795 us +/- 0.571 us, var 1.441 | 2.672 us +/- 0.041 us, var 0.046 | 14.52x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 88.492 us +/- 0.805 us, var 4.102 | 7.442 us +/- 0.033 us, var 0.087 | 11.89x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.892 us +/- 0.168 us, var 0.052 | 2.021 us +/- 0.017 us, var 0.002 | 2.92x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.303 us +/- 0.010 us, var 0.000 | 1.975 us +/- 0.053 us, var 0.164 | 0.66x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.170 us +/- 0.081 us, var 0.034 | 3.113 us +/- 0.026 us, var 0.022 | 1.34x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 3.797 us +/- 0.061 us, var 0.015 | 2.456 us +/- 0.021 us, var 0.005 | 1.55x | `14264808484827237231`/`14264808484827237231` |
| `singleton_contiguous_8192` | singleton | `torch.concatenate` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 38.482 us +/- 0.342 us, var 3.333 | 2.511 us +/- 0.063 us, var 0.261 | 15.32x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concatenate` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.972 us +/- 0.058 us, var 0.016 | 2.119 us +/- 0.025 us, var 0.001 | 2.35x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concatenate` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 8.310 us +/- 0.060 us, var 0.023 | 3.527 us +/- 0.020 us, var 0.012 | 2.36x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concatenate` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.145 us +/- 0.006 us, var 0.002 | 1.545 us +/- 0.008 us, var 0.005 | 0.74x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concatenate` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 38.534 us +/- 0.381 us, var 2.203 | 2.637 us +/- 0.032 us, var 0.007 | 14.61x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concatenate` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 95.141 us +/- 0.686 us, var 1.281 | 7.486 us +/- 0.071 us, var 0.264 | 12.71x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concatenate` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 5.757 us +/- 0.038 us, var 0.029 | 1.987 us +/- 0.014 us, var 0.002 | 2.90x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concatenate` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.339 us +/- 0.009 us, var 0.002 | 1.926 us +/- 0.011 us, var 0.002 | 0.70x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 4.150 us +/- 0.070 us, var 0.014 | 3.079 us +/- 0.019 us, var 0.009 | 1.35x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 3.882 us +/- 0.040 us, var 0.012 | 2.465 us +/- 0.024 us, var 0.077 | 1.57x | `14264808484827237231`/`14264808484827237231` |

## Zero-Credit Unsupported Cells

These cells are not timed because `torch_rs` cannot execute the equivalent PyTorch operation. They are preserved as zero-credit cells instead of being removed from the evidence set.

| Workload | Category | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `top_level_torch_cat_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_cat_rank3_dim0` | rank >=3 cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 2, 3), stride (6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_cat_rank3_dim2` | rank >=3 cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 3), stride (9, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_rank3_dim0` | rank >=3 cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 2, 3), stride (6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concat_rank3_dim2` | rank >=3 cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 3), stride (9, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_out_1d` | concrete out | `RuntimeError: cat(): the 'out' argument is not supported` | `supported (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_rank3_dim0` | rank >=3 cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (3, 2, 3), stride (6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_concatenate_rank3_dim2` | rank >=3 cat | `NotImplementedError: cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported` | `supported (2, 3, 3), stride (9, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
