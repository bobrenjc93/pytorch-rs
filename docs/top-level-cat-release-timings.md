# `torch.cat`, `torch.concat`, and `torch.concatenate` Supported CPU Release Timings

Date: 2026-09-07

Benchmark artifact base head:
`a7f1caf00d2018349143aa3538fcb0fc8340ca82`. The benchmark evidence was
refreshed after integration so rank-1, rank-2, and backward `cat` cells are
timed, while true remaining gaps are retained as zero-credit unsupported cells.
The raw JSON records git provenance, including the worktree diff visible to the
timing driver. It does not claim to be measured from a committed exact-head
state; Burner creates that commit after review.

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
`docs/benchmark-data/top-level-cat-release-timings.json` in 47.08 seconds with two
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
  `cffc49c0ac74b1a206b07475cceab64cb588d2b060e6c0040e4bdb38b949c1c5`
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
- Build: successful editable release extension rebuild before timing

Inputs are created outside timed regions from deterministic CPU `float32`
values with fixed NumPy seeds recorded per workload in the JSON artifact. Every
supported cell first compares `torch_rs` against PyTorch for shape, stride,
storage offset, contiguity, dtype, device, `requires_grad`, leaf status, and
exact logical value bits. Active-autograd cells time forward graph construction
only; backward cells create fresh leaves per timed repeat, run a scalar sum
backward pass, and materialize the output plus leaf gradients. No-grad cells use
pre-created `requires_grad=True` leaves and require fresh `requires_grad=False`
leaf outputs. Every warmup and measured block materializes its final bundle as a
64-bit BLAKE2b checksum over tensor metadata and logical bytes. The artifact
validates stable equal checksum sets for `torch_rs` and PyTorch.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/top-level-cat-release-timings.json`
- Benchmark: `top_level_cat_cpu_supported_benchmark_v2`
- Timed supported cells: 42 (3 APIs x 14 workload shapes and modes)
- Zero-credit unsupported cells: 9
- Implementation orders: torch_rs then pytorch, pytorch then torch_rs; each implementation appears once before and once after the other implementation
- Warmup and sampling: 15 untimed warmup blocks and 81 measured blocks per implementation pass
- CPU affinity: selected CPU 24, pinned affinity [24]; threads=1
- All supported cells: 0.59x uncapped, 0.59x capped
- `torch.cat` cells: 0.60x uncapped, 0.60x capped
- `torch.concat` cells: 0.59x uncapped, 0.59x capped
- `torch.concatenate` cells: 0.59x uncapped, 0.59x capped
- Singleton cells: 0.74x uncapped, 0.74x capped
- Multi-input cells: 0.62x uncapped, 0.62x capped
- Empty-operand cells: 0.52x uncapped, 0.52x capped
- Offset cells: 0.70x uncapped, 0.70x capped
- Noncontiguous cells: 1.10x uncapped, 1.10x capped
- Axis-keyword cells: 0.56x uncapped, 0.56x capped
- Active-autograd cells: 0.63x uncapped, 0.63x capped
- `no_grad` cells: 0.47x uncapped, 0.47x capped
- Rank-2 cells: 0.74x uncapped, 0.74x capped
- Backward cells: 0.36x uncapped, 0.36x capped

Unsupported cells below are retained as feature/API coverage evidence and are not included in the performance aggregates.

## Supported Timed Cells

| Workload | Category | API | Call form | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `singleton_contiguous_8192` | singleton | `torch.cat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.841 us +/- 0.040 us, var 0.028 | 2.469 us +/- 0.029 us, var 0.047 | 0.75x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.cat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.294 us +/- 0.023 us, var 0.026 | 2.130 us +/- 0.017 us, var 0.004 | 0.61x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.cat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.331 us +/- 0.041 us, var 0.047 | 3.446 us +/- 0.028 us, var 0.116 | 0.39x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.cat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.109 us +/- 0.007 us, var 0.011 | 1.564 us +/- 0.011 us, var 0.008 | 0.71x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.cat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.884 us +/- 0.021 us, var 0.017 | 2.655 us +/- 0.035 us, var 0.041 | 0.71x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.cat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 8.167 us +/- 0.043 us, var 1.891 | 7.409 us +/- 0.033 us, var 0.030 | 1.10x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.cat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.305 us +/- 0.020 us, var 0.039 | 2.048 us +/- 0.018 us, var 0.053 | 0.64x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.cat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.085 us +/- 0.007 us, var 0.016 | 1.932 us +/- 0.027 us, var 0.054 | 0.56x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.487 us +/- 0.012 us, var 0.024 | 3.119 us +/- 0.018 us, var 0.101 | 0.48x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 1.508 us +/- 0.017 us, var 0.002 | 2.400 us +/- 0.032 us, var 0.125 | 0.63x | `14264808484827237231`/`14264808484827237231` |
| `rank2_dim0_generated_23_19x37` | rank-2 | `torch.cat` | list dim=0 | held-out generated rank-2 inputs with shapes (23, 37) and (19, 37) | rank-2 row concatenation output; (42, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.370 us +/- 0.044 us, var 0.042 | 2.130 us +/- 0.037 us, var 0.105 | 0.64x | `15329540218385864427`/`15329540218385864427` |
| `rank2_dim1_generated_31x17_13_11` | rank-2 | `torch.cat` | list dim=1 | held-out generated rank-2 inputs with shapes (31, 17), (31, 13), and (31, 11) | rank-2 column concatenation output; (31, 41), stride (41, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 2.289 us +/- 0.042 us, var 0.130 | 2.426 us +/- 0.034 us, var 0.004 | 0.94x | `12710136227280332661`/`12710136227280332661` |
| `backward_rank1_generated_113_127` | backward | `torch.cat` | list dim=0 | held-out generated grad-requiring 1-D inputs with lengths 113 and 127; sum backward | forward output plus leaf gradients; (240,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 54.778 us +/- 0.616 us, var 17.336 | 152.986 us +/- 1.635 us, var 53.752 | 0.36x | `11635413765118461767`/`11635413765118461767` |
| `backward_rank2_dim1_generated_7x11_5` | backward | `torch.cat` | list dim=1 | held-out generated grad-requiring rank-2 inputs with shapes (7, 11) and (7, 5); sum backward | forward output plus leaf gradients; (7, 16), stride (16, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 55.699 us +/- 1.090 us, var 36.606 | 150.787 us +/- 1.620 us, var 27.300 | 0.37x | `14564349192754315822`/`14564349192754315822` |
| `singleton_contiguous_8192` | singleton | `torch.concat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.879 us +/- 0.027 us, var 0.003 | 2.547 us +/- 0.043 us, var 0.097 | 0.74x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.338 us +/- 0.022 us, var 0.016 | 2.153 us +/- 0.015 us, var 0.004 | 0.62x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.403 us +/- 0.017 us, var 0.005 | 3.629 us +/- 0.026 us, var 0.011 | 0.39x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.103 us +/- 0.007 us, var 0.000 | 1.629 us +/- 0.010 us, var 0.003 | 0.68x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.863 us +/- 0.032 us, var 0.006 | 2.711 us +/- 0.043 us, var 0.012 | 0.69x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 8.170 us +/- 0.062 us, var 0.108 | 7.502 us +/- 0.031 us, var 0.148 | 1.09x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.312 us +/- 0.027 us, var 0.003 | 2.054 us +/- 0.016 us, var 0.003 | 0.64x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.100 us +/- 0.007 us, var 0.000 | 1.972 us +/- 0.012 us, var 0.000 | 0.56x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.469 us +/- 0.017 us, var 0.001 | 3.215 us +/- 0.036 us, var 0.004 | 0.46x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 1.556 us +/- 0.013 us, var 0.001 | 2.457 us +/- 0.028 us, var 0.073 | 0.63x | `14264808484827237231`/`14264808484827237231` |
| `rank2_dim0_generated_23_19x37` | rank-2 | `torch.concat` | list dim=0 | held-out generated rank-2 inputs with shapes (23, 37) and (19, 37) | rank-2 row concatenation output; (42, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.313 us +/- 0.015 us, var 0.004 | 2.149 us +/- 0.032 us, var 0.021 | 0.61x | `15329540218385864427`/`15329540218385864427` |
| `rank2_dim1_generated_31x17_13_11` | rank-2 | `torch.concat` | list dim=1 | held-out generated rank-2 inputs with shapes (31, 17), (31, 13), and (31, 11) | rank-2 column concatenation output; (31, 41), stride (41, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 2.289 us +/- 0.042 us, var 0.013 | 2.524 us +/- 0.034 us, var 0.064 | 0.91x | `12710136227280332661`/`12710136227280332661` |
| `backward_rank1_generated_113_127` | backward | `torch.concat` | list dim=0 | held-out generated grad-requiring 1-D inputs with lengths 113 and 127; sum backward | forward output plus leaf gradients; (240,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 55.543 us +/- 0.838 us, var 70.599 | 153.150 us +/- 1.126 us, var 4.870 | 0.36x | `11635413765118461767`/`11635413765118461767` |
| `backward_rank2_dim1_generated_7x11_5` | backward | `torch.concat` | list dim=1 | held-out generated grad-requiring rank-2 inputs with shapes (7, 11) and (7, 5); sum backward | forward output plus leaf gradients; (7, 16), stride (16, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 55.274 us +/- 0.477 us, var 1.308 | 149.949 us +/- 1.208 us, var 5.647 | 0.37x | `14564349192754315822`/`14564349192754315822` |
| `singleton_contiguous_8192` | singleton | `torch.concatenate` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.824 us +/- 0.035 us, var 0.005 | 2.519 us +/- 0.033 us, var 0.011 | 0.72x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concatenate` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.261 us +/- 0.012 us, var 0.001 | 2.164 us +/- 0.016 us, var 0.001 | 0.58x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concatenate` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.351 us +/- 0.025 us, var 0.002 | 3.543 us +/- 0.022 us, var 0.003 | 0.38x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concatenate` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.106 us +/- 0.007 us, var 0.000 | 1.577 us +/- 0.014 us, var 0.013 | 0.70x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concatenate` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.854 us +/- 0.030 us, var 0.078 | 2.688 us +/- 0.033 us, var 0.014 | 0.69x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concatenate` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 8.206 us +/- 0.048 us, var 0.117 | 7.479 us +/- 0.047 us, var 0.504 | 1.10x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concatenate` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.268 us +/- 0.010 us, var 0.003 | 2.042 us +/- 0.013 us, var 0.001 | 0.62x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concatenate` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.100 us +/- 0.008 us, var 0.002 | 1.944 us +/- 0.008 us, var 0.006 | 0.57x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.480 us +/- 0.026 us, var 0.006 | 3.122 us +/- 0.020 us, var 0.002 | 0.47x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 1.581 us +/- 0.013 us, var 0.001 | 2.534 us +/- 0.021 us, var 0.006 | 0.62x | `14264808484827237231`/`14264808484827237231` |
| `rank2_dim0_generated_23_19x37` | rank-2 | `torch.concatenate` | list dim=0 | held-out generated rank-2 inputs with shapes (23, 37) and (19, 37) | rank-2 row concatenation output; (42, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.306 us +/- 0.034 us, var 0.003 | 2.418 us +/- 0.199 us, var 0.045 | 0.54x | `15329540218385864427`/`15329540218385864427` |
| `rank2_dim1_generated_31x17_13_11` | rank-2 | `torch.concatenate` | list dim=1 | held-out generated rank-2 inputs with shapes (31, 17), (31, 13), and (31, 11) | rank-2 column concatenation output; (31, 41), stride (41, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 2.279 us +/- 0.028 us, var 0.004 | 2.479 us +/- 0.030 us, var 0.003 | 0.92x | `12710136227280332661`/`12710136227280332661` |
| `backward_rank1_generated_113_127` | backward | `torch.concatenate` | list dim=0 | held-out generated grad-requiring 1-D inputs with lengths 113 and 127; sum backward | forward output plus leaf gradients; (240,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 54.833 us +/- 0.646 us, var 1.310 | 152.832 us +/- 1.086 us, var 55.837 | 0.36x | `11635413765118461767`/`11635413765118461767` |
| `backward_rank2_dim1_generated_7x11_5` | backward | `torch.concatenate` | list dim=1 | held-out generated grad-requiring rank-2 inputs with shapes (7, 11) and (7, 5); sum backward | forward output plus leaf gradients; (7, 16), stride (16, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 54.639 us +/- 0.469 us, var 1.974 | 149.565 us +/- 1.143 us, var 9.435 | 0.37x | `14564349192754315822`/`14564349192754315822` |

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
