# `torch.cat`, `torch.concat`, and `torch.concatenate` Supported CPU Release Timings

Date: 2026-09-07

Benchmark artifact exact head:
`0e7cc2ddc2de24156096b4f6b874264906c549c5`. The benchmark evidence was refreshed from a clean composite worktree so
rank-1, rank-2, and backward `cat` cells are timed, while true remaining gaps
are retained as zero-credit unsupported cells. The raw JSON records clean git
provenance with empty `status_short` and `diff_stat` fields captured before the
artifact write.

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
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_autograd.AutogradApiTests.test_set_grad_enabled_decorator_uses_subclass_clone \
  tests.test_autograd.AutogradReferenceTests.test_set_grad_enabled_subclass_decorator_matches_pytorch_2_13 \
  tests.test_no_grad_namespace.NoGradNamespaceTests.test_aliases_preserve_context_decorator_generator_and_thread_behavior
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --all-targets
git diff --check
```

Results: the checked-in driver produced
`docs/benchmark-data/top-level-cat-release-timings.json` in 48.17 seconds with two
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
  `8ff32a313e386cccd0a5356753cc642dfb73831008670def5f266c0b896881b1`
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
only; backward cells pre-create the exact timed operands outside the measured
region, checksum those operands before and after each timed block, run a scalar
sum backward pass, and materialize the output plus leaf gradients. No-grad cells
use pre-created `requires_grad=True` leaves and require fresh
`requires_grad=False` leaf outputs. Every warmup and measured block materializes
its final bundle as a 64-bit BLAKE2b checksum over tensor metadata and logical
bytes. The artifact validates stable equal checksum sets for `torch_rs` and
PyTorch.

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
- All supported cells: 0.54x uncapped, 0.54x capped
- `torch.cat` cells: 0.55x uncapped, 0.55x capped
- `torch.concat` cells: 0.53x uncapped, 0.53x capped
- `torch.concatenate` cells: 0.53x uncapped, 0.53x capped
- Singleton cells: 0.75x uncapped, 0.75x capped
- Multi-input cells: 0.61x uncapped, 0.61x capped
- Empty-operand cells: 0.51x uncapped, 0.51x capped
- Offset cells: 0.69x uncapped, 0.69x capped
- Noncontiguous cells: 1.09x uncapped, 1.09x capped
- Axis-keyword cells: 0.57x uncapped, 0.57x capped
- Active-autograd cells: 0.63x uncapped, 0.63x capped
- `no_grad` cells: 0.48x uncapped, 0.48x capped
- Rank-2 cells: 0.76x uncapped, 0.76x capped
- Backward cells: 0.17x uncapped, 0.17x capped

Unsupported cells below are retained as feature/API coverage evidence and are not included in the performance aggregates.

## Supported Timed Cells

| Workload | Category | API | Call form | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `singleton_contiguous_8192` | singleton | `torch.cat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.875 us +/- 0.040 us, var 0.005 | 2.437 us +/- 0.031 us, var 0.006 | 0.77x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.cat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.274 us +/- 0.008 us, var 0.002 | 2.043 us +/- 0.012 us, var 0.000 | 0.62x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.cat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.327 us +/- 0.017 us, var 0.017 | 3.429 us +/- 0.027 us, var 0.009 | 0.39x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.cat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.069 us +/- 0.005 us, var 0.003 | 1.499 us +/- 0.008 us, var 0.000 | 0.71x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.cat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.854 us +/- 0.022 us, var 0.004 | 2.613 us +/- 0.048 us, var 0.006 | 0.71x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.cat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 8.136 us +/- 0.038 us, var 0.024 | 7.409 us +/- 0.045 us, var 0.064 | 1.10x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.cat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.279 us +/- 0.026 us, var 0.001 | 1.963 us +/- 0.020 us, var 0.005 | 0.65x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.cat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.071 us +/- 0.005 us, var 0.000 | 1.866 us +/- 0.011 us, var 0.001 | 0.57x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.488 us +/- 0.012 us, var 0.001 | 3.055 us +/- 0.021 us, var 0.008 | 0.49x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.cat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 1.476 us +/- 0.017 us, var 0.003 | 2.319 us +/- 0.027 us, var 0.003 | 0.64x | `14264808484827237231`/`14264808484827237231` |
| `rank2_dim0_generated_23_19x37` | rank-2 | `torch.cat` | list dim=0 | held-out generated rank-2 inputs with shapes (23, 37) and (19, 37) | rank-2 row concatenation output; (42, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.321 us +/- 0.017 us, var 0.003 | 2.039 us +/- 0.028 us, var 0.004 | 0.65x | `15329540218385864427`/`15329540218385864427` |
| `rank2_dim1_generated_31x17_13_11` | rank-2 | `torch.cat` | list dim=1 | held-out generated rank-2 inputs with shapes (31, 17), (31, 13), and (31, 11) | rank-2 column concatenation output; (31, 41), stride (41, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 2.304 us +/- 0.026 us, var 0.015 | 2.371 us +/- 0.033 us, var 0.016 | 0.97x | `12710136227280332661`/`12710136227280332661` |
| `backward_rank1_generated_113_127` | backward | `torch.cat` | list dim=0 | held-out generated grad-requiring 1-D inputs with lengths 113 and 127; sum backward | forward output plus leaf gradients; (240,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 6.476 us +/- 0.128 us, var 0.039 | 38.072 us +/- 0.449 us, var 1.375 | 0.17x | `11635413765118461767`/`11635413765118461767` |
| `backward_rank2_dim1_generated_7x11_5` | backward | `torch.cat` | list dim=1 | held-out generated grad-requiring rank-2 inputs with shapes (7, 11) and (7, 5); sum backward | forward output plus leaf gradients; (7, 16), stride (16, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 6.692 us +/- 0.125 us, var 0.167 | 38.765 us +/- 0.526 us, var 7.335 | 0.17x | `14564349192754315822`/`14564349192754315822` |
| `singleton_contiguous_8192` | singleton | `torch.concat` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.795 us +/- 0.032 us, var 0.002 | 2.487 us +/- 0.033 us, var 0.061 | 0.72x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concat` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.242 us +/- 0.018 us, var 0.001 | 2.124 us +/- 0.014 us, var 0.013 | 0.58x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concat` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.304 us +/- 0.016 us, var 0.003 | 3.594 us +/- 0.023 us, var 0.007 | 0.36x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concat` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.072 us +/- 0.006 us, var 0.000 | 1.551 us +/- 0.008 us, var 0.001 | 0.69x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concat` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.837 us +/- 0.042 us, var 0.006 | 2.679 us +/- 0.034 us, var 0.006 | 0.69x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concat` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 8.111 us +/- 0.055 us, var 0.055 | 7.517 us +/- 0.027 us, var 0.110 | 1.08x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concat` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.223 us +/- 0.010 us, var 0.001 | 2.010 us +/- 0.015 us, var 0.012 | 0.61x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concat` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.070 us +/- 0.006 us, var 0.003 | 1.908 us +/- 0.012 us, var 0.003 | 0.56x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.457 us +/- 0.011 us, var 0.001 | 3.072 us +/- 0.019 us, var 0.003 | 0.47x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.concat` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 1.504 us +/- 0.016 us, var 0.039 | 2.406 us +/- 0.016 us, var 0.003 | 0.63x | `14264808484827237231`/`14264808484827237231` |
| `rank2_dim0_generated_23_19x37` | rank-2 | `torch.concat` | list dim=0 | held-out generated rank-2 inputs with shapes (23, 37) and (19, 37) | rank-2 row concatenation output; (42, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.289 us +/- 0.014 us, var 0.001 | 2.125 us +/- 0.028 us, var 0.005 | 0.61x | `15329540218385864427`/`15329540218385864427` |
| `rank2_dim1_generated_31x17_13_11` | rank-2 | `torch.concat` | list dim=1 | held-out generated rank-2 inputs with shapes (31, 17), (31, 13), and (31, 11) | rank-2 column concatenation output; (31, 41), stride (41, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 2.266 us +/- 0.028 us, var 0.003 | 2.480 us +/- 0.027 us, var 0.004 | 0.91x | `12710136227280332661`/`12710136227280332661` |
| `backward_rank1_generated_113_127` | backward | `torch.concat` | list dim=0 | held-out generated grad-requiring 1-D inputs with lengths 113 and 127; sum backward | forward output plus leaf gradients; (240,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 6.454 us +/- 0.077 us, var 0.065 | 38.618 us +/- 0.531 us, var 8.115 | 0.17x | `11635413765118461767`/`11635413765118461767` |
| `backward_rank2_dim1_generated_7x11_5` | backward | `torch.concat` | list dim=1 | held-out generated grad-requiring rank-2 inputs with shapes (7, 11) and (7, 5); sum backward | forward output plus leaf gradients; (7, 16), stride (16, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 6.710 us +/- 0.143 us, var 0.122 | 39.057 us +/- 0.495 us, var 4.298 | 0.17x | `14564349192754315822`/`14564349192754315822` |
| `singleton_contiguous_8192` | singleton | `torch.concatenate` | list dim=0 | one contiguous input (8192,), stride (1,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.866 us +/- 0.069 us, var 0.013 | 2.477 us +/- 0.028 us, var 0.013 | 0.75x | `18053859545652983804`/`18053859545652983804` |
| `multi_input_contiguous_257_263_269` | multi-input | `torch.concatenate` | list dim=0 | three contiguous inputs with lengths 257, 263, and 269 | concatenation output; (789,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.292 us +/- 0.020 us, var 0.006 | 2.112 us +/- 0.021 us, var 0.005 | 0.61x | `10775443274831830041`/`10775443274831830041` |
| `empty_operand_middle_1024_0_511` | empty operand | `torch.concatenate` | list dim=0 | contiguous inputs with lengths 1024, 0, and 511 | concatenation output; (1535,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 1.303 us +/- 0.018 us, var 0.005 | 3.559 us +/- 0.022 us, var 0.038 | 0.37x | `9278694249625300899`/`9278694249625300899` |
| `all_empty_tuple_dim_negative_one` | empty operand | `torch.concatenate` | tuple dim=-1 | two empty 1-D inputs with length 0 | empty concatenation output; (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.072 us +/- 0.006 us, var 0.003 | 1.542 us +/- 0.007 us, var 0.001 | 0.70x | `8195591020010394303`/`8195591020010394303` |
| `offset_contiguous_views_4096` | offset | `torch.concatenate` | list dim=0 | left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 1.845 us +/- 0.049 us, var 0.014 | 2.707 us +/- 0.056 us, var 0.023 | 0.68x | `4041411121873054337`/`4041411121873054337` |
| `noncontiguous_stride2_views_4096` | noncontiguous | `torch.concatenate` | list dim=0 | left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,) | concatenation output; (8192,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 128 | 8.226 us +/- 0.123 us, var 0.778 | 7.522 us +/- 0.043 us, var 0.454 | 1.09x | `11758073942313516581`/`11758073942313516581` |
| `tuple_dim_negative_one_513_509` | multi-input | `torch.concatenate` | tuple dim=-1 | two contiguous tuple inputs with lengths 513 and 509 | concatenation output; (1022,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.232 us +/- 0.015 us, var 0.002 | 2.028 us +/- 0.015 us, var 0.005 | 0.61x | `14974692540956724659`/`14974692540956724659` |
| `axis_keyword_17_19` | axis keyword | `torch.concatenate` | list axis=0 | two contiguous inputs with lengths 17 and 19 | concatenation output; (36,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.087 us +/- 0.009 us, var 0.002 | 1.918 us +/- 0.011 us, var 0.002 | 0.57x | `13790662490979461913`/`13790662490979461913` |
| `no_grad_grad_inputs_257_263` | no_grad | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 1024 | 1.493 us +/- 0.023 us, var 0.007 | 3.134 us +/- 0.047 us, var 0.022 | 0.48x | `4113829215800195259`/`4113829215800195259` |
| `active_autograd_1d_257_263` | active autograd | `torch.concatenate` | list dim=0 | two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only | concatenation output; (520,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 512 | 1.519 us +/- 0.012 us, var 0.002 | 2.426 us +/- 0.022 us, var 0.003 | 0.63x | `14264808484827237231`/`14264808484827237231` |
| `rank2_dim0_generated_23_19x37` | rank-2 | `torch.concatenate` | list dim=0 | held-out generated rank-2 inputs with shapes (23, 37) and (19, 37) | rank-2 row concatenation output; (42, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 1.275 us +/- 0.022 us, var 0.003 | 2.098 us +/- 0.025 us, var 0.002 | 0.61x | `15329540218385864427`/`15329540218385864427` |
| `rank2_dim1_generated_31x17_13_11` | rank-2 | `torch.concatenate` | list dim=1 | held-out generated rank-2 inputs with shapes (31, 17), (31, 13), and (31, 11) | rank-2 column concatenation output; (31, 41), stride (41, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 256 | 2.241 us +/- 0.033 us, var 0.013 | 2.442 us +/- 0.042 us, var 0.101 | 0.92x | `12710136227280332661`/`12710136227280332661` |
| `backward_rank1_generated_113_127` | backward | `torch.concatenate` | list dim=0 | held-out generated grad-requiring 1-D inputs with lengths 113 and 127; sum backward | forward output plus leaf gradients; (240,), stride (1,), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 6.518 us +/- 0.115 us, var 0.053 | 38.289 us +/- 0.360 us, var 7.993 | 0.17x | `11635413765118461767`/`11635413765118461767` |
| `backward_rank2_dim1_generated_7x11_5` | backward | `torch.concatenate` | list dim=1 | held-out generated grad-requiring rank-2 inputs with shapes (7, 11) and (7, 5); sum backward | forward output plus leaf gradients; (7, 16), stride (16, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 64 | 6.691 us +/- 0.139 us, var 0.559 | 39.084 us +/- 0.392 us, var 5.320 | 0.17x | `14564349192754315822`/`14564349192754315822` |

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
