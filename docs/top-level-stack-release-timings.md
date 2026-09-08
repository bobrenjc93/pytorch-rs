# `torch.stack` Release Timings

Date: 2026-09-07

Candidate provenance: current branch head
`a21ffd5a74d65986ef4ae22659a42ef4143617d1`. The raw JSON artifact records
the exact git head, worktree status, driver checksum, Python, PyTorch, Rust,
CPU, thread, and affinity provenance captured when the benchmark ran.

Exact build, check, and timing commands were run from the repository root. The
benchmark used the worktree-local `.venv` with pinned PyTorch 2.13.0 and did
not install packages outside the worktree. `CUDA_VISIBLE_DEVICES=` kept this
CPU-only benchmark from selecting the host GPUs.

```bash
uv venv --clear --python 3.12
uv sync --locked --no-install-project --group dev --group reference
env -u CONDA_PREFIX \
  TMPDIR="$PWD/target" \
  CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --out target/stack-wheel
uv pip install --python "$PWD/.venv/bin/python" --force-reinstall --no-deps \
  target/stack-wheel/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
env PYTHONNOUSERSITE=1 .venv/bin/python .github/scripts/verify_native_extension.py
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  .venv/bin/python scripts/benchmark_top_level_stack.py \
  --cpu 24 --threads 1 \
  --output docs/benchmark-data/top-level-stack-release-timings.json
.venv/bin/python scripts/benchmark_top_level_stack.py \
  --render-markdown-summary \
  docs/benchmark-data/top-level-stack-release-timings.json \
  > target/top-level-stack-summary.md
```

Checks run for this evidence:

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/benchmark_top_level_stack.py --validate-artifact
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  .venv/bin/python -m unittest \
  tests.test_top_level_stack tests.test_top_level_stack_reference \
  tests.test_top_level_stack_benchmark_artifact tests.test_readme_quickstart
cargo fmt --check
cargo test --locked --all-targets
git diff --check
```

Results: the checked-in driver produced
`docs/benchmark-data/top-level-stack-release-timings.json` with two
implementation orders, 15 untimed warmup blocks, and 81 measured blocks per
implementation pass. The generated markdown below is validated byte-for-byte
against that artifact by `scripts/benchmark_top_level_stack.py
--validate-artifact` and `tests.test_top_level_stack_benchmark_artifact`.

Inputs are created outside timed regions from deterministic CPU `float32`
values with fixed seeds. Every supported cell first compares `torch_rs` against
PyTorch for shape, stride, storage offset, contiguity, dtype, device, layout,
`requires_grad`, leaf status, and exact logical value bits. Backward cells
materialize the stack output plus leaf gradients. Every warmup and measured
block materializes its final output bundle as a 64-bit BLAKE2b checksum over
output metadata and logical bytes; the artifact validates stable equal
checksum sets for `torch_rs` and PyTorch.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/top-level-stack-release-timings.json`
- Benchmark: `top_level_stack_cpu_benchmark_v1`
- Timed supported cells: 8 (1 API x 8 workload shapes and modes)
- Zero-credit unsupported cells: 4
- Implementation orders: torch_rs then pytorch, pytorch then torch_rs; each implementation appears once before and once after the other implementation
- Warmup and sampling: 15 untimed warmup blocks and 81 measured blocks per implementation pass
- CPU affinity: selected CPU 24, pinned affinity [24]; threads=1
- All supported cells: 3.63x uncapped, 2.36x capped
- Scalar cells: 0.69x uncapped, 0.69x capped
- Vector cells: 1.87x uncapped, 1.87x capped
- Matrix cells: 39.14x uncapped, 10.00x capped
- Empty cells: 0.64x uncapped, 0.64x capped
- Offset cells: 28.58x uncapped, 10.00x capped
- Noncontiguous cells: 2.34x uncapped, 2.34x capped
- Autograd forward cells: 27.76x uncapped, 10.00x capped
- Autograd forward+backward cells: 0.50x uncapped, 0.50x capped

Including the unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 3.82x.

## Supported Timed Cells

| Workload | Category | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_three_inputs_dim0` | scalar | three scalar tensors, dim=0 | stack output; (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.652 us +/- 0.034 us, var 0.068 | 2.392 us +/- 0.016 us, var 0.007 | 0.69x | `3595086523326908924`/`3595086523326908924` |
| `vector_three_inputs_dim_neg1_257` | vector | three vectors of shape (257,), dim=-1 | stack output; (257, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 6.909 us +/- 0.061 us, var 0.024 | 3.702 us +/- 0.029 us, var 0.003 | 1.87x | `9666124477339715250`/`9666124477339715250` |
| `matrix_three_inputs_dim1_257x263` | matrix | three matrices of shape (257, 263), dim=1 | stack output; (257, 3, 263), stride (789, 263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 8 | 939.681 us +/- 32.069 us, var 3034.478 | 24.009 us +/- 0.643 us, var 157.938 | 39.14x | `9196514359419668404`/`9196514359419668404` |
| `empty_two_inputs_dim2_2x0x3` | empty | two empty tensors of shape (2, 0, 3), dim=2 | stack output; (2, 0, 2, 3), stride (6, 6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.089 us +/- 0.007 us, var 0.002 | 1.704 us +/- 0.008 us, var 0.002 | 0.64x | `5914968957525217100`/`5914968957525217100` |
| `offset_two_inputs_dim0_127x131` | offset | two nonzero-storage-offset views from tensor((3, 127, 131))[1], dim=0 | stack output; (2, 127, 131), stride (16637, 131, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 20 | 137.481 us +/- 4.584 us, var 46.796 | 4.811 us +/- 0.081 us, var 0.198 | 28.58x | `4576909815035170523`/`4576909815035170523` |
| `noncontig_two_inputs_dim0_512x1024` | noncontiguous | two transposed views from tensor((1024, 512)).transpose(0, 1), dim=0 | stack output; (2, 512, 1024), stride (524288, 1024, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 3 | 5530.650 us +/- 368.800 us, var 261857.148 | 2367.100 us +/- 19.009 us, var 3573.454 | 2.34x | `16879733367057555647`/`16879733367057555647` |
| `autograd_forward_two_inputs_dim1_127x131` | autograd forward | two requires_grad=True leaves of shape (127, 131), dim=1; forward construction only | stack output; (127, 2, 131), stride (262, 131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 146.741 us +/- 7.072 us, var 152.471 | 5.286 us +/- 0.060 us, var 0.136 | 27.76x | `1017021100399807393`/`1017021100399807393` |
| `autograd_forward_backward_repeated_dim1_32x33` | autograd forward+backward | left/right/left requires_grad=True leaves of shape (32, 33), dim=1; timed stack(...).sum().backward() | stack output plus accumulated leaf gradients; (32, 3, 33), stride (99, 33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 25.609 us +/- 0.497 us, var 6.451 | 50.909 us +/- 0.947 us, var 9.936 | 0.50x | `15114439219529365358`/`15114439219529365358` |

## Zero-Credit Unsupported Cells

These cells are not timed. They are preserved as zero-credit denominator entries instead of being removed from the evidence set.

| Workload | Input | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `top_level_torch_stack_empty_input_sequence` | empty TensorList | `RuntimeError: stack expects a non-empty TensorList` | `RuntimeError: stack expects a non-empty TensorList` | zero |
| `top_level_torch_stack_mixed_shapes` | same-rank tensors with shapes (1,) and (2,) | `RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 1` | `RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 1` | zero |
| `top_level_torch_stack_mixed_metadata` | float32 tensor and float64 tensor requiring dtype promotion | `AttributeError: module 'torch_rs' has no attribute 'float64'` | `supported (2, 1), stride (1, 1), offset 0, torch.float64, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_stack_concrete_out` | same-shape tensor sequence with a concrete out tensor | `RuntimeError: stack(): the 'out' argument is not supported` | `supported (2, 1), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |
