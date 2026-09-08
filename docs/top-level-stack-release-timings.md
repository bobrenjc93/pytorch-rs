# `torch.stack` Release Timings

Date: 2026-09-07

Measured provenance: git head
`5ee92720ad010c106143d4d6962ac9f6ddbeb78f`. The raw JSON artifact records
the exact git head, worktree status, driver checksum, Python, PyTorch, Rust,
CPU, thread, and affinity provenance captured when the benchmark ran. Its
`git.status_short` and `git.diff_stat` were empty at benchmark capture.

Exact build, check, and timing commands were run from the repository root. The
benchmark used the worktree-local `.venv` with pinned PyTorch 2.13.0 and did
not install packages outside the worktree. `CUDA_VISIBLE_DEVICES=` kept this
CPU-only benchmark from selecting the host GPUs.

```bash
PYTHONNOUSERSITE=1 UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv venv --clear --python 3.12
PYTHONNOUSERSITE=1 \
UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv sync --locked --no-install-project --group dev --group reference
PYTHONNOUSERSITE=1 \
  CONDA_PREFIX= \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  TMPDIR="$PWD/target" \
  CARGO_TARGET_DIR="$PWD/target/stack-release-build-timing" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --out target/stack-wheel
PYTHONNOUSERSITE=1 \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
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
implementation pass. The raw JSON records setup timing evidence under
`environment.setup_timings`. The generated markdown below is validated
byte-for-byte against that artifact by
`scripts/benchmark_top_level_stack.py --validate-artifact` and
`tests.test_top_level_stack_benchmark_artifact`.

Setup timing evidence:

- Virtualenv creation: `uv venv --clear --python 3.12` completed in 0.610s.
- Dependency installation: locked `uv sync` completed in 1.159s wall time;
  `uv` reported resolving 36 packages in 27 ms and installing 31 packages in
  1.02s.
- Release build: fresh `maturin build --release --locked` in
  `target/stack-release-build-timing` completed in 63.360s wall time; Cargo
  reported `Finished release profile [optimized] target(s) in 1m 03s`.
- Release wheel reinstall: `uv pip install --force-reinstall --no-deps`
  completed in 0.870s wall time; `uv` reported resolving in 7 ms, preparing in
  384 ms, and installing in 87 ms.
- Native-extension verification completed in 0.424s.

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
Only PyTorch-supported boundary gaps are included as zero-credit denominator
rows. Boundary inputs that both PyTorch and `torch_rs` reject with the same
error are reported below as error-parity checks and are not included in the
zero-credit aggregate.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/top-level-stack-release-timings.json`
- Benchmark: `top_level_stack_cpu_benchmark_v2`
- Timed supported cells: 8 (1 API x 8 workload shapes and modes)
- Zero-credit unsupported cells: 2
- Error-parity boundary cells: 2
- Implementation orders: torch_rs then pytorch, pytorch then torch_rs; each implementation appears once before and once after the other implementation
- Warmup and sampling: 15 untimed warmup blocks and 81 measured blocks per implementation pass
- CPU affinity: selected CPU 24, pinned affinity [24]; threads=1
- All supported cells: 3.26x uncapped, 2.25x capped
- Scalar cells: 0.59x uncapped, 0.59x capped
- Vector cells: 1.67x uncapped, 1.67x capped
- Matrix cells: 30.81x uncapped, 10.00x capped
- Empty cells: 0.55x uncapped, 0.55x capped
- Offset cells: 26.83x uncapped, 10.00x capped
- Noncontiguous cells: 3.45x uncapped, 3.45x capped
- Autograd forward cells: 23.00x uncapped, 10.00x capped
- Autograd forward+backward cells: 0.36x uncapped, 0.36x capped

Including the PyTorch-supported unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 3.04x.

## Supported Timed Cells

| Workload | Category | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_three_inputs_dim0` | scalar | three scalar tensors, dim=0 | stack output; (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 2.778 us +/- 0.223 us, var 2.548 | 4.741 us +/- 0.320 us, var 4.675 | 0.59x | `3595086523326908924`/`3595086523326908924` |
| `vector_three_inputs_dim_neg1_257` | vector | three vectors of shape (257,), dim=-1 | stack output; (257, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 11.131 us +/- 0.427 us, var 165.416 | 6.655 us +/- 0.337 us, var 22.247 | 1.67x | `9666124477339715250`/`9666124477339715250` |
| `matrix_three_inputs_dim1_257x263` | matrix | three matrices of shape (257, 263), dim=1 | stack output; (257, 3, 263), stride (789, 263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 8 | 1255.703 us +/- 63.586 us, var 914487.307 | 40.757 us +/- 2.778 us, var 233.411 | 30.81x | `9196514359419668404`/`9196514359419668404` |
| `empty_two_inputs_dim2_2x0x3` | empty | two empty tensors of shape (2, 0, 3), dim=2 | stack output; (2, 0, 2, 3), stride (6, 6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 1.708 us +/- 0.071 us, var 0.414 | 3.103 us +/- 0.305 us, var 0.610 | 0.55x | `5914968957525217100`/`5914968957525217100` |
| `offset_two_inputs_dim0_127x131` | offset | two nonzero-storage-offset views from tensor((3, 127, 131))[1], dim=0 | stack output; (2, 127, 131), stride (16637, 131, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 20 | 232.530 us +/- 8.432 us, var 9762.994 | 8.667 us +/- 0.329 us, var 7.475 | 26.83x | `4576909815035170523`/`4576909815035170523` |
| `noncontig_two_inputs_dim0_512x1024` | noncontiguous | two transposed views from tensor((1024, 512)).transpose(0, 1), dim=0 | stack output; (2, 512, 1024), stride (524288, 1024, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 3 | 10200.179 us +/- 716.278 us, var 4905431.695 | 2954.765 us +/- 310.351 us, var 53864454.838 | 3.45x | `16879733367057555647`/`16879733367057555647` |
| `autograd_forward_two_inputs_dim1_127x131` | autograd forward | two requires_grad=True leaves of shape (127, 131), dim=1; forward construction only | stack output; (127, 2, 131), stride (262, 131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 268.048 us +/- 9.418 us, var 1996.793 | 11.652 us +/- 0.582 us, var 19.994 | 23.00x | `1017021100399807393`/`1017021100399807393` |
| `autograd_forward_backward_repeated_dim1_32x33` | autograd forward+backward | left/right/left requires_grad=True leaves of shape (32, 33), dim=1; timed stack(...).sum().backward() | stack output plus accumulated leaf gradients; (32, 3, 33), stride (99, 33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 54.633 us +/- 2.811 us, var 216.839 | 152.499 us +/- 10.420 us, var 1087883.238 | 0.36x | `15114439219529365358`/`15114439219529365358` |

## Zero-Credit Unsupported Cells

These cells are not timed. PyTorch supports them and torch_rs rejects them, so they are preserved as zero-credit denominator entries instead of being removed from the evidence set.

| Workload | Input | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `top_level_torch_stack_mixed_metadata` | float32 tensor and float64 tensor requiring dtype promotion | `AttributeError: module 'torch_rs' has no attribute 'float64'` | `supported (2, 1), stride (1, 1), offset 0, torch.float64, cpu, requires_grad=False, leaf=True` | zero |
| `top_level_torch_stack_concrete_out` | same-shape tensor sequence with a concrete out tensor | `RuntimeError: stack(): the 'out' argument is not supported` | `supported (2, 1), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True` | zero |

## Error-Parity Boundary Cells

These cells are checked but excluded from the zero-credit denominator because PyTorch rejects the same invalid inputs.

| Workload | Input | `torch_rs` status | PyTorch status | Credit |
| --- | --- | --- | --- | --- |
| `top_level_torch_stack_empty_input_sequence` | empty TensorList | `RuntimeError: stack expects a non-empty TensorList` | `RuntimeError: stack expects a non-empty TensorList` | error_parity |
| `top_level_torch_stack_mixed_shapes` | same-rank tensors with shapes (1,) and (2,) | `RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 1` | `RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 1` | error_parity |
