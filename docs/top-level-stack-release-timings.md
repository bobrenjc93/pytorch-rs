# `torch.stack` Release Timings

Date: 2026-09-08

Measured provenance: git head
`5013f61ba6db1761e4b55a6f8b4a1fd8f3709df4`. The raw JSON artifact records
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
  CARGO_TARGET_DIR="$PWD/target/stack-wheel-build" \
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
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest tests.test_autograd tests.test_no_grad_namespace
cargo fmt --check
CARGO_TARGET_DIR="$PWD/target/stack-wheel-build" \
  cargo test --locked --all-targets
git diff --check
```

Results: the checked-in driver produced
`docs/benchmark-data/top-level-stack-release-timings.json` with two
implementation orders, 15 untimed warmup blocks, and 81 measured blocks per
implementation pass. The generated markdown below is validated byte-for-byte
against that artifact by `scripts/benchmark_top_level_stack.py
--validate-artifact` and `tests.test_top_level_stack_benchmark_artifact`.

Setup evidence from regeneration:

- Virtualenv creation used CPython 3.12.12 and recreated `.venv` inside this
  worktree.
- Locked dependency installation resolved 36 packages in 27 ms, prepared 31
  packages in 16.73s, and installed them in 1.33s.
- Release build used `target/stack-wheel-build`; Cargo reported
  `Finished release profile [optimized] target(s) in 0.02s`.
- Release wheel reinstall reported resolving in 2 ms, preparing in 48 ms, and
  installing in 19 ms.
- Native-extension verification completed successfully and reported import paths
  under this worktree-local `.venv`.

Scope: this artifact times the eight same-shape `dim=` workloads listed below:
scalar, vector, matrix, empty, offset, noncontiguous, autograd forward, and
autograd forward+backward. `torch.no_grad()` behavior for grad-requiring
operands and PyTorch 2.13's `axis=` alias are covered by correctness tests, but
they are not included in this timing denominator.

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
- All supported cells: 0.72x uncapped, 0.72x capped
- Scalar cells: 0.54x uncapped, 0.54x capped
- Vector cells: 1.38x uncapped, 1.38x capped
- Matrix cells: 1.21x uncapped, 1.21x capped
- Empty cells: 0.50x uncapped, 0.50x capped
- Offset cells: 0.78x uncapped, 0.78x capped
- Noncontiguous cells: 1.03x uncapped, 1.03x capped
- Autograd forward cells: 0.93x uncapped, 0.93x capped
- Autograd forward+backward cells: 0.22x uncapped, 0.22x capped

Including the PyTorch-supported unsupported cells below as zero-credit denominator entries with a 10.00x capped penalty gives a combined capped aggregate of 1.22x.

## Supported Timed Cells

| Workload | Category | Input / mode | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Materialized checksums |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `scalar_three_inputs_dim0` | scalar | three scalar tensors, dim=0 | stack output; (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 10000 | 1.277 us +/- 0.005 us, var 0.000 | 2.384 us +/- 0.015 us, var 0.001 | 0.54x | `3595086523326908924`/`3595086523326908924` |
| `vector_three_inputs_dim_neg1_257` | vector | three vectors of shape (257,), dim=-1 | stack output; (257, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 512 | 5.159 us +/- 0.034 us, var 0.010 | 3.726 us +/- 0.025 us, var 0.018 | 1.38x | `9666124477339715250`/`9666124477339715250` |
| `matrix_three_inputs_dim1_257x263` | matrix | three matrices of shape (257, 263), dim=1 | stack output; (257, 3, 263), stride (789, 263, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 8 | 28.726 us +/- 1.055 us, var 3.923 | 23.695 us +/- 0.545 us, var 71.765 | 1.21x | `9196514359419668404`/`9196514359419668404` |
| `empty_two_inputs_dim2_2x0x3` | empty | two empty tensors of shape (2, 0, 3), dim=2 | stack output; (2, 0, 2, 3), stride (6, 6, 3, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 5000 | 0.853 us +/- 0.006 us, var 0.001 | 1.715 us +/- 0.017 us, var 0.007 | 0.50x | `5914968957525217100`/`5914968957525217100` |
| `offset_two_inputs_dim0_127x131` | offset | two nonzero-storage-offset views from tensor((3, 127, 131))[1], dim=0 | stack output; (2, 127, 131), stride (16637, 131, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 20 | 3.713 us +/- 0.046 us, var 0.051 | 4.731 us +/- 0.065 us, var 0.124 | 0.78x | `4576909815035170523`/`4576909815035170523` |
| `noncontig_two_inputs_dim0_512x1024` | noncontiguous | two transposed views from tensor((1024, 512)).transpose(0, 1), dim=0 | stack output; (2, 512, 1024), stride (524288, 1024, 1), offset 0, torch.float32, cpu, requires_grad=False, leaf=True | 3 | 2773.220 us +/- 289.152 us, var 153844.460 | 2691.978 us +/- 260.417 us, var 241261.351 | 1.03x | `16879733367057555647`/`16879733367057555647` |
| `autograd_forward_two_inputs_dim1_127x131` | autograd forward | two requires_grad=True leaves of shape (127, 131), dim=1; forward construction only | stack output; (127, 2, 131), stride (262, 131, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 20 | 4.866 us +/- 0.062 us, var 0.106 | 5.251 us +/- 0.041 us, var 0.083 | 0.93x | `1017021100399807393`/`1017021100399807393` |
| `autograd_forward_backward_repeated_dim1_32x33` | autograd forward+backward | left/right/left requires_grad=True leaves of shape (32, 33), dim=1; timed stack(...).sum().backward() | stack output plus accumulated leaf gradients; (32, 3, 33), stride (99, 33, 1), offset 0, torch.float32, cpu, requires_grad=True, leaf=False | 5 | 11.540 us +/- 0.263 us, var 0.960 | 52.675 us +/- 1.597 us, var 9.461 | 0.22x | `15114439219529365358`/`15114439219529365358` |

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
