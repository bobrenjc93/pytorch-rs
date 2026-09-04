# `torch.compile` Eager CPU Release Timings

Date: 2026-09-04

Candidate provenance: source snapshot based on
`f89b884e8bc67d0ebedb3f4c42d8a8574338e786`, plus the worktree changes that
add native zero-argument `Tensor.t()` view compilation, bump the compile
corpus to v5, and require explicit view-alias oracle checks in the evaluator
and CPU benchmark.

Exact setup, build, check, and timing commands were run from the repository
root. The reusable timing driver is checked in as
`scripts/benchmark_compile_cpu.py`; its complete raw JSON output is committed
at `docs/benchmark-data/torch-compile-cpu-v5.json`. The active Conda
environment held ambient PyTorch 2.14, so the PyTorch 2.13 reference evidence
used this worktree's local `.venv`; uv and Cargo state were redirected under
`target/`.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv venv --clear --python \
  /home/bobren/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14 \
  .venv
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv sync --locked --python "$PWD/.venv/bin/python" \
  --no-install-project --group reference
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --out target/wheels
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv pip install --python "$PWD/.venv/bin/python" --force-reinstall --no-deps \
  target/wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" PYTHONPATH= \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_compile_corpus.CompileCorpusMetadataTests \
  tests.test_compile_corpus.CompileCorpusTraceTests \
  tests.test_compile_corpus.CompileRecompilationGuardCorpusTests \
  tests.test_top_level_compile.TorchCompileEntrypointTests \
  tests.test_torch_compile_coverage_evaluator tests.test_readme_quickstart
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" PYTHONPATH= \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/evaluate_torch_compile_coverage.py --subset public
cargo fmt --check
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python scripts/benchmark_compile_cpu.py \
  --require-single-cpu-affinity \
  --output docs/benchmark-data/torch-compile-cpu-v5.json
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python scripts/benchmark_compile_cpu.py \
  --render-markdown-summary docs/benchmark-data/torch-compile-cpu-v5.json \
  > target/torch-compile-cpu-v5-summary.md
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python scripts/benchmark_compile_cpu.py --validate-artifact
```

Results: the focused evaluator and benchmark alias-oracle tests passed under
pinned PyTorch 2.13. The public coverage evaluator reported 32.0/100 with all
13 reference-eligible public cases passing and explicit alias evidence for the
view case. The CPU timing artifact validated against this markdown report. The
benchmark uses fixed single-core affinity, 7 warmups, and 31 measured samples
for each implementation order.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.14.5
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.14/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `.venv/lib/python3.14/site-packages/torch_rs/__init__.py`
- Device/dtype: CPU float32
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`

The benchmark uses the checked-in `torch_compile_corpus_v5` programs. The timed
supported set contains every public native compile case: five one-input
tensor-arithmetic programs, four two-input broadcasting programs, one
mutation/aliasing/view program, and three recompilation-guard programs.
One-input programs run across the corpus default input plus scalar, vector,
row-major matrix, larger row-major matrix, empty, and non-contiguous transpose
inputs. Two-input programs run across the corpus default input plus row-major
matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor,
empty broadcast, and non-contiguous matrix/vector broadcast inputs.
Recompilation-guard programs run across shape, stride, and `requires_grad`
metadata variants; separate guard-sequence rows exercise cache reuse, bounded
`recompile_limit` behavior, `torch.compiler.reset()` semantics, and both
implementation orders. Inputs are created outside timed regions from
deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock
`torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the
native guarded eager/fullgraph path. Both implementations run in both orders:
`torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant
compiler state for cold timing, measures the first materialized compiled call
separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured
block repeats the operation according to the table's `Repeats` column; medians
below are microseconds per compiled call. The CPU workload has no asynchronous
device queue, but the driver still calls synchronization hooks when an
implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, shape, stride,
storage offset, contiguity, dtype, device, and `requires_grad` against the same
eager program. View cells also assert that the compiled output is set to the
same eager output view and preserves the input storage relationship; this alias
evidence is stored in the raw artifact. The `torch_rs` result is also checked
against the PyTorch result. After every warmup and measured block, the driver
materializes the last output and records a 64-bit BLAKE2b checksum over values
and metadata. All 91 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is
generated by the reusable fixed-affinity driver, uses equivalent work in both
implementation orders, pins the reference version, materializes and checks
outputs instead of timing dead code, asserts view alias observables for timed
view cells, keeps held-out corpus cases in differential tests, validates guard
sequences separately from timed cells, and retains every unsupported category in
the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v5.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v5`
- Cold first compiled call: 0.018x uncapped, 0.111x capped
- Steady-state materialized compiled call: 1.571x uncapped, 1.571x capped
- Timed supported cells: 91 (35 tensor-arithmetic, 28 broadcasting, 7 mutation/aliasing/view, 21 recompilation-guard)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 32.0% supported by native compile cases, 68% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 325.217 | 182002.013 | 0.002x | 28.324 +/- 0.380 | 28.211 +/- 0.401 | 1.004x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 233.323 | 47982.748 | 0.005x | 26.986 +/- 0.647 | 24.466 +/- 0.724 | 1.103x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 241.675 | 44953.397 | 0.005x | 38.961 +/- 0.332 | 24.409 +/- 0.533 | 1.596x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 2668.604 | 49754.300 | 0.054x | 61.650 +/- 4.258 | 35.745 +/- 6.028 | 1.725x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 15874.118 | 61033.272 | 0.260x | 1074.032 +/- 74.009 | 899.075 +/- 37.838 | 1.195x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 266.252 | 50434.863 | 0.005x | 47.137 +/- 3.689 | 23.363 +/- 0.134 | 2.018x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 2835.216 | 53454.344 | 0.053x | 64.431 +/- 0.362 | 47.724 +/- 1.530 | 1.350x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 2253.672 | 41066.081 | 0.055x | 44.060 +/- 0.494 | 26.836 +/- 0.327 | 1.642x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 207.498 | 42362.821 | 0.005x | 28.080 +/- 0.404 | 21.875 +/- 0.620 | 1.284x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 2210.091 | 41072.496 | 0.054x | 31.007 +/- 0.152 | 22.495 +/- 0.262 | 1.378x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 635.515 | 48864.845 | 0.013x | 51.488 +/- 0.704 | 34.436 +/- 9.132 | 1.495x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 12740.881 | 55810.771 | 0.228x | 914.731 +/- 9.953 | 914.630 +/- 8.302 | 1.000x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 248.125 | 44927.438 | 0.006x | 31.555 +/- 0.184 | 21.682 +/- 1.530 | 1.455x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 687.735 | 48610.190 | 0.014x | 54.116 +/- 0.404 | 14.957 +/- 0.745 | 3.618x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 234.310 | 46514.121 | 0.005x | 35.909 +/- 0.308 | 28.472 +/- 0.425 | 1.261x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 228.505 | 43230.823 | 0.005x | 39.626 +/- 4.487 | 24.670 +/- 0.679 | 1.606x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 241.625 | 44529.977 | 0.005x | 38.748 +/- 0.127 | 24.724 +/- 1.508 | 1.567x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 684.530 | 54344.829 | 0.013x | 55.625 +/- 0.303 | 48.342 +/- 2.098 | 1.151x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 12435.218 | 66641.599 | 0.187x | 903.985 +/- 7.380 | 897.052 +/- 51.147 | 1.008x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 273.930 | 43861.668 | 0.006x | 40.369 +/- 0.700 | 23.306 +/- 0.124 | 1.732x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 2687.563 | 48217.757 | 0.056x | 57.403 +/- 0.807 | 47.689 +/- 1.567 | 1.204x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 278.637 | 47585.081 | 0.006x | 62.178 +/- 0.560 | 32.326 +/- 0.411 | 1.923x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 339.628 | 47639.188 | 0.007x | 61.300 +/- 0.759 | 33.248 +/- 0.803 | 1.844x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 301.601 | 49786.649 | 0.006x | 65.572 +/- 2.170 | 32.384 +/- 1.229 | 2.025x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 738.677 | 47684.406 | 0.015x | 71.533 +/- 0.911 | 51.344 +/- 1.778 | 1.393x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 12909.110 | 61055.459 | 0.211x | 932.450 +/- 7.273 | 913.955 +/- 6.540 | 1.020x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 377.596 | 54671.804 | 0.007x | 67.155 +/- 0.571 | 30.023 +/- 0.732 | 2.237x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 915.058 | 51627.732 | 0.018x | 94.507 +/- 14.506 | 52.200 +/- 0.452 | 1.810x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 391.181 | 51696.416 | 0.008x | 67.483 +/- 1.213 | 30.804 +/- 0.196 | 2.191x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 2344.344 | 47607.965 | 0.049x | 62.776 +/- 0.647 | 34.127 +/- 0.718 | 1.839x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 351.672 | 50965.196 | 0.007x | 69.774 +/- 0.837 | 32.797 +/- 0.925 | 2.127x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 845.067 | 52700.910 | 0.016x | 75.923 +/- 2.944 | 54.004 +/- 2.594 | 1.406x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 13114.726 | 68343.006 | 0.192x | 943.458 +/- 44.694 | 939.096 +/- 14.112 | 1.005x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 400.345 | 51245.990 | 0.008x | 70.315 +/- 0.334 | 30.649 +/- 0.827 | 2.294x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 837.631 | 49184.404 | 0.017x | 95.615 +/- 2.271 | 52.722 +/- 0.617 | 1.814x | `fedf1f495675c5ac` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 324.991 | 48606.325 | 0.007x | 67.314 +/- 0.874 | 31.654 +/- 0.293 | 2.127x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 2804.450 | 52137.137 | 0.054x | 71.950 +/- 0.950 | 51.166 +/- 0.585 | 1.406x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 14826.314 | 62430.683 | 0.237x | 945.544 +/- 3.828 | 920.244 +/- 11.470 | 1.027x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 4856.121 | 51026.383 | 0.095x | 71.837 +/- 1.591 | 50.819 +/- 0.615 | 1.414x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 752.077 | 59633.540 | 0.013x | 72.128 +/- 1.950 | 50.835 +/- 0.498 | 1.419x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 287.440 | 48586.194 | 0.006x | 69.576 +/- 0.839 | 28.135 +/- 0.695 | 2.473x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 794.326 | 55648.155 | 0.014x | 125.768 +/- 0.929 | 52.309 +/- 3.383 | 2.404x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 246.473 | 54539.875 | 0.005x | 56.905 +/- 0.863 | 29.699 +/- 0.272 | 1.916x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 708.556 | 51489.363 | 0.014x | 61.169 +/- 0.270 | 48.509 +/- 1.087 | 1.261x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 12644.456 | 65698.489 | 0.192x | 928.833 +/- 14.196 | 925.080 +/- 9.300 | 1.004x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 2758.004 | 45063.428 | 0.061x | 59.905 +/- 0.306 | 47.820 +/- 1.388 | 1.253x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 665.851 | 51777.143 | 0.013x | 61.812 +/- 0.713 | 44.600 +/- 5.118 | 1.386x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 2352.902 | 52010.050 | 0.045x | 50.874 +/- 0.208 | 24.091 +/- 0.222 | 2.112x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 753.920 | 53394.855 | 0.014x | 112.630 +/- 1.048 | 48.789 +/- 4.630 | 2.309x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 249.077 | 47560.213 | 0.005x | 56.955 +/- 1.076 | 30.247 +/- 0.290 | 1.883x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 2701.384 | 44566.583 | 0.061x | 61.723 +/- 0.343 | 48.232 +/- 2.706 | 1.280x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 12651.251 | 67791.563 | 0.187x | 933.678 +/- 26.375 | 927.817 +/- 18.853 | 1.006x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 765.012 | 50259.564 | 0.015x | 64.951 +/- 0.804 | 48.442 +/- 1.508 | 1.341x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 767.094 | 50833.011 | 0.015x | 61.379 +/- 0.619 | 47.999 +/- 2.556 | 1.279x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 248.285 | 50198.271 | 0.005x | 51.005 +/- 0.246 | 24.449 +/- 0.566 | 2.086x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 754.801 | 54687.788 | 0.014x | 114.776 +/- 1.065 | 49.391 +/- 0.981 | 2.324x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 257.635 | 46495.293 | 0.006x | 57.623 +/- 0.793 | 29.235 +/- 0.300 | 1.971x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 713.308 | 47602.917 | 0.015x | 61.961 +/- 0.542 | 43.627 +/- 5.313 | 1.420x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 14666.477 | 59421.245 | 0.247x | 922.547 +/- 14.668 | 895.331 +/- 28.984 | 1.030x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 752.588 | 50329.615 | 0.015x | 60.612 +/- 0.358 | 47.270 +/- 1.419 | 1.282x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 721.375 | 56622.163 | 0.013x | 61.657 +/- 0.366 | 44.493 +/- 6.015 | 1.386x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 244.060 | 51692.240 | 0.005x | 50.730 +/- 0.189 | 23.560 +/- 0.205 | 2.153x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 760.174 | 52728.457 | 0.014x | 112.568 +/- 2.286 | 47.945 +/- 1.235 | 2.348x | `2bd384aefcaaa397` |
| `cpu_float32_t_view` | `case_default` | 1 | 256 | shape (3, 2), stride (1, 3), offset 0, torch.float32, cpu, requires_grad=True | 201.325 | 44227.556 | 0.005x | 31.146 +/- 0.267 | 26.255 +/- 0.302 | 1.186x | `9382f02b63ce0129` |
| `cpu_float32_t_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 184.899 | 43766.964 | 0.004x | 26.693 +/- 0.189 | 19.517 +/- 0.049 | 1.368x | `e75a1d3233117514` |
| `cpu_float32_t_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 202.451 | 40199.611 | 0.005x | 34.545 +/- 1.051 | 19.715 +/- 0.824 | 1.752x | `4c3dc265c5b9d697` |
| `cpu_float32_t_view` | `matrix_31x37` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 2646.110 | 43578.389 | 0.061x | 49.772 +/- 0.652 | 14.517 +/- 1.302 | 3.429x | `4ba5419e2e3f2393` |
| `cpu_float32_t_view` | `matrix_127x131` | 1 | 16 | shape (131, 127), stride (1, 131), offset 0, torch.float32, cpu, requires_grad=False | 12631.355 | 59909.598 | 0.211x | 906.222 +/- 24.448 | 868.349 +/- 44.335 | 1.044x | `992050cd3907bd7f` |
| `cpu_float32_t_view` | `empty_2x0` | 1 | 2048 | shape (0, 2), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 252.582 | 44061.774 | 0.006x | 30.535 +/- 0.829 | 19.768 +/- 0.104 | 1.545x | `97d8efc321a43d95` |
| `cpu_float32_t_view` | `transpose_37x31` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 759.819 | 45115.827 | 0.017x | 54.015 +/- 0.859 | 14.227 +/- 0.854 | 3.797x | `5ccc89fb94f689e5` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 321.856 | 48363.503 | 0.007x | 63.821 +/- 2.254 | 29.910 +/- 0.202 | 2.134x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 316.278 | 52633.268 | 0.006x | 46.282 +/- 1.317 | 30.072 +/- 1.736 | 1.539x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 300.299 | 48965.883 | 0.006x | 48.507 +/- 1.202 | 29.792 +/- 0.779 | 1.628x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 579.240 | 49194.543 | 0.012x | 60.923 +/- 0.456 | 48.166 +/- 0.950 | 1.265x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8359.233 | 57882.427 | 0.144x | 538.582 +/- 8.456 | 517.972 +/- 3.021 | 1.040x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 336.338 | 44235.943 | 0.008x | 50.495 +/- 0.380 | 26.500 +/- 0.931 | 1.905x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 669.877 | 51918.205 | 0.013x | 65.848 +/- 1.452 | 48.864 +/- 2.552 | 1.348x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 300.079 | 44998.475 | 0.007x | 62.157 +/- 0.763 | 30.559 +/- 0.316 | 2.034x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 738.427 | 47056.426 | 0.016x | 66.388 +/- 0.298 | 49.682 +/- 1.131 | 1.336x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 14819.503 | 59762.003 | 0.248x | 937.787 +/- 4.893 | 916.178 +/- 22.354 | 1.024x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 3836.415 | 51392.807 | 0.075x | 65.445 +/- 0.362 | 49.319 +/- 0.850 | 1.327x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 736.013 | 44653.213 | 0.016x | 66.354 +/- 0.311 | 49.342 +/- 0.781 | 1.345x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 289.523 | 45084.060 | 0.006x | 61.293 +/- 1.815 | 26.869 +/- 0.178 | 2.281x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 787.545 | 52453.860 | 0.015x | 119.242 +/- 1.104 | 49.942 +/- 1.002 | 2.388x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 278.211 | 46430.926 | 0.006x | 57.079 +/- 1.266 | 29.917 +/- 0.248 | 1.908x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 260.709 | 43887.886 | 0.006x | 45.942 +/- 0.510 | 27.815 +/- 0.428 | 1.652x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 277.515 | 41255.878 | 0.007x | 48.256 +/- 0.634 | 29.427 +/- 1.209 | 1.640x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 560.673 | 45568.067 | 0.012x | 60.680 +/- 0.397 | 47.912 +/- 2.873 | 1.266x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8162.535 | 55759.963 | 0.146x | 530.023 +/- 3.640 | 515.021 +/- 2.069 | 1.029x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 316.379 | 44917.393 | 0.007x | 50.488 +/- 0.223 | 26.496 +/- 0.501 | 1.905x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 4652.073 | 44277.115 | 0.105x | 64.993 +/- 0.452 | 48.467 +/- 0.763 | 1.341x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 861.492 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 4800.778 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 535.279 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 325140.969 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 252765.528 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 163020.830 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 216111.417 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 230708.705 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 153623.044 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1029.467 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 789.524 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 581.940 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, mutation/aliasing/views, and recompilation guards. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_t_view` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `inference` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `training_autograd` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `python_control_flow` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `containers_pytrees` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `decompositions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `custom_functions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dtype_device_transitions` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 32 / 100. Zero-credit unsupported category weight: 68 / 100.
The v5 corpus also keeps 2 held-out broadcasting programs, 1 held-out mutation/aliasing/view program, and 2 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
