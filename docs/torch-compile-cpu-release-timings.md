# `torch.compile` Eager CPU Release Timings

Date: 2026-09-04

Candidate provenance: source snapshot based on
`f89b884e8bc67d0ebedb3f4c42d8a8574338e786`, plus the worktree changes that
add native zero-argument `Tensor.t()` view compilation and bump the compile
corpus to v5.

Exact setup, build, check, and timing commands were run from the repository
root. The reusable timing driver is checked in as
`scripts/benchmark_compile_cpu.py`; its complete raw JSON output is committed
at `docs/benchmark-data/torch-compile-cpu-v5.json`. The active Conda
environment held ambient PyTorch 2.14, so the PyTorch 2.13 reference evidence
used this worktree's local `.venv`; uv and Cargo state were redirected under
`target/`.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12 .venv
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --python "$PWD/.venv/bin/python" \
  --no-install-project --group dev --group reference
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  PYO3_PYTHON="$(which python)" \
  maturin build --release --locked --out target/test-wheels
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv pip install --python "$PWD/.venv/bin/python" --reinstall --no-deps \
  target/test-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
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

Results: the focused compile, top-level `torch.compile`, evaluator, and README
smoke tests passed 95 tests under pinned PyTorch 2.13. The public coverage
evaluator reported 32.0/100 with all 13 reference-eligible public cases passing.
`cargo fmt --check` passed. The CPU timing artifact validated against this
markdown report. The benchmark uses fixed single-core affinity, 7 warmups, and
31 measured samples for each implementation order.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.12.14+meta
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `.venv/lib/python3.12/site-packages/torch_rs/__init__.py`
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
eager program. The `torch_rs` result is also checked against the PyTorch result.
After every warmup and measured block, the driver materializes the last output
and records a 64-bit BLAKE2b checksum over values and metadata. All 91 timed
cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is
generated by the reusable fixed-affinity driver, uses equivalent work in both
implementation orders, pins the reference version, materializes and checks
outputs instead of timing dead code, keeps held-out corpus cases in differential
tests, validates guard sequences separately from timed cells, and retains every
unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v5.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v5`
- Cold first compiled call: 0.026x uncapped, 0.114x capped
- Steady-state materialized compiled call: 1.818x uncapped, 1.818x capped
- Timed supported cells: 91 (35 tensor-arithmetic, 28 broadcasting, 7 mutation/aliasing/view, 21 recompilation-guard)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 32.0% supported by native compile cases, 68% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 362.704 | 81685.529 | 0.004x | 25.119 +/- 0.135 | 13.581 +/- 0.162 | 1.850x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 217.929 | 24186.409 | 0.009x | 21.669 +/- 0.087 | 13.390 +/- 0.120 | 1.618x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 241.330 | 25338.781 | 0.010x | 23.625 +/- 0.115 | 13.580 +/- 0.211 | 1.740x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 829.109 | 25066.990 | 0.033x | 29.781 +/- 0.188 | 18.471 +/- 0.148 | 1.612x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8624.237 | 33522.077 | 0.257x | 540.710 +/- 3.258 | 538.408 +/- 6.367 | 1.004x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 259.337 | 26634.440 | 0.010x | 24.349 +/- 0.180 | 12.817 +/- 0.050 | 1.900x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 850.325 | 28941.032 | 0.029x | 31.782 +/- 0.261 | 19.123 +/- 0.138 | 1.662x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 194.128 | 23600.287 | 0.008x | 19.868 +/- 0.118 | 12.155 +/- 0.096 | 1.635x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 169.471 | 23166.622 | 0.007x | 17.139 +/- 0.068 | 12.034 +/- 0.082 | 1.424x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 189.656 | 22467.179 | 0.008x | 18.953 +/- 0.135 | 11.923 +/- 0.148 | 1.590x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 800.961 | 22625.600 | 0.035x | 25.287 +/- 0.207 | 17.179 +/- 0.177 | 1.472x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9080.077 | 35366.821 | 0.257x | 565.588 +/- 4.886 | 555.203 +/- 5.852 | 1.019x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 221.480 | 25553.816 | 0.009x | 19.476 +/- 0.097 | 11.680 +/- 0.051 | 1.668x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 880.641 | 23606.062 | 0.037x | 27.253 +/- 0.238 | 17.140 +/- 0.147 | 1.590x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 218.310 | 21850.873 | 0.010x | 24.873 +/- 0.209 | 13.717 +/- 0.182 | 1.813x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 195.546 | 21228.838 | 0.009x | 21.603 +/- 0.107 | 13.421 +/- 0.147 | 1.610x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 227.940 | 21460.593 | 0.011x | 23.760 +/- 0.093 | 13.496 +/- 0.168 | 1.761x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 806.274 | 23440.702 | 0.034x | 29.705 +/- 0.220 | 18.435 +/- 0.221 | 1.611x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8467.806 | 32515.416 | 0.260x | 569.619 +/- 29.927 | 534.316 +/- 4.720 | 1.066x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 317.796 | 22903.640 | 0.014x | 24.357 +/- 0.272 | 12.885 +/- 0.043 | 1.890x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 822.027 | 25877.957 | 0.032x | 31.305 +/- 0.140 | 18.860 +/- 0.159 | 1.660x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 289.543 | 24023.092 | 0.012x | 36.421 +/- 0.131 | 17.089 +/- 0.168 | 2.131x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 271.435 | 25568.874 | 0.011x | 36.629 +/- 0.146 | 17.072 +/- 0.349 | 2.146x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 312.057 | 25897.130 | 0.012x | 40.562 +/- 0.236 | 17.179 +/- 0.357 | 2.361x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 886.831 | 27737.502 | 0.032x | 48.298 +/- 0.344 | 22.720 +/- 0.288 | 2.126x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8516.840 | 33879.993 | 0.251x | 561.534 +/- 2.667 | 544.176 +/- 6.050 | 1.032x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 322.157 | 25084.302 | 0.013x | 41.061 +/- 0.108 | 15.922 +/- 0.059 | 2.579x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 922.044 | 26289.679 | 0.035x | 51.989 +/- 0.330 | 23.367 +/- 0.248 | 2.225x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 330.320 | 25696.542 | 0.013x | 42.937 +/- 0.203 | 16.240 +/- 0.162 | 2.644x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 308.587 | 24481.220 | 0.013x | 38.397 +/- 0.083 | 17.663 +/- 0.283 | 2.174x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 336.619 | 26139.937 | 0.013x | 42.234 +/- 0.257 | 17.602 +/- 0.388 | 2.399x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 951.694 | 29794.673 | 0.032x | 50.523 +/- 0.298 | 23.333 +/- 0.447 | 2.165x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8947.407 | 37217.023 | 0.240x | 587.803 +/- 5.499 | 619.819 +/- 20.716 | 0.948x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 368.882 | 27795.711 | 0.013x | 43.095 +/- 0.186 | 15.960 +/- 0.065 | 2.700x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 967.628 | 27202.664 | 0.036x | 57.112 +/- 0.213 | 23.948 +/- 0.191 | 2.385x | `fedf1f495675c5ac` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 298.452 | 25100.165 | 0.012x | 42.939 +/- 0.210 | 16.802 +/- 0.422 | 2.556x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 891.122 | 27947.971 | 0.032x | 48.460 +/- 0.298 | 21.870 +/- 0.230 | 2.216x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8779.207 | 35271.211 | 0.249x | 589.836 +/- 4.689 | 571.338 +/- 3.704 | 1.032x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 993.442 | 26189.037 | 0.038x | 47.621 +/- 0.351 | 21.981 +/- 0.205 | 2.166x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 925.338 | 28812.142 | 0.032x | 46.209 +/- 0.235 | 22.082 +/- 0.177 | 2.093x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 287.841 | 25286.381 | 0.011x | 42.456 +/- 0.123 | 15.230 +/- 0.055 | 2.788x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 939.370 | 28745.632 | 0.033x | 70.716 +/- 0.348 | 22.744 +/- 0.140 | 3.109x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 263.394 | 24601.602 | 0.011x | 30.996 +/- 0.118 | 14.833 +/- 0.174 | 2.090x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 874.207 | 24021.670 | 0.036x | 36.290 +/- 0.250 | 19.299 +/- 0.117 | 1.880x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8854.275 | 35348.633 | 0.250x | 581.548 +/- 11.826 | 557.246 +/- 3.598 | 1.044x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 883.986 | 28383.725 | 0.031x | 35.022 +/- 0.206 | 19.419 +/- 0.217 | 1.803x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 833.154 | 26688.156 | 0.031x | 36.129 +/- 0.178 | 19.213 +/- 0.128 | 1.880x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 253.103 | 25135.104 | 0.010x | 30.939 +/- 0.137 | 13.305 +/- 0.070 | 2.325x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 883.155 | 27883.483 | 0.032x | 55.980 +/- 0.344 | 19.685 +/- 0.147 | 2.844x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 246.933 | 23609.847 | 0.010x | 30.983 +/- 0.105 | 15.327 +/- 0.146 | 2.021x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 872.825 | 23791.687 | 0.037x | 36.584 +/- 0.201 | 19.512 +/- 0.139 | 1.875x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8931.757 | 31895.639 | 0.280x | 581.169 +/- 4.534 | 573.612 +/- 3.701 | 1.013x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 893.045 | 24687.707 | 0.036x | 35.993 +/- 0.227 | 19.298 +/- 0.128 | 1.865x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 836.940 | 23457.592 | 0.036x | 36.164 +/- 0.260 | 19.368 +/- 0.170 | 1.867x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 300.655 | 22813.563 | 0.013x | 31.091 +/- 0.171 | 13.446 +/- 0.077 | 2.312x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 883.120 | 24758.809 | 0.036x | 57.764 +/- 0.443 | 20.367 +/- 0.184 | 2.836x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 255.436 | 23636.503 | 0.011x | 31.321 +/- 0.161 | 14.476 +/- 0.131 | 2.164x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 828.582 | 23842.314 | 0.035x | 35.744 +/- 0.239 | 18.586 +/- 0.158 | 1.923x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8443.809 | 32605.722 | 0.259x | 550.885 +/- 2.283 | 531.514 +/- 3.892 | 1.036x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 881.863 | 24207.992 | 0.036x | 35.135 +/- 0.283 | 18.664 +/- 0.188 | 1.882x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 860.311 | 26539.491 | 0.032x | 36.170 +/- 0.166 | 18.866 +/- 0.232 | 1.917x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 243.168 | 25661.508 | 0.009x | 30.921 +/- 0.231 | 12.981 +/- 0.048 | 2.382x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1062.341 | 26596.247 | 0.040x | 56.606 +/- 0.939 | 19.297 +/- 0.306 | 2.933x | `2bd384aefcaaa397` |
| `cpu_float32_t_view` | `case_default` | 1 | 256 | shape (3, 2), stride (1, 3), offset 0, torch.float32, cpu, requires_grad=True | 182.636 | 24311.088 | 0.008x | 18.421 +/- 0.116 | 12.535 +/- 0.808 | 1.470x | `9382f02b63ce0129` |
| `cpu_float32_t_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 169.266 | 22543.445 | 0.008x | 15.834 +/- 0.069 | 11.263 +/- 0.134 | 1.406x | `e75a1d3233117514` |
| `cpu_float32_t_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 194.980 | 24698.588 | 0.008x | 17.220 +/- 0.119 | 11.159 +/- 0.110 | 1.543x | `4c3dc265c5b9d697` |
| `cpu_float32_t_view` | `matrix_31x37` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 789.749 | 23010.461 | 0.034x | 22.854 +/- 0.139 | 16.304 +/- 0.210 | 1.402x | `4ba5419e2e3f2393` |
| `cpu_float32_t_view` | `matrix_127x131` | 1 | 16 | shape (131, 127), stride (1, 131), offset 0, torch.float32, cpu, requires_grad=False | 8756.283 | 31419.929 | 0.279x | 540.139 +/- 3.699 | 541.172 +/- 5.384 | 0.998x | `992050cd3907bd7f` |
| `cpu_float32_t_view` | `empty_2x0` | 1 | 2048 | shape (0, 2), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 223.893 | 24560.570 | 0.009x | 18.200 +/- 0.140 | 11.267 +/- 0.074 | 1.615x | `97d8efc321a43d95` |
| `cpu_float32_t_view` | `transpose_37x31` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 794.732 | 24451.470 | 0.033x | 22.898 +/- 0.202 | 16.277 +/- 0.122 | 1.407x | `5ccc89fb94f689e5` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 277.915 | 25739.302 | 0.011x | 31.536 +/- 0.193 | 15.106 +/- 0.183 | 2.088x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 259.702 | 24629.213 | 0.011x | 27.244 +/- 0.065 | 15.147 +/- 0.186 | 1.799x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 281.466 | 23287.771 | 0.012x | 29.984 +/- 0.086 | 15.060 +/- 0.159 | 1.991x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 647.684 | 24336.245 | 0.027x | 35.335 +/- 0.208 | 18.566 +/- 0.200 | 1.903x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5382.166 | 30595.363 | 0.176x | 350.838 +/- 2.116 | 344.896 +/- 4.248 | 1.017x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 288.642 | 24124.626 | 0.012x | 30.606 +/- 0.160 | 14.349 +/- 0.093 | 2.133x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 692.482 | 25988.814 | 0.027x | 39.010 +/- 0.277 | 19.231 +/- 0.196 | 2.029x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 284.775 | 28554.027 | 0.010x | 37.030 +/- 0.193 | 15.871 +/- 0.246 | 2.333x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 872.950 | 26827.216 | 0.033x | 42.231 +/- 0.210 | 20.910 +/- 0.270 | 2.020x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8806.142 | 34366.469 | 0.256x | 575.636 +/- 2.724 | 564.845 +/- 4.916 | 1.019x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 936.255 | 26406.926 | 0.035x | 40.672 +/- 0.261 | 20.703 +/- 0.159 | 1.965x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 855.589 | 26227.719 | 0.033x | 41.936 +/- 0.232 | 20.530 +/- 0.187 | 2.043x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 272.522 | 28003.109 | 0.010x | 36.239 +/- 0.117 | 14.348 +/- 0.088 | 2.526x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 924.558 | 28148.789 | 0.033x | 62.949 +/- 0.360 | 21.359 +/- 0.222 | 2.947x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 266.678 | 23557.609 | 0.011x | 31.251 +/- 0.143 | 15.196 +/- 0.153 | 2.056x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 242.416 | 24574.881 | 0.010x | 26.957 +/- 0.055 | 15.018 +/- 0.188 | 1.795x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 274.895 | 24796.191 | 0.011x | 29.686 +/- 0.074 | 15.394 +/- 0.672 | 1.928x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 633.642 | 23781.287 | 0.027x | 41.888 +/- 0.791 | 18.644 +/- 0.261 | 2.247x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6492.570 | 35287.531 | 0.184x | 415.422 +/- 8.629 | 341.015 +/- 3.350 | 1.218x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 287.254 | 23546.556 | 0.012x | 30.412 +/- 0.105 | 14.130 +/- 0.101 | 2.152x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 660.473 | 29141.740 | 0.023x | 38.610 +/- 0.157 | 19.117 +/- 0.161 | 2.020x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 900.551 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 764.274 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 544.955 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 164517.020 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 133412.127 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 77368.863 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 135594.045 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 108215.004 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 74751.176 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 963.637 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 817.967 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 585.424 |

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
