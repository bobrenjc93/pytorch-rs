# `torch.compile` Eager CPU Release Timings

Date: 2026-09-04

Candidate provenance: source snapshot refreshed against
`c418ac98e715f1b4833ca61fcebb50aa493444a9`, plus the worktree changes that
add one exact same-module helper-call inline path for `torch.compile` and
refresh the raw benchmark artifact for corpus v9.

Exact setup, build, focused check, and timing commands were run from the
repository root. The reusable timing driver is checked in as
`scripts/benchmark_compile_cpu.py`; its complete raw JSON output is committed
at `docs/benchmark-data/torch-compile-cpu-v4.json`. The PyTorch 2.13 reference
evidence used this worktree's local `.venv`; uv and Cargo state were redirected
under `target/`.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv sync --locked --no-install-project --group dev --group reference
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_top_level_compile tests.test_compile_corpus \
  tests.test_torch_compile_coverage_evaluator tests.test_readme_quickstart
bash scripts/evaluate_torch_compile_coverage.sh
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  TMPDIR="$PWD/target" XDG_CACHE_HOME="$PWD/target/xdg-cache" \
  TORCHINDUCTOR_CACHE_DIR="$PWD/target/torchinductor-cache" \
  TRITON_CACHE_DIR="$PWD/target/triton-cache" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python scripts/benchmark_compile_cpu.py \
  --require-single-cpu-affinity \
  --output docs/benchmark-data/torch-compile-cpu-v4.json
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python scripts/benchmark_compile_cpu.py \
  --render-markdown-summary docs/benchmark-data/torch-compile-cpu-v4.json \
  > target/torch-compile-cpu-v4-summary.md
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python scripts/benchmark_compile_cpu.py --validate-artifact
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_compile_benchmark_artifact tests.test_compile_corpus
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m py_compile \
  python/torch_rs/__init__.py python/torch_rs/_compile_bytecode.py \
  python/torch_rs/_compile_trace.py scripts/evaluate_torch_compile_coverage.py \
  scripts/benchmark_compile_cpu.py tests/test_compile_benchmark_artifact.py \
  tests/test_compile_corpus.py tests/test_top_level_compile.py \
  tests/test_torch_compile_coverage_evaluator.py tests/test_readme_quickstart.py
cargo fmt --check
```

Checks run for this evidence:

```bash
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python scripts/benchmark_compile_cpu.py --validate-artifact
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_compile_benchmark_artifact tests.test_compile_corpus
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m py_compile \
  python/torch_rs/__init__.py python/torch_rs/_compile_bytecode.py \
  python/torch_rs/_compile_trace.py scripts/evaluate_torch_compile_coverage.py \
  scripts/benchmark_compile_cpu.py tests/test_compile_benchmark_artifact.py \
  tests/test_compile_corpus.py tests/test_top_level_compile.py \
  tests/test_torch_compile_coverage_evaluator.py tests/test_readme_quickstart.py
cargo fmt --check
```

Results: the full compile-coverage evaluator, fixed-affinity CPU benchmark,
raw-artifact/markdown validation, focused compile/docs unittest suite, focused
compile-corpus plus benchmark-artifact unittest suite, Python bytecode
compilation, and `cargo fmt --check` passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_052f01ed/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_052f01ed/python/torch_rs/__init__.py`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`,
  `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` used the worktree-local uv cache
- Build time: release editable build completed in 36.43s

The benchmark uses the checked-in `torch_compile_corpus_v9` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input training-autograd program, one one-input square decomposition program, one one-input custom-function helper-inline program, four two-input broadcasting programs, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Detach cells return shared-storage aliases with `requires_grad=False`. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Custom-function cells verify the same-module helper inline path over tensor proxy arguments with `neg`, `abs`, `add`, `relu`, and `detach` operations. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 119 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v9`
- Cold first compiled call: 0.028x uncapped, 0.114x capped
- Steady-state materialized compiled call: 3.705x uncapped, 3.705x capped
- Timed supported cells: 119 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 decomposition, 7 custom-functions, 21 recompilation-guard, 7 mutation_aliasing_views)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 58.0% supported by native compile cases, 42% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 376.440 | 74406.311 | 0.005x | 59.177 +/- 0.325 | 13.770 +/- 0.136 | 4.297x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 230.478 | 21699.727 | 0.011x | 54.845 +/- 0.315 | 13.458 +/- 0.117 | 4.075x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 216.722 | 22595.296 | 0.010x | 57.558 +/- 0.281 | 13.626 +/- 0.206 | 4.224x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 815.548 | 22673.805 | 0.036x | 64.021 +/- 0.286 | 18.725 +/- 0.103 | 3.419x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8541.252 | 30202.691 | 0.283x | 575.649 +/- 3.593 | 531.816 +/- 3.647 | 1.082x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 261.981 | 21593.526 | 0.012x | 58.274 +/- 0.532 | 13.151 +/- 0.120 | 4.431x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 822.078 | 23722.336 | 0.035x | 65.186 +/- 0.409 | 19.205 +/- 0.160 | 3.394x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 196.667 | 21052.208 | 0.009x | 42.372 +/- 0.388 | 12.255 +/- 0.048 | 3.458x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 156.601 | 20816.492 | 0.008x | 40.386 +/- 1.272 | 12.069 +/- 0.116 | 3.346x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 197.639 | 21198.098 | 0.009x | 45.524 +/- 0.255 | 12.152 +/- 0.135 | 3.746x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 827.957 | 21439.178 | 0.039x | 52.727 +/- 0.353 | 17.091 +/- 0.100 | 3.085x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9557.289 | 30145.946 | 0.317x | 639.885 +/- 6.134 | 546.454 +/- 2.849 | 1.171x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 232.376 | 22693.515 | 0.010x | 42.306 +/- 0.477 | 11.836 +/- 0.041 | 3.574x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 786.654 | 22255.156 | 0.035x | 49.742 +/- 0.491 | 17.133 +/- 0.137 | 2.903x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 245.322 | 21365.766 | 0.011x | 59.124 +/- 0.322 | 13.889 +/- 0.106 | 4.257x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 204.435 | 20999.819 | 0.010x | 54.911 +/- 0.241 | 13.610 +/- 0.108 | 4.035x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 218.245 | 21336.673 | 0.010x | 57.649 +/- 0.257 | 13.678 +/- 0.119 | 4.215x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 811.512 | 22173.488 | 0.037x | 64.340 +/- 0.414 | 18.425 +/- 0.155 | 3.492x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8404.130 | 32645.446 | 0.257x | 587.512 +/- 6.191 | 529.539 +/- 2.854 | 1.109x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 297.174 | 21196.426 | 0.014x | 58.052 +/- 0.444 | 13.058 +/- 0.047 | 4.446x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 872.545 | 23146.300 | 0.038x | 65.232 +/- 0.270 | 20.294 +/- 0.673 | 3.214x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 310.459 | 22972.107 | 0.014x | 98.763 +/- 0.784 | 18.461 +/- 0.859 | 5.350x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 317.691 | 22149.472 | 0.014x | 98.556 +/- 0.650 | 17.154 +/- 0.215 | 5.745x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 318.372 | 22876.122 | 0.014x | 104.730 +/- 0.969 | 17.098 +/- 0.237 | 6.125x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 898.323 | 23618.328 | 0.038x | 112.940 +/- 0.798 | 22.574 +/- 0.385 | 5.003x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8585.894 | 30874.362 | 0.278x | 628.121 +/- 4.456 | 539.328 +/- 4.104 | 1.165x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 385.953 | 22819.160 | 0.017x | 103.865 +/- 0.511 | 15.943 +/- 0.087 | 6.515x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 913.025 | 23607.637 | 0.039x | 115.283 +/- 0.708 | 23.172 +/- 0.108 | 4.975x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 389.118 | 22993.468 | 0.017x | 121.775 +/- 0.539 | 16.155 +/- 0.104 | 7.538x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 351.672 | 22803.371 | 0.015x | 114.563 +/- 0.696 | 17.529 +/- 0.289 | 6.536x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 385.939 | 23633.712 | 0.016x | 121.311 +/- 0.771 | 17.493 +/- 0.409 | 6.935x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1017.363 | 23433.409 | 0.043x | 130.993 +/- 0.720 | 22.888 +/- 0.188 | 5.723x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8889.349 | 31397.523 | 0.283x | 670.708 +/- 4.388 | 562.402 +/- 5.095 | 1.193x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 470.872 | 22611.556 | 0.021x | 122.402 +/- 0.944 | 16.113 +/- 0.124 | 7.597x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1038.771 | 25112.093 | 0.041x | 137.240 +/- 1.099 | 23.562 +/- 0.136 | 5.825x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 195.962 | 22201.751 | 0.009x | 43.840 +/- 0.249 | 13.545 +/- 0.088 | 3.237x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 167.373 | 21001.841 | 0.008x | 40.946 +/- 0.178 | 13.451 +/- 0.166 | 3.044x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 186.907 | 22482.466 | 0.008x | 43.061 +/- 0.262 | 13.416 +/- 0.108 | 3.210x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 545.189 | 22048.711 | 0.025x | 47.106 +/- 0.615 | 16.486 +/- 0.148 | 2.857x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5213.985 | 26445.834 | 0.197x | 356.071 +/- 2.318 | 328.506 +/- 2.361 | 1.084x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 202.361 | 21311.019 | 0.009x | 43.444 +/- 0.328 | 13.084 +/- 0.072 | 3.321x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 556.055 | 22914.344 | 0.024x | 47.418 +/- 0.366 | 16.540 +/- 0.192 | 2.867x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 172.570 | 22735.474 | 0.008x | 41.224 +/- 0.205 | 11.153 +/- 0.115 | 3.696x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 159.746 | 21873.375 | 0.007x | 39.911 +/- 0.770 | 11.005 +/- 0.062 | 3.627x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 174.880 | 22698.597 | 0.008x | 41.202 +/- 0.208 | 11.053 +/- 0.052 | 3.728x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 805.237 | 21595.119 | 0.037x | 46.514 +/- 0.303 | 15.967 +/- 0.115 | 2.913x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8859.474 | 29727.562 | 0.298x | 559.077 +/- 2.596 | 530.345 +/- 7.432 | 1.054x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 240.794 | 21448.923 | 0.011x | 41.958 +/- 0.344 | 11.187 +/- 0.089 | 3.751x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 774.907 | 23247.302 | 0.033x | 46.430 +/- 0.198 | 15.984 +/- 0.208 | 2.905x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 347.947 | 23290.077 | 0.015x | 110.623 +/- 0.875 | 18.104 +/- 0.180 | 6.111x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 387.356 | 21795.407 | 0.018x | 101.610 +/- 0.566 | 16.263 +/- 0.233 | 6.248x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 340.059 | 23501.096 | 0.014x | 106.272 +/- 0.376 | 16.329 +/- 0.259 | 6.508x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 713.278 | 22764.097 | 0.031x | 113.766 +/- 0.710 | 19.922 +/- 0.153 | 5.711x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5384.713 | 27967.772 | 0.193x | 436.228 +/- 3.251 | 341.627 +/- 4.599 | 1.277x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 356.794 | 22787.527 | 0.016x | 106.704 +/- 0.572 | 15.580 +/- 0.509 | 6.849x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 718.271 | 26152.336 | 0.027x | 116.259 +/- 0.620 | 21.355 +/- 0.495 | 5.444x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 285.272 | 23048.567 | 0.012x | 80.750 +/- 0.250 | 16.541 +/- 0.335 | 4.882x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 256.793 | 27603.771 | 0.009x | 81.067 +/- 0.802 | 15.492 +/- 0.165 | 5.233x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 319.252 | 23269.066 | 0.014x | 84.735 +/- 0.697 | 15.242 +/- 0.141 | 5.559x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 905.415 | 23126.319 | 0.039x | 91.963 +/- 0.453 | 20.830 +/- 0.168 | 4.415x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8921.597 | 31842.987 | 0.280x | 636.866 +/- 4.447 | 569.090 +/- 3.842 | 1.119x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 394.006 | 22878.410 | 0.017x | 85.420 +/- 0.721 | 14.511 +/- 0.096 | 5.887x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 933.292 | 24417.262 | 0.038x | 96.030 +/- 0.654 | 21.468 +/- 0.106 | 4.473x | `68585b64809ef02a` |
| `cpu_float32_custom_function_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 448.007 | 38428.036 | 0.012x | 99.632 +/- 0.494 | 18.467 +/- 0.219 | 5.395x | `d16fd2f4dd199523` |
| `cpu_float32_custom_function_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 368.332 | 23838.757 | 0.015x | 91.719 +/- 0.846 | 18.248 +/- 0.269 | 5.026x | `5c2ffe407931c8ee` |
| `cpu_float32_custom_function_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 384.591 | 24676.424 | 0.016x | 96.603 +/- 0.599 | 18.348 +/- 0.312 | 5.265x | `d85643b7b66a7ca9` |
| `cpu_float32_custom_function_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 984.905 | 25257.096 | 0.039x | 105.960 +/- 0.733 | 24.216 +/- 0.150 | 4.376x | `414eafab6fd10fb4` |
| `cpu_float32_custom_function_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8888.688 | 32912.539 | 0.270x | 642.768 +/- 5.192 | 557.375 +/- 3.508 | 1.153x | `7863bb8d1d98f49b` |
| `cpu_float32_custom_function_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 435.619 | 24586.543 | 0.018x | 98.400 +/- 0.385 | 17.015 +/- 0.085 | 5.783x | `e99a6c9902c3119e` |
| `cpu_float32_custom_function_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1024.900 | 29388.655 | 0.035x | 111.152 +/- 0.696 | 30.551 +/- 0.558 | 3.638x | `188c6817fce2e1e1` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 361.732 | 26728.942 | 0.014x | 96.917 +/- 0.593 | 20.758 +/- 0.617 | 4.669x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 912.956 | 26145.840 | 0.035x | 102.302 +/- 0.561 | 27.490 +/- 2.477 | 3.721x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8934.782 | 37874.018 | 0.236x | 651.345 +/- 3.716 | 669.229 +/- 37.845 | 0.973x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1000.964 | 27794.535 | 0.036x | 102.428 +/- 1.054 | 28.242 +/- 0.377 | 3.627x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 913.977 | 24069.922 | 0.038x | 100.974 +/- 1.124 | 21.953 +/- 0.182 | 4.599x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 310.084 | 23481.181 | 0.013x | 96.079 +/- 0.517 | 15.425 +/- 0.135 | 6.229x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 975.236 | 27632.925 | 0.035x | 124.087 +/- 0.995 | 25.609 +/- 0.446 | 4.845x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 255.782 | 25724.749 | 0.010x | 69.686 +/- 0.646 | 17.036 +/- 0.291 | 4.090x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 849.550 | 25391.405 | 0.033x | 74.805 +/- 0.537 | 22.132 +/- 0.300 | 3.380x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8843.539 | 34505.298 | 0.256x | 617.439 +/- 3.858 | 607.356 +/- 7.741 | 1.017x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 922.656 | 24997.554 | 0.037x | 72.836 +/- 0.505 | 22.040 +/- 0.949 | 3.305x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 846.029 | 23607.693 | 0.036x | 75.121 +/- 0.761 | 19.338 +/- 0.125 | 3.885x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 249.863 | 22650.390 | 0.011x | 69.198 +/- 0.344 | 13.671 +/- 0.065 | 5.062x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 879.645 | 23776.753 | 0.037x | 94.417 +/- 0.693 | 20.196 +/- 0.209 | 4.675x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 263.008 | 22678.848 | 0.012x | 64.844 +/- 0.552 | 15.457 +/- 0.136 | 4.195x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 872.439 | 22384.433 | 0.039x | 70.074 +/- 0.570 | 19.703 +/- 0.183 | 3.557x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8947.095 | 31033.633 | 0.288x | 621.630 +/- 4.658 | 569.814 +/- 5.462 | 1.091x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 950.964 | 23196.495 | 0.041x | 69.052 +/- 0.644 | 19.440 +/- 0.171 | 3.552x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 838.322 | 22671.297 | 0.037x | 69.145 +/- 0.609 | 19.412 +/- 0.101 | 3.562x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 256.648 | 21865.708 | 0.012x | 64.250 +/- 0.268 | 15.263 +/- 0.555 | 4.209x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 884.533 | 23318.391 | 0.038x | 90.465 +/- 1.077 | 20.290 +/- 0.181 | 4.459x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 263.404 | 23840.670 | 0.011x | 69.274 +/- 0.576 | 14.753 +/- 0.133 | 4.696x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 864.062 | 23303.693 | 0.037x | 74.407 +/- 0.673 | 19.296 +/- 0.499 | 3.856x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8452.733 | 33406.091 | 0.253x | 594.378 +/- 6.491 | 533.721 +/- 5.559 | 1.114x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 957.883 | 23581.083 | 0.041x | 76.585 +/- 1.207 | 18.756 +/- 0.254 | 4.083x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 847.167 | 28622.557 | 0.030x | 75.094 +/- 1.190 | 18.622 +/- 0.113 | 4.033x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 247.779 | 23360.083 | 0.011x | 69.401 +/- 0.430 | 13.214 +/- 0.057 | 5.252x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 881.823 | 23615.279 | 0.037x | 93.893 +/- 0.843 | 19.185 +/- 0.224 | 4.894x | `2bd384aefcaaa397` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 283.714 | 21827.671 | 0.013x | 86.843 +/- 0.781 | 15.367 +/- 0.211 | 5.651x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 264.586 | 22183.459 | 0.012x | 79.981 +/- 0.296 | 15.277 +/- 0.180 | 5.235x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 276.418 | 22879.757 | 0.012x | 84.101 +/- 0.444 | 15.146 +/- 0.127 | 5.553x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 641.346 | 22647.496 | 0.028x | 90.843 +/- 0.750 | 18.688 +/- 0.285 | 4.861x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5390.106 | 27683.521 | 0.195x | 419.007 +/- 7.581 | 343.162 +/- 4.062 | 1.221x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 427.221 | 23417.385 | 0.018x | 85.757 +/- 0.633 | 14.382 +/- 0.109 | 5.963x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 665.041 | 23538.734 | 0.028x | 94.167 +/- 0.852 | 19.520 +/- 0.210 | 4.824x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 314.506 | 23075.988 | 0.014x | 86.966 +/- 0.516 | 16.094 +/- 0.285 | 5.404x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 906.882 | 23764.524 | 0.038x | 93.309 +/- 0.723 | 21.010 +/- 0.240 | 4.441x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8941.898 | 31773.417 | 0.281x | 630.795 +/- 4.677 | 559.507 +/- 3.716 | 1.127x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1033.202 | 23900.190 | 0.043x | 92.084 +/- 0.741 | 20.763 +/- 0.143 | 4.435x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 896.156 | 24017.502 | 0.037x | 94.002 +/- 0.400 | 20.629 +/- 0.148 | 4.557x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 297.635 | 23092.067 | 0.013x | 86.318 +/- 0.308 | 14.572 +/- 0.078 | 5.923x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 996.096 | 23802.172 | 0.042x | 113.033 +/- 0.673 | 21.125 +/- 0.187 | 5.351x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 304.155 | 21782.677 | 0.014x | 86.414 +/- 0.455 | 15.269 +/- 0.139 | 5.659x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 271.400 | 21517.501 | 0.013x | 80.581 +/- 0.722 | 15.103 +/- 0.167 | 5.335x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 282.762 | 22385.554 | 0.013x | 84.478 +/- 0.388 | 15.036 +/- 0.152 | 5.618x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 649.813 | 22879.916 | 0.028x | 90.403 +/- 0.528 | 19.027 +/- 0.668 | 4.751x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5870.754 | 29061.275 | 0.202x | 410.136 +/- 2.670 | 342.806 +/- 3.364 | 1.196x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 316.649 | 22496.622 | 0.014x | 85.293 +/- 0.457 | 14.198 +/- 0.066 | 6.007x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 674.870 | 23386.804 | 0.029x | 93.629 +/- 0.534 | 18.907 +/- 0.166 | 4.952x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1105.520 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 1047.415 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 899.391 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 155125.892 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 125470.805 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 75595.847 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 113759.822 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 109601.836 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 72991.719 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1375.960 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 1147.187 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 929.545 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, inference, training autograd, mutation_aliasing_views, decompositions, custom functions, and recompilation guards. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `decompositions` | 6 | Supported and timed public cases: `cpu_float32_decomposition_square_scalar` |
| `custom_functions` | 6 | Supported and timed public cases: `cpu_float32_custom_function_unary` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `python_control_flow` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `containers_pytrees` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dtype_device_transitions` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 58 / 100. Zero-credit unsupported category weight: 42 / 100.
The torch_compile_corpus_v9 corpus also keeps 2 held-out broadcasting programs, 1 held-out custom-function program, 1 held-out decomposition program, 1 held-out inference program, 1 held-out mutation_aliasing_views program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 2 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
