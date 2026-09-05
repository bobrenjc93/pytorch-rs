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
- Cold first compiled call: 0.030x uncapped, 0.115x capped
- Steady-state materialized compiled call: 1.855x uncapped, 1.855x capped
- Timed supported cells: 119 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 decomposition, 7 custom-functions, 21 recompilation-guard, 7 mutation_aliasing_views)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 58.0% supported by native compile cases, 42% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 412.649 | 75815.147 | 0.005x | 27.314 +/- 0.159 | 14.550 +/- 0.091 | 1.877x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 262.397 | 22907.362 | 0.011x | 24.026 +/- 0.160 | 14.261 +/- 0.098 | 1.685x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 267.825 | 21770.734 | 0.012x | 26.053 +/- 0.241 | 14.303 +/- 0.193 | 1.821x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 925.424 | 22431.859 | 0.041x | 31.990 +/- 0.217 | 19.089 +/- 0.172 | 1.676x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8630.351 | 30163.285 | 0.286x | 544.427 +/- 3.679 | 534.896 +/- 3.327 | 1.018x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 284.836 | 21610.521 | 0.013x | 26.313 +/- 0.137 | 13.814 +/- 0.164 | 1.905x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 866.881 | 24272.702 | 0.036x | 33.658 +/- 0.220 | 20.207 +/- 0.292 | 1.666x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 219.637 | 21255.804 | 0.010x | 22.142 +/- 0.095 | 12.938 +/- 0.135 | 1.711x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 202.746 | 21711.073 | 0.009x | 22.359 +/- 1.112 | 12.811 +/- 0.147 | 1.745x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 220.995 | 20811.482 | 0.011x | 21.009 +/- 0.071 | 12.669 +/- 0.105 | 1.658x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 844.603 | 25735.404 | 0.033x | 27.150 +/- 0.113 | 17.746 +/- 0.178 | 1.530x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8808.436 | 30175.649 | 0.292x | 565.539 +/- 4.803 | 555.195 +/- 4.730 | 1.019x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 251.771 | 21408.755 | 0.012x | 21.789 +/- 0.128 | 12.335 +/- 0.048 | 1.767x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 850.456 | 22485.599 | 0.038x | 29.489 +/- 0.223 | 18.524 +/- 0.769 | 1.592x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 280.328 | 23690.842 | 0.012x | 26.855 +/- 0.141 | 16.196 +/- 0.294 | 1.658x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 259.267 | 23575.188 | 0.011x | 23.744 +/- 0.085 | 14.186 +/- 0.139 | 1.674x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 259.403 | 22342.649 | 0.012x | 25.716 +/- 0.137 | 14.327 +/- 0.203 | 1.795x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 839.349 | 23158.778 | 0.036x | 31.874 +/- 0.172 | 19.154 +/- 0.274 | 1.664x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8611.803 | 30618.403 | 0.281x | 541.650 +/- 2.319 | 544.808 +/- 9.146 | 0.994x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 283.223 | 22328.687 | 0.013x | 26.301 +/- 0.094 | 13.735 +/- 0.111 | 1.915x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 865.870 | 25527.870 | 0.034x | 33.742 +/- 0.308 | 19.569 +/- 0.192 | 1.724x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 354.430 | 26321.149 | 0.013x | 39.023 +/- 0.121 | 17.985 +/- 0.284 | 2.170x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 336.183 | 23782.981 | 0.014x | 38.951 +/- 0.124 | 17.967 +/- 0.288 | 2.168x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 378.137 | 22597.454 | 0.017x | 42.831 +/- 0.267 | 17.750 +/- 0.199 | 2.413x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 975.310 | 23921.101 | 0.041x | 51.308 +/- 0.540 | 22.843 +/- 0.198 | 2.246x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9431.629 | 32274.052 | 0.292x | 564.589 +/- 3.797 | 540.268 +/- 3.302 | 1.045x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 391.486 | 22345.342 | 0.018x | 43.787 +/- 0.240 | 16.741 +/- 0.170 | 2.615x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 976.817 | 25125.798 | 0.039x | 55.226 +/- 0.818 | 23.942 +/- 0.256 | 2.307x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 425.538 | 24755.071 | 0.017x | 45.561 +/- 0.245 | 16.923 +/- 0.083 | 2.692x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 411.031 | 23068.681 | 0.018x | 40.707 +/- 0.141 | 18.325 +/- 0.250 | 2.221x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 428.443 | 23444.730 | 0.018x | 44.508 +/- 0.197 | 18.172 +/- 0.357 | 2.449x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1025.867 | 23798.510 | 0.043x | 53.506 +/- 0.341 | 24.118 +/- 0.532 | 2.218x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9016.776 | 31959.928 | 0.282x | 589.797 +/- 4.121 | 574.336 +/- 12.010 | 1.027x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 481.578 | 23401.404 | 0.021x | 45.390 +/- 0.156 | 16.813 +/- 0.061 | 2.700x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1072.717 | 24932.370 | 0.043x | 60.070 +/- 0.263 | 24.834 +/- 0.519 | 2.419x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 275.682 | 22263.213 | 0.012x | 21.313 +/- 0.108 | 14.040 +/- 0.118 | 1.518x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 208.430 | 20933.662 | 0.010x | 19.058 +/- 0.092 | 13.808 +/- 0.085 | 1.380x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 228.445 | 24132.566 | 0.009x | 20.536 +/- 0.132 | 13.901 +/- 0.198 | 1.477x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 587.704 | 22057.768 | 0.027x | 24.556 +/- 0.172 | 16.843 +/- 0.177 | 1.458x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5313.466 | 26040.911 | 0.204x | 337.535 +/- 3.188 | 327.454 +/- 1.800 | 1.031x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 248.585 | 20913.552 | 0.012x | 21.110 +/- 0.097 | 13.538 +/- 0.056 | 1.559x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 590.378 | 23147.345 | 0.026x | 25.574 +/- 0.274 | 16.855 +/- 0.103 | 1.517x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 217.935 | 21663.101 | 0.010x | 19.240 +/- 0.059 | 11.525 +/- 0.093 | 1.669x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 227.124 | 20472.390 | 0.011x | 18.091 +/- 0.078 | 11.354 +/- 0.042 | 1.593x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 220.428 | 20922.796 | 0.011x | 19.542 +/- 0.105 | 11.537 +/- 0.073 | 1.694x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 803.485 | 22867.848 | 0.035x | 24.794 +/- 0.157 | 16.178 +/- 0.123 | 1.533x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8667.407 | 29917.208 | 0.290x | 544.438 +/- 4.142 | 531.513 +/- 2.901 | 1.024x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 269.032 | 22604.343 | 0.012x | 20.192 +/- 0.134 | 11.517 +/- 0.067 | 1.753x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 851.893 | 26654.904 | 0.032x | 24.778 +/- 0.258 | 16.541 +/- 0.375 | 1.498x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 386.113 | 25771.924 | 0.015x | 40.631 +/- 0.153 | 18.932 +/- 0.230 | 2.146x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 417.081 | 22384.241 | 0.019x | 34.843 +/- 0.175 | 16.917 +/- 0.288 | 2.060x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 378.773 | 24176.185 | 0.016x | 38.409 +/- 0.143 | 16.927 +/- 0.252 | 2.269x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 761.150 | 24352.052 | 0.031x | 44.008 +/- 0.199 | 20.257 +/- 0.114 | 2.172x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5479.341 | 28281.645 | 0.194x | 363.433 +/- 2.348 | 341.302 +/- 2.273 | 1.065x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 415.989 | 23146.098 | 0.018x | 38.825 +/- 0.259 | 15.741 +/- 0.059 | 2.466x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 783.354 | 24570.457 | 0.032x | 49.013 +/- 0.266 | 21.062 +/- 0.136 | 2.327x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 312.337 | 23216.680 | 0.013x | 30.308 +/- 0.125 | 16.141 +/- 0.126 | 1.878x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 299.884 | 22273.599 | 0.013x | 30.411 +/- 0.161 | 16.186 +/- 0.187 | 1.879x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 337.606 | 25230.961 | 0.013x | 32.901 +/- 0.154 | 16.035 +/- 0.144 | 2.052x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 948.935 | 23280.927 | 0.041x | 40.215 +/- 0.156 | 21.350 +/- 0.148 | 1.884x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9145.865 | 31613.232 | 0.289x | 583.576 +/- 3.856 | 565.997 +/- 2.115 | 1.031x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 387.160 | 23659.114 | 0.016x | 33.441 +/- 0.131 | 15.704 +/- 0.435 | 2.129x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 964.880 | 24178.885 | 0.040x | 43.861 +/- 0.196 | 22.120 +/- 0.117 | 1.983x | `68585b64809ef02a` |
| `cpu_float32_custom_function_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 438.649 | 41151.242 | 0.011x | 54.480 +/- 1.295 | 19.191 +/- 0.221 | 2.839x | `d16fd2f4dd199523` |
| `cpu_float32_custom_function_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 455.659 | 23809.957 | 0.019x | 46.674 +/- 0.177 | 19.246 +/- 0.308 | 2.425x | `5c2ffe407931c8ee` |
| `cpu_float32_custom_function_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 493.702 | 24731.436 | 0.020x | 50.936 +/- 0.289 | 19.075 +/- 0.287 | 2.670x | `d85643b7b66a7ca9` |
| `cpu_float32_custom_function_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1032.967 | 24869.249 | 0.042x | 59.453 +/- 0.261 | 24.366 +/- 0.163 | 2.440x | `414eafab6fd10fb4` |
| `cpu_float32_custom_function_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8978.863 | 34401.259 | 0.261x | 594.871 +/- 4.589 | 565.443 +/- 4.558 | 1.052x | `7863bb8d1d98f49b` |
| `cpu_float32_custom_function_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 465.139 | 24656.793 | 0.019x | 52.524 +/- 0.491 | 17.536 +/- 0.190 | 2.995x | `e99a6c9902c3119e` |
| `cpu_float32_custom_function_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1066.333 | 26389.734 | 0.040x | 65.203 +/- 0.753 | 25.832 +/- 0.256 | 2.524x | `188c6817fce2e1e1` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 354.125 | 25867.283 | 0.014x | 45.438 +/- 0.260 | 17.730 +/- 0.179 | 2.563x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 949.721 | 27777.927 | 0.034x | 51.199 +/- 0.244 | 22.671 +/- 0.235 | 2.258x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9041.148 | 34371.885 | 0.263x | 594.938 +/- 3.312 | 567.223 +/- 3.851 | 1.049x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1025.595 | 25028.014 | 0.041x | 49.677 +/- 0.378 | 22.696 +/- 0.239 | 2.189x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1057.569 | 24811.612 | 0.043x | 49.222 +/- 0.409 | 22.471 +/- 0.156 | 2.190x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 335.953 | 24085.088 | 0.014x | 44.937 +/- 0.228 | 16.082 +/- 0.098 | 2.794x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1038.801 | 25987.531 | 0.040x | 71.865 +/- 0.505 | 23.574 +/- 0.171 | 3.049x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 292.733 | 23049.442 | 0.013x | 33.911 +/- 0.507 | 15.811 +/- 0.188 | 2.145x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 900.827 | 24620.058 | 0.037x | 38.795 +/- 0.419 | 20.015 +/- 0.129 | 1.938x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9419.145 | 32429.478 | 0.290x | 590.535 +/- 4.431 | 562.162 +/- 5.867 | 1.050x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 945.820 | 22880.122 | 0.041x | 37.651 +/- 0.280 | 20.056 +/- 0.126 | 1.877x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 896.114 | 23385.556 | 0.038x | 38.523 +/- 0.192 | 19.795 +/- 0.156 | 1.946x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 280.900 | 22786.570 | 0.012x | 33.467 +/- 0.244 | 14.190 +/- 0.054 | 2.358x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 957.934 | 24477.953 | 0.039x | 57.056 +/- 0.369 | 20.349 +/- 0.140 | 2.804x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 283.784 | 22813.245 | 0.012x | 33.588 +/- 0.253 | 16.099 +/- 0.106 | 2.086x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 903.246 | 23175.037 | 0.039x | 39.411 +/- 0.390 | 20.254 +/- 0.115 | 1.946x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9463.221 | 31200.203 | 0.303x | 598.827 +/- 4.312 | 565.113 +/- 3.540 | 1.060x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 952.816 | 22984.490 | 0.041x | 38.748 +/- 0.311 | 20.163 +/- 0.263 | 1.922x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 909.545 | 22645.090 | 0.040x | 38.726 +/- 0.212 | 20.127 +/- 0.268 | 1.924x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 296.214 | 22584.735 | 0.013x | 33.492 +/- 0.143 | 14.322 +/- 0.066 | 2.339x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 918.328 | 24099.726 | 0.038x | 59.043 +/- 0.304 | 21.216 +/- 0.153 | 2.783x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 327.345 | 23313.783 | 0.014x | 33.738 +/- 0.227 | 15.364 +/- 0.153 | 2.196x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 864.842 | 22976.888 | 0.038x | 38.521 +/- 0.169 | 19.500 +/- 0.254 | 1.975x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8515.212 | 31918.846 | 0.267x | 555.003 +/- 2.855 | 541.837 +/- 4.021 | 1.024x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 927.138 | 24205.771 | 0.038x | 37.412 +/- 0.247 | 19.374 +/- 0.160 | 1.931x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 881.287 | 23547.621 | 0.037x | 38.652 +/- 0.214 | 19.327 +/- 0.113 | 2.000x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 279.823 | 22435.615 | 0.012x | 33.186 +/- 0.137 | 13.744 +/- 0.093 | 2.415x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 939.660 | 24701.331 | 0.038x | 57.323 +/- 0.378 | 19.799 +/- 0.142 | 2.895x | `2bd384aefcaaa397` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 337.020 | 22201.861 | 0.015x | 34.259 +/- 0.183 | 16.167 +/- 0.190 | 2.119x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 310.439 | 23122.643 | 0.013x | 29.834 +/- 0.096 | 15.920 +/- 0.210 | 1.874x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 319.243 | 23343.928 | 0.014x | 32.535 +/- 0.142 | 15.786 +/- 0.148 | 2.061x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 707.019 | 23490.359 | 0.030x | 37.799 +/- 0.190 | 19.167 +/- 0.124 | 1.972x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5598.457 | 28746.398 | 0.195x | 359.632 +/- 2.118 | 343.816 +/- 2.632 | 1.046x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 338.863 | 21986.119 | 0.015x | 33.383 +/- 0.100 | 14.940 +/- 0.054 | 2.234x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 748.322 | 23958.262 | 0.031x | 41.598 +/- 0.258 | 19.994 +/- 0.159 | 2.080x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 346.374 | 23131.372 | 0.015x | 39.400 +/- 0.270 | 16.709 +/- 0.129 | 2.358x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 954.088 | 23989.885 | 0.040x | 44.955 +/- 0.329 | 21.567 +/- 0.155 | 2.084x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9136.432 | 31912.656 | 0.286x | 590.735 +/- 3.696 | 560.041 +/- 3.391 | 1.055x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1035.146 | 23592.374 | 0.044x | 43.644 +/- 0.335 | 21.506 +/- 0.127 | 2.029x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 922.145 | 23488.026 | 0.039x | 44.680 +/- 0.276 | 21.476 +/- 0.151 | 2.080x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 427.617 | 25831.014 | 0.017x | 38.868 +/- 0.187 | 15.283 +/- 0.130 | 2.543x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 980.372 | 26944.953 | 0.036x | 64.106 +/- 0.438 | 21.991 +/- 0.171 | 2.915x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 333.038 | 25597.912 | 0.013x | 33.779 +/- 0.186 | 22.309 +/- 0.660 | 1.514x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 311.876 | 22169.467 | 0.014x | 29.535 +/- 0.108 | 15.862 +/- 0.205 | 1.862x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 323.630 | 22932.371 | 0.014x | 32.288 +/- 0.243 | 15.608 +/- 0.147 | 2.069x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 696.368 | 22749.294 | 0.031x | 37.551 +/- 0.318 | 19.605 +/- 0.373 | 1.915x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5479.521 | 27862.211 | 0.197x | 357.623 +/- 2.950 | 339.432 +/- 2.623 | 1.054x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 343.459 | 22730.970 | 0.015x | 33.040 +/- 0.170 | 14.819 +/- 0.071 | 2.230x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 701.711 | 27693.986 | 0.025x | 41.327 +/- 0.258 | 19.894 +/- 0.169 | 2.077x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 985.500 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 824.917 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 629.292 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 166523.673 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 132977.173 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 74709.283 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 114173.495 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 105417.082 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 71356.292 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 995.375 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 884.289 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 650.985 |

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
