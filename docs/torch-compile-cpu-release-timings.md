# `torch.compile` Eager CPU Release Timings

Date: 2026-09-05

Candidate provenance: source snapshot refreshed against this worktree, including zero-argument `Tensor.float()` identity graphlets in the dtype/device-transition category, one top-level `requires_grad` branch graphlet in the Python-control-flow category, module-global exact native Tensor constant capture in the modules/parameters/buffers category, and no-break `fullgraph=False` graphlets in the graph-breaks/fullgraph category. The raw benchmark artifact is refreshed for `torch_compile_corpus_v12` with the current supported public cases included.

The setup, build, focused check, and timing commands below reproduce this evidence from the repository root. The reusable timing driver is checked in as `scripts/benchmark_compile_cpu.py`; its complete raw JSON output is committed at `docs/benchmark-data/torch-compile-cpu-v4.json`. The PyTorch 2.13 reference evidence used this worktree's local `.venv`; uv and Cargo state were redirected under `target/`.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --no-install-project --group dev --group reference
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
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
  tests.test_compile_benchmark_artifact tests.test_compile_corpus \
  tests.test_top_level_compile tests.test_torch_compile_coverage_evaluator
bash scripts/evaluate_torch_compile_coverage.sh
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m py_compile \
  python/torch_rs/__init__.py python/torch_rs/_compile_bytecode.py \
  python/torch_rs/_compile_trace.py scripts/evaluate_torch_compile_coverage.py \
  scripts/benchmark_compile_cpu.py tests/test_compile_benchmark_artifact.py \
  tests/test_compile_corpus.py tests/test_top_level_compile.py \
  tests/test_torch_compile_coverage_evaluator.py
cargo fmt --check
git diff --check
```

Checks run for this evidence:

```bash
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python scripts/benchmark_compile_cpu.py --validate-artifact
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_compile_benchmark_artifact tests.test_compile_corpus \
  tests.test_top_level_compile tests.test_torch_compile_coverage_evaluator
bash scripts/evaluate_torch_compile_coverage.sh
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m py_compile \
  python/torch_rs/__init__.py python/torch_rs/_compile_bytecode.py \
  python/torch_rs/_compile_trace.py scripts/evaluate_torch_compile_coverage.py \
  scripts/benchmark_compile_cpu.py tests/test_compile_benchmark_artifact.py \
  tests/test_compile_corpus.py tests/test_top_level_compile.py \
  tests/test_torch_compile_coverage_evaluator.py
cargo fmt --check
git diff --check
```

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.12.12
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`, `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/composite_3fb574b9/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/composite_3fb574b9/python/torch_rs/__init__.py`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`; `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both reported 1
- Dependency installation: locked `uv sync` used the worktree-local uv cache
- Build: release editable wheel installed in the worktree-local `.venv`

The benchmark uses the checked-in `torch_compile_corpus_v12` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input `Tensor.float()` identity dtype/device-transition program, one one-input training-autograd program, one one-input `requires_grad` branch Python-control-flow program, one one-input no-break `fullgraph=False` graph-breaks/fullgraph program, one one-input square decomposition program, one one-input custom-function helper-inline program, one one-input module-global Tensor buffer program, four two-input broadcasting programs, one two-input containers-pytrees program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Detach cells return shared-storage aliases with `requires_grad=False`. `Tensor.float()` identity cells preserve values, shape, stride, storage offset, device, dtype, and `requires_grad`. `requires_grad` branch cells select the branch from input metadata, lower only that branch, and preserve the selected branch's output metadata. No-break `fullgraph=False` cells use the same guarded native lowering path as equivalent fullgraph graphlets and do not exercise eager fallback. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Custom-function cells verify the same-module helper inline path over tensor proxy arguments with `neg`, `abs`, `add`, `relu`, `detach`, and `float` operations. Module-global Tensor buffer cells verify captured read-only global values and metadata across the same one-input variants. Tuple/list output cells preserve container structure and record per-tensor metadata for each output leaf. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, `requires_grad` branch cache specialization, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses each corpus case's checked-in compile kwargs, including stock `torch.compile(backend="eager", fullgraph=True)` for fullgraph cases and `fullgraph=False` for the no-break graphlet. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, tuple/list container structure, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 154 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.


## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v12`
- Cold first compiled call: 0.030x uncapped, 0.114x capped
- Steady-state materialized compiled call: 1.948x uncapped, 1.948x capped
- Timed supported cells: 154 (35 tensor-arithmetic, 28 broadcasting, 7 modules-parameters-buffers, 7 inference, 7 training-autograd, 7 python-control-flow, 7 containers-pytrees, 7 decomposition, 7 custom-functions, 21 recompilation-guard, 7 dtype-device-transitions, 7 graph_breaks_fullgraph, 7 mutation_aliasing_views)
- Recompilation guard sequences: 16 rows, 72 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 92.0% supported by native compile cases, 8% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 361.351 | 76946.202 | 0.005x | 26.810 +/- 0.496 | 13.039 +/- 0.165 | 2.056x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 260.293 | 19881.467 | 0.013x | 21.660 +/- 0.870 | 12.897 +/- 0.123 | 1.679x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 270.058 | 19655.095 | 0.014x | 22.795 +/- 0.109 | 12.696 +/- 0.170 | 1.795x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 782.178 | 19661.064 | 0.040x | 28.311 +/- 0.197 | 17.118 +/- 0.202 | 1.654x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7745.466 | 26944.180 | 0.287x | 489.233 +/- 2.609 | 471.220 +/- 2.012 | 1.038x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 280.464 | 19881.177 | 0.014x | 23.315 +/- 0.083 | 11.993 +/- 0.056 | 1.944x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 782.163 | 21250.058 | 0.037x | 29.702 +/- 0.176 | 17.496 +/- 0.150 | 1.698x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 220.834 | 18922.096 | 0.012x | 19.511 +/- 0.129 | 11.410 +/- 0.108 | 1.710x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 196.787 | 18474.760 | 0.011x | 17.040 +/- 0.119 | 12.456 +/- 0.228 | 1.368x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 215.505 | 20639.509 | 0.010x | 18.595 +/- 0.111 | 12.559 +/- 0.130 | 1.481x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 721.167 | 20179.487 | 0.036x | 23.726 +/- 0.177 | 17.026 +/- 0.366 | 1.394x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7387.515 | 28429.488 | 0.260x | 472.369 +/- 3.100 | 459.416 +/- 1.894 | 1.028x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 263.769 | 19701.440 | 0.013x | 18.992 +/- 0.077 | 11.099 +/- 0.144 | 1.711x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 717.105 | 20758.504 | 0.035x | 25.419 +/- 0.138 | 15.430 +/- 0.141 | 1.647x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 243.989 | 18670.776 | 0.013x | 23.547 +/- 0.104 | 12.964 +/- 0.153 | 1.816x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 268.145 | 18775.920 | 0.014x | 20.533 +/- 0.085 | 12.773 +/- 0.128 | 1.608x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 229.662 | 19532.008 | 0.012x | 22.702 +/- 0.137 | 12.799 +/- 0.159 | 1.774x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 760.030 | 19939.325 | 0.038x | 28.260 +/- 0.242 | 17.011 +/- 0.178 | 1.661x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7659.410 | 26725.950 | 0.287x | 492.195 +/- 4.220 | 471.669 +/- 3.051 | 1.044x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 270.465 | 19527.603 | 0.014x | 23.188 +/- 0.116 | 12.189 +/- 0.101 | 1.902x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 773.315 | 20318.843 | 0.038x | 29.665 +/- 0.256 | 17.601 +/- 0.120 | 1.685x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 312.693 | 20005.215 | 0.016x | 33.536 +/- 0.185 | 16.190 +/- 0.204 | 2.071x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 293.589 | 20160.069 | 0.015x | 33.546 +/- 0.134 | 16.340 +/- 0.208 | 2.053x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 311.120 | 21748.327 | 0.014x | 37.402 +/- 0.166 | 16.077 +/- 0.230 | 2.326x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 827.787 | 21554.719 | 0.038x | 44.888 +/- 0.225 | 22.012 +/- 0.909 | 2.039x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7778.011 | 28635.233 | 0.272x | 511.687 +/- 1.960 | 478.572 +/- 2.097 | 1.069x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 356.484 | 20298.823 | 0.018x | 38.238 +/- 0.217 | 15.072 +/- 0.096 | 2.537x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 856.781 | 22233.525 | 0.039x | 47.752 +/- 0.316 | 21.757 +/- 0.167 | 2.195x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 363.895 | 21249.361 | 0.017x | 40.081 +/- 0.278 | 15.709 +/- 0.126 | 2.551x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 347.135 | 20560.585 | 0.017x | 35.129 +/- 0.106 | 17.024 +/- 0.209 | 2.064x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 371.977 | 20867.268 | 0.018x | 39.186 +/- 0.176 | 16.852 +/- 0.236 | 2.325x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 877.351 | 21084.668 | 0.042x | 46.773 +/- 0.233 | 21.731 +/- 0.144 | 2.152x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7620.882 | 28326.726 | 0.269x | 508.342 +/- 3.139 | 474.066 +/- 1.930 | 1.072x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 403.671 | 21035.372 | 0.019x | 40.057 +/- 0.102 | 15.587 +/- 0.106 | 2.570x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 912.154 | 21539.871 | 0.042x | 52.497 +/- 0.465 | 22.319 +/- 0.151 | 2.352x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 244.925 | 19225.410 | 0.013x | 18.753 +/- 0.102 | 12.310 +/- 0.075 | 1.523x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 201.665 | 18886.883 | 0.011x | 16.443 +/- 0.076 | 12.138 +/- 0.066 | 1.355x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 207.344 | 19485.083 | 0.011x | 17.906 +/- 0.106 | 12.161 +/- 0.138 | 1.472x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 529.741 | 19517.171 | 0.027x | 21.412 +/- 0.096 | 14.809 +/- 0.116 | 1.446x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4655.787 | 25383.529 | 0.183x | 299.950 +/- 1.428 | 291.080 +/- 1.085 | 1.030x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 231.200 | 18710.967 | 0.012x | 18.509 +/- 0.086 | 11.912 +/- 0.059 | 1.554x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 544.047 | 19987.708 | 0.027x | 22.162 +/- 0.092 | 14.973 +/- 0.098 | 1.480x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 189.747 | 19904.086 | 0.010x | 16.818 +/- 0.064 | 9.975 +/- 0.125 | 1.686x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 185.856 | 18703.651 | 0.010x | 15.533 +/- 0.063 | 11.311 +/- 0.133 | 1.373x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 202.687 | 20658.117 | 0.010x | 16.817 +/- 0.092 | 11.499 +/- 0.660 | 1.463x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 742.638 | 20425.385 | 0.036x | 21.672 +/- 0.135 | 14.050 +/- 0.137 | 1.543x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7668.975 | 27063.661 | 0.283x | 482.400 +/- 1.910 | 477.116 +/- 3.953 | 1.011x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 224.370 | 23320.229 | 0.010x | 17.506 +/- 0.088 | 9.881 +/- 0.095 | 1.772x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 735.953 | 21362.232 | 0.034x | 21.703 +/- 0.148 | 14.063 +/- 0.100 | 1.543x | `4ba5419e2e3f2393` |
| `cpu_float32_float_identity_view` | `case_default` | 1 | 256 | shape (3,), stride (4,), offset 1, torch.float32, cpu, requires_grad=True | 208.305 | 19993.912 | 0.010x | 16.611 +/- 0.096 | 9.657 +/- 0.062 | 1.720x | `58df67cd172620c1` |
| `cpu_float32_float_identity_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 188.960 | 18953.324 | 0.010x | 15.544 +/- 0.083 | 9.600 +/- 0.073 | 1.619x | `e75a1d3233117514` |
| `cpu_float32_float_identity_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 223.793 | 19183.071 | 0.012x | 16.707 +/- 0.082 | 9.583 +/- 0.046 | 1.743x | `4c3dc265c5b9d697` |
| `cpu_float32_float_identity_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 723.173 | 19362.037 | 0.037x | 21.294 +/- 0.143 | 13.725 +/- 0.117 | 1.551x | `5ccc89fb94f689e5` |
| `cpu_float32_float_identity_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7697.188 | 27655.787 | 0.278x | 487.472 +/- 2.368 | 477.189 +/- 2.776 | 1.022x | `91fa5699b26ca1b8` |
| `cpu_float32_float_identity_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 220.945 | 19120.126 | 0.012x | 17.325 +/- 0.099 | 9.618 +/- 0.060 | 1.801x | `e99a6c9902c3119e` |
| `cpu_float32_float_identity_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 751.558 | 20017.222 | 0.038x | 21.314 +/- 0.074 | 13.673 +/- 0.093 | 1.559x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 373.630 | 21268.476 | 0.018x | 35.757 +/- 0.187 | 17.483 +/- 0.147 | 2.045x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 361.091 | 20391.158 | 0.018x | 29.969 +/- 0.167 | 15.674 +/- 0.222 | 1.912x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 340.625 | 20289.094 | 0.017x | 33.403 +/- 0.288 | 15.763 +/- 0.213 | 2.119x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 629.187 | 20807.573 | 0.030x | 38.575 +/- 0.227 | 18.528 +/- 0.189 | 2.082x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4782.068 | 25090.641 | 0.191x | 316.797 +/- 2.270 | 289.758 +/- 1.416 | 1.093x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 347.892 | 20102.917 | 0.017x | 34.029 +/- 0.132 | 14.421 +/- 0.072 | 2.360x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 666.274 | 22500.755 | 0.030x | 42.065 +/- 0.159 | 19.248 +/- 0.236 | 2.185x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 287.419 | 20025.806 | 0.014x | 25.739 +/- 0.155 | 14.679 +/- 0.115 | 1.753x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 278.537 | 19672.486 | 0.014x | 25.739 +/- 0.070 | 14.852 +/- 0.183 | 1.733x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 290.474 | 20723.171 | 0.014x | 28.269 +/- 0.193 | 14.758 +/- 0.221 | 1.916x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 837.251 | 20923.854 | 0.040x | 34.664 +/- 0.226 | 18.911 +/- 0.146 | 1.833x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7648.379 | 28540.926 | 0.268x | 491.461 +/- 3.925 | 482.247 +/- 4.898 | 1.019x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 337.130 | 20153.975 | 0.017x | 29.158 +/- 0.179 | 13.746 +/- 0.058 | 2.121x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 866.551 | 21102.965 | 0.041x | 38.258 +/- 0.223 | 19.724 +/- 0.165 | 1.940x | `68585b64809ef02a` |
| `cpu_float32_custom_function_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 486.361 | 33638.731 | 0.014x | 115.750 +/- 0.689 | 18.166 +/- 0.189 | 6.372x | `d16fd2f4dd199523` |
| `cpu_float32_custom_function_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 388.848 | 21533.201 | 0.018x | 108.624 +/- 0.525 | 18.181 +/- 0.230 | 5.975x | `5c2ffe407931c8ee` |
| `cpu_float32_custom_function_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 430.201 | 22728.794 | 0.019x | 112.836 +/- 0.819 | 17.927 +/- 0.248 | 6.294x | `d85643b7b66a7ca9` |
| `cpu_float32_custom_function_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 887.422 | 23049.129 | 0.039x | 121.238 +/- 1.353 | 22.354 +/- 0.171 | 5.424x | `414eafab6fd10fb4` |
| `cpu_float32_custom_function_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7537.171 | 29868.789 | 0.252x | 577.779 +/- 5.556 | 470.796 +/- 2.206 | 1.227x | `7863bb8d1d98f49b` |
| `cpu_float32_custom_function_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 514.803 | 21800.545 | 0.024x | 113.728 +/- 0.407 | 16.554 +/- 0.076 | 6.870x | `e99a6c9902c3119e` |
| `cpu_float32_custom_function_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 944.569 | 24598.282 | 0.038x | 125.371 +/- 0.753 | 23.771 +/- 0.430 | 5.274x | `188c6817fce2e1e1` |
| `cpu_float32_requires_grad_branch_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 312.036 | 21283.092 | 0.015x | 24.717 +/- 0.188 | 12.882 +/- 0.127 | 1.919x | `43e5fdfc5aec3505` |
| `cpu_float32_requires_grad_branch_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 329.754 | 19176.827 | 0.017x | 21.371 +/- 0.078 | 12.821 +/- 0.174 | 1.667x | `e75a1d3233117514` |
| `cpu_float32_requires_grad_branch_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 332.208 | 20850.448 | 0.016x | 23.549 +/- 0.092 | 12.812 +/- 0.168 | 1.838x | `47aef822223dbae7` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 856.235 | 20593.003 | 0.042x | 29.601 +/- 0.149 | 16.858 +/- 0.116 | 1.756x | `2148badcc2b9e4ce` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7620.823 | 27667.579 | 0.275x | 489.856 +/- 2.138 | 484.737 +/- 3.846 | 1.011x | `d53163cb2693cd35` |
| `cpu_float32_requires_grad_branch_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 384.802 | 19820.365 | 0.019x | 24.071 +/- 0.065 | 12.053 +/- 0.075 | 1.997x | `e99a6c9902c3119e` |
| `cpu_float32_requires_grad_branch_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 876.957 | 22091.871 | 0.040x | 31.752 +/- 0.159 | 17.239 +/- 0.152 | 1.842x | `372841b6f1764798` |
| `cpu_float32_fullgraph_false_no_break_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 347.039 | 21277.905 | 0.016x | 35.017 +/- 0.194 | 13.966 +/- 0.197 | 2.507x | `81b5b7b86e1ba824` |
| `cpu_float32_fullgraph_false_no_break_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 305.467 | 21105.263 | 0.014x | 30.064 +/- 0.161 | 13.766 +/- 0.181 | 2.184x | `96474978e4b2c20f` |
| `cpu_float32_fullgraph_false_no_break_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 324.031 | 21918.022 | 0.015x | 33.130 +/- 0.100 | 13.780 +/- 0.181 | 2.404x | `3e3e1d5aa2a3441f` |
| `cpu_float32_fullgraph_false_no_break_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 833.410 | 21748.768 | 0.038x | 40.114 +/- 0.241 | 18.305 +/- 0.169 | 2.191x | `1d762530ef6c58be` |
| `cpu_float32_fullgraph_false_no_break_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7510.781 | 29058.584 | 0.258x | 491.425 +/- 2.888 | 477.167 +/- 3.324 | 1.030x | `f2db5d5c08e66799` |
| `cpu_float32_fullgraph_false_no_break_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 341.847 | 21685.946 | 0.016x | 34.116 +/- 0.201 | 12.716 +/- 0.053 | 2.683x | `e99a6c9902c3119e` |
| `cpu_float32_fullgraph_false_no_break_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 836.461 | 23358.782 | 0.036x | 43.517 +/- 0.199 | 19.055 +/- 0.260 | 2.284x | `7dd49a516859a9cd` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 315.957 | 22353.276 | 0.014x | 39.544 +/- 0.201 | 16.331 +/- 0.173 | 2.421x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 805.433 | 21976.662 | 0.037x | 44.108 +/- 0.222 | 20.537 +/- 0.157 | 2.148x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7529.254 | 29123.456 | 0.259x | 493.364 +/- 2.035 | 476.704 +/- 3.306 | 1.035x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 812.609 | 21860.611 | 0.037x | 42.942 +/- 0.250 | 20.317 +/- 0.202 | 2.114x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 818.193 | 21866.099 | 0.037x | 42.132 +/- 0.269 | 20.266 +/- 0.191 | 2.079x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 301.671 | 21578.749 | 0.014x | 38.994 +/- 0.133 | 14.781 +/- 0.069 | 2.638x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 876.582 | 22868.861 | 0.038x | 63.690 +/- 0.421 | 21.113 +/- 0.163 | 3.017x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 268.451 | 20502.031 | 0.013x | 29.075 +/- 0.181 | 14.598 +/- 0.115 | 1.992x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 816.460 | 21318.892 | 0.038x | 33.299 +/- 0.176 | 18.024 +/- 0.095 | 1.847x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7506.189 | 28095.917 | 0.267x | 484.164 +/- 3.836 | 467.754 +/- 2.871 | 1.035x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 829.374 | 20895.681 | 0.040x | 32.352 +/- 0.321 | 17.797 +/- 0.138 | 1.818x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 760.721 | 21174.628 | 0.036x | 33.494 +/- 0.266 | 17.746 +/- 0.138 | 1.887x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 259.188 | 20607.045 | 0.013x | 28.957 +/- 0.172 | 12.816 +/- 0.099 | 2.259x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 812.609 | 22360.518 | 0.036x | 50.925 +/- 0.397 | 18.058 +/- 0.132 | 2.820x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 257.925 | 20500.724 | 0.013x | 29.038 +/- 0.140 | 15.095 +/- 0.134 | 1.924x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 751.887 | 20685.499 | 0.036x | 33.818 +/- 0.174 | 18.127 +/- 0.160 | 1.866x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7436.959 | 27903.596 | 0.267x | 487.010 +/- 2.548 | 472.064 +/- 2.489 | 1.032x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 825.764 | 20892.387 | 0.040x | 33.164 +/- 0.243 | 17.850 +/- 0.219 | 1.858x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 739.328 | 20891.636 | 0.035x | 33.610 +/- 0.215 | 17.871 +/- 0.186 | 1.881x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 245.637 | 20035.836 | 0.012x | 29.070 +/- 0.099 | 13.095 +/- 0.070 | 2.220x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 788.202 | 21488.898 | 0.037x | 51.637 +/- 0.246 | 18.555 +/- 0.115 | 2.783x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 263.423 | 20718.173 | 0.013x | 29.401 +/- 0.141 | 13.961 +/- 0.103 | 2.106x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 767.387 | 20653.234 | 0.037x | 33.174 +/- 0.178 | 17.208 +/- 0.109 | 1.928x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7364.043 | 27397.762 | 0.269x | 474.888 +/- 2.547 | 448.538 +/- 1.955 | 1.059x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 826.189 | 20692.019 | 0.040x | 32.166 +/- 0.154 | 17.132 +/- 0.131 | 1.878x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 763.605 | 20627.931 | 0.037x | 33.622 +/- 0.258 | 17.253 +/- 0.202 | 1.949x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 262.512 | 20020.858 | 0.013x | 28.655 +/- 0.066 | 12.366 +/- 0.071 | 2.317x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 833.702 | 21557.743 | 0.039x | 52.150 +/- 0.322 | 17.602 +/- 0.169 | 2.963x | `2bd384aefcaaa397` |
| `cpu_float32_global_buffer_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 318.057 | 20663.449 | 0.015x | 60.761 +/- 0.361 | 14.188 +/- 0.184 | 4.283x | `dd1428515dc76c04` |
| `cpu_float32_global_buffer_add` | `scalar` | 1 | 2048 | shape (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 249.528 | 20761.133 | 0.012x | 58.992 +/- 0.225 | 13.667 +/- 0.098 | 4.316x | `5214bceaa64234ff` |
| `cpu_float32_global_buffer_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 266.488 | 21552.681 | 0.012x | 59.291 +/- 0.224 | 13.879 +/- 0.127 | 4.272x | `dbed541b43896343` |
| `cpu_float32_global_buffer_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 824.998 | 21572.535 | 0.038x | 73.607 +/- 0.518 | 18.069 +/- 0.104 | 4.074x | `e4ebd180a49a9ea8` |
| `cpu_float32_global_buffer_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8004.828 | 28974.371 | 0.276x | 654.331 +/- 3.818 | 491.428 +/- 2.636 | 1.331x | `aed7b1c611594d2a` |
| `cpu_float32_global_buffer_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 330.655 | 20598.026 | 0.016x | 61.205 +/- 0.354 | 13.739 +/- 0.086 | 4.455x | `e99a6c9902c3119e` |
| `cpu_float32_global_buffer_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 838.869 | 21713.959 | 0.039x | 84.106 +/- 0.573 | 18.347 +/- 0.175 | 4.584x | `630802db112622aa` |
| `cpu_float32_tuple_list_output_pytree` | `case_default` | 2 | 256 | tuple[shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True, list[shape (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True]] | 397.967 | 22663.525 | 0.018x | 48.282 +/- 0.385 | 18.956 +/- 0.176 | 2.547x | `a62dacb062c1ed92` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1331.630 | 22985.307 | 0.058x | 55.589 +/- 0.421 | 23.383 +/- 0.155 | 2.377x | `3bce94d7e523bafe` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_127x131_by_131` | 2 | 16 | tuple[shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (131,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 12137.530 | 33869.164 | 0.358x | 794.876 +/- 9.544 | 764.147 +/- 3.255 | 1.040x | `022557af0d301f5e` |
| `cpu_float32_tuple_list_output_pytree` | `tensor_scalar_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1293.717 | 22882.762 | 0.057x | 53.683 +/- 0.307 | 23.016 +/- 0.180 | 2.332x | `f4ff04ee55c4e2cd` |
| `cpu_float32_tuple_list_output_pytree` | `scalar_tensor_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False]] | 1379.812 | 22571.311 | 0.061x | 55.076 +/- 0.522 | 24.076 +/- 0.184 | 2.288x | `f1950b665bfdc9f1` |
| `cpu_float32_tuple_list_output_pytree` | `empty_2x0_by_0` | 2 | 2048 | tuple[shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 415.323 | 21402.939 | 0.019x | 47.034 +/- 0.195 | 14.749 +/- 0.078 | 3.189x | `e89cfed7478c41fa` |
| `cpu_float32_tuple_list_output_pytree` | `transpose_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False]] | 1315.555 | 24630.535 | 0.053x | 75.959 +/- 0.677 | 23.662 +/- 0.155 | 3.210x | `776bd23d05673f66` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 304.631 | 20074.780 | 0.015x | 29.656 +/- 0.139 | 14.658 +/- 0.116 | 2.023x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 278.466 | 20056.908 | 0.014x | 25.511 +/- 0.119 | 14.447 +/- 0.190 | 1.766x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 301.435 | 20291.287 | 0.015x | 28.319 +/- 0.091 | 14.376 +/- 0.126 | 1.970x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 629.372 | 20185.808 | 0.031x | 33.224 +/- 0.196 | 17.180 +/- 0.159 | 1.934x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4759.374 | 24769.696 | 0.192x | 307.908 +/- 3.286 | 293.440 +/- 3.541 | 1.049x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 342.382 | 21337.985 | 0.016x | 28.902 +/- 0.115 | 13.350 +/- 0.069 | 2.165x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 615.201 | 21749.168 | 0.028x | 35.914 +/- 0.223 | 17.840 +/- 0.137 | 2.013x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 304.946 | 20945.998 | 0.015x | 33.986 +/- 0.134 | 15.350 +/- 0.143 | 2.214x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 791.597 | 21190.207 | 0.037x | 38.940 +/- 0.197 | 19.557 +/- 0.235 | 1.991x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7479.789 | 28739.787 | 0.260x | 491.619 +/- 2.490 | 470.222 +/- 3.057 | 1.046x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 835.874 | 21050.236 | 0.040x | 37.645 +/- 0.235 | 19.230 +/- 0.162 | 1.958x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 821.142 | 21308.651 | 0.039x | 38.880 +/- 0.215 | 19.222 +/- 0.217 | 2.023x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 322.738 | 20788.535 | 0.016x | 33.784 +/- 0.142 | 13.758 +/- 0.085 | 2.456x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 850.171 | 21646.177 | 0.039x | 57.510 +/- 0.456 | 19.641 +/- 0.167 | 2.928x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 293.489 | 20133.940 | 0.015x | 29.734 +/- 0.116 | 14.638 +/- 0.197 | 2.031x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 271.076 | 19943.662 | 0.014x | 25.580 +/- 0.129 | 14.339 +/- 0.169 | 1.784x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 324.666 | 20884.816 | 0.016x | 28.612 +/- 0.332 | 14.181 +/- 0.147 | 2.018x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 601.765 | 19983.467 | 0.030x | 33.794 +/- 0.502 | 17.117 +/- 0.082 | 1.974x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5002.548 | 24637.938 | 0.203x | 312.930 +/- 3.358 | 291.091 +/- 2.017 | 1.075x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 310.736 | 20106.579 | 0.015x | 29.679 +/- 0.363 | 13.400 +/- 0.061 | 2.215x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 653.088 | 21657.504 | 0.030x | 36.279 +/- 0.518 | 17.716 +/- 0.143 | 2.048x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 816.846 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 742.093 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 418.484 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 548.330 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 142344.027 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 121310.629 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 40990.689 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 72823.301 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 109500.515 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 107388.900 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 43904.796 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 83420.643 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 989.657 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 872.310 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 481.928 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 621.901 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, modules, parameters, and buffers, inference, training autograd, Python control flow, graph_breaks_fullgraph, mutation_aliasing_views, containers and pytrees, decompositions, custom functions, recompilation guards, and dtype/device transitions. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `modules_parameters_buffers` | 8 | Supported and timed public cases: `cpu_float32_global_buffer_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `python_control_flow` | 8 | Supported and timed public cases: `cpu_float32_requires_grad_branch_unary` |
| `graph_breaks_fullgraph` | 8 | Supported and timed public cases: `cpu_float32_fullgraph_false_no_break_unary` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `containers_pytrees` | 6 | Supported and timed public cases: `cpu_float32_tuple_list_output_pytree` |
| `decompositions` | 6 | Supported and timed public cases: `cpu_float32_decomposition_square_scalar` |
| `custom_functions` | 6 | Supported and timed public cases: `cpu_float32_custom_function_unary` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `dtype_device_transitions` | 4 | Supported and timed public cases: `cpu_float32_float_identity_view` |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 92 / 100. Zero-credit unsupported category weight: 8 / 100.
The torch_compile_corpus_v12 corpus also keeps 2 held-out broadcasting programs, 1 held-out containers-pytrees program, 1 held-out custom-function program, 1 held-out decomposition program, 1 held-out dtype/device-transition program, 1 held-out graph_breaks_fullgraph program, 1 held-out inference program, 1 held-out modules/parameters/buffers program, 1 held-out mutation_aliasing_views program, 1 held-out Python-control-flow program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 3 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
