# `torch.compile` Eager CPU Release Timings

Date: 2026-09-05

Candidate provenance: source snapshot refreshed against this worktree, including zero-argument `Tensor.float()` identity graphlets in the dtype/device-transition category and one top-level `requires_grad` branch graphlet in the Python-control-flow category. The raw benchmark artifact is refreshed for `torch_compile_corpus_v10` with the current supported public cases included.

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
- Python: 3.12.14+meta
- NumPy: 2.2.6
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`, `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_7ce3ff95/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_7ce3ff95/.venv/lib/python3.12/site-packages/torch_rs/__init__.py`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`; `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both reported 1
- Dependency installation: locked `uv sync` used the worktree-local uv cache
- Build: release editable wheel installed in the worktree-local `.venv`

The benchmark uses the checked-in `torch_compile_corpus_v10` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input `Tensor.float()` identity dtype/device-transition program, one one-input training-autograd program, one one-input `requires_grad` branch Python-control-flow program, one one-input square decomposition program, one one-input custom-function helper-inline program, four two-input broadcasting programs, one two-input containers-pytrees program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Detach cells return shared-storage aliases with `requires_grad=False`. `Tensor.float()` identity cells preserve values, shape, stride, storage offset, device, dtype, and `requires_grad`. `requires_grad` branch cells select the branch from input metadata, lower only that branch, and preserve the selected branch's output metadata. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Custom-function cells verify the same-module helper inline path over tensor proxy arguments with `neg`, `abs`, `add`, `relu`, `detach`, and `float` operations. Tuple/list output cells preserve container structure and record per-tensor metadata for each output leaf. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, `requires_grad` branch cache specialization, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, tuple/list container structure, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 140 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v10`
- Cold first compiled call: 0.031x uncapped, 0.115x capped
- Steady-state materialized compiled call: 1.902x uncapped, 1.902x capped
- Timed supported cells: 140 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 python-control-flow, 7 containers-pytrees, 7 decomposition, 7 custom-functions, 21 recompilation-guard, 7 dtype-device-transitions, 7 mutation_aliasing_views)
- Recompilation guard sequences: 16 rows, 72 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 76.0% supported by native compile cases, 24% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 375.613 | 85352.208 | 0.004x | 27.390 +/- 0.076 | 14.102 +/- 0.110 | 1.942x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 263.644 | 22441.022 | 0.012x | 24.150 +/- 0.104 | 13.950 +/- 0.192 | 1.731x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 270.479 | 22260.264 | 0.012x | 26.337 +/- 0.230 | 13.780 +/- 0.176 | 1.911x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 889.439 | 23031.690 | 0.039x | 32.350 +/- 0.269 | 18.716 +/- 0.152 | 1.728x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8879.851 | 30291.802 | 0.293x | 542.360 +/- 2.851 | 525.021 +/- 3.498 | 1.033x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 365.452 | 22015.018 | 0.017x | 26.839 +/- 0.174 | 13.173 +/- 0.063 | 2.037x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 909.324 | 26743.649 | 0.034x | 34.024 +/- 0.222 | 19.235 +/- 0.214 | 1.769x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 229.402 | 21056.725 | 0.011x | 22.563 +/- 0.130 | 12.539 +/- 0.128 | 1.799x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 212.171 | 21235.008 | 0.010x | 19.772 +/- 0.077 | 12.337 +/- 0.123 | 1.603x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 238.370 | 24197.482 | 0.010x | 21.557 +/- 0.135 | 13.256 +/- 0.992 | 1.626x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 861.427 | 24323.431 | 0.035x | 27.652 +/- 0.245 | 17.275 +/- 0.220 | 1.601x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8858.032 | 29651.469 | 0.299x | 563.132 +/- 4.085 | 547.240 +/- 6.841 | 1.029x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 291.425 | 23187.370 | 0.013x | 22.062 +/- 0.104 | 13.108 +/- 0.816 | 1.683x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 883.179 | 24071.296 | 0.037x | 29.572 +/- 0.174 | 20.934 +/- 0.288 | 1.413x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 279.948 | 24269.035 | 0.012x | 27.131 +/- 0.132 | 16.416 +/- 0.400 | 1.653x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 237.699 | 23716.590 | 0.010x | 24.108 +/- 0.100 | 16.329 +/- 0.547 | 1.476x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 275.111 | 22319.893 | 0.012x | 25.980 +/- 0.110 | 14.059 +/- 0.445 | 1.848x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 872.909 | 25708.009 | 0.034x | 32.207 +/- 0.291 | 22.030 +/- 0.307 | 1.462x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8662.983 | 33358.687 | 0.260x | 542.877 +/- 3.071 | 576.093 +/- 38.805 | 0.942x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 328.391 | 22742.583 | 0.014x | 26.462 +/- 0.112 | 14.980 +/- 1.025 | 1.766x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 868.327 | 23625.777 | 0.037x | 33.720 +/- 0.222 | 19.695 +/- 0.359 | 1.712x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 361.987 | 23723.485 | 0.015x | 39.113 +/- 0.248 | 17.483 +/- 0.144 | 2.237x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 346.364 | 22693.879 | 0.015x | 39.358 +/- 0.317 | 17.569 +/- 0.246 | 2.240x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 363.369 | 24327.248 | 0.015x | 46.558 +/- 1.383 | 17.491 +/- 0.245 | 2.662x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 968.549 | 23550.488 | 0.041x | 56.110 +/- 1.035 | 22.779 +/- 0.221 | 2.463x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9573.734 | 31455.415 | 0.304x | 629.479 +/- 3.828 | 540.410 +/- 5.206 | 1.165x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 469.149 | 23760.686 | 0.020x | 43.739 +/- 0.171 | 16.284 +/- 0.186 | 2.686x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1011.899 | 24831.865 | 0.041x | 55.378 +/- 0.595 | 23.298 +/- 0.113 | 2.377x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 420.435 | 23111.210 | 0.018x | 45.518 +/- 0.312 | 16.442 +/- 0.096 | 2.768x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 434.382 | 23021.183 | 0.019x | 40.747 +/- 0.169 | 17.648 +/- 0.187 | 2.309x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 430.140 | 23446.927 | 0.018x | 44.661 +/- 0.164 | 17.651 +/- 0.253 | 2.530x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1021.178 | 25468.262 | 0.040x | 53.141 +/- 0.271 | 23.307 +/- 0.179 | 2.280x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8937.949 | 32195.124 | 0.278x | 589.812 +/- 4.183 | 557.068 +/- 5.138 | 1.059x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 451.453 | 24674.347 | 0.018x | 45.623 +/- 0.142 | 16.369 +/- 0.103 | 2.787x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1038.774 | 26666.457 | 0.039x | 59.536 +/- 0.246 | 23.735 +/- 0.229 | 2.508x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 245.616 | 23107.840 | 0.011x | 21.451 +/- 0.109 | 13.675 +/- 0.101 | 1.569x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 212.181 | 22367.571 | 0.009x | 19.305 +/- 0.056 | 13.623 +/- 0.089 | 1.417x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 226.588 | 22186.863 | 0.010x | 20.734 +/- 0.091 | 13.602 +/- 0.174 | 1.524x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 615.240 | 21869.202 | 0.028x | 24.646 +/- 0.171 | 16.467 +/- 0.216 | 1.497x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5486.426 | 26699.833 | 0.205x | 336.042 +/- 2.939 | 324.893 +/- 3.484 | 1.034x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 252.022 | 21553.946 | 0.012x | 21.379 +/- 0.108 | 13.295 +/- 0.115 | 1.608x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 596.061 | 23443.266 | 0.025x | 25.587 +/- 0.125 | 16.618 +/- 0.217 | 1.540x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 219.117 | 22949.431 | 0.010x | 19.502 +/- 0.122 | 11.344 +/- 0.112 | 1.719x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 217.093 | 21880.255 | 0.010x | 18.293 +/- 0.060 | 11.216 +/- 0.067 | 1.631x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 227.018 | 21492.153 | 0.011x | 19.461 +/- 0.068 | 11.306 +/- 0.054 | 1.721x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 807.660 | 22624.209 | 0.036x | 24.803 +/- 0.131 | 16.018 +/- 0.118 | 1.548x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8767.546 | 30729.393 | 0.285x | 539.354 +/- 4.046 | 539.226 +/- 7.524 | 1.000x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 256.528 | 21501.798 | 0.012x | 20.206 +/- 0.076 | 11.299 +/- 0.098 | 1.788x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 846.659 | 24263.496 | 0.035x | 25.030 +/- 0.248 | 16.226 +/- 0.276 | 1.543x | `4ba5419e2e3f2393` |
| `cpu_float32_float_identity_view` | `case_default` | 1 | 256 | shape (3,), stride (4,), offset 1, torch.float32, cpu, requires_grad=True | 229.697 | 23557.078 | 0.010x | 19.337 +/- 0.120 | 10.923 +/- 0.145 | 1.770x | `58df67cd172620c1` |
| `cpu_float32_float_identity_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 219.362 | 22058.899 | 0.010x | 18.085 +/- 0.067 | 10.887 +/- 0.067 | 1.661x | `e75a1d3233117514` |
| `cpu_float32_float_identity_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 222.531 | 22440.075 | 0.010x | 19.412 +/- 0.130 | 11.169 +/- 0.153 | 1.738x | `4c3dc265c5b9d697` |
| `cpu_float32_float_identity_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 805.718 | 22405.708 | 0.036x | 24.596 +/- 0.160 | 15.832 +/- 0.236 | 1.554x | `5ccc89fb94f689e5` |
| `cpu_float32_float_identity_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8549.957 | 30818.503 | 0.277x | 540.841 +/- 4.112 | 526.450 +/- 3.605 | 1.027x | `91fa5699b26ca1b8` |
| `cpu_float32_float_identity_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 285.897 | 21601.814 | 0.013x | 20.254 +/- 0.258 | 11.022 +/- 0.164 | 1.838x | `e99a6c9902c3119e` |
| `cpu_float32_float_identity_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 822.784 | 23885.845 | 0.034x | 24.739 +/- 0.152 | 15.579 +/- 0.114 | 1.588x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 399.909 | 24251.473 | 0.016x | 40.781 +/- 0.192 | 18.653 +/- 0.138 | 2.186x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 425.202 | 22801.306 | 0.019x | 35.187 +/- 0.263 | 16.528 +/- 0.180 | 2.129x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 399.708 | 23693.489 | 0.017x | 38.065 +/- 0.153 | 16.298 +/- 0.225 | 2.336x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 758.316 | 23085.426 | 0.033x | 44.080 +/- 0.202 | 19.768 +/- 0.084 | 2.230x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5492.562 | 27645.688 | 0.199x | 363.859 +/- 3.215 | 335.532 +/- 1.811 | 1.084x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 408.092 | 22596.938 | 0.018x | 38.856 +/- 0.119 | 15.252 +/- 0.062 | 2.548x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 776.168 | 24150.815 | 0.032x | 48.335 +/- 0.219 | 20.396 +/- 0.146 | 2.370x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 318.381 | 22716.318 | 0.014x | 29.604 +/- 0.097 | 15.730 +/- 0.150 | 1.882x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 345.256 | 21918.932 | 0.016x | 29.802 +/- 0.069 | 15.532 +/- 0.188 | 1.919x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 358.992 | 22771.506 | 0.016x | 35.897 +/- 0.616 | 15.506 +/- 0.139 | 2.315x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1103.617 | 24201.197 | 0.046x | 44.018 +/- 0.669 | 20.792 +/- 0.129 | 2.117x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9790.877 | 32117.406 | 0.305x | 584.924 +/- 2.630 | 561.545 +/- 3.534 | 1.042x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 374.721 | 23122.446 | 0.016x | 33.323 +/- 0.101 | 14.608 +/- 0.059 | 2.281x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 975.705 | 24399.232 | 0.040x | 44.001 +/- 0.275 | 21.606 +/- 0.212 | 2.037x | `68585b64809ef02a` |
| `cpu_float32_custom_function_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 488.759 | 38522.406 | 0.013x | 53.453 +/- 0.205 | 18.893 +/- 0.101 | 2.829x | `d16fd2f4dd199523` |
| `cpu_float32_custom_function_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 423.225 | 24492.742 | 0.017x | 46.693 +/- 0.158 | 19.145 +/- 0.311 | 2.439x | `5c2ffe407931c8ee` |
| `cpu_float32_custom_function_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 460.191 | 24396.883 | 0.019x | 50.819 +/- 0.180 | 18.701 +/- 0.301 | 2.718x | `d85643b7b66a7ca9` |
| `cpu_float32_custom_function_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1040.743 | 25461.526 | 0.041x | 59.389 +/- 0.227 | 23.946 +/- 0.129 | 2.480x | `414eafab6fd10fb4` |
| `cpu_float32_custom_function_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9011.836 | 33673.819 | 0.268x | 595.589 +/- 5.516 | 553.635 +/- 3.731 | 1.076x | `7863bb8d1d98f49b` |
| `cpu_float32_custom_function_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 508.949 | 24684.507 | 0.021x | 51.832 +/- 0.183 | 17.187 +/- 0.091 | 3.016x | `e99a6c9902c3119e` |
| `cpu_float32_custom_function_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1076.838 | 25472.764 | 0.042x | 65.174 +/- 0.655 | 24.886 +/- 0.170 | 2.619x | `188c6817fce2e1e1` |
| `cpu_float32_requires_grad_branch_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 391.442 | 22526.731 | 0.017x | 28.798 +/- 0.318 | 13.817 +/- 0.123 | 2.084x | `43e5fdfc5aec3505` |
| `cpu_float32_requires_grad_branch_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 370.360 | 21901.692 | 0.017x | 25.062 +/- 0.157 | 13.716 +/- 0.186 | 1.827x | `e75a1d3233117514` |
| `cpu_float32_requires_grad_branch_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 383.830 | 21775.225 | 0.018x | 27.225 +/- 0.178 | 13.737 +/- 0.271 | 1.982x | `47aef822223dbae7` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1073.092 | 22779.823 | 0.047x | 33.867 +/- 0.111 | 18.471 +/- 0.097 | 1.834x | `2148badcc2b9e4ce` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8735.443 | 31187.821 | 0.280x | 563.627 +/- 4.885 | 545.990 +/- 4.467 | 1.032x | `d53163cb2693cd35` |
| `cpu_float32_requires_grad_branch_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 438.518 | 22319.298 | 0.020x | 27.958 +/- 0.229 | 13.122 +/- 0.111 | 2.131x | `e99a6c9902c3119e` |
| `cpu_float32_requires_grad_branch_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1014.654 | 23482.125 | 0.043x | 36.908 +/- 0.217 | 18.930 +/- 0.183 | 1.950x | `372841b6f1764798` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 370.470 | 24368.710 | 0.015x | 46.004 +/- 0.179 | 17.081 +/- 0.243 | 2.693x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 971.418 | 25183.266 | 0.039x | 51.673 +/- 0.279 | 22.146 +/- 0.219 | 2.333x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9368.004 | 33421.893 | 0.280x | 594.533 +/- 4.588 | 560.967 +/- 3.572 | 1.060x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1002.565 | 24590.660 | 0.041x | 49.798 +/- 0.370 | 22.042 +/- 0.193 | 2.259x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 975.068 | 25058.342 | 0.039x | 49.038 +/- 0.289 | 21.936 +/- 0.187 | 2.235x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 345.172 | 24536.538 | 0.014x | 45.037 +/- 0.298 | 15.683 +/- 0.078 | 2.872x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1028.795 | 26547.062 | 0.039x | 73.043 +/- 0.348 | 22.817 +/- 0.159 | 3.201x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 324.470 | 24414.579 | 0.013x | 33.538 +/- 0.141 | 15.322 +/- 0.114 | 2.189x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 883.670 | 33377.476 | 0.026x | 38.464 +/- 0.376 | 19.791 +/- 0.163 | 1.944x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9095.547 | 36296.938 | 0.251x | 585.619 +/- 5.461 | 553.423 +/- 3.404 | 1.058x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 949.965 | 24216.901 | 0.039x | 37.552 +/- 0.195 | 19.472 +/- 0.223 | 1.929x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 889.144 | 23355.213 | 0.038x | 38.353 +/- 0.223 | 19.270 +/- 0.135 | 1.990x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 299.588 | 24246.901 | 0.012x | 33.152 +/- 0.128 | 13.665 +/- 0.053 | 2.426x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 936.706 | 24331.488 | 0.038x | 58.470 +/- 0.193 | 19.902 +/- 0.139 | 2.938x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 284.165 | 22787.079 | 0.012x | 33.510 +/- 0.148 | 15.606 +/- 0.185 | 2.147x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 891.678 | 23208.948 | 0.038x | 39.106 +/- 0.192 | 19.683 +/- 0.204 | 1.987x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9098.717 | 32341.430 | 0.281x | 580.973 +/- 3.053 | 556.882 +/- 2.428 | 1.043x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 918.112 | 23258.882 | 0.039x | 38.176 +/- 0.122 | 19.347 +/- 0.164 | 1.973x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 895.649 | 23235.417 | 0.039x | 38.553 +/- 0.233 | 19.483 +/- 0.151 | 1.979x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 275.381 | 22807.936 | 0.012x | 33.613 +/- 0.211 | 13.898 +/- 0.082 | 2.419x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 927.818 | 24629.815 | 0.038x | 59.982 +/- 0.360 | 20.266 +/- 0.124 | 2.960x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 305.632 | 24303.942 | 0.013x | 33.574 +/- 0.138 | 14.955 +/- 0.175 | 2.245x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 879.594 | 26337.435 | 0.033x | 38.198 +/- 0.167 | 19.222 +/- 0.525 | 1.987x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8477.358 | 35055.567 | 0.242x | 549.815 +/- 2.375 | 567.776 +/- 11.699 | 0.968x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 906.365 | 25014.847 | 0.036x | 37.438 +/- 0.203 | 20.056 +/- 0.841 | 1.867x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 873.195 | 25458.122 | 0.034x | 38.663 +/- 0.225 | 19.097 +/- 0.428 | 2.025x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 283.534 | 26115.134 | 0.011x | 33.276 +/- 0.175 | 13.683 +/- 0.495 | 2.432x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1124.269 | 27571.039 | 0.041x | 69.740 +/- 2.828 | 19.380 +/- 0.315 | 3.599x | `2bd384aefcaaa397` |
| `cpu_float32_tuple_list_output_pytree` | `case_default` | 2 | 256 | tuple[shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True, list[shape (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True]] | 496.851 | 25697.217 | 0.019x | 56.007 +/- 0.152 | 19.847 +/- 0.389 | 2.822x | `a62dacb062c1ed92` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1566.572 | 25749.897 | 0.061x | 64.516 +/- 0.308 | 26.003 +/- 0.276 | 2.481x | `3bce94d7e523bafe` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_127x131_by_131` | 2 | 16 | tuple[shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (131,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 14136.640 | 40288.250 | 0.351x | 916.453 +/- 7.643 | 911.192 +/- 7.615 | 1.006x | `022557af0d301f5e` |
| `cpu_float32_tuple_list_output_pytree` | `tensor_scalar_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1505.867 | 25354.946 | 0.059x | 62.945 +/- 0.515 | 25.468 +/- 0.332 | 2.472x | `f4ff04ee55c4e2cd` |
| `cpu_float32_tuple_list_output_pytree` | `scalar_tensor_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False]] | 1661.221 | 27645.352 | 0.060x | 63.893 +/- 0.316 | 27.150 +/- 0.246 | 2.353x | `f1950b665bfdc9f1` |
| `cpu_float32_tuple_list_output_pytree` | `empty_2x0_by_0` | 2 | 2048 | tuple[shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 464.857 | 26906.215 | 0.017x | 54.928 +/- 0.149 | 15.806 +/- 0.179 | 3.475x | `e89cfed7478c41fa` |
| `cpu_float32_tuple_list_output_pytree` | `transpose_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False]] | 1544.444 | 27952.326 | 0.055x | 86.987 +/- 0.293 | 25.638 +/- 0.255 | 3.393x | `776bd23d05673f66` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 331.981 | 25189.997 | 0.013x | 34.244 +/- 0.170 | 15.564 +/- 0.137 | 2.200x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 308.757 | 23691.006 | 0.013x | 30.250 +/- 0.217 | 15.444 +/- 0.234 | 1.959x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 334.706 | 23399.220 | 0.014x | 32.915 +/- 0.221 | 15.113 +/- 0.158 | 2.178x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 749.497 | 23495.340 | 0.032x | 38.030 +/- 0.377 | 18.681 +/- 0.150 | 2.036x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5525.356 | 28340.052 | 0.195x | 359.247 +/- 2.128 | 337.943 +/- 2.576 | 1.063x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 343.629 | 22586.512 | 0.015x | 33.055 +/- 0.137 | 14.437 +/- 0.089 | 2.290x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 732.052 | 23673.018 | 0.031x | 41.305 +/- 0.172 | 19.421 +/- 0.175 | 2.127x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 328.567 | 23539.967 | 0.014x | 39.484 +/- 0.188 | 16.275 +/- 0.147 | 2.426x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 966.541 | 23868.960 | 0.040x | 45.167 +/- 0.439 | 21.061 +/- 0.168 | 2.145x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8991.234 | 33485.644 | 0.269x | 596.844 +/- 10.118 | 560.456 +/- 4.935 | 1.065x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 988.444 | 23761.397 | 0.042x | 43.685 +/- 0.222 | 21.303 +/- 0.373 | 2.051x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 917.822 | 24913.439 | 0.037x | 44.612 +/- 0.262 | 21.146 +/- 0.340 | 2.110x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 336.298 | 23174.611 | 0.015x | 39.211 +/- 0.202 | 14.837 +/- 0.202 | 2.643x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1007.463 | 24761.383 | 0.041x | 65.722 +/- 0.342 | 21.398 +/- 0.128 | 3.071x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 357.691 | 22290.244 | 0.016x | 33.808 +/- 0.153 | 15.412 +/- 0.133 | 2.194x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 310.815 | 22426.850 | 0.014x | 29.895 +/- 0.158 | 15.348 +/- 0.198 | 1.948x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 325.292 | 22792.258 | 0.014x | 32.500 +/- 0.188 | 15.226 +/- 0.296 | 2.134x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 699.778 | 23077.128 | 0.030x | 37.510 +/- 0.303 | 18.517 +/- 0.156 | 2.026x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5501.344 | 29248.740 | 0.188x | 358.401 +/- 4.528 | 337.698 +/- 2.218 | 1.061x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 374.857 | 22514.157 | 0.017x | 33.281 +/- 0.162 | 14.336 +/- 0.070 | 2.322x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 713.984 | 23623.834 | 0.030x | 41.518 +/- 0.239 | 19.123 +/- 0.164 | 2.171x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 961.263 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 836.114 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 496.762 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 621.150 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 156770.968 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 127794.153 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 43888.341 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 75290.841 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 113516.695 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 111035.247 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 44600.920 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 74872.010 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1025.030 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 900.591 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 528.749 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 658.735 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, inference, training autograd, Python control flow, mutation_aliasing_views, containers and pytrees, decompositions, custom functions, recompilation guards, and dtype/device transitions. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `python_control_flow` | 8 | Supported and timed public cases: `cpu_float32_requires_grad_branch_unary` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `containers_pytrees` | 6 | Supported and timed public cases: `cpu_float32_tuple_list_output_pytree` |
| `decompositions` | 6 | Supported and timed public cases: `cpu_float32_decomposition_square_scalar` |
| `custom_functions` | 6 | Supported and timed public cases: `cpu_float32_custom_function_unary` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `dtype_device_transitions` | 4 | Supported and timed public cases: `cpu_float32_float_identity_view` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 76 / 100. Zero-credit unsupported category weight: 24 / 100.
The torch_compile_corpus_v10 corpus also keeps 2 held-out broadcasting programs, 1 held-out containers-pytrees program, 1 held-out custom-function program, 1 held-out decomposition program, 1 held-out dtype/device-transition program, 1 held-out inference program, 1 held-out mutation_aliasing_views program, 1 held-out Python-control-flow program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 3 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
