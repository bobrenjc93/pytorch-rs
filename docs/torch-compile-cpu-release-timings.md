# `torch.compile` Eager CPU Release Timings

Date: 2026-09-05

Candidate provenance: source snapshot refreshed against this worktree, including dynamic `torch.compile(..., backend="eager", fullgraph=True, dynamic=True/False)` graphlets that recompile by concrete shape/stride metadata, zero-argument `Tensor.float()` identity graphlets in the dtype/device-transition category, and one top-level `requires_grad` branch graphlet in the Python-control-flow category. The raw benchmark artifact is refreshed for `torch_compile_corpus_v11` with the current supported public cases included.

The setup, build, focused check, and timing commands below reproduce this evidence from the repository root. The reusable timing driver is checked in as `scripts/benchmark_compile_cpu.py`; its complete raw JSON output is committed at `docs/benchmark-data/torch-compile-cpu-v4.json`. The PyTorch 2.13 reference evidence used the worktree-local `target/torch-compile-coverage/venv`; uv and Cargo state were redirected under `target/`.

```bash
bash scripts/evaluate_torch_compile_coverage.sh --subset public
env -u CONDA_PREFIX PATH="$PWD/target/torch-compile-coverage/venv/bin:$PATH" \
  TMPDIR="$PWD/target" XDG_CACHE_HOME="$PWD/target/xdg-cache" \
  TORCHINDUCTOR_CACHE_DIR="$PWD/target/torchinductor-cache" \
  TRITON_CACHE_DIR="$PWD/target/triton-cache" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 target/torch-compile-coverage/venv/bin/python scripts/benchmark_compile_cpu.py \
  --require-single-cpu-affinity \
  --output docs/benchmark-data/torch-compile-cpu-v4.json
env -u CONDA_PREFIX PATH="$PWD/target/torch-compile-coverage/venv/bin:$PATH" \
  target/torch-compile-coverage/venv/bin/python scripts/benchmark_compile_cpu.py \
  --render-markdown-summary docs/benchmark-data/torch-compile-cpu-v4.json \
  > target/torch-compile-cpu-v4-summary.md
env -u CONDA_PREFIX PATH="$PWD/target/torch-compile-coverage/venv/bin:$PATH" \
  target/torch-compile-coverage/venv/bin/python scripts/benchmark_compile_cpu.py --validate-artifact
env -u CONDA_PREFIX PATH="$PWD/target/torch-compile-coverage/venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  target/torch-compile-coverage/venv/bin/python -m unittest \
  tests.test_compile_benchmark_artifact tests.test_compile_corpus \
  tests.test_top_level_compile tests.test_torch_compile_coverage_evaluator
bash scripts/evaluate_torch_compile_coverage.sh
env -u CONDA_PREFIX PATH="$PWD/target/torch-compile-coverage/venv/bin:$PATH" \
  target/torch-compile-coverage/venv/bin/python -m py_compile \
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
env -u CONDA_PREFIX PATH="$PWD/target/torch-compile-coverage/venv/bin:$PATH" \
  target/torch-compile-coverage/venv/bin/python scripts/benchmark_compile_cpu.py --validate-artifact
env -u CONDA_PREFIX PATH="$PWD/target/torch-compile-coverage/venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  target/torch-compile-coverage/venv/bin/python -m unittest \
  tests.test_compile_benchmark_artifact tests.test_compile_corpus \
  tests.test_top_level_compile tests.test_torch_compile_coverage_evaluator
bash scripts/evaluate_torch_compile_coverage.sh
env -u CONDA_PREFIX PATH="$PWD/target/torch-compile-coverage/venv/bin:$PATH" \
  target/torch-compile-coverage/venv/bin/python -m py_compile \
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
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_d0fc26fc/target/torch-compile-coverage/venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_d0fc26fc/target/torch-compile-coverage/venv/lib/python3.12/site-packages/torch_rs/__init__.py`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`; `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both reported 1
- Dependency installation: locked `uv sync` used the worktree-local uv cache and dedicated compile-coverage virtual environment
- Build: release wheel installed in the worktree-local `target/torch-compile-coverage/venv`

The benchmark uses the checked-in `torch_compile_corpus_v11` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input `dynamic=True` shape/stride graphlet, one one-input `Tensor.float()` identity dtype/device-transition program, one one-input training-autograd program, one one-input `requires_grad` branch Python-control-flow program, one one-input square decomposition program, one one-input custom-function helper-inline program, four two-input broadcasting programs, one two-input containers-pytrees program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Dynamic-shape cells use the same compiled wrapper across shape and stride variants in the coverage evaluator and the benchmark times the checked-in `dynamic=True` public case across all compatible input variants. Detach cells return shared-storage aliases with `requires_grad=False`. `Tensor.float()` identity cells preserve values, shape, stride, storage offset, device, dtype, and `requires_grad`. `requires_grad` branch cells select the branch from input metadata, lower only that branch, and preserve the selected branch's output metadata. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Custom-function cells verify the same-module helper inline path over tensor proxy arguments with `neg`, `abs`, `add`, `relu`, `detach`, and `float` operations. Tuple/list output cells preserve container structure and record per-tensor metadata for each output leaf. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, `requires_grad` branch cache specialization, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile` with each case's checked-in `backend`, `fullgraph`, and optional `dynamic` flag. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, tuple/list container structure, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 147 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus and dynamic shape/stride variants in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v11`
- Cold first compiled call: 0.031x uncapped, 0.115x capped
- Steady-state materialized compiled call: 1.912x uncapped, 1.912x capped
- Timed supported cells: 147 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 python-control-flow, 7 dynamic-shape, 7 containers-pytrees, 7 decomposition, 7 custom-functions, 21 recompilation-guard, 7 dtype-device-transitions, 7 mutation_aliasing_views)
- Recompilation guard sequences: 16 rows, 72 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 84.0% supported by native compile cases, 16% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 383.415 | 82452.152 | 0.005x | 27.479 +/- 0.082 | 14.107 +/- 0.143 | 1.948x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 269.974 | 22404.277 | 0.012x | 24.275 +/- 0.098 | 14.175 +/- 0.186 | 1.713x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 276.087 | 22759.429 | 0.012x | 26.287 +/- 0.092 | 13.950 +/- 0.228 | 1.884x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 859.059 | 22445.605 | 0.038x | 32.295 +/- 0.165 | 18.770 +/- 0.165 | 1.721x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8533.842 | 30913.674 | 0.276x | 540.325 +/- 3.617 | 523.952 +/- 3.452 | 1.031x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 341.606 | 21952.569 | 0.016x | 27.097 +/- 0.354 | 13.272 +/- 0.071 | 2.042x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 877.562 | 24312.173 | 0.036x | 34.315 +/- 0.247 | 19.310 +/- 0.185 | 1.777x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 241.755 | 21463.339 | 0.011x | 22.826 +/- 0.235 | 12.458 +/- 0.100 | 1.832x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 220.309 | 21250.227 | 0.010x | 19.992 +/- 0.100 | 12.339 +/- 0.090 | 1.620x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 234.435 | 21297.488 | 0.011x | 21.727 +/- 0.103 | 12.371 +/- 0.129 | 1.756x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 835.799 | 21967.893 | 0.038x | 27.837 +/- 0.207 | 17.201 +/- 0.108 | 1.618x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8764.366 | 30586.223 | 0.287x | 564.502 +/- 8.514 | 541.614 +/- 3.748 | 1.042x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 263.813 | 21073.734 | 0.013x | 22.026 +/- 0.088 | 11.963 +/- 0.056 | 1.841x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 840.827 | 23962.479 | 0.035x | 29.654 +/- 0.190 | 17.279 +/- 0.129 | 1.716x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 274.645 | 21341.050 | 0.013x | 27.191 +/- 0.128 | 13.960 +/- 0.108 | 1.948x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 243.007 | 21970.957 | 0.011x | 24.298 +/- 0.092 | 13.844 +/- 0.158 | 1.755x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 301.265 | 22523.107 | 0.013x | 26.179 +/- 0.119 | 14.005 +/- 0.183 | 1.869x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 909.236 | 22101.164 | 0.041x | 33.003 +/- 0.210 | 18.774 +/- 0.120 | 1.758x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8485.520 | 29849.393 | 0.284x | 553.025 +/- 3.766 | 527.058 +/- 3.867 | 1.049x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 300.064 | 22068.990 | 0.014x | 27.620 +/- 0.118 | 13.264 +/- 0.077 | 2.082x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 875.409 | 22770.667 | 0.038x | 35.163 +/- 0.273 | 19.099 +/- 0.165 | 1.841x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 362.603 | 22548.746 | 0.016x | 40.986 +/- 0.194 | 17.376 +/- 0.140 | 2.359x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 368.993 | 22175.020 | 0.017x | 40.960 +/- 0.271 | 17.487 +/- 0.242 | 2.342x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 381.567 | 22933.609 | 0.017x | 44.062 +/- 0.178 | 17.460 +/- 0.270 | 2.524x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 978.234 | 24147.949 | 0.041x | 52.022 +/- 0.507 | 22.706 +/- 0.186 | 2.291x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8573.337 | 31592.525 | 0.271x | 574.858 +/- 9.404 | 530.454 +/- 3.342 | 1.084x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 392.949 | 23785.532 | 0.017x | 44.340 +/- 0.230 | 16.270 +/- 0.119 | 2.725x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1004.104 | 25778.136 | 0.039x | 55.421 +/- 0.347 | 23.422 +/- 0.148 | 2.366x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 492.976 | 23051.506 | 0.021x | 46.266 +/- 0.172 | 16.592 +/- 0.375 | 2.788x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 392.394 | 23952.975 | 0.016x | 41.286 +/- 0.283 | 17.574 +/- 0.174 | 2.349x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 426.745 | 23203.962 | 0.018x | 45.430 +/- 0.153 | 17.807 +/- 0.368 | 2.551x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1028.991 | 23468.432 | 0.044x | 54.108 +/- 0.268 | 23.157 +/- 0.222 | 2.337x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8883.777 | 32155.086 | 0.276x | 589.086 +/- 2.724 | 556.981 +/- 4.633 | 1.058x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 461.137 | 26399.515 | 0.017x | 46.098 +/- 0.185 | 16.282 +/- 0.208 | 2.831x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1061.766 | 24540.949 | 0.043x | 60.428 +/- 0.319 | 23.916 +/- 0.222 | 2.527x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 244.490 | 21653.277 | 0.011x | 21.583 +/- 0.115 | 13.618 +/- 0.086 | 1.585x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 219.261 | 21271.033 | 0.010x | 19.367 +/- 0.061 | 13.317 +/- 0.068 | 1.454x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 232.382 | 21075.257 | 0.011x | 20.940 +/- 0.088 | 13.530 +/- 0.190 | 1.548x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 603.047 | 21602.008 | 0.028x | 24.962 +/- 0.169 | 16.620 +/- 0.124 | 1.502x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5319.992 | 26618.666 | 0.200x | 335.773 +/- 2.456 | 320.679 +/- 2.234 | 1.047x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 258.567 | 21452.338 | 0.012x | 21.431 +/- 0.061 | 13.128 +/- 0.050 | 1.632x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 589.607 | 22838.690 | 0.026x | 25.589 +/- 0.143 | 16.549 +/- 0.129 | 1.546x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 228.245 | 22355.844 | 0.010x | 19.668 +/- 0.134 | 11.194 +/- 0.075 | 1.757x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 222.937 | 21110.506 | 0.011x | 18.554 +/- 0.159 | 11.104 +/- 0.048 | 1.671x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 223.153 | 21579.059 | 0.010x | 19.813 +/- 0.078 | 11.122 +/- 0.047 | 1.781x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 808.092 | 21737.463 | 0.037x | 25.221 +/- 0.175 | 15.812 +/- 0.088 | 1.595x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8485.019 | 29571.302 | 0.287x | 536.366 +/- 3.487 | 527.202 +/- 2.872 | 1.017x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 281.125 | 22482.281 | 0.013x | 20.691 +/- 0.252 | 11.164 +/- 0.045 | 1.853x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 819.439 | 22308.933 | 0.037x | 25.332 +/- 0.373 | 16.018 +/- 0.120 | 1.581x | `4ba5419e2e3f2393` |
| `cpu_float32_float_identity_view` | `case_default` | 1 | 256 | shape (3,), stride (4,), offset 1, torch.float32, cpu, requires_grad=True | 216.248 | 21804.024 | 0.010x | 19.376 +/- 0.096 | 10.850 +/- 0.059 | 1.786x | `58df67cd172620c1` |
| `cpu_float32_float_identity_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 210.323 | 20534.635 | 0.010x | 18.454 +/- 0.148 | 10.776 +/- 0.045 | 1.712x | `e75a1d3233117514` |
| `cpu_float32_float_identity_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 227.344 | 21017.840 | 0.011x | 19.646 +/- 0.094 | 10.862 +/- 0.047 | 1.809x | `4c3dc265c5b9d697` |
| `cpu_float32_float_identity_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 826.455 | 21189.996 | 0.039x | 24.787 +/- 0.165 | 15.531 +/- 0.127 | 1.596x | `5ccc89fb94f689e5` |
| `cpu_float32_float_identity_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8422.509 | 29502.299 | 0.285x | 538.377 +/- 3.007 | 524.562 +/- 3.207 | 1.026x | `91fa5699b26ca1b8` |
| `cpu_float32_float_identity_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 286.743 | 20829.621 | 0.014x | 20.240 +/- 0.066 | 10.787 +/- 0.060 | 1.876x | `e99a6c9902c3119e` |
| `cpu_float32_float_identity_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 841.503 | 22544.910 | 0.037x | 24.778 +/- 0.119 | 15.349 +/- 0.093 | 1.614x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 409.935 | 23623.342 | 0.017x | 41.155 +/- 0.122 | 18.092 +/- 0.108 | 2.275x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 445.474 | 22850.507 | 0.019x | 35.036 +/- 0.148 | 16.377 +/- 0.152 | 2.139x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 402.003 | 22237.845 | 0.018x | 38.309 +/- 0.061 | 16.384 +/- 0.211 | 2.338x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 779.730 | 22664.281 | 0.034x | 44.148 +/- 0.284 | 19.691 +/- 0.132 | 2.242x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5464.335 | 28111.039 | 0.194x | 364.995 +/- 2.526 | 336.336 +/- 1.988 | 1.085x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 404.307 | 23058.693 | 0.018x | 39.546 +/- 0.298 | 15.170 +/- 0.070 | 2.607x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 800.291 | 23764.325 | 0.034x | 48.863 +/- 0.299 | 20.242 +/- 0.170 | 2.414x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 340.149 | 22471.508 | 0.015x | 29.930 +/- 0.116 | 15.516 +/- 0.123 | 1.929x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 339.143 | 21923.525 | 0.015x | 29.971 +/- 0.100 | 15.616 +/- 0.207 | 1.919x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 398.377 | 23192.128 | 0.017x | 32.752 +/- 0.087 | 15.473 +/- 0.200 | 2.117x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 955.500 | 23251.164 | 0.041x | 39.846 +/- 0.190 | 20.941 +/- 0.130 | 1.903x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9075.812 | 31943.020 | 0.284x | 582.899 +/- 2.792 | 559.582 +/- 4.199 | 1.042x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 372.874 | 22585.942 | 0.017x | 33.287 +/- 0.080 | 14.676 +/- 0.054 | 2.268x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 984.404 | 23731.399 | 0.041x | 44.422 +/- 0.204 | 21.537 +/- 0.115 | 2.063x | `68585b64809ef02a` |
| `cpu_float32_custom_function_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 464.322 | 39297.765 | 0.012x | 53.601 +/- 0.271 | 18.704 +/- 0.201 | 2.866x | `d16fd2f4dd199523` |
| `cpu_float32_custom_function_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 430.091 | 24461.264 | 0.018x | 47.043 +/- 0.148 | 18.599 +/- 0.225 | 2.529x | `5c2ffe407931c8ee` |
| `cpu_float32_custom_function_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 433.365 | 23790.865 | 0.018x | 51.265 +/- 0.234 | 18.523 +/- 0.242 | 2.768x | `d85643b7b66a7ca9` |
| `cpu_float32_custom_function_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1093.659 | 24640.765 | 0.044x | 59.808 +/- 0.410 | 23.815 +/- 0.112 | 2.511x | `414eafab6fd10fb4` |
| `cpu_float32_custom_function_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 11756.082 | 32855.115 | 0.358x | 596.127 +/- 10.063 | 550.470 +/- 2.979 | 1.083x | `7863bb8d1d98f49b` |
| `cpu_float32_custom_function_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 599.281 | 23964.433 | 0.025x | 51.904 +/- 0.325 | 17.153 +/- 0.071 | 3.026x | `e99a6c9902c3119e` |
| `cpu_float32_custom_function_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1065.246 | 25780.094 | 0.041x | 65.713 +/- 0.555 | 24.882 +/- 0.131 | 2.641x | `188c6817fce2e1e1` |
| `cpu_float32_requires_grad_branch_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 429.985 | 21699.031 | 0.020x | 28.882 +/- 0.159 | 13.815 +/- 0.090 | 2.091x | `43e5fdfc5aec3505` |
| `cpu_float32_requires_grad_branch_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 357.491 | 21541.552 | 0.017x | 25.331 +/- 0.171 | 13.670 +/- 0.162 | 1.853x | `e75a1d3233117514` |
| `cpu_float32_requires_grad_branch_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 384.873 | 21990.887 | 0.018x | 27.403 +/- 0.147 | 13.739 +/- 0.238 | 1.995x | `47aef822223dbae7` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 983.968 | 23269.281 | 0.042x | 33.963 +/- 0.112 | 18.697 +/- 0.165 | 1.816x | `2148badcc2b9e4ce` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9226.386 | 31448.031 | 0.293x | 561.096 +/- 3.575 | 545.063 +/- 3.994 | 1.029x | `d53163cb2693cd35` |
| `cpu_float32_requires_grad_branch_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 422.579 | 21483.850 | 0.020x | 28.101 +/- 0.172 | 13.099 +/- 0.111 | 2.145x | `e99a6c9902c3119e` |
| `cpu_float32_requires_grad_branch_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1001.445 | 23700.713 | 0.042x | 36.862 +/- 0.144 | 18.886 +/- 0.126 | 1.952x | `372841b6f1764798` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 374.000 | 54214.446 | 0.007x | 40.485 +/- 0.177 | 22.959 +/- 0.621 | 1.763x | `78824b56236781de` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 360.440 | 22465.300 | 0.016x | 35.372 +/- 0.191 | 19.165 +/- 0.189 | 1.846x | `96474978e4b2c20f` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 374.562 | 32599.368 | 0.011x | 38.827 +/- 0.227 | 20.686 +/- 0.189 | 1.877x | `3e3e1d5aa2a3441f` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1120.700 | 39076.510 | 0.029x | 46.865 +/- 0.563 | 27.339 +/- 0.152 | 1.714x | `1d762530ef6c58be` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9738.731 | 46916.863 | 0.208x | 574.532 +/- 4.477 | 551.062 +/- 3.521 | 1.043x | `f2db5d5c08e66799` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 401.132 | 32714.177 | 0.012x | 39.144 +/- 0.270 | 19.616 +/- 0.095 | 1.995x | `e99a6c9902c3119e` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1012.241 | 50171.020 | 0.020x | 50.706 +/- 0.304 | 28.471 +/- 0.190 | 1.781x | `7dd49a516859a9cd` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 368.007 | 23936.675 | 0.015x | 45.560 +/- 0.225 | 17.086 +/- 0.178 | 2.666x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1104.791 | 24818.745 | 0.045x | 51.195 +/- 0.361 | 22.237 +/- 0.161 | 2.302x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9059.153 | 33779.724 | 0.268x | 599.634 +/- 4.691 | 558.774 +/- 2.731 | 1.073x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1028.211 | 23977.677 | 0.043x | 50.370 +/- 0.222 | 22.221 +/- 0.220 | 2.267x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1169.554 | 23999.149 | 0.049x | 49.305 +/- 0.269 | 22.190 +/- 0.232 | 2.222x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 373.064 | 23913.139 | 0.016x | 45.063 +/- 0.154 | 15.710 +/- 0.069 | 2.869x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1022.867 | 30706.294 | 0.033x | 73.462 +/- 0.262 | 23.279 +/- 0.410 | 3.156x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 305.312 | 24530.409 | 0.012x | 33.706 +/- 0.160 | 16.014 +/- 0.403 | 2.105x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 937.108 | 23769.507 | 0.039x | 38.748 +/- 0.207 | 19.635 +/- 0.133 | 1.973x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9040.720 | 31499.880 | 0.287x | 575.974 +/- 3.055 | 551.653 +/- 4.840 | 1.044x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 958.690 | 22795.099 | 0.042x | 37.610 +/- 0.220 | 19.588 +/- 0.192 | 1.920x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 917.523 | 22986.974 | 0.040x | 38.531 +/- 0.210 | 19.471 +/- 0.182 | 1.979x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 340.791 | 23135.414 | 0.015x | 33.444 +/- 0.138 | 13.689 +/- 0.045 | 2.443x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 940.207 | 25243.427 | 0.037x | 59.055 +/- 0.358 | 19.873 +/- 0.196 | 2.972x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 289.048 | 22682.834 | 0.013x | 33.806 +/- 0.266 | 15.556 +/- 0.199 | 2.173x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 885.194 | 22810.847 | 0.039x | 39.273 +/- 0.169 | 19.681 +/- 0.120 | 1.995x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9064.891 | 31802.788 | 0.285x | 582.097 +/- 3.174 | 564.447 +/- 6.843 | 1.031x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 953.587 | 23028.677 | 0.041x | 38.724 +/- 0.235 | 19.549 +/- 0.168 | 1.981x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 870.827 | 23496.389 | 0.037x | 38.857 +/- 0.209 | 19.530 +/- 0.144 | 1.990x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 271.326 | 22562.667 | 0.012x | 33.849 +/- 0.177 | 13.899 +/- 0.111 | 2.435x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 919.661 | 24868.990 | 0.037x | 60.492 +/- 0.249 | 20.413 +/- 0.126 | 2.963x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 300.625 | 22983.233 | 0.013x | 33.671 +/- 0.142 | 14.873 +/- 0.140 | 2.264x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 873.170 | 23081.752 | 0.038x | 38.616 +/- 0.260 | 18.906 +/- 0.185 | 2.043x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8395.890 | 30742.304 | 0.273x | 550.128 +/- 3.828 | 528.003 +/- 3.193 | 1.042x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 950.057 | 23528.853 | 0.040x | 37.726 +/- 0.218 | 18.905 +/- 0.162 | 1.996x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 886.766 | 22672.624 | 0.039x | 38.741 +/- 0.148 | 18.823 +/- 0.170 | 2.058x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 296.589 | 22244.936 | 0.013x | 33.542 +/- 0.153 | 13.384 +/- 0.119 | 2.506x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 948.185 | 26794.032 | 0.035x | 59.274 +/- 0.314 | 19.487 +/- 0.333 | 3.042x | `2bd384aefcaaa397` |
| `cpu_float32_tuple_list_output_pytree` | `case_default` | 2 | 256 | tuple[shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True, list[shape (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True]] | 493.947 | 25114.416 | 0.020x | 56.505 +/- 0.265 | 19.368 +/- 0.259 | 2.917x | `a62dacb062c1ed92` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1475.021 | 25003.179 | 0.059x | 64.683 +/- 0.250 | 25.362 +/- 0.266 | 2.550x | `3bce94d7e523bafe` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_127x131_by_131` | 2 | 16 | tuple[shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (131,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 14339.234 | 40618.212 | 0.353x | 923.197 +/- 4.142 | 983.500 +/- 36.083 | 0.939x | `022557af0d301f5e` |
| `cpu_float32_tuple_list_output_pytree` | `tensor_scalar_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1507.485 | 25583.506 | 0.059x | 63.423 +/- 0.358 | 27.914 +/- 1.325 | 2.272x | `f4ff04ee55c4e2cd` |
| `cpu_float32_tuple_list_output_pytree` | `scalar_tensor_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False]] | 1662.465 | 25184.562 | 0.066x | 64.117 +/- 0.240 | 31.520 +/- 0.974 | 2.034x | `f1950b665bfdc9f1` |
| `cpu_float32_tuple_list_output_pytree` | `empty_2x0_by_0` | 2 | 2048 | tuple[shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 636.969 | 27271.825 | 0.023x | 55.626 +/- 0.311 | 15.595 +/- 0.073 | 3.567x | `e89cfed7478c41fa` |
| `cpu_float32_tuple_list_output_pytree` | `transpose_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False]] | 1564.226 | 27498.022 | 0.057x | 87.335 +/- 0.454 | 25.710 +/- 0.236 | 3.397x | `776bd23d05673f66` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 331.937 | 22007.132 | 0.015x | 34.334 +/- 0.113 | 15.561 +/- 0.139 | 2.206x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 311.957 | 22081.354 | 0.014x | 29.919 +/- 0.103 | 15.541 +/- 0.202 | 1.925x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 334.391 | 22369.014 | 0.015x | 32.956 +/- 0.262 | 15.308 +/- 0.133 | 2.153x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 712.843 | 22931.169 | 0.031x | 38.041 +/- 0.234 | 18.514 +/- 0.121 | 2.055x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5512.779 | 27712.978 | 0.199x | 355.873 +/- 2.122 | 336.932 +/- 3.615 | 1.056x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 363.300 | 22034.478 | 0.016x | 33.587 +/- 0.244 | 14.447 +/- 0.061 | 2.325x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 732.237 | 24664.972 | 0.030x | 41.703 +/- 0.256 | 19.227 +/- 0.121 | 2.169x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 362.538 | 24009.180 | 0.015x | 39.480 +/- 0.187 | 16.060 +/- 0.251 | 2.458x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 930.622 | 23741.169 | 0.039x | 44.850 +/- 0.177 | 21.054 +/- 0.216 | 2.130x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8899.115 | 32154.876 | 0.277x | 581.029 +/- 3.032 | 559.696 +/- 7.598 | 1.038x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 967.382 | 24304.817 | 0.040x | 43.614 +/- 0.226 | 20.740 +/- 0.110 | 2.103x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 917.738 | 23530.370 | 0.039x | 44.858 +/- 0.208 | 20.706 +/- 0.159 | 2.166x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 344.742 | 23058.612 | 0.015x | 39.013 +/- 0.085 | 14.656 +/- 0.066 | 2.662x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 997.529 | 24212.893 | 0.041x | 66.252 +/- 0.320 | 21.403 +/- 0.175 | 3.095x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 331.802 | 22532.662 | 0.015x | 34.210 +/- 0.179 | 15.386 +/- 0.111 | 2.223x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 323.489 | 22118.685 | 0.015x | 30.013 +/- 0.162 | 15.296 +/- 0.213 | 1.962x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 346.299 | 23339.888 | 0.015x | 32.687 +/- 0.190 | 15.244 +/- 0.190 | 2.144x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 722.328 | 23154.282 | 0.031x | 38.389 +/- 0.483 | 18.668 +/- 0.156 | 2.056x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5490.710 | 27578.028 | 0.199x | 356.051 +/- 3.671 | 339.899 +/- 4.967 | 1.048x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 352.509 | 22561.481 | 0.016x | 33.367 +/- 0.094 | 14.264 +/- 0.075 | 2.339x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 716.734 | 24514.439 | 0.029x | 41.724 +/- 0.214 | 19.469 +/- 0.348 | 2.143x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 972.501 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 830.957 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 494.508 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 612.826 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 123977.606 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 124800.160 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 41353.983 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 76741.548 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 113668.354 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 103429.445 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 44602.475 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 78273.774 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1081.937 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 931.248 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 551.495 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 673.708 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, inference, training autograd, Python control flow, dynamic shapes, mutation_aliasing_views, containers and pytrees, decompositions, custom functions, recompilation guards, and dtype/device transitions. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `python_control_flow` | 8 | Supported and timed public cases: `cpu_float32_requires_grad_branch_unary` |
| `dynamic_shapes_symbolics` | 8 | Supported and timed public cases: `cpu_float32_dynamic_true_shape_stride_unary` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `containers_pytrees` | 6 | Supported and timed public cases: `cpu_float32_tuple_list_output_pytree` |
| `decompositions` | 6 | Supported and timed public cases: `cpu_float32_decomposition_square_scalar` |
| `custom_functions` | 6 | Supported and timed public cases: `cpu_float32_custom_function_unary` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `dtype_device_transitions` | 4 | Supported and timed public cases: `cpu_float32_float_identity_view` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 84 / 100. Zero-credit unsupported category weight: 16 / 100.
The torch_compile_corpus_v11 corpus also keeps 2 held-out broadcasting programs, 1 held-out containers-pytrees program, 1 held-out custom-function program, 1 held-out decomposition program, 1 held-out dtype/device-transition program, 1 held-out dynamic-shape program, 1 held-out inference program, 1 held-out mutation_aliasing_views program, 1 held-out Python-control-flow program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 3 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
