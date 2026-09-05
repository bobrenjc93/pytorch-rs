# `torch.compile` Eager CPU Release Timings

Date: 2026-09-05

Candidate provenance: source snapshot refreshed against this worktree, including dynamic `torch.compile(..., backend="eager", fullgraph=True, dynamic=True/False)` graphlets where `dynamic=True` reuses same-rank shape/stride variants, zero-argument `Tensor.float()` identity graphlets in the dtype/device-transition category, and one top-level `requires_grad` branch graphlet in the Python-control-flow category. The raw benchmark artifact is refreshed for `torch_compile_corpus_v12` with the current supported public cases included.

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

The benchmark uses the checked-in `torch_compile_corpus_v12` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input `dynamic=True` shape/stride graphlet, one one-input `Tensor.float()` identity dtype/device-transition program, one one-input training-autograd program, one one-input `requires_grad` branch Python-control-flow program, one one-input square decomposition program, one one-input custom-function helper-inline program, four two-input broadcasting programs, one two-input containers-pytrees program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Dynamic-shape cells use the same compiled wrapper across shape and stride variants in the coverage evaluator, with `dynamic=True` requiring same-rank graph reuse and `dynamic=False` preserving metadata-specialized recompiles; the benchmark times the checked-in `dynamic=True` public case across all compatible input variants. Detach cells return shared-storage aliases with `requires_grad=False`. `Tensor.float()` identity cells preserve values, shape, stride, storage offset, device, dtype, and `requires_grad`. `requires_grad` branch cells select the branch from input metadata, lower only that branch, and preserve the selected branch's output metadata. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Custom-function cells verify the same-module helper inline path over tensor proxy arguments with `neg`, `abs`, `add`, `relu`, `detach`, and `float` operations. Tuple/list output cells preserve container structure and record per-tensor metadata for each output leaf. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, `requires_grad` branch cache specialization, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile` with each case's checked-in `backend`, `fullgraph`, and optional `dynamic` flag. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, tuple/list container structure, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 147 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus and dynamic shape/stride variants in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v12`
- Cold first compiled call: 0.030x uncapped, 0.115x capped
- Steady-state materialized compiled call: 1.894x uncapped, 1.894x capped
- Timed supported cells: 147 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 python-control-flow, 7 dynamic-shape, 7 containers-pytrees, 7 decomposition, 7 custom-functions, 21 recompilation-guard, 7 dtype-device-transitions, 7 mutation_aliasing_views)
- Recompilation guard sequences: 16 rows, 72 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 84.0% supported by native compile cases, 16% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 397.867 | 75285.033 | 0.005x | 28.381 +/- 0.161 | 13.817 +/- 0.097 | 2.054x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 263.058 | 21823.476 | 0.012x | 24.800 +/- 0.124 | 13.700 +/- 0.129 | 1.810x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 270.339 | 22090.639 | 0.012x | 26.763 +/- 0.173 | 13.823 +/- 0.200 | 1.936x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 861.483 | 22746.667 | 0.038x | 32.844 +/- 0.189 | 18.843 +/- 0.166 | 1.743x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8723.295 | 30608.194 | 0.285x | 542.199 +/- 2.513 | 527.961 +/- 3.746 | 1.027x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 300.805 | 22265.769 | 0.014x | 27.192 +/- 0.155 | 13.242 +/- 0.055 | 2.054x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 888.339 | 23788.573 | 0.037x | 34.430 +/- 0.173 | 19.360 +/- 0.151 | 1.778x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 234.049 | 21777.536 | 0.011x | 22.805 +/- 0.106 | 12.544 +/- 0.110 | 1.818x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 204.704 | 22563.315 | 0.009x | 20.250 +/- 0.092 | 12.333 +/- 0.085 | 1.642x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 231.796 | 22258.703 | 0.010x | 21.822 +/- 0.130 | 12.304 +/- 0.149 | 1.774x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 823.826 | 22222.599 | 0.037x | 27.920 +/- 0.175 | 17.405 +/- 0.146 | 1.604x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8797.307 | 30863.559 | 0.285x | 559.420 +/- 4.142 | 546.455 +/- 3.610 | 1.024x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 309.333 | 21606.667 | 0.014x | 22.430 +/- 0.061 | 11.961 +/- 0.047 | 1.875x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 846.981 | 22543.890 | 0.038x | 30.090 +/- 0.230 | 17.391 +/- 0.109 | 1.730x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 276.529 | 21957.238 | 0.013x | 27.566 +/- 0.131 | 13.944 +/- 0.113 | 1.977x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 248.530 | 21636.052 | 0.011x | 24.695 +/- 0.200 | 13.680 +/- 0.120 | 1.805x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 293.224 | 21898.620 | 0.013x | 26.626 +/- 0.164 | 14.007 +/- 0.265 | 1.901x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 860.832 | 22597.081 | 0.038x | 33.287 +/- 0.557 | 18.466 +/- 0.096 | 1.803x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8642.858 | 29889.166 | 0.289x | 566.560 +/- 15.510 | 527.762 +/- 3.677 | 1.074x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 330.956 | 21646.107 | 0.015x | 29.413 +/- 2.224 | 13.114 +/- 0.061 | 2.243x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1142.948 | 23281.226 | 0.049x | 43.980 +/- 4.782 | 19.082 +/- 0.159 | 2.305x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 348.692 | 22779.807 | 0.015x | 40.251 +/- 0.334 | 17.561 +/- 0.172 | 2.292x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 346.915 | 23858.959 | 0.015x | 40.242 +/- 0.163 | 17.477 +/- 0.170 | 2.303x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 367.040 | 23065.503 | 0.016x | 44.020 +/- 0.221 | 17.409 +/- 0.268 | 2.529x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 968.340 | 23215.066 | 0.042x | 51.681 +/- 0.230 | 22.640 +/- 0.148 | 2.283x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8763.591 | 31402.474 | 0.279x | 570.322 +/- 3.815 | 533.384 +/- 3.638 | 1.069x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 420.577 | 24319.957 | 0.017x | 44.679 +/- 0.172 | 16.091 +/- 0.059 | 2.777x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 971.784 | 23880.442 | 0.041x | 55.715 +/- 0.182 | 23.404 +/- 0.197 | 2.381x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 414.587 | 23188.560 | 0.018x | 46.360 +/- 0.158 | 16.430 +/- 0.101 | 2.822x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 410.201 | 23058.648 | 0.018x | 41.888 +/- 0.152 | 19.038 +/- 1.165 | 2.200x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 515.926 | 25195.281 | 0.020x | 45.839 +/- 0.366 | 19.699 +/- 0.477 | 2.327x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1028.716 | 25431.403 | 0.040x | 54.149 +/- 0.218 | 26.015 +/- 0.334 | 2.081x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9071.607 | 34518.889 | 0.263x | 591.405 +/- 3.231 | 612.338 +/- 4.061 | 0.966x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 449.550 | 24624.738 | 0.018x | 46.514 +/- 0.215 | 18.520 +/- 0.384 | 2.512x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1064.089 | 26994.001 | 0.039x | 60.666 +/- 0.286 | 27.334 +/- 0.336 | 2.219x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 258.071 | 23447.206 | 0.011x | 22.035 +/- 0.147 | 13.666 +/- 0.088 | 1.612x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 202.346 | 21085.213 | 0.010x | 19.783 +/- 0.122 | 15.059 +/- 0.354 | 1.314x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 234.474 | 23245.191 | 0.010x | 21.227 +/- 0.115 | 15.198 +/- 0.168 | 1.397x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 572.841 | 23630.834 | 0.024x | 25.216 +/- 0.166 | 18.240 +/- 0.978 | 1.382x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5270.712 | 28860.014 | 0.183x | 336.343 +/- 2.541 | 353.392 +/- 7.974 | 0.952x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 291.145 | 21407.787 | 0.014x | 22.187 +/- 0.307 | 15.042 +/- 0.197 | 1.475x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 599.662 | 25660.464 | 0.023x | 26.810 +/- 0.278 | 19.105 +/- 0.363 | 1.403x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 219.151 | 24504.946 | 0.009x | 20.026 +/- 0.093 | 12.520 +/- 0.162 | 1.600x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 218.270 | 22634.803 | 0.010x | 18.967 +/- 0.244 | 12.306 +/- 0.072 | 1.541x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 238.606 | 22633.100 | 0.011x | 20.051 +/- 0.079 | 12.412 +/- 0.095 | 1.615x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 807.147 | 25328.903 | 0.032x | 25.466 +/- 0.194 | 17.575 +/- 0.705 | 1.449x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8815.169 | 29967.574 | 0.294x | 547.834 +/- 4.929 | 585.191 +/- 12.860 | 0.936x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 265.192 | 23150.953 | 0.011x | 20.912 +/- 0.230 | 12.371 +/- 0.096 | 1.690x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 848.488 | 24486.859 | 0.035x | 25.350 +/- 0.187 | 19.545 +/- 0.222 | 1.297x | `4ba5419e2e3f2393` |
| `cpu_float32_float_identity_view` | `case_default` | 1 | 256 | shape (3,), stride (4,), offset 1, torch.float32, cpu, requires_grad=True | 253.208 | 25835.755 | 0.010x | 19.834 +/- 0.107 | 13.178 +/- 0.052 | 1.505x | `58df67cd172620c1` |
| `cpu_float32_float_identity_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 209.406 | 26387.569 | 0.008x | 18.714 +/- 0.098 | 10.667 +/- 0.046 | 1.754x | `e75a1d3233117514` |
| `cpu_float32_float_identity_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 225.671 | 21235.200 | 0.011x | 20.109 +/- 0.221 | 10.773 +/- 0.030 | 1.867x | `4c3dc265c5b9d697` |
| `cpu_float32_float_identity_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 828.609 | 21361.547 | 0.039x | 25.415 +/- 0.237 | 15.367 +/- 0.080 | 1.654x | `5ccc89fb94f689e5` |
| `cpu_float32_float_identity_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8559.652 | 29928.911 | 0.286x | 540.913 +/- 2.696 | 538.614 +/- 4.033 | 1.004x | `91fa5699b26ca1b8` |
| `cpu_float32_float_identity_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 259.758 | 21363.480 | 0.012x | 20.681 +/- 0.181 | 10.752 +/- 0.033 | 1.924x | `e99a6c9902c3119e` |
| `cpu_float32_float_identity_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 827.622 | 22786.122 | 0.036x | 25.577 +/- 0.565 | 15.463 +/- 0.132 | 1.654x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 388.618 | 24402.737 | 0.016x | 41.541 +/- 0.218 | 18.164 +/- 0.150 | 2.287x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 422.749 | 22453.128 | 0.019x | 35.748 +/- 0.216 | 16.305 +/- 0.138 | 2.192x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 394.768 | 22183.801 | 0.018x | 39.250 +/- 0.204 | 16.291 +/- 0.165 | 2.409x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 772.233 | 23057.627 | 0.033x | 45.114 +/- 0.373 | 19.707 +/- 0.123 | 2.289x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5662.796 | 27478.314 | 0.206x | 365.497 +/- 2.092 | 338.180 +/- 2.786 | 1.081x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 402.279 | 22345.690 | 0.018x | 39.621 +/- 0.130 | 15.269 +/- 0.054 | 2.595x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 805.839 | 24364.133 | 0.033x | 49.460 +/- 0.260 | 20.931 +/- 0.220 | 2.363x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 319.413 | 22365.525 | 0.014x | 30.422 +/- 0.102 | 15.722 +/- 0.123 | 1.935x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 310.000 | 22162.853 | 0.014x | 30.603 +/- 0.190 | 15.741 +/- 0.214 | 1.944x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 344.060 | 22371.844 | 0.015x | 33.344 +/- 0.193 | 15.873 +/- 0.332 | 2.101x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 967.243 | 24200.650 | 0.040x | 40.515 +/- 0.192 | 21.634 +/- 0.420 | 1.873x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9076.043 | 34338.679 | 0.264x | 585.907 +/- 4.039 | 565.870 +/- 3.022 | 1.035x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 390.381 | 22275.579 | 0.018x | 34.033 +/- 0.139 | 14.695 +/- 0.093 | 2.316x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 961.004 | 25432.593 | 0.038x | 44.906 +/- 0.320 | 21.812 +/- 0.280 | 2.059x | `68585b64809ef02a` |
| `cpu_float32_custom_function_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 454.833 | 39359.655 | 0.012x | 54.188 +/- 0.218 | 18.847 +/- 0.149 | 2.875x | `d16fd2f4dd199523` |
| `cpu_float32_custom_function_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 408.002 | 24040.237 | 0.017x | 47.715 +/- 0.163 | 18.681 +/- 0.168 | 2.554x | `5c2ffe407931c8ee` |
| `cpu_float32_custom_function_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 429.539 | 25607.423 | 0.017x | 51.705 +/- 0.321 | 18.488 +/- 0.229 | 2.797x | `d85643b7b66a7ca9` |
| `cpu_float32_custom_function_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1074.656 | 24844.754 | 0.043x | 59.972 +/- 0.359 | 24.316 +/- 0.150 | 2.466x | `414eafab6fd10fb4` |
| `cpu_float32_custom_function_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8971.581 | 32583.880 | 0.275x | 592.229 +/- 4.614 | 552.458 +/- 2.462 | 1.072x | `7863bb8d1d98f49b` |
| `cpu_float32_custom_function_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 498.969 | 26683.336 | 0.019x | 52.775 +/- 0.118 | 17.326 +/- 0.133 | 3.046x | `e99a6c9902c3119e` |
| `cpu_float32_custom_function_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1079.918 | 26544.320 | 0.041x | 65.579 +/- 0.331 | 25.078 +/- 0.264 | 2.615x | `188c6817fce2e1e1` |
| `cpu_float32_requires_grad_branch_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 393.611 | 22015.135 | 0.018x | 29.316 +/- 0.180 | 14.035 +/- 0.145 | 2.089x | `43e5fdfc5aec3505` |
| `cpu_float32_requires_grad_branch_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 373.290 | 22032.270 | 0.017x | 25.701 +/- 0.093 | 13.780 +/- 0.260 | 1.865x | `e75a1d3233117514` |
| `cpu_float32_requires_grad_branch_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 389.899 | 22454.228 | 0.017x | 27.833 +/- 0.086 | 13.452 +/- 0.113 | 2.069x | `47aef822223dbae7` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 994.350 | 24374.152 | 0.041x | 34.405 +/- 0.329 | 18.560 +/- 0.097 | 1.854x | `2148badcc2b9e4ce` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8968.947 | 30839.722 | 0.291x | 564.586 +/- 2.898 | 542.399 +/- 2.665 | 1.041x | `d53163cb2693cd35` |
| `cpu_float32_requires_grad_branch_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 441.933 | 21657.403 | 0.020x | 28.521 +/- 0.137 | 12.959 +/- 0.049 | 2.201x | `e99a6c9902c3119e` |
| `cpu_float32_requires_grad_branch_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1022.692 | 23157.001 | 0.044x | 37.981 +/- 0.301 | 18.670 +/- 0.103 | 2.034x | `372841b6f1764798` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 375.513 | 53687.777 | 0.007x | 37.107 +/- 0.210 | 22.450 +/- 0.314 | 1.653x | `78824b56236781de` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 349.273 | 22478.516 | 0.016x | 31.756 +/- 0.186 | 19.302 +/- 0.174 | 1.645x | `96474978e4b2c20f` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 371.522 | 33024.802 | 0.011x | 34.816 +/- 0.176 | 20.860 +/- 0.331 | 1.669x | `3e3e1d5aa2a3441f` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1133.309 | 38448.726 | 0.029x | 42.850 +/- 0.596 | 27.709 +/- 0.218 | 1.546x | `1d762530ef6c58be` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8738.729 | 46772.386 | 0.187x | 570.287 +/- 3.316 | 555.008 +/- 5.611 | 1.028x | `f2db5d5c08e66799` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 404.047 | 32128.827 | 0.013x | 35.200 +/- 0.065 | 19.639 +/- 0.081 | 1.792x | `e99a6c9902c3119e` |
| `cpu_float32_dynamic_true_shape_stride_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 984.900 | 48998.668 | 0.020x | 46.595 +/- 0.174 | 28.907 +/- 0.211 | 1.612x | `7dd49a516859a9cd` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 372.559 | 25036.259 | 0.015x | 45.989 +/- 0.168 | 17.256 +/- 0.238 | 2.665x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 995.932 | 24902.516 | 0.040x | 51.623 +/- 0.198 | 22.781 +/- 0.423 | 2.266x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9056.885 | 33023.705 | 0.274x | 595.929 +/- 3.059 | 564.439 +/- 3.343 | 1.056x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1019.862 | 24212.853 | 0.042x | 50.782 +/- 0.249 | 22.352 +/- 0.221 | 2.272x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 950.162 | 24128.380 | 0.039x | 49.952 +/- 0.252 | 22.293 +/- 0.157 | 2.241x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 343.760 | 24288.919 | 0.014x | 45.487 +/- 0.108 | 15.631 +/- 0.077 | 2.910x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1014.334 | 29393.565 | 0.035x | 74.067 +/- 0.343 | 23.046 +/- 0.229 | 3.214x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 305.281 | 23938.534 | 0.013x | 34.136 +/- 0.135 | 15.329 +/- 0.164 | 2.227x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 877.918 | 23344.696 | 0.038x | 39.118 +/- 0.156 | 19.861 +/- 0.130 | 1.970x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8873.467 | 31981.078 | 0.277x | 578.113 +/- 3.586 | 554.330 +/- 3.138 | 1.043x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 970.132 | 25428.291 | 0.038x | 38.076 +/- 0.208 | 24.346 +/- 0.848 | 1.564x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 872.184 | 29041.737 | 0.030x | 39.258 +/- 0.223 | 23.971 +/- 1.410 | 1.638x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 296.038 | 27084.271 | 0.011x | 33.898 +/- 0.119 | 16.968 +/- 3.203 | 1.998x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 948.831 | 24999.148 | 0.038x | 59.550 +/- 0.270 | 20.050 +/- 0.122 | 2.970x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 278.521 | 22696.931 | 0.012x | 34.055 +/- 0.100 | 15.889 +/- 0.163 | 2.143x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 898.103 | 22808.865 | 0.039x | 39.804 +/- 0.200 | 19.953 +/- 0.142 | 1.995x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9008.161 | 31922.219 | 0.282x | 586.408 +/- 3.304 | 567.597 +/- 6.933 | 1.033x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 937.098 | 23094.066 | 0.041x | 39.028 +/- 0.162 | 19.852 +/- 0.211 | 1.966x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 875.745 | 23008.357 | 0.038x | 39.364 +/- 0.224 | 19.804 +/- 0.172 | 1.988x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 274.145 | 22307.522 | 0.012x | 34.314 +/- 0.118 | 13.892 +/- 0.120 | 2.470x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 923.533 | 23970.166 | 0.039x | 63.859 +/- 0.296 | 20.446 +/- 0.185 | 3.123x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 299.393 | 22594.251 | 0.013x | 34.326 +/- 0.129 | 14.810 +/- 0.113 | 2.318x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 869.535 | 22742.109 | 0.038x | 39.128 +/- 0.177 | 18.805 +/- 0.090 | 2.081x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8499.192 | 33360.320 | 0.255x | 550.228 +/- 2.163 | 530.883 +/- 3.143 | 1.036x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 953.853 | 22488.184 | 0.042x | 38.196 +/- 0.245 | 18.954 +/- 0.173 | 2.015x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 878.333 | 25758.732 | 0.034x | 39.277 +/- 0.158 | 18.872 +/- 0.139 | 2.081x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 289.273 | 23205.876 | 0.012x | 33.819 +/- 0.101 | 13.257 +/- 0.069 | 2.551x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 924.534 | 24283.921 | 0.038x | 59.349 +/- 0.390 | 19.179 +/- 0.147 | 3.095x | `2bd384aefcaaa397` |
| `cpu_float32_tuple_list_output_pytree` | `case_default` | 2 | 256 | tuple[shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True, list[shape (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True]] | 486.816 | 24284.517 | 0.020x | 57.505 +/- 0.171 | 19.289 +/- 0.210 | 2.981x | `a62dacb062c1ed92` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1481.271 | 25243.487 | 0.059x | 66.031 +/- 0.362 | 25.512 +/- 0.155 | 2.588x | `3bce94d7e523bafe` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_127x131_by_131` | 2 | 16 | tuple[shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (131,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 15074.257 | 38249.656 | 0.394x | 925.574 +/- 5.630 | 875.115 +/- 7.004 | 1.058x | `022557af0d301f5e` |
| `cpu_float32_tuple_list_output_pytree` | `tensor_scalar_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1525.132 | 25067.010 | 0.061x | 64.450 +/- 0.320 | 25.112 +/- 0.117 | 2.567x | `f4ff04ee55c4e2cd` |
| `cpu_float32_tuple_list_output_pytree` | `scalar_tensor_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False]] | 1663.677 | 24632.213 | 0.068x | 65.814 +/- 0.339 | 26.967 +/- 0.209 | 2.441x | `f1950b665bfdc9f1` |
| `cpu_float32_tuple_list_output_pytree` | `empty_2x0_by_0` | 2 | 2048 | tuple[shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 483.046 | 23863.410 | 0.020x | 56.480 +/- 0.272 | 15.633 +/- 0.080 | 3.613x | `e89cfed7478c41fa` |
| `cpu_float32_tuple_list_output_pytree` | `transpose_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False]] | 1560.776 | 27554.883 | 0.057x | 88.900 +/- 0.343 | 26.059 +/- 0.224 | 3.411x | `776bd23d05673f66` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 352.423 | 22346.781 | 0.016x | 34.783 +/- 0.139 | 15.769 +/- 0.160 | 2.206x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 309.334 | 21768.050 | 0.014x | 30.664 +/- 0.110 | 15.501 +/- 0.240 | 1.978x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 363.065 | 22854.288 | 0.016x | 33.335 +/- 0.146 | 15.589 +/- 0.340 | 2.138x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 702.803 | 22279.203 | 0.032x | 38.444 +/- 0.180 | 18.815 +/- 0.264 | 2.043x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5508.968 | 27774.901 | 0.198x | 360.169 +/- 2.516 | 344.148 +/- 2.607 | 1.047x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 400.636 | 22617.591 | 0.018x | 34.684 +/- 0.293 | 14.439 +/- 0.080 | 2.402x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 752.283 | 23770.744 | 0.032x | 42.865 +/- 0.340 | 19.288 +/- 0.150 | 2.222x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 345.157 | 22973.474 | 0.015x | 40.712 +/- 0.431 | 16.243 +/- 0.129 | 2.506x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 970.808 | 23907.792 | 0.041x | 46.559 +/- 0.387 | 21.098 +/- 0.159 | 2.207x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9216.001 | 33500.827 | 0.275x | 595.164 +/- 3.727 | 560.901 +/- 3.691 | 1.061x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 990.083 | 23882.484 | 0.041x | 45.342 +/- 0.306 | 20.945 +/- 0.114 | 2.165x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 929.101 | 23467.141 | 0.040x | 46.227 +/- 0.327 | 20.800 +/- 0.175 | 2.222x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 339.749 | 23378.436 | 0.015x | 40.685 +/- 0.224 | 14.710 +/- 0.069 | 2.766x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1022.071 | 26176.524 | 0.039x | 66.853 +/- 0.249 | 21.332 +/- 0.156 | 3.134x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 338.322 | 22880.739 | 0.015x | 35.461 +/- 0.234 | 15.403 +/- 0.118 | 2.302x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 370.702 | 22128.630 | 0.017x | 30.798 +/- 0.189 | 15.480 +/- 0.233 | 1.990x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 361.347 | 23058.493 | 0.016x | 33.350 +/- 0.162 | 15.156 +/- 0.220 | 2.200x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 700.394 | 23004.070 | 0.030x | 39.338 +/- 0.699 | 18.715 +/- 0.158 | 2.102x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5638.013 | 28915.401 | 0.195x | 365.093 +/- 2.742 | 335.625 +/- 1.906 | 1.088x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 410.020 | 22074.539 | 0.019x | 34.465 +/- 0.262 | 14.532 +/- 0.112 | 2.372x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 733.034 | 23423.094 | 0.031x | 42.794 +/- 0.556 | 19.139 +/- 0.145 | 2.236x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1038.101 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 861.213 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 496.661 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 627.840 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 129461.077 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 132266.830 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 43284.242 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 77142.376 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 116400.230 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 106463.948 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 41809.021 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 73755.604 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1053.043 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 882.355 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 520.348 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 672.078 |

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
The torch_compile_corpus_v12 corpus also keeps 2 held-out broadcasting programs, 1 held-out containers-pytrees program, 1 held-out custom-function program, 1 held-out decomposition program, 1 held-out dtype/device-transition program, 1 held-out dynamic-shape program, 1 held-out inference program, 1 held-out mutation_aliasing_views program, 1 held-out Python-control-flow program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 3 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
