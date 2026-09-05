# `torch.compile` Eager CPU Release Timings

Date: 2026-09-05

Candidate provenance: source snapshot refreshed against this worktree, including zero-argument `Tensor.float()` identity graphlets in the dtype/device-transition category, one top-level `requires_grad` branch graphlet in the Python-control-flow category, and module-global exact native Tensor constant capture in the modules/parameters/buffers category. The raw benchmark artifact is refreshed for `torch_compile_corpus_v11` with the current supported public cases included.

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
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`, `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_7e85fbd5/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_7e85fbd5/python/torch_rs/__init__.py`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`; `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both reported 1
- Dependency installation: locked `uv sync` used the worktree-local uv cache
- Build: release editable wheel installed in the worktree-local `.venv`

The benchmark uses the checked-in `torch_compile_corpus_v11` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input `Tensor.float()` identity dtype/device-transition program, one one-input training-autograd program, one one-input `requires_grad` branch Python-control-flow program, one one-input square decomposition program, one one-input custom-function helper-inline program, one one-input module-global Tensor buffer program, four two-input broadcasting programs, one two-input containers-pytrees program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Detach cells return shared-storage aliases with `requires_grad=False`. `Tensor.float()` identity cells preserve values, shape, stride, storage offset, device, dtype, and `requires_grad`. `requires_grad` branch cells select the branch from input metadata, lower only that branch, and preserve the selected branch's output metadata. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Custom-function cells verify the same-module helper inline path over tensor proxy arguments with `neg`, `abs`, `add`, `relu`, `detach`, and `float` operations. Module-global Tensor buffer cells verify captured read-only global values and metadata across the same one-input variants. Tuple/list output cells preserve container structure and record per-tensor metadata for each output leaf. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, `requires_grad` branch cache specialization, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, tuple/list container structure, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 147 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.


## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v11`
- Cold first compiled call: 0.031x uncapped, 0.115x capped
- Steady-state materialized compiled call: 1.890x uncapped, 1.890x capped
- Timed supported cells: 147 (35 tensor-arithmetic, 28 broadcasting, 7 modules-parameters-buffers, 7 inference, 7 training-autograd, 7 python-control-flow, 7 containers-pytrees, 7 decomposition, 7 custom-functions, 21 recompilation-guard, 7 dtype-device-transitions, 7 mutation_aliasing_views)
- Recompilation guard sequences: 16 rows, 72 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 84.0% supported by native compile cases, 16% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 365.002 | 71379.880 | 0.005x | 24.113 +/- 0.190 | 12.807 +/- 0.149 | 1.883x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 246.673 | 19933.232 | 0.012x | 20.805 +/- 0.072 | 12.516 +/- 0.093 | 1.662x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 237.990 | 19831.718 | 0.012x | 22.906 +/- 0.092 | 12.720 +/- 0.186 | 1.801x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 751.502 | 20582.910 | 0.037x | 28.240 +/- 0.122 | 17.301 +/- 0.203 | 1.632x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7598.899 | 27116.462 | 0.280x | 482.557 +/- 2.008 | 469.847 +/- 2.435 | 1.027x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 282.151 | 19691.997 | 0.014x | 23.255 +/- 0.130 | 12.115 +/- 0.042 | 1.920x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 798.327 | 21274.881 | 0.038x | 29.664 +/- 0.169 | 17.555 +/- 0.144 | 1.690x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 211.860 | 19856.035 | 0.011x | 19.595 +/- 0.119 | 11.277 +/- 0.068 | 1.738x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 202.707 | 19372.839 | 0.010x | 17.163 +/- 0.105 | 11.232 +/- 0.079 | 1.528x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 200.068 | 21456.541 | 0.009x | 18.561 +/- 0.068 | 11.279 +/- 0.175 | 1.646x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 702.658 | 19879.039 | 0.035x | 23.731 +/- 0.085 | 15.074 +/- 0.130 | 1.574x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7328.640 | 26169.980 | 0.280x | 466.888 +/- 1.488 | 452.208 +/- 2.496 | 1.032x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 243.488 | 19123.011 | 0.013x | 19.104 +/- 0.101 | 10.720 +/- 0.064 | 1.782x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 709.833 | 19989.216 | 0.036x | 25.493 +/- 0.128 | 15.231 +/- 0.129 | 1.674x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 247.429 | 19281.285 | 0.013x | 23.790 +/- 0.105 | 12.892 +/- 0.136 | 1.845x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 215.240 | 18857.454 | 0.011x | 20.668 +/- 0.107 | 12.599 +/- 0.147 | 1.640x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 236.793 | 19674.070 | 0.012x | 22.701 +/- 0.109 | 12.569 +/- 0.145 | 1.806x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 753.409 | 20314.774 | 0.037x | 28.289 +/- 0.176 | 16.853 +/- 0.113 | 1.679x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7592.094 | 26405.868 | 0.288x | 486.003 +/- 2.383 | 468.258 +/- 2.320 | 1.038x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 285.397 | 19501.064 | 0.015x | 23.194 +/- 0.080 | 11.938 +/- 0.050 | 1.943x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 768.748 | 20689.100 | 0.037x | 29.467 +/- 0.158 | 17.454 +/- 0.162 | 1.688x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 304.456 | 20339.095 | 0.015x | 33.612 +/- 0.173 | 15.947 +/- 0.134 | 2.108x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 329.358 | 20082.232 | 0.016x | 33.581 +/- 0.163 | 16.102 +/- 0.177 | 2.086x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 313.418 | 20494.942 | 0.015x | 37.364 +/- 0.127 | 16.086 +/- 0.203 | 2.323x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 836.185 | 21663.739 | 0.039x | 44.826 +/- 0.222 | 20.742 +/- 0.093 | 2.161x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7657.007 | 27813.787 | 0.275x | 506.661 +/- 2.566 | 478.690 +/- 3.171 | 1.058x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 333.454 | 20207.732 | 0.017x | 38.159 +/- 0.158 | 14.955 +/- 0.047 | 2.552x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 858.023 | 21882.515 | 0.039x | 47.932 +/- 0.227 | 21.455 +/- 0.215 | 2.234x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 357.086 | 20574.842 | 0.017x | 40.033 +/- 0.204 | 15.363 +/- 0.093 | 2.606x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 348.683 | 19939.331 | 0.017x | 35.204 +/- 0.177 | 16.623 +/- 0.184 | 2.118x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 371.627 | 20799.698 | 0.018x | 39.040 +/- 0.151 | 16.586 +/- 0.259 | 2.354x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 862.830 | 21341.662 | 0.040x | 46.381 +/- 0.259 | 21.121 +/- 0.146 | 2.196x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7519.144 | 27895.421 | 0.270x | 505.197 +/- 7.960 | 470.902 +/- 2.447 | 1.073x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 407.411 | 20293.506 | 0.020x | 39.827 +/- 0.106 | 15.427 +/- 0.050 | 2.582x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 910.707 | 21880.567 | 0.042x | 52.319 +/- 0.381 | 22.337 +/- 0.134 | 2.342x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 224.825 | 19147.628 | 0.012x | 18.812 +/- 0.126 | 12.319 +/- 0.095 | 1.527x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 195.811 | 19006.074 | 0.010x | 16.602 +/- 0.077 | 12.052 +/- 0.064 | 1.378x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 199.613 | 19147.002 | 0.010x | 18.159 +/- 0.116 | 12.271 +/- 0.116 | 1.480x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 530.232 | 19392.258 | 0.027x | 21.629 +/- 0.150 | 14.698 +/- 0.085 | 1.472x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4555.516 | 24027.505 | 0.190x | 296.210 +/- 2.112 | 285.865 +/- 1.642 | 1.036x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 240.959 | 19510.477 | 0.012x | 18.480 +/- 0.091 | 11.845 +/- 0.045 | 1.560x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 525.630 | 20807.384 | 0.025x | 22.181 +/- 0.112 | 14.940 +/- 0.124 | 1.485x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 217.830 | 19962.050 | 0.011x | 16.794 +/- 0.072 | 9.916 +/- 0.081 | 1.694x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 181.114 | 18635.594 | 0.010x | 15.664 +/- 0.064 | 9.785 +/- 0.033 | 1.601x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 202.923 | 18947.170 | 0.011x | 16.936 +/- 0.069 | 9.919 +/- 0.048 | 1.707x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 725.497 | 19853.696 | 0.037x | 21.552 +/- 0.119 | 13.912 +/- 0.094 | 1.549x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7574.327 | 26872.183 | 0.282x | 476.842 +/- 2.447 | 469.498 +/- 2.543 | 1.016x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 256.568 | 21983.883 | 0.012x | 17.433 +/- 0.095 | 9.779 +/- 0.055 | 1.783x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 738.722 | 21082.104 | 0.035x | 21.593 +/- 0.139 | 13.989 +/- 0.094 | 1.544x | `4ba5419e2e3f2393` |
| `cpu_float32_float_identity_view` | `case_default` | 1 | 256 | shape (3,), stride (4,), offset 1, torch.float32, cpu, requires_grad=True | 201.484 | 19499.601 | 0.010x | 16.712 +/- 0.061 | 9.472 +/- 0.072 | 1.764x | `58df67cd172620c1` |
| `cpu_float32_float_identity_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 190.058 | 18183.029 | 0.010x | 15.688 +/- 0.091 | 9.367 +/- 0.040 | 1.675x | `e75a1d3233117514` |
| `cpu_float32_float_identity_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 202.226 | 18440.844 | 0.011x | 16.824 +/- 0.134 | 9.483 +/- 0.018 | 1.774x | `4c3dc265c5b9d697` |
| `cpu_float32_float_identity_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 722.523 | 18786.046 | 0.038x | 21.963 +/- 0.470 | 13.557 +/- 0.107 | 1.620x | `5ccc89fb94f689e5` |
| `cpu_float32_float_identity_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7561.198 | 26441.552 | 0.286x | 480.686 +/- 4.305 | 468.178 +/- 2.487 | 1.027x | `91fa5699b26ca1b8` |
| `cpu_float32_float_identity_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 237.159 | 18827.029 | 0.013x | 17.307 +/- 0.083 | 9.437 +/- 0.030 | 1.834x | `e99a6c9902c3119e` |
| `cpu_float32_float_identity_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 734.351 | 20243.206 | 0.036x | 21.310 +/- 0.101 | 13.661 +/- 0.134 | 1.560x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 347.785 | 20632.359 | 0.017x | 35.869 +/- 0.119 | 17.401 +/- 0.205 | 2.061x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 380.832 | 20035.145 | 0.019x | 30.268 +/- 0.133 | 15.264 +/- 0.146 | 1.983x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 328.401 | 20001.155 | 0.016x | 33.726 +/- 0.187 | 15.189 +/- 0.178 | 2.220x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 635.837 | 20519.108 | 0.031x | 39.561 +/- 0.213 | 18.200 +/- 0.177 | 2.174x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4693.700 | 25062.711 | 0.187x | 312.634 +/- 1.559 | 289.378 +/- 1.385 | 1.080x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 354.056 | 20088.536 | 0.018x | 34.299 +/- 0.222 | 14.138 +/- 0.078 | 2.426x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 656.738 | 21882.761 | 0.030x | 42.579 +/- 0.256 | 18.836 +/- 0.106 | 2.261x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 278.050 | 19936.467 | 0.014x | 26.226 +/- 0.141 | 14.525 +/- 0.117 | 1.806x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 273.303 | 19011.427 | 0.014x | 26.158 +/- 0.080 | 14.664 +/- 0.196 | 1.784x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 288.186 | 20107.254 | 0.014x | 28.472 +/- 0.081 | 14.593 +/- 0.178 | 1.951x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 824.852 | 20713.512 | 0.040x | 35.158 +/- 0.188 | 19.059 +/- 0.152 | 1.845x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7586.586 | 27654.982 | 0.274x | 488.100 +/- 2.175 | 468.503 +/- 3.210 | 1.042x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 330.079 | 19932.295 | 0.017x | 29.124 +/- 0.087 | 13.685 +/- 0.123 | 2.128x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 827.797 | 21414.688 | 0.039x | 38.569 +/- 0.258 | 19.906 +/- 0.313 | 1.938x | `68585b64809ef02a` |
| `cpu_float32_custom_function_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 445.964 | 34392.473 | 0.013x | 77.637 +/- 0.359 | 18.223 +/- 0.252 | 4.260x | `d16fd2f4dd199523` |
| `cpu_float32_custom_function_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 371.091 | 22377.174 | 0.017x | 71.493 +/- 0.598 | 17.917 +/- 0.224 | 3.990x | `5c2ffe407931c8ee` |
| `cpu_float32_custom_function_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 376.319 | 22051.656 | 0.017x | 75.176 +/- 0.391 | 17.961 +/- 0.295 | 4.186x | `d85643b7b66a7ca9` |
| `cpu_float32_custom_function_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 882.795 | 22222.109 | 0.040x | 84.011 +/- 0.931 | 22.492 +/- 0.156 | 3.735x | `414eafab6fd10fb4` |
| `cpu_float32_custom_function_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7554.192 | 29190.775 | 0.259x | 540.003 +/- 5.673 | 463.815 +/- 2.580 | 1.164x | `7863bb8d1d98f49b` |
| `cpu_float32_custom_function_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 484.333 | 21540.918 | 0.022x | 75.961 +/- 0.357 | 16.460 +/- 0.051 | 4.615x | `e99a6c9902c3119e` |
| `cpu_float32_custom_function_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 907.487 | 23236.248 | 0.039x | 87.815 +/- 0.543 | 23.496 +/- 0.181 | 3.737x | `188c6817fce2e1e1` |
| `cpu_float32_requires_grad_branch_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 323.489 | 19365.438 | 0.017x | 24.832 +/- 0.088 | 12.866 +/- 0.109 | 1.930x | `43e5fdfc5aec3505` |
| `cpu_float32_requires_grad_branch_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 321.006 | 19259.898 | 0.017x | 21.715 +/- 0.055 | 12.847 +/- 0.167 | 1.690x | `e75a1d3233117514` |
| `cpu_float32_requires_grad_branch_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 338.397 | 20431.636 | 0.017x | 23.689 +/- 0.048 | 12.866 +/- 0.200 | 1.841x | `47aef822223dbae7` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 846.280 | 20863.609 | 0.041x | 29.435 +/- 0.136 | 16.792 +/- 0.103 | 1.753x | `2148badcc2b9e4ce` |
| `cpu_float32_requires_grad_branch_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7646.166 | 26760.188 | 0.286x | 494.868 +/- 8.112 | 470.703 +/- 2.140 | 1.051x | `d53163cb2693cd35` |
| `cpu_float32_requires_grad_branch_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 392.413 | 19055.800 | 0.021x | 24.220 +/- 0.080 | 11.993 +/- 0.055 | 2.019x | `e99a6c9902c3119e` |
| `cpu_float32_requires_grad_branch_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 860.687 | 20460.955 | 0.042x | 31.821 +/- 0.176 | 16.997 +/- 0.111 | 1.872x | `372841b6f1764798` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 331.942 | 21081.538 | 0.016x | 40.363 +/- 0.288 | 16.206 +/- 0.131 | 2.491x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 879.721 | 21955.951 | 0.040x | 44.968 +/- 0.297 | 20.446 +/- 0.163 | 2.199x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7555.064 | 29012.880 | 0.260x | 505.890 +/- 5.596 | 475.874 +/- 2.686 | 1.063x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 853.626 | 21243.484 | 0.040x | 44.397 +/- 0.785 | 19.969 +/- 0.191 | 2.223x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 823.211 | 21409.550 | 0.038x | 42.958 +/- 0.318 | 19.802 +/- 0.162 | 2.169x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 308.562 | 20862.773 | 0.015x | 39.340 +/- 0.154 | 14.657 +/- 0.164 | 2.684x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 875.555 | 23334.416 | 0.038x | 63.467 +/- 0.253 | 21.066 +/- 0.190 | 3.013x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 277.599 | 20913.184 | 0.013x | 29.457 +/- 0.148 | 14.249 +/- 0.135 | 2.067x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 757.761 | 20957.902 | 0.036x | 33.843 +/- 0.292 | 17.828 +/- 0.213 | 1.898x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7416.970 | 27409.260 | 0.271x | 485.138 +/- 1.950 | 466.339 +/- 2.634 | 1.040x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 805.524 | 20524.901 | 0.039x | 32.714 +/- 0.293 | 17.452 +/- 0.107 | 1.875x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 886.250 | 20946.200 | 0.042x | 34.322 +/- 0.489 | 17.686 +/- 0.198 | 1.941x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 255.507 | 20281.895 | 0.013x | 29.690 +/- 0.171 | 12.570 +/- 0.077 | 2.362x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 825.033 | 28170.912 | 0.029x | 52.287 +/- 0.661 | 18.082 +/- 0.134 | 2.892x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 262.272 | 20731.840 | 0.013x | 29.418 +/- 0.125 | 15.276 +/- 0.220 | 1.926x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 758.958 | 20881.747 | 0.036x | 34.297 +/- 0.265 | 18.314 +/- 0.222 | 1.873x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7540.426 | 27651.817 | 0.273x | 490.713 +/- 3.718 | 473.516 +/- 3.164 | 1.036x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 825.769 | 20379.763 | 0.041x | 33.639 +/- 0.351 | 17.548 +/- 0.156 | 1.917x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 782.559 | 20045.711 | 0.039x | 33.946 +/- 0.128 | 17.544 +/- 0.253 | 1.935x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 249.863 | 19586.222 | 0.013x | 29.501 +/- 0.096 | 12.886 +/- 0.075 | 2.289x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 815.644 | 22552.529 | 0.036x | 52.470 +/- 0.344 | 18.624 +/- 0.240 | 2.817x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 281.145 | 20479.783 | 0.014x | 29.522 +/- 0.152 | 13.901 +/- 0.216 | 2.124x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 746.479 | 20641.253 | 0.036x | 33.562 +/- 0.243 | 16.918 +/- 0.151 | 1.984x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7235.951 | 27021.093 | 0.268x | 470.067 +/- 2.915 | 452.453 +/- 1.423 | 1.039x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 813.916 | 20134.215 | 0.040x | 32.391 +/- 0.222 | 16.716 +/- 0.183 | 1.938x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 748.061 | 20147.631 | 0.037x | 33.803 +/- 0.203 | 16.931 +/- 0.205 | 1.997x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 279.563 | 19599.376 | 0.014x | 29.118 +/- 0.099 | 12.148 +/- 0.057 | 2.397x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 821.283 | 21972.466 | 0.037x | 51.225 +/- 0.238 | 17.380 +/- 0.196 | 2.947x | `2bd384aefcaaa397` |
| `cpu_float32_global_buffer_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 285.808 | 20697.583 | 0.014x | 30.612 +/- 0.123 | 13.785 +/- 0.152 | 2.221x | `dd1428515dc76c04` |
| `cpu_float32_global_buffer_add` | `scalar` | 1 | 2048 | shape (1,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 261.746 | 20316.561 | 0.013x | 28.564 +/- 0.072 | 13.797 +/- 0.425 | 2.070x | `5214bceaa64234ff` |
| `cpu_float32_global_buffer_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 271.175 | 20391.750 | 0.013x | 29.009 +/- 0.128 | 13.645 +/- 0.097 | 2.126x | `dbed541b43896343` |
| `cpu_float32_global_buffer_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 810.365 | 20782.697 | 0.039x | 42.598 +/- 0.230 | 17.904 +/- 0.152 | 2.379x | `e4ebd180a49a9ea8` |
| `cpu_float32_global_buffer_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7937.832 | 28832.112 | 0.275x | 613.798 +/- 3.646 | 483.967 +/- 3.281 | 1.268x | `aed7b1c611594d2a` |
| `cpu_float32_global_buffer_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 337.120 | 20505.783 | 0.016x | 30.234 +/- 0.141 | 13.455 +/- 0.122 | 2.247x | `e99a6c9902c3119e` |
| `cpu_float32_global_buffer_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 856.450 | 24036.749 | 0.036x | 52.630 +/- 0.259 | 17.955 +/- 0.208 | 2.931x | `630802db112622aa` |
| `cpu_float32_tuple_list_output_pytree` | `case_default` | 2 | 256 | tuple[shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True, list[shape (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True]] | 432.489 | 25719.164 | 0.017x | 48.371 +/- 0.253 | 18.609 +/- 0.191 | 2.599x | `a62dacb062c1ed92` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1284.513 | 22960.059 | 0.056x | 55.798 +/- 0.386 | 23.292 +/- 0.207 | 2.396x | `3bce94d7e523bafe` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_127x131_by_131` | 2 | 16 | tuple[shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (131,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 12468.766 | 33897.384 | 0.368x | 785.372 +/- 2.959 | 752.870 +/- 4.976 | 1.043x | `022557af0d301f5e` |
| `cpu_float32_tuple_list_output_pytree` | `tensor_scalar_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1305.475 | 23728.027 | 0.055x | 54.288 +/- 0.312 | 23.367 +/- 0.582 | 2.323x | `f4ff04ee55c4e2cd` |
| `cpu_float32_tuple_list_output_pytree` | `scalar_tensor_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False]] | 1376.893 | 23390.016 | 0.059x | 54.409 +/- 0.261 | 24.465 +/- 0.286 | 2.224x | `f1950b665bfdc9f1` |
| `cpu_float32_tuple_list_output_pytree` | `empty_2x0_by_0` | 2 | 2048 | tuple[shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 414.683 | 21592.582 | 0.019x | 47.069 +/- 0.162 | 14.802 +/- 0.111 | 3.180x | `e89cfed7478c41fa` |
| `cpu_float32_tuple_list_output_pytree` | `transpose_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False]] | 1323.526 | 24956.050 | 0.053x | 75.009 +/- 0.244 | 23.959 +/- 0.155 | 3.131x | `776bd23d05673f66` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 297.755 | 20671.018 | 0.014x | 29.775 +/- 0.143 | 14.940 +/- 0.245 | 1.993x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 279.328 | 19681.441 | 0.014x | 25.590 +/- 0.089 | 14.665 +/- 0.227 | 1.745x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 294.615 | 20717.418 | 0.014x | 28.386 +/- 0.098 | 14.398 +/- 0.193 | 1.971x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 601.059 | 20609.745 | 0.029x | 32.966 +/- 0.191 | 17.248 +/- 0.231 | 1.911x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4579.702 | 24852.448 | 0.184x | 305.695 +/- 2.885 | 289.150 +/- 2.342 | 1.057x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 332.558 | 19692.503 | 0.017x | 29.026 +/- 0.105 | 13.429 +/- 0.074 | 2.161x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 623.563 | 20738.345 | 0.030x | 36.318 +/- 0.323 | 17.812 +/- 0.105 | 2.039x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 301.716 | 20849.478 | 0.014x | 34.602 +/- 0.201 | 15.069 +/- 0.150 | 2.296x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 793.135 | 21277.371 | 0.037x | 39.153 +/- 0.269 | 19.111 +/- 0.156 | 2.049x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7543.977 | 28228.450 | 0.267x | 529.120 +/- 3.144 | 471.792 +/- 2.925 | 1.122x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 906.641 | 21106.121 | 0.043x | 41.694 +/- 0.465 | 18.859 +/- 0.159 | 2.211x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 860.076 | 20783.167 | 0.041x | 42.799 +/- 0.384 | 19.050 +/- 0.184 | 2.247x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 342.128 | 20757.594 | 0.016x | 34.008 +/- 0.097 | 13.639 +/- 0.059 | 2.493x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 883.131 | 22070.415 | 0.040x | 57.207 +/- 0.335 | 19.490 +/- 0.136 | 2.935x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 310.349 | 20584.141 | 0.015x | 29.866 +/- 0.141 | 14.656 +/- 0.140 | 2.038x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 273.118 | 19741.477 | 0.014x | 25.899 +/- 0.278 | 14.405 +/- 0.135 | 1.798x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 352.363 | 20395.967 | 0.017x | 28.698 +/- 0.180 | 14.177 +/- 0.221 | 2.024x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 599.477 | 20545.553 | 0.029x | 33.163 +/- 0.155 | 17.283 +/- 0.165 | 1.919x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4601.721 | 24734.630 | 0.186x | 305.326 +/- 2.033 | 292.026 +/- 1.601 | 1.046x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 339.293 | 20450.690 | 0.017x | 29.028 +/- 0.108 | 13.300 +/- 0.063 | 2.183x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 629.361 | 21319.078 | 0.030x | 36.125 +/- 0.187 | 17.783 +/- 0.125 | 2.031x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 843.536 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 750.014 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 435.860 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 545.786 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 143512.903 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 117099.039 |
| `requires_grad_branch_unary_cache` | `torch_rs,pytorch` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 36485.568 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 67675.148 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 107443.653 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 94622.130 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `pytorch` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 37657.161 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 66744.419 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 948.686 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 790.988 |
| `requires_grad_branch_unary_cache` | `pytorch,torch_rs` | `torch_rs` | None | false_branch ok(initial); same_false_metadata ok(same_metadata); true_branch ok(requires_grad) | 457.892 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 571.954 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, modules, parameters, and buffers, inference, training autograd, Python control flow, mutation_aliasing_views, containers and pytrees, decompositions, custom functions, recompilation guards, and dtype/device transitions. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `modules_parameters_buffers` | 8 | Supported and timed public cases: `cpu_float32_global_buffer_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `python_control_flow` | 8 | Supported and timed public cases: `cpu_float32_requires_grad_branch_unary` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `containers_pytrees` | 6 | Supported and timed public cases: `cpu_float32_tuple_list_output_pytree` |
| `decompositions` | 6 | Supported and timed public cases: `cpu_float32_decomposition_square_scalar` |
| `custom_functions` | 6 | Supported and timed public cases: `cpu_float32_custom_function_unary` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `dtype_device_transitions` | 4 | Supported and timed public cases: `cpu_float32_float_identity_view` |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 84 / 100. Zero-credit unsupported category weight: 16 / 100.
The torch_compile_corpus_v11 corpus also keeps 2 held-out broadcasting programs, 1 held-out containers-pytrees program, 1 held-out custom-function program, 1 held-out decomposition program, 1 held-out dtype/device-transition program, 1 held-out inference program, 1 held-out modules/parameters/buffers program, 1 held-out mutation_aliasing_views program, 1 held-out Python-control-flow program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 3 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
