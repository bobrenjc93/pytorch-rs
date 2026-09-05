# `torch.compile` Eager CPU Release Timings

Date: 2026-09-05

Candidate provenance: source snapshot refreshed against this worktree, including zero-argument `Tensor.float()` identity graphlets in the dtype/device-transition category. The raw benchmark artifact is refreshed for `torch_compile_corpus_v9` with the current supported public cases included.

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
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_ca518172/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_ca518172/python/torch_rs/__init__.py`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`; `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both reported 1
- Dependency installation: locked `uv sync` used the worktree-local uv cache
- Build: release editable wheel installed in the worktree-local `.venv`

The benchmark uses the checked-in `torch_compile_corpus_v9` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input `Tensor.float()` identity dtype/device-transition program, one one-input training-autograd program, one one-input square decomposition program, one one-input custom-function helper-inline program, four two-input broadcasting programs, one two-input containers-pytrees program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Detach cells return shared-storage aliases with `requires_grad=False`. `Tensor.float()` identity cells preserve values, shape, stride, storage offset, device, dtype, and `requires_grad`. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Custom-function cells verify the same-module helper inline path over tensor proxy arguments with `neg`, `abs`, `add`, `relu`, `detach`, and `float` operations. Tuple/list output cells preserve container structure and record per-tensor metadata for each output leaf. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, tuple/list container structure, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 133 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v9`
- Cold first compiled call: 0.031x uncapped, 0.115x capped
- Steady-state materialized compiled call: 1.935x uncapped, 1.935x capped
- Timed supported cells: 133 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 containers-pytrees, 7 decomposition, 7 custom-functions, 21 recompilation-guard, 7 dtype-device-transitions, 7 mutation_aliasing_views)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 68.0% supported by native compile cases, 32% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 388.722 | 93503.109 | 0.004x | 27.764 +/- 0.119 | 16.941 +/- 0.130 | 1.639x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 268.315 | 23652.069 | 0.011x | 24.161 +/- 0.082 | 14.406 +/- 0.786 | 1.677x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 290.449 | 21728.301 | 0.013x | 26.131 +/- 0.085 | 13.820 +/- 0.182 | 1.891x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 894.251 | 22208.171 | 0.040x | 32.370 +/- 0.219 | 18.565 +/- 0.082 | 1.744x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8572.851 | 29903.541 | 0.287x | 535.836 +/- 1.835 | 524.514 +/- 2.160 | 1.022x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 294.836 | 21617.158 | 0.014x | 26.698 +/- 0.075 | 13.183 +/- 0.044 | 2.025x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 973.992 | 23672.444 | 0.041x | 34.148 +/- 0.277 | 19.232 +/- 0.131 | 1.776x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 249.742 | 21357.746 | 0.012x | 22.599 +/- 0.098 | 12.401 +/- 0.077 | 1.822x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 212.261 | 21478.809 | 0.010x | 19.789 +/- 0.047 | 12.248 +/- 0.081 | 1.616x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 237.219 | 21165.891 | 0.011x | 21.594 +/- 0.129 | 12.164 +/- 0.156 | 1.775x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 879.394 | 21600.584 | 0.041x | 27.483 +/- 0.184 | 17.246 +/- 0.137 | 1.594x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8764.069 | 29823.010 | 0.294x | 558.484 +/- 3.868 | 544.463 +/- 4.831 | 1.026x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 287.129 | 22013.843 | 0.013x | 22.221 +/- 0.062 | 11.855 +/- 0.048 | 1.874x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 856.555 | 23397.028 | 0.037x | 29.713 +/- 0.124 | 17.165 +/- 0.121 | 1.731x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 275.887 | 21395.213 | 0.013x | 27.410 +/- 0.117 | 14.080 +/- 0.070 | 1.947x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 248.781 | 21381.412 | 0.012x | 24.158 +/- 0.108 | 13.684 +/- 0.084 | 1.765x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 346.583 | 21637.194 | 0.016x | 26.002 +/- 0.159 | 13.763 +/- 0.150 | 1.889x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 851.272 | 23024.511 | 0.037x | 32.371 +/- 0.173 | 18.440 +/- 0.071 | 1.755x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8416.224 | 29960.618 | 0.281x | 539.615 +/- 2.861 | 522.427 +/- 2.915 | 1.033x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 306.223 | 21755.567 | 0.014x | 26.610 +/- 0.092 | 13.064 +/- 0.052 | 2.037x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 872.328 | 23892.708 | 0.037x | 34.053 +/- 0.199 | 19.116 +/- 0.198 | 1.781x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 351.986 | 22199.618 | 0.016x | 39.579 +/- 0.207 | 17.236 +/- 0.170 | 2.296x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 356.149 | 22268.057 | 0.016x | 39.609 +/- 0.221 | 17.599 +/- 0.180 | 2.251x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 373.895 | 25082.605 | 0.015x | 43.139 +/- 0.269 | 17.425 +/- 0.224 | 2.476x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 950.837 | 23105.959 | 0.041x | 51.172 +/- 0.327 | 22.471 +/- 0.147 | 2.277x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8773.369 | 32529.254 | 0.270x | 563.793 +/- 2.870 | 537.999 +/- 5.115 | 1.048x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 392.574 | 23159.530 | 0.017x | 44.078 +/- 0.246 | 16.204 +/- 0.112 | 2.720x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1008.238 | 24565.609 | 0.041x | 55.527 +/- 0.400 | 23.517 +/- 0.138 | 2.361x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 435.933 | 23133.630 | 0.019x | 45.681 +/- 0.160 | 16.570 +/- 0.135 | 2.757x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 397.997 | 23033.163 | 0.017x | 40.615 +/- 0.103 | 17.890 +/- 0.224 | 2.270x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 437.336 | 23346.266 | 0.019x | 44.612 +/- 0.126 | 17.773 +/- 0.303 | 2.510x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1032.255 | 23537.906 | 0.044x | 53.277 +/- 0.285 | 22.908 +/- 0.116 | 2.326x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8948.854 | 32615.078 | 0.274x | 590.681 +/- 3.793 | 553.795 +/- 3.053 | 1.067x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 461.202 | 23194.151 | 0.020x | 45.822 +/- 0.146 | 16.304 +/- 0.097 | 2.810x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1097.684 | 24895.728 | 0.044x | 60.031 +/- 0.220 | 23.975 +/- 0.218 | 2.504x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 245.626 | 22325.869 | 0.011x | 21.676 +/- 0.104 | 13.653 +/- 0.162 | 1.588x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 209.131 | 21506.185 | 0.010x | 19.395 +/- 0.050 | 13.362 +/- 0.102 | 1.452x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 241.005 | 22332.710 | 0.011x | 20.954 +/- 0.097 | 13.382 +/- 0.159 | 1.566x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 580.297 | 22419.501 | 0.026x | 25.531 +/- 0.609 | 16.531 +/- 0.188 | 1.544x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5370.136 | 26881.389 | 0.200x | 332.349 +/- 2.083 | 326.066 +/- 3.735 | 1.019x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 242.291 | 21535.720 | 0.011x | 21.509 +/- 0.109 | 13.157 +/- 0.084 | 1.635x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 717.339 | 22829.200 | 0.031x | 26.600 +/- 0.630 | 16.488 +/- 0.123 | 1.613x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 225.550 | 22034.334 | 0.010x | 20.184 +/- 0.484 | 11.125 +/- 0.088 | 1.814x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 208.410 | 21126.307 | 0.010x | 18.515 +/- 0.092 | 10.957 +/- 0.053 | 1.690x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 232.983 | 21590.403 | 0.011x | 19.718 +/- 0.157 | 11.117 +/- 0.075 | 1.774x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 821.712 | 26421.398 | 0.031x | 24.914 +/- 0.144 | 15.941 +/- 0.194 | 1.563x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8432.038 | 30734.311 | 0.274x | 538.944 +/- 4.115 | 528.354 +/- 2.615 | 1.020x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 252.452 | 21392.168 | 0.012x | 20.505 +/- 0.163 | 11.093 +/- 0.060 | 1.848x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 818.321 | 26693.364 | 0.031x | 25.344 +/- 0.351 | 15.924 +/- 0.218 | 1.592x | `4ba5419e2e3f2393` |
| `cpu_float32_float_identity_view` | `case_default` | 1 | 256 | shape (3,), stride (4,), offset 1, torch.float32, cpu, requires_grad=True | 215.951 | 22731.461 | 0.010x | 19.885 +/- 0.317 | 10.951 +/- 0.129 | 1.816x | `58df67cd172620c1` |
| `cpu_float32_float_identity_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 200.167 | 21154.979 | 0.009x | 18.386 +/- 0.074 | 10.739 +/- 0.047 | 1.712x | `e75a1d3233117514` |
| `cpu_float32_float_identity_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 220.544 | 21022.128 | 0.010x | 19.561 +/- 0.087 | 10.852 +/- 0.050 | 1.803x | `4c3dc265c5b9d697` |
| `cpu_float32_float_identity_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 828.868 | 21982.800 | 0.038x | 25.123 +/- 0.306 | 15.330 +/- 0.127 | 1.639x | `5ccc89fb94f689e5` |
| `cpu_float32_float_identity_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8905.734 | 29152.981 | 0.305x | 542.843 +/- 5.562 | 528.056 +/- 4.172 | 1.028x | `91fa5699b26ca1b8` |
| `cpu_float32_float_identity_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 260.449 | 21867.732 | 0.012x | 20.430 +/- 0.171 | 10.857 +/- 0.054 | 1.882x | `e99a6c9902c3119e` |
| `cpu_float32_float_identity_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 804.285 | 22441.353 | 0.036x | 24.888 +/- 0.170 | 15.427 +/- 0.117 | 1.613x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 401.542 | 23290.852 | 0.017x | 40.909 +/- 0.241 | 18.235 +/- 0.080 | 2.243x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 442.815 | 22706.654 | 0.020x | 34.761 +/- 0.104 | 16.502 +/- 0.252 | 2.106x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 383.539 | 23436.196 | 0.016x | 38.139 +/- 0.107 | 16.348 +/- 0.248 | 2.333x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 763.769 | 23401.990 | 0.033x | 44.044 +/- 0.172 | 19.938 +/- 0.269 | 2.209x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5536.326 | 27797.492 | 0.199x | 363.280 +/- 2.468 | 336.821 +/- 2.360 | 1.079x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 402.053 | 22377.862 | 0.018x | 38.850 +/- 0.082 | 15.338 +/- 0.068 | 2.533x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 786.804 | 25006.179 | 0.031x | 48.747 +/- 0.235 | 20.480 +/- 0.181 | 2.380x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 323.463 | 22846.540 | 0.014x | 30.101 +/- 0.119 | 15.889 +/- 0.171 | 1.894x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 352.262 | 21973.126 | 0.016x | 30.307 +/- 0.163 | 15.503 +/- 0.153 | 1.955x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 347.740 | 23364.849 | 0.015x | 32.714 +/- 0.106 | 15.483 +/- 0.207 | 2.113x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 937.076 | 23168.442 | 0.040x | 39.941 +/- 0.141 | 20.777 +/- 0.147 | 1.922x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9264.341 | 31077.745 | 0.298x | 579.362 +/- 1.747 | 558.376 +/- 2.147 | 1.038x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 365.903 | 22572.370 | 0.016x | 33.467 +/- 0.091 | 14.610 +/- 0.060 | 2.291x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 969.666 | 23586.548 | 0.041x | 44.207 +/- 0.218 | 21.604 +/- 0.277 | 2.046x | `68585b64809ef02a` |
| `cpu_float32_custom_function_unary` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 452.358 | 39224.806 | 0.012x | 53.286 +/- 0.159 | 18.900 +/- 0.291 | 2.819x | `d16fd2f4dd199523` |
| `cpu_float32_custom_function_unary` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 415.784 | 23918.325 | 0.017x | 46.635 +/- 0.117 | 18.580 +/- 0.164 | 2.510x | `5c2ffe407931c8ee` |
| `cpu_float32_custom_function_unary` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 428.357 | 23801.263 | 0.018x | 50.701 +/- 0.132 | 18.417 +/- 0.201 | 2.753x | `d85643b7b66a7ca9` |
| `cpu_float32_custom_function_unary` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1046.617 | 24592.999 | 0.043x | 59.469 +/- 0.278 | 23.943 +/- 0.217 | 2.484x | `414eafab6fd10fb4` |
| `cpu_float32_custom_function_unary` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8953.951 | 32453.744 | 0.276x | 590.232 +/- 3.211 | 550.801 +/- 2.518 | 1.072x | `7863bb8d1d98f49b` |
| `cpu_float32_custom_function_unary` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 483.385 | 26634.429 | 0.018x | 52.053 +/- 0.263 | 17.461 +/- 0.600 | 2.981x | `e99a6c9902c3119e` |
| `cpu_float32_custom_function_unary` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 1069.461 | 26404.257 | 0.041x | 65.364 +/- 0.678 | 24.744 +/- 0.172 | 2.642x | `188c6817fce2e1e1` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 366.163 | 23881.885 | 0.015x | 45.656 +/- 0.201 | 16.984 +/- 0.148 | 2.688x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 982.290 | 24333.238 | 0.040x | 51.188 +/- 0.293 | 22.423 +/- 0.338 | 2.283x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8995.975 | 32438.926 | 0.277x | 594.860 +/- 7.122 | 557.905 +/- 2.743 | 1.066x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1044.363 | 25219.846 | 0.041x | 50.311 +/- 0.277 | 25.023 +/- 0.673 | 2.011x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 975.820 | 27259.640 | 0.036x | 49.649 +/- 0.422 | 25.020 +/- 0.262 | 1.984x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 361.947 | 26007.763 | 0.014x | 45.289 +/- 0.139 | 15.636 +/- 0.094 | 2.897x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1003.757 | 26229.363 | 0.038x | 73.744 +/- 0.339 | 23.035 +/- 0.210 | 3.201x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 303.273 | 23831.538 | 0.013x | 33.887 +/- 0.094 | 15.433 +/- 0.243 | 2.196x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 889.093 | 23581.395 | 0.038x | 38.761 +/- 0.205 | 20.005 +/- 0.180 | 1.938x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8842.263 | 32576.073 | 0.271x | 577.157 +/- 4.638 | 552.643 +/- 4.546 | 1.044x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 945.223 | 23577.971 | 0.040x | 37.812 +/- 0.313 | 19.487 +/- 0.157 | 1.940x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 883.676 | 25754.149 | 0.034x | 39.047 +/- 0.213 | 19.317 +/- 0.132 | 2.021x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 291.756 | 23382.922 | 0.012x | 33.651 +/- 0.122 | 13.680 +/- 0.042 | 2.460x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1008.374 | 25709.172 | 0.039x | 59.326 +/- 0.341 | 19.960 +/- 0.129 | 2.972x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 303.434 | 22407.082 | 0.014x | 33.893 +/- 0.198 | 15.595 +/- 0.118 | 2.173x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 933.305 | 23281.592 | 0.040x | 39.673 +/- 0.373 | 19.613 +/- 0.120 | 2.023x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9091.940 | 31270.050 | 0.291x | 588.281 +/- 3.999 | 560.125 +/- 4.182 | 1.050x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 940.000 | 23538.912 | 0.040x | 38.933 +/- 0.281 | 19.506 +/- 0.116 | 1.996x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 899.754 | 22788.969 | 0.039x | 39.769 +/- 0.429 | 19.453 +/- 0.110 | 2.044x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 282.707 | 22560.052 | 0.013x | 34.565 +/- 0.325 | 13.752 +/- 0.073 | 2.513x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 923.666 | 23760.151 | 0.039x | 60.747 +/- 0.390 | 20.421 +/- 0.131 | 2.975x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 303.559 | 22553.282 | 0.013x | 34.154 +/- 0.189 | 14.886 +/- 0.138 | 2.294x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 873.340 | 24346.753 | 0.036x | 38.700 +/- 0.266 | 18.818 +/- 0.107 | 2.057x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8461.947 | 31200.465 | 0.271x | 552.069 +/- 3.755 | 532.000 +/- 4.080 | 1.038x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 995.384 | 24149.239 | 0.041x | 37.877 +/- 0.274 | 19.186 +/- 0.266 | 1.974x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 891.167 | 23589.243 | 0.038x | 38.879 +/- 0.220 | 19.175 +/- 0.171 | 2.028x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 293.749 | 22980.297 | 0.013x | 33.491 +/- 0.118 | 13.433 +/- 0.107 | 2.493x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 939.439 | 25228.100 | 0.037x | 58.994 +/- 0.328 | 19.255 +/- 0.132 | 3.064x | `2bd384aefcaaa397` |
| `cpu_float32_tuple_list_output_pytree` | `case_default` | 2 | 256 | tuple[shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True, list[shape (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True]] | 498.258 | 24533.169 | 0.020x | 56.554 +/- 0.337 | 19.498 +/- 0.157 | 2.901x | `a62dacb062c1ed92` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1484.484 | 25671.273 | 0.058x | 64.969 +/- 0.331 | 25.325 +/- 0.137 | 2.565x | `3bce94d7e523bafe` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_127x131_by_131` | 2 | 16 | tuple[shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (131,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 14193.468 | 38250.053 | 0.371x | 916.108 +/- 5.310 | 874.281 +/- 6.307 | 1.048x | `022557af0d301f5e` |
| `cpu_float32_tuple_list_output_pytree` | `tensor_scalar_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1534.900 | 24772.681 | 0.062x | 63.211 +/- 0.372 | 25.082 +/- 0.208 | 2.520x | `f4ff04ee55c4e2cd` |
| `cpu_float32_tuple_list_output_pytree` | `scalar_tensor_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False]] | 1660.730 | 27125.036 | 0.061x | 64.228 +/- 0.299 | 26.797 +/- 0.181 | 2.397x | `f1950b665bfdc9f1` |
| `cpu_float32_tuple_list_output_pytree` | `empty_2x0_by_0` | 2 | 2048 | tuple[shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 486.315 | 23863.883 | 0.020x | 55.400 +/- 0.212 | 15.729 +/- 0.103 | 3.522x | `e89cfed7478c41fa` |
| `cpu_float32_tuple_list_output_pytree` | `transpose_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False]] | 1545.060 | 27690.255 | 0.056x | 88.723 +/- 0.726 | 25.964 +/- 0.290 | 3.417x | `776bd23d05673f66` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 345.317 | 22570.358 | 0.015x | 34.869 +/- 0.292 | 15.603 +/- 0.120 | 2.235x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 317.925 | 22226.228 | 0.014x | 30.035 +/- 0.107 | 15.584 +/- 0.372 | 1.927x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 338.747 | 23284.047 | 0.015x | 32.735 +/- 0.187 | 15.255 +/- 0.165 | 2.146x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 710.384 | 23480.313 | 0.030x | 38.075 +/- 0.277 | 18.792 +/- 0.162 | 2.026x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5475.870 | 29224.008 | 0.187x | 358.727 +/- 3.115 | 333.347 +/- 2.080 | 1.076x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 360.429 | 21958.558 | 0.016x | 33.713 +/- 0.138 | 14.446 +/- 0.056 | 2.334x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 772.622 | 24727.057 | 0.031x | 42.324 +/- 0.325 | 19.223 +/- 0.215 | 2.202x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 346.544 | 22986.417 | 0.015x | 40.400 +/- 0.260 | 16.141 +/- 0.173 | 2.503x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 942.109 | 25081.738 | 0.038x | 45.705 +/- 0.363 | 20.861 +/- 0.138 | 2.191x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8978.823 | 32266.141 | 0.278x | 585.710 +/- 2.872 | 554.705 +/- 3.484 | 1.056x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1024.388 | 27451.840 | 0.037x | 44.503 +/- 0.218 | 27.263 +/- 0.136 | 1.632x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 938.353 | 27254.327 | 0.034x | 45.668 +/- 0.365 | 27.120 +/- 0.148 | 1.684x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 342.908 | 26626.783 | 0.013x | 43.205 +/- 3.696 | 14.713 +/- 0.066 | 2.936x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1136.833 | 25320.253 | 0.045x | 79.227 +/- 5.302 | 21.445 +/- 0.119 | 3.694x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 329.954 | 22368.844 | 0.015x | 34.796 +/- 0.174 | 15.504 +/- 0.124 | 2.244x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 346.043 | 21765.782 | 0.016x | 30.196 +/- 0.340 | 15.415 +/- 0.370 | 1.959x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 333.534 | 22924.808 | 0.015x | 38.825 +/- 0.756 | 15.171 +/- 0.125 | 2.559x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 796.017 | 22436.281 | 0.035x | 44.968 +/- 0.250 | 18.568 +/- 0.110 | 2.422x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5455.509 | 27129.689 | 0.201x | 353.311 +/- 1.465 | 338.731 +/- 2.632 | 1.043x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 415.108 | 22177.344 | 0.019x | 36.569 +/- 3.050 | 14.280 +/- 0.051 | 2.561x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 721.656 | 24026.028 | 0.030x | 49.406 +/- 0.350 | 19.170 +/- 0.102 | 2.577x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 994.363 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 852.318 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 639.036 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 161746.732 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 126709.559 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 77529.634 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 114447.921 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 105848.124 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 71863.770 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1086.021 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 974.152 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 701.502 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, inference, training autograd, mutation_aliasing_views, containers and pytrees, decompositions, custom functions, recompilation guards, and dtype/device transitions. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `containers_pytrees` | 6 | Supported and timed public cases: `cpu_float32_tuple_list_output_pytree` |
| `decompositions` | 6 | Supported and timed public cases: `cpu_float32_decomposition_square_scalar` |
| `custom_functions` | 6 | Supported and timed public cases: `cpu_float32_custom_function_unary` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `dtype_device_transitions` | 4 | Supported and timed public cases: `cpu_float32_float_identity_view` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `python_control_flow` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 68 / 100. Zero-credit unsupported category weight: 32 / 100.
The torch_compile_corpus_v9 corpus also keeps 2 held-out broadcasting programs, 1 held-out containers-pytrees program, 1 held-out custom-function program, 1 held-out decomposition program, 1 held-out dtype/device-transition program, 1 held-out inference program, 1 held-out mutation_aliasing_views program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 2 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
