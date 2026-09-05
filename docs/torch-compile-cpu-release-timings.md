# `torch.compile` Eager CPU Release Timings

Date: 2026-09-04

Candidate provenance: source snapshot refreshed against `b97db2fd`, plus the worktree changes that add compiled tuple/list output pytrees while preserving main's `Tensor.square()` decomposition corpus updates. The raw benchmark artifact is refreshed for `torch_compile_corpus_v9`.

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
  tests.test_torch_compile_coverage_evaluator
bash scripts/evaluate_torch_compile_coverage.sh
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m py_compile \
  python/torch_rs/__init__.py python/torch_rs/_compile_bytecode.py \
  python/torch_rs/_compile_trace.py scripts/evaluate_torch_compile_coverage.py \
  scripts/benchmark_compile_cpu.py tests/test_compile_benchmark_artifact.py \
  tests/test_compile_corpus.py tests/test_torch_compile_coverage_evaluator.py
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
  tests.test_torch_compile_coverage_evaluator
bash scripts/evaluate_torch_compile_coverage.sh
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m py_compile \
  python/torch_rs/__init__.py python/torch_rs/_compile_bytecode.py \
  python/torch_rs/_compile_trace.py scripts/evaluate_torch_compile_coverage.py \
  scripts/benchmark_compile_cpu.py tests/test_compile_benchmark_artifact.py \
  tests/test_compile_corpus.py tests/test_torch_compile_coverage_evaluator.py
cargo fmt --check
git diff --check
```

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.14.7+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`, `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_9629eb76/.venv/lib/python3.14/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_9629eb76/python/torch_rs/__init__.py`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`; `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both reported 1
- Dependency installation: locked `uv sync` used the worktree-local uv cache
- Build: release editable wheel installed in the worktree-local `.venv`

The benchmark uses the checked-in `torch_compile_corpus_v9` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input training-autograd program, one one-input square decomposition program, four two-input broadcasting programs, one two-input containers-pytrees program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Detach cells return shared-storage aliases with `requires_grad=False`. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Tuple/list output cells preserve container structure and record per-tensor metadata for each output leaf. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, tuple/list container structure, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 119 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v9`
- Cold first compiled call: 0.024x uncapped, 0.111x capped
- Steady-state materialized compiled call: 1.683x uncapped, 1.683x capped
- Timed supported cells: 119 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 containers-pytrees, 7 decomposition, 21 recompilation-guard, 7 mutation_aliasing_views)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 58.0% supported by native compile cases, 42% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 367.406 | 90970.381 | 0.004x | 21.484 +/- 0.103 | 13.520 +/- 0.298 | 1.589x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 286.519 | 24609.622 | 0.012x | 18.897 +/- 0.056 | 13.101 +/- 0.173 | 1.442x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 247.209 | 30604.940 | 0.008x | 20.700 +/- 0.089 | 13.507 +/- 0.552 | 1.533x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 869.650 | 31403.548 | 0.028x | 25.683 +/- 0.161 | 18.305 +/- 1.135 | 1.403x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7032.453 | 36175.762 | 0.194x | 443.899 +/- 3.286 | 460.504 +/- 9.449 | 0.964x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 295.548 | 29142.608 | 0.010x | 20.976 +/- 0.077 | 13.872 +/- 0.455 | 1.512x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 776.755 | 33203.931 | 0.023x | 26.978 +/- 0.133 | 20.405 +/- 0.609 | 1.322x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 219.267 | 29395.255 | 0.007x | 17.222 +/- 0.096 | 11.713 +/- 0.221 | 1.470x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 192.140 | 28825.168 | 0.007x | 15.144 +/- 0.061 | 12.906 +/- 0.156 | 1.173x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 210.964 | 29340.803 | 0.007x | 16.603 +/- 0.068 | 11.341 +/- 0.117 | 1.464x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 718.121 | 24882.660 | 0.029x | 21.544 +/- 0.206 | 15.526 +/- 0.095 | 1.388x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7154.497 | 29823.844 | 0.240x | 456.804 +/- 4.105 | 450.985 +/- 2.455 | 1.013x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 245.922 | 23167.660 | 0.011x | 16.937 +/- 0.051 | 10.933 +/- 0.039 | 1.549x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 721.597 | 23728.083 | 0.030x | 23.421 +/- 0.164 | 15.634 +/- 0.180 | 1.498x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 235.726 | 22923.345 | 0.010x | 21.410 +/- 0.140 | 13.327 +/- 0.163 | 1.607x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 233.694 | 22435.237 | 0.010x | 18.961 +/- 0.046 | 12.985 +/- 0.114 | 1.460x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 243.133 | 24881.077 | 0.010x | 20.598 +/- 0.081 | 13.056 +/- 0.174 | 1.578x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 748.753 | 25796.486 | 0.029x | 25.870 +/- 0.174 | 16.954 +/- 0.139 | 1.526x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6979.073 | 30100.357 | 0.232x | 445.123 +/- 1.867 | 440.048 +/- 2.291 | 1.012x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 284.350 | 23181.005 | 0.012x | 20.834 +/- 0.063 | 12.222 +/- 0.139 | 1.705x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 741.902 | 26925.104 | 0.028x | 26.886 +/- 0.180 | 17.546 +/- 0.187 | 1.532x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 292.683 | 25517.299 | 0.011x | 31.866 +/- 0.161 | 16.500 +/- 0.198 | 1.931x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 282.372 | 25732.230 | 0.011x | 31.775 +/- 0.066 | 16.690 +/- 0.115 | 1.904x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 329.483 | 26169.651 | 0.013x | 35.077 +/- 0.112 | 16.728 +/- 0.308 | 2.097x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 797.701 | 24633.618 | 0.032x | 41.723 +/- 0.300 | 20.736 +/- 0.179 | 2.012x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7372.137 | 30971.179 | 0.238x | 466.588 +/- 3.486 | 450.811 +/- 5.091 | 1.035x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 375.322 | 27275.158 | 0.014x | 35.802 +/- 0.294 | 15.482 +/- 0.164 | 2.313x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 815.995 | 30794.336 | 0.026x | 44.894 +/- 0.239 | 21.812 +/- 0.173 | 2.058x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 346.579 | 26660.299 | 0.013x | 37.123 +/- 0.331 | 15.521 +/- 0.077 | 2.392x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 335.703 | 28789.589 | 0.012x | 33.105 +/- 0.203 | 17.047 +/- 0.242 | 1.942x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 365.934 | 28124.208 | 0.013x | 36.679 +/- 0.272 | 17.019 +/- 0.291 | 2.155x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 898.204 | 26463.471 | 0.034x | 44.134 +/- 0.431 | 21.468 +/- 0.132 | 2.056x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7272.896 | 33924.742 | 0.214x | 480.794 +/- 3.955 | 462.201 +/- 4.229 | 1.040x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 399.073 | 28207.980 | 0.014x | 37.442 +/- 0.305 | 15.508 +/- 0.077 | 2.414x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 884.152 | 30367.742 | 0.029x | 49.726 +/- 0.491 | 22.783 +/- 0.624 | 2.183x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 233.633 | 24383.815 | 0.010x | 17.166 +/- 0.500 | 12.330 +/- 0.139 | 1.392x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 219.918 | 27356.877 | 0.008x | 14.994 +/- 0.081 | 12.151 +/- 0.093 | 1.234x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 216.762 | 27509.768 | 0.008x | 16.147 +/- 0.070 | 12.104 +/- 0.136 | 1.334x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 516.327 | 28344.630 | 0.018x | 19.429 +/- 0.217 | 14.580 +/- 0.089 | 1.333x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4377.797 | 31351.909 | 0.140x | 274.510 +/- 1.801 | 271.597 +/- 2.296 | 1.011x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 243.788 | 26292.180 | 0.009x | 16.519 +/- 0.061 | 11.747 +/- 0.047 | 1.406x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 528.394 | 24125.508 | 0.022x | 20.219 +/- 0.223 | 14.707 +/- 0.128 | 1.375x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 204.969 | 24587.858 | 0.008x | 15.018 +/- 0.095 | 10.016 +/- 0.063 | 1.499x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 202.020 | 21691.377 | 0.009x | 14.013 +/- 0.072 | 9.848 +/- 0.085 | 1.423x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 210.027 | 22760.229 | 0.009x | 15.005 +/- 0.058 | 10.009 +/- 0.064 | 1.499x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 710.825 | 23768.138 | 0.030x | 19.335 +/- 0.224 | 13.955 +/- 0.161 | 1.386x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6973.974 | 30241.505 | 0.231x | 440.270 +/- 2.679 | 437.093 +/- 3.282 | 1.007x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 292.237 | 23401.474 | 0.012x | 15.505 +/- 0.061 | 10.007 +/- 0.081 | 1.549x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 705.562 | 24686.091 | 0.029x | 19.319 +/- 0.128 | 13.822 +/- 0.098 | 1.398x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 341.272 | 24572.711 | 0.014x | 33.047 +/- 0.173 | 17.951 +/- 0.201 | 1.841x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 376.690 | 23633.831 | 0.016x | 28.093 +/- 0.200 | 15.750 +/- 0.156 | 1.784x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 329.673 | 27517.435 | 0.012x | 30.688 +/- 0.103 | 15.802 +/- 0.262 | 1.942x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 649.232 | 25488.997 | 0.025x | 35.705 +/- 0.158 | 18.702 +/- 0.174 | 1.909x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4564.725 | 31032.857 | 0.147x | 301.216 +/- 2.049 | 287.675 +/- 2.627 | 1.047x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 352.123 | 27215.333 | 0.013x | 31.374 +/- 0.129 | 14.550 +/- 0.078 | 2.156x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 680.459 | 27335.930 | 0.025x | 39.749 +/- 0.228 | 19.666 +/- 0.218 | 2.021x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 287.611 | 24401.041 | 0.012x | 23.821 +/- 0.126 | 15.084 +/- 0.155 | 1.579x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 283.188 | 22784.421 | 0.012x | 23.759 +/- 0.106 | 15.234 +/- 0.189 | 1.560x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 299.513 | 24903.476 | 0.012x | 33.570 +/- 0.449 | 15.069 +/- 0.315 | 2.228x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 969.266 | 24781.262 | 0.039x | 41.469 +/- 0.268 | 19.613 +/- 0.107 | 2.114x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9133.151 | 30968.876 | 0.295x | 471.081 +/- 3.141 | 462.206 +/- 3.935 | 1.019x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 384.957 | 25243.671 | 0.015x | 26.460 +/- 0.072 | 14.031 +/- 0.084 | 1.886x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 821.748 | 27636.935 | 0.030x | 35.521 +/- 0.171 | 20.437 +/- 0.170 | 1.738x | `68585b64809ef02a` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 324.896 | 32076.762 | 0.010x | 36.812 +/- 0.205 | 16.428 +/- 0.180 | 2.241x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 839.154 | 27966.805 | 0.030x | 41.987 +/- 0.533 | 20.631 +/- 0.152 | 2.035x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7254.168 | 33221.959 | 0.218x | 484.847 +/- 4.218 | 460.772 +/- 2.862 | 1.052x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 883.726 | 27547.525 | 0.032x | 40.714 +/- 0.400 | 20.452 +/- 0.167 | 1.991x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 807.707 | 31256.932 | 0.026x | 39.927 +/- 0.242 | 20.113 +/- 0.115 | 1.985x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 306.615 | 29477.669 | 0.010x | 36.365 +/- 0.314 | 14.867 +/- 0.053 | 2.446x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 856.336 | 29040.404 | 0.029x | 61.413 +/- 0.460 | 22.107 +/- 0.409 | 2.778x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 289.513 | 29459.673 | 0.010x | 26.956 +/- 0.204 | 14.646 +/- 0.104 | 1.840x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 750.195 | 25831.800 | 0.029x | 30.878 +/- 0.231 | 18.026 +/- 0.114 | 1.713x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7266.618 | 35319.317 | 0.206x | 465.992 +/- 2.508 | 455.206 +/- 2.512 | 1.024x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 812.174 | 26179.952 | 0.031x | 29.771 +/- 0.193 | 17.857 +/- 0.156 | 1.667x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 751.763 | 27791.084 | 0.027x | 31.076 +/- 0.327 | 17.846 +/- 0.198 | 1.741x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 261.130 | 24617.077 | 0.011x | 26.524 +/- 0.137 | 12.916 +/- 0.055 | 2.054x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 790.966 | 26251.735 | 0.030x | 48.773 +/- 0.324 | 18.491 +/- 0.137 | 2.638x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 268.236 | 24134.568 | 0.011x | 26.755 +/- 0.120 | 14.929 +/- 0.104 | 1.792x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 740.895 | 26393.169 | 0.028x | 31.304 +/- 0.170 | 17.993 +/- 0.122 | 1.740x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7323.403 | 32062.935 | 0.228x | 471.317 +/- 2.400 | 459.019 +/- 2.688 | 1.027x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 817.938 | 27444.179 | 0.030x | 30.742 +/- 0.211 | 18.323 +/- 0.278 | 1.678x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 737.826 | 25339.726 | 0.029x | 30.962 +/- 0.150 | 18.045 +/- 0.223 | 1.716x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 249.652 | 24432.894 | 0.010x | 26.601 +/- 0.102 | 13.081 +/- 0.077 | 2.034x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 787.371 | 25861.891 | 0.030x | 49.557 +/- 0.282 | 19.361 +/- 0.298 | 2.560x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 286.143 | 27595.458 | 0.010x | 26.974 +/- 0.129 | 14.292 +/- 0.197 | 1.887x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 756.179 | 25785.860 | 0.029x | 30.928 +/- 0.241 | 17.166 +/- 0.155 | 1.802x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6965.381 | 30828.648 | 0.226x | 455.314 +/- 2.940 | 446.324 +/- 4.376 | 1.020x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 822.569 | 24850.255 | 0.033x | 29.806 +/- 0.183 | 17.285 +/- 0.219 | 1.724x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 746.038 | 24856.124 | 0.030x | 30.941 +/- 0.224 | 17.301 +/- 0.190 | 1.788x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 274.776 | 23966.283 | 0.011x | 26.608 +/- 0.155 | 12.501 +/- 0.193 | 2.128x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 812.174 | 27431.946 | 0.030x | 49.717 +/- 1.534 | 17.884 +/- 0.284 | 2.780x | `2bd384aefcaaa397` |
| `cpu_float32_tuple_list_output_pytree` | `case_default` | 2 | 256 | tuple[shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True, list[shape (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True]] | 451.798 | 29519.503 | 0.015x | 45.619 +/- 0.225 | 19.393 +/- 0.254 | 2.352x | `a62dacb062c1ed92` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1223.626 | 30152.391 | 0.041x | 52.621 +/- 0.236 | 23.683 +/- 0.194 | 2.222x | `3bce94d7e523bafe` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_127x131_by_131` | 2 | 16 | tuple[shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (131,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 11991.414 | 40438.260 | 0.297x | 769.242 +/- 9.872 | 751.273 +/- 7.191 | 1.024x | `022557af0d301f5e` |
| `cpu_float32_tuple_list_output_pytree` | `tensor_scalar_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1290.513 | 27740.983 | 0.047x | 51.679 +/- 0.286 | 23.341 +/- 0.249 | 2.214x | `f4ff04ee55c4e2cd` |
| `cpu_float32_tuple_list_output_pytree` | `scalar_tensor_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False]] | 1384.544 | 26848.463 | 0.052x | 52.506 +/- 0.371 | 24.824 +/- 0.230 | 2.115x | `f1950b665bfdc9f1` |
| `cpu_float32_tuple_list_output_pytree` | `empty_2x0_by_0` | 2 | 2048 | tuple[shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 447.792 | 26382.703 | 0.017x | 48.474 +/- 1.623 | 14.992 +/- 0.108 | 3.233x | `e89cfed7478c41fa` |
| `cpu_float32_tuple_list_output_pytree` | `transpose_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False]] | 1459.137 | 31772.582 | 0.046x | 80.451 +/- 0.684 | 24.342 +/- 0.317 | 3.305x | `776bd23d05673f66` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 286.142 | 26981.339 | 0.011x | 30.264 +/- 0.422 | 14.914 +/- 0.161 | 2.029x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 295.227 | 24308.686 | 0.012x | 24.019 +/- 0.532 | 14.625 +/- 0.115 | 1.642x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 310.315 | 25789.481 | 0.012x | 26.111 +/- 0.200 | 14.757 +/- 0.262 | 1.769x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 608.776 | 26488.143 | 0.023x | 30.840 +/- 0.332 | 17.656 +/- 0.238 | 1.747x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4598.697 | 29867.629 | 0.154x | 308.316 +/- 8.452 | 292.661 +/- 2.049 | 1.053x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 390.771 | 24867.471 | 0.016x | 26.730 +/- 0.283 | 13.810 +/- 0.089 | 1.936x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 619.292 | 25824.980 | 0.024x | 33.755 +/- 0.254 | 18.805 +/- 0.289 | 1.795x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 296.869 | 25526.579 | 0.012x | 31.665 +/- 0.192 | 15.710 +/- 0.198 | 2.016x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 811.322 | 25927.410 | 0.031x | 36.486 +/- 0.239 | 19.662 +/- 0.191 | 1.856x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 7323.507 | 37212.350 | 0.197x | 474.458 +/- 1.947 | 463.700 +/- 4.768 | 1.023x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 854.869 | 26154.262 | 0.033x | 35.346 +/- 0.172 | 19.268 +/- 0.210 | 1.834x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 785.814 | 28696.814 | 0.027x | 36.191 +/- 0.194 | 19.159 +/- 0.147 | 1.889x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 295.552 | 25702.845 | 0.011x | 31.187 +/- 0.078 | 14.078 +/- 0.065 | 2.215x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 878.424 | 30245.752 | 0.029x | 54.662 +/- 0.316 | 20.301 +/- 0.174 | 2.693x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 297.700 | 23806.976 | 0.013x | 30.539 +/- 0.463 | 15.096 +/- 0.184 | 2.023x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 285.783 | 26940.853 | 0.011x | 24.159 +/- 0.408 | 14.784 +/- 0.096 | 1.634x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 306.087 | 26010.941 | 0.012x | 26.280 +/- 0.255 | 14.868 +/- 0.249 | 1.768x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 605.325 | 25832.802 | 0.023x | 31.231 +/- 0.353 | 17.319 +/- 0.108 | 1.803x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4482.967 | 30378.097 | 0.148x | 294.699 +/- 3.253 | 285.176 +/- 1.908 | 1.033x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 329.694 | 24641.159 | 0.013x | 27.816 +/- 1.231 | 13.677 +/- 0.065 | 2.034x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 645.642 | 25646.775 | 0.025x | 33.951 +/- 0.211 | 18.302 +/- 0.178 | 1.855x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 943.197 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 761.382 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 565.874 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 172352.691 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 147449.112 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 89373.417 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 131677.585 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 122145.518 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 80567.398 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1142.508 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 864.308 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 634.110 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, inference, training autograd, mutation_aliasing_views, containers and pytrees, decompositions, and recompilation guards. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `containers_pytrees` | 6 | Supported and timed public cases: `cpu_float32_tuple_list_output_pytree` |
| `decompositions` | 6 | Supported and timed public cases: `cpu_float32_decomposition_square_scalar` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `python_control_flow` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `custom_functions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dtype_device_transitions` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 58 / 100. Zero-credit unsupported category weight: 42 / 100.
The torch_compile_corpus_v9 corpus also keeps 2 held-out broadcasting programs, 1 held-out containers-pytrees program, 1 held-out decomposition program, 1 held-out inference program, 1 held-out mutation_aliasing_views program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 2 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
