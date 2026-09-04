# `torch.compile` Eager CPU Release Timings

Date: 2026-09-04

Candidate provenance: source snapshot refreshed against
`c418ac98e715f1b4833ca61fcebb50aa493444a9`, plus the worktree changes that
add compiled `Tensor.square()` decomposition graphlets and refresh the raw
benchmark artifact for corpus v8.

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
- Python: 3.14.5
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_72f4192f/.venv/lib/python3.14/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_72f4192f/.venv/lib/python3.14/site-packages/torch_rs/__init__.py`
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

The benchmark uses the checked-in `torch_compile_corpus_v8` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input training-autograd program, one one-input square decomposition program, four two-input broadcasting programs, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Detach cells return shared-storage aliases with `requires_grad=False`. Decomposition cells verify square-derived values and metadata across scalar, empty, and non-contiguous inputs. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 112 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v8`
- Cold first compiled call: 0.026x uncapped, 0.111x capped
- Steady-state materialized compiled call: 1.676x uncapped, 1.676x capped
- Timed supported cells: 112 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 decomposition, 21 recompilation-guard, 7 mutation_aliasing_views)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 52.0% supported by native compile cases, 48% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 386.960 | 90433.464 | 0.004x | 21.334 +/- 0.151 | 13.101 +/- 0.293 | 1.628x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 241.745 | 22413.428 | 0.011x | 18.685 +/- 0.146 | 12.909 +/- 0.121 | 1.447x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 259.572 | 22036.372 | 0.012x | 20.451 +/- 0.233 | 13.081 +/- 0.202 | 1.563x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 683.269 | 24174.817 | 0.028x | 25.153 +/- 0.236 | 16.809 +/- 0.428 | 1.496x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6659.694 | 31050.458 | 0.214x | 390.841 +/- 2.531 | 395.647 +/- 3.409 | 0.988x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 291.200 | 21703.760 | 0.013x | 20.748 +/- 0.183 | 12.590 +/- 0.408 | 1.648x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 732.027 | 25117.953 | 0.029x | 26.537 +/- 0.205 | 17.116 +/- 0.372 | 1.550x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 229.502 | 22656.360 | 0.010x | 16.892 +/- 0.088 | 11.598 +/- 0.319 | 1.456x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 200.007 | 22792.857 | 0.009x | 14.937 +/- 0.055 | 11.489 +/- 0.096 | 1.300x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 211.836 | 24959.594 | 0.008x | 16.338 +/- 0.113 | 11.312 +/- 0.134 | 1.444x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 659.267 | 22047.569 | 0.030x | 20.832 +/- 0.142 | 15.084 +/- 0.354 | 1.381x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6401.167 | 27967.654 | 0.229x | 403.846 +/- 3.282 | 408.502 +/- 4.846 | 0.989x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 273.624 | 21491.573 | 0.013x | 16.622 +/- 0.090 | 10.994 +/- 0.075 | 1.512x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 694.255 | 23185.821 | 0.030x | 22.836 +/- 0.159 | 14.777 +/- 0.129 | 1.545x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 265.296 | 20591.497 | 0.013x | 20.959 +/- 0.090 | 12.988 +/- 0.155 | 1.614x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 221.279 | 20802.101 | 0.011x | 18.440 +/- 0.108 | 12.517 +/- 0.094 | 1.473x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 269.973 | 21504.302 | 0.013x | 20.305 +/- 0.106 | 12.871 +/- 0.270 | 1.578x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 669.938 | 21377.666 | 0.031x | 25.016 +/- 0.207 | 16.384 +/- 0.182 | 1.527x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6360.752 | 26322.315 | 0.242x | 395.653 +/- 3.071 | 387.681 +/- 3.792 | 1.021x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 323.429 | 20087.484 | 0.016x | 20.756 +/- 0.118 | 12.040 +/- 0.110 | 1.724x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 684.325 | 24107.810 | 0.028x | 26.260 +/- 0.186 | 17.068 +/- 0.135 | 1.539x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 320.880 | 21340.755 | 0.015x | 31.167 +/- 0.119 | 16.313 +/- 0.187 | 1.911x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 285.101 | 21660.709 | 0.013x | 31.261 +/- 0.113 | 16.328 +/- 0.184 | 1.915x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 308.146 | 21829.715 | 0.014x | 34.600 +/- 0.153 | 16.300 +/- 0.311 | 2.123x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 744.552 | 22424.269 | 0.033x | 40.853 +/- 0.346 | 19.933 +/- 0.145 | 2.050x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6325.183 | 32111.803 | 0.197x | 412.620 +/- 2.577 | 398.017 +/- 4.262 | 1.037x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 342.488 | 26514.457 | 0.013x | 35.372 +/- 0.099 | 15.037 +/- 0.111 | 2.352x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 788.928 | 24709.676 | 0.032x | 44.437 +/- 0.330 | 21.099 +/- 0.226 | 2.106x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 359.714 | 22083.093 | 0.016x | 37.230 +/- 1.040 | 15.720 +/- 0.346 | 2.368x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 378.633 | 22761.419 | 0.017x | 32.800 +/- 0.251 | 16.671 +/- 0.168 | 1.968x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 368.006 | 22579.238 | 0.016x | 35.919 +/- 0.128 | 16.623 +/- 0.289 | 2.161x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 791.011 | 23279.618 | 0.034x | 42.465 +/- 0.175 | 20.855 +/- 0.159 | 2.036x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6531.525 | 28617.326 | 0.228x | 426.007 +/- 2.603 | 407.836 +/- 2.636 | 1.045x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 413.591 | 22692.971 | 0.018x | 36.386 +/- 0.132 | 15.275 +/- 0.104 | 2.382x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 845.153 | 26930.525 | 0.031x | 48.991 +/- 0.332 | 21.500 +/- 0.161 | 2.279x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 237.579 | 21219.798 | 0.011x | 16.400 +/- 0.120 | 12.247 +/- 0.094 | 1.339x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 205.631 | 20750.778 | 0.010x | 14.459 +/- 0.040 | 12.138 +/- 0.130 | 1.191x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 227.529 | 20516.588 | 0.011x | 15.739 +/- 0.060 | 12.030 +/- 0.114 | 1.308x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 495.600 | 21314.806 | 0.023x | 18.886 +/- 0.183 | 14.390 +/- 0.089 | 1.312x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4034.887 | 24143.970 | 0.167x | 256.067 +/- 2.265 | 246.011 +/- 1.418 | 1.041x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 251.886 | 20424.604 | 0.012x | 16.320 +/- 0.104 | 11.730 +/- 0.069 | 1.391x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 530.537 | 21526.356 | 0.025x | 19.605 +/- 0.199 | 14.597 +/- 0.120 | 1.343x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 218.926 | 20930.339 | 0.010x | 14.799 +/- 0.077 | 9.850 +/- 0.069 | 1.502x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 192.591 | 19422.564 | 0.010x | 13.752 +/- 0.068 | 9.861 +/- 0.054 | 1.395x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 208.446 | 21132.540 | 0.010x | 14.783 +/- 0.095 | 9.960 +/- 0.092 | 1.484x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 725.107 | 21003.390 | 0.035x | 18.992 +/- 0.284 | 13.943 +/- 0.459 | 1.362x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6367.561 | 27261.611 | 0.234x | 388.910 +/- 4.003 | 387.132 +/- 5.293 | 1.005x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 226.909 | 21074.628 | 0.011x | 15.278 +/- 0.117 | 11.158 +/- 0.606 | 1.369x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 664.795 | 27016.476 | 0.025x | 18.812 +/- 0.165 | 13.691 +/- 0.128 | 1.374x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 347.075 | 29451.333 | 0.012x | 32.679 +/- 0.449 | 17.711 +/- 0.355 | 1.845x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 386.103 | 22949.800 | 0.017x | 27.686 +/- 0.184 | 15.429 +/- 0.163 | 1.794x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 338.162 | 22774.363 | 0.015x | 30.437 +/- 0.171 | 15.355 +/- 0.254 | 1.982x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 604.485 | 23786.650 | 0.025x | 35.198 +/- 0.194 | 18.118 +/- 0.207 | 1.943x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4225.832 | 25654.776 | 0.165x | 275.752 +/- 1.438 | 259.180 +/- 1.823 | 1.064x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 351.261 | 21813.997 | 0.016x | 31.142 +/- 0.189 | 14.258 +/- 0.134 | 2.184x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 651.831 | 23018.077 | 0.028x | 39.394 +/- 0.260 | 19.365 +/- 0.702 | 2.034x | `9348bfb9afa1f8c3` |
| `cpu_float32_decomposition_square_scalar` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 285.006 | 21843.226 | 0.013x | 23.745 +/- 0.158 | 14.645 +/- 0.262 | 1.621x | `028c65ba60e5aa0c` |
| `cpu_float32_decomposition_square_scalar` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 275.692 | 21729.418 | 0.013x | 23.608 +/- 0.085 | 14.581 +/- 0.203 | 1.619x | `649cd45c79b56805` |
| `cpu_float32_decomposition_square_scalar` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 299.378 | 21767.916 | 0.014x | 25.759 +/- 0.085 | 14.704 +/- 0.304 | 1.752x | `ca82da4f9d91253a` |
| `cpu_float32_decomposition_square_scalar` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 768.422 | 22812.947 | 0.034x | 31.547 +/- 0.204 | 18.424 +/- 0.127 | 1.712x | `e5d475561c8b39c9` |
| `cpu_float32_decomposition_square_scalar` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6553.644 | 27594.255 | 0.238x | 418.570 +/- 3.544 | 405.478 +/- 2.307 | 1.032x | `490ae4034ccb3f1f` |
| `cpu_float32_decomposition_square_scalar` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 335.452 | 21627.925 | 0.016x | 26.141 +/- 0.091 | 13.678 +/- 0.203 | 1.911x | `e99a6c9902c3119e` |
| `cpu_float32_decomposition_square_scalar` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 768.322 | 23815.974 | 0.032x | 35.493 +/- 0.206 | 19.418 +/- 0.275 | 1.828x | `68585b64809ef02a` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 333.259 | 25220.489 | 0.013x | 36.313 +/- 0.140 | 16.108 +/- 0.445 | 2.254x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 788.988 | 24983.976 | 0.032x | 41.029 +/- 0.263 | 19.610 +/- 0.147 | 2.092x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6510.783 | 28813.469 | 0.226x | 429.252 +/- 2.149 | 406.885 +/- 1.669 | 1.055x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 787.601 | 24782.136 | 0.032x | 39.920 +/- 0.259 | 21.857 +/- 0.859 | 1.826x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 751.928 | 25757.166 | 0.029x | 38.868 +/- 0.211 | 22.314 +/- 0.497 | 1.742x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 310.109 | 24005.872 | 0.013x | 36.398 +/- 0.226 | 14.641 +/- 0.085 | 2.486x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 824.692 | 25506.643 | 0.032x | 59.602 +/- 0.604 | 20.914 +/- 0.249 | 2.850x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 289.849 | 21814.783 | 0.013x | 26.673 +/- 0.198 | 14.138 +/- 0.162 | 1.887x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 737.385 | 22643.976 | 0.033x | 30.327 +/- 0.133 | 17.164 +/- 0.125 | 1.767x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6494.754 | 31447.289 | 0.207x | 414.406 +/- 2.368 | 400.468 +/- 2.971 | 1.035x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 718.762 | 24049.618 | 0.030x | 29.230 +/- 0.257 | 17.210 +/- 0.174 | 1.698x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 698.351 | 22249.896 | 0.031x | 30.332 +/- 0.295 | 16.982 +/- 0.134 | 1.786x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 254.500 | 21884.564 | 0.012x | 26.514 +/- 0.140 | 12.802 +/- 0.112 | 2.071x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 758.402 | 23713.825 | 0.032x | 47.512 +/- 0.283 | 17.529 +/- 0.100 | 2.710x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 278.306 | 22379.462 | 0.012x | 26.722 +/- 0.353 | 14.709 +/- 0.219 | 1.817x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 874.973 | 23050.087 | 0.038x | 38.004 +/- 7.123 | 17.239 +/- 0.112 | 2.205x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8864.178 | 27721.102 | 0.320x | 488.180 +/- 5.285 | 405.663 +/- 2.360 | 1.203x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 883.546 | 22409.673 | 0.039x | 39.621 +/- 2.343 | 17.289 +/- 0.127 | 2.292x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 864.453 | 21924.834 | 0.039x | 40.401 +/- 0.679 | 17.341 +/- 0.143 | 2.330x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 288.872 | 21413.566 | 0.013x | 26.522 +/- 0.160 | 12.809 +/- 0.078 | 2.071x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 751.661 | 24398.365 | 0.031x | 48.715 +/- 0.286 | 18.341 +/- 0.171 | 2.656x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 275.361 | 22753.468 | 0.012x | 26.649 +/- 0.106 | 13.711 +/- 0.179 | 1.944x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 702.668 | 25016.986 | 0.028x | 30.094 +/- 0.153 | 16.405 +/- 0.107 | 1.834x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6197.865 | 27814.904 | 0.223x | 400.225 +/- 2.749 | 383.635 +/- 1.697 | 1.043x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 751.562 | 21579.587 | 0.035x | 29.295 +/- 0.280 | 16.225 +/- 0.103 | 1.806x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 726.944 | 22122.403 | 0.033x | 30.509 +/- 0.229 | 16.478 +/- 0.134 | 1.852x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 259.918 | 21922.651 | 0.012x | 26.373 +/- 0.124 | 12.046 +/- 0.100 | 2.189x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 771.626 | 23318.422 | 0.033x | 46.150 +/- 0.277 | 16.709 +/- 0.115 | 2.762x | `2bd384aefcaaa397` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 310.990 | 22008.120 | 0.014x | 26.826 +/- 0.143 | 14.597 +/- 0.157 | 1.838x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 284.054 | 21260.148 | 0.013x | 23.909 +/- 0.468 | 14.299 +/- 0.143 | 1.672x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 360.731 | 21943.397 | 0.016x | 25.693 +/- 0.092 | 14.339 +/- 0.224 | 1.792x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 586.492 | 22203.030 | 0.026x | 29.915 +/- 0.128 | 17.113 +/- 0.276 | 1.748x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4108.069 | 27087.639 | 0.152x | 268.679 +/- 1.720 | 258.058 +/- 1.976 | 1.041x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 308.997 | 22867.280 | 0.014x | 26.205 +/- 0.107 | 13.389 +/- 0.073 | 1.957x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 588.765 | 25020.842 | 0.024x | 33.539 +/- 0.293 | 17.654 +/- 0.121 | 1.900x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 312.232 | 22038.256 | 0.014x | 31.458 +/- 0.197 | 15.063 +/- 0.204 | 2.088x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 744.120 | 26476.609 | 0.028x | 35.499 +/- 0.237 | 18.705 +/- 0.132 | 1.898x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6578.801 | 32266.558 | 0.204x | 420.896 +/- 2.094 | 409.328 +/- 3.536 | 1.028x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 781.297 | 23746.791 | 0.033x | 34.178 +/- 0.157 | 18.426 +/- 0.119 | 1.855x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 720.239 | 22502.613 | 0.032x | 35.719 +/- 0.216 | 18.459 +/- 0.235 | 1.935x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 291.781 | 22380.319 | 0.013x | 31.130 +/- 0.113 | 13.710 +/- 0.064 | 2.271x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 784.922 | 23652.322 | 0.033x | 52.865 +/- 0.613 | 19.149 +/- 0.132 | 2.761x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 302.288 | 21042.715 | 0.014x | 26.827 +/- 0.148 | 14.351 +/- 0.287 | 1.869x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 269.593 | 22434.310 | 0.012x | 23.569 +/- 0.118 | 14.343 +/- 0.128 | 1.643x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 345.057 | 21529.857 | 0.016x | 26.036 +/- 0.410 | 14.388 +/- 0.209 | 1.810x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 627.755 | 22698.174 | 0.028x | 29.864 +/- 0.127 | 16.616 +/- 0.164 | 1.797x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4108.239 | 24523.986 | 0.168x | 267.818 +/- 2.162 | 255.883 +/- 1.906 | 1.047x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 316.158 | 20954.437 | 0.015x | 26.750 +/- 0.261 | 13.235 +/- 0.057 | 2.021x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 615.776 | 22245.815 | 0.028x | 34.409 +/- 0.722 | 17.483 +/- 0.132 | 1.968x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 893.311 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 712.939 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 535.701 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 146479.190 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 120705.304 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 70528.937 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 104465.979 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 100095.340 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 67243.146 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1007.523 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 855.545 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 619.508 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, inference, training autograd, mutation_aliasing_views, decompositions, and recompilation guards. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `decompositions` | 6 | Supported and timed public cases: `cpu_float32_decomposition_square_scalar` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `python_control_flow` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `containers_pytrees` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `custom_functions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dtype_device_transitions` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 52 / 100. Zero-credit unsupported category weight: 48 / 100.
The torch_compile_corpus_v8 corpus also keeps 2 held-out broadcasting programs, 1 held-out decomposition program, 1 held-out inference program, 1 held-out mutation_aliasing_views program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 2 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
