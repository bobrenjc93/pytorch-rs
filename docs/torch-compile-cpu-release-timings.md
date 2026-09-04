# `torch.compile` Eager CPU Release Timings

Date: 2026-09-04

Candidate provenance: source snapshot based on
`12ac4fcb8a1b0ac6b965fcafaa8cdb8a9800b076`, plus the worktree changes that
restore strict current-corpus benchmark validation and refresh the raw benchmark
artifact for corpus v8.

The setup, build, focused check, and timing commands below reproduce this
evidence from the repository root. The reusable timing driver is checked in as
`scripts/benchmark_compile_cpu.py`; its complete raw JSON output is committed
at `docs/benchmark-data/torch-compile-cpu-v4.json`. The PyTorch 2.13 reference
evidence used this worktree's local `.venv`; uv and Cargo state were redirected
under `target/`.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv venv --clear --python 3.12
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv sync --locked --no-install-project --group dev --group reference
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --out target/wheels
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv pip install --python "$PWD/.venv/bin/python" \
  --force-reinstall --no-deps target/wheels/torch_rs-*.whl
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
  scripts/benchmark_compile_cpu.py tests/test_compile_benchmark_artifact.py
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
  scripts/benchmark_compile_cpu.py tests/test_compile_benchmark_artifact.py
cargo fmt --check
```

Results: the full fixed-affinity CPU benchmark, raw-artifact/markdown
validation, and focused compile-corpus plus benchmark-artifact unittest suite
passed. The focused unittest run passed 56 tests. Python bytecode compilation
for the edited benchmark/test files and `cargo fmt --check` also passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_9629eb76/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_9629eb76/.venv/lib/python3.12/site-packages/torch_rs/__init__.py`
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
- Build: release wheel installed in the worktree-local `.venv`

The benchmark uses the checked-in `torch_compile_corpus_v8` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, one one-input no-grad inference program, one one-input storage-aliasing detach program, one one-input training-autograd program, four two-input broadcasting programs, one two-input containers-pytrees program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Detach cells return shared-storage aliases with `requires_grad=False`. Tuple/list output cells preserve container structure and record per-tensor metadata for each output leaf. Grad-enabled training-autograd cells validate forward output metadata and expected input gradients after backward through a materialized sum, and assert measured and reference inputs remain unchanged after backward. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, tuple/list container structure, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. For cases marked `backward_through_sum`, grad-enabled cells compare leaf-input gradients after backward through a materialized sum and verify input values and metadata are unchanged by backward. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 112 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v8`
- Cold first compiled call: 0.027x uncapped, 0.114x capped
- Steady-state materialized compiled call: 1.816x uncapped, 1.816x capped
- Timed supported cells: 112 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 7 containers-pytrees, 21 recompilation-guard, 7 mutation_aliasing_views)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 52.0% supported by native compile cases, 48% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 346.575 | 83257.751 | 0.004x | 26.069 +/- 0.146 | 14.307 +/- 0.150 | 1.822x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 236.368 | 22890.508 | 0.010x | 22.421 +/- 0.118 | 14.167 +/- 0.184 | 1.583x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 234.976 | 22734.103 | 0.010x | 24.597 +/- 0.194 | 14.183 +/- 0.169 | 1.734x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 826.065 | 22642.460 | 0.036x | 30.688 +/- 0.272 | 19.295 +/- 0.255 | 1.590x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8574.941 | 32212.135 | 0.266x | 538.779 +/- 2.606 | 528.344 +/- 3.555 | 1.020x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 242.362 | 24062.642 | 0.010x | 24.946 +/- 0.141 | 13.452 +/- 0.079 | 1.854x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 836.225 | 25772.238 | 0.032x | 32.182 +/- 0.291 | 19.471 +/- 0.152 | 1.653x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 208.516 | 22573.444 | 0.009x | 20.676 +/- 0.095 | 12.744 +/- 0.109 | 1.622x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 193.623 | 22776.151 | 0.009x | 17.828 +/- 0.111 | 12.562 +/- 0.079 | 1.419x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 202.697 | 22018.510 | 0.009x | 19.591 +/- 0.117 | 12.676 +/- 0.162 | 1.546x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 808.984 | 22569.799 | 0.036x | 25.915 +/- 0.309 | 17.701 +/- 0.120 | 1.464x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8765.818 | 31820.086 | 0.275x | 557.665 +/- 3.560 | 550.602 +/- 6.547 | 1.013x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 226.673 | 23091.127 | 0.010x | 20.255 +/- 0.144 | 12.286 +/- 0.136 | 1.649x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 815.393 | 24794.844 | 0.033x | 27.653 +/- 0.233 | 17.616 +/- 0.121 | 1.570x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 230.353 | 21449.099 | 0.011x | 25.312 +/- 0.157 | 14.354 +/- 0.203 | 1.763x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 211.469 | 24240.351 | 0.009x | 22.107 +/- 0.119 | 14.133 +/- 0.188 | 1.564x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 246.443 | 24145.147 | 0.010x | 24.123 +/- 0.214 | 14.090 +/- 0.120 | 1.712x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 808.442 | 25060.217 | 0.032x | 30.231 +/- 0.153 | 18.961 +/- 0.140 | 1.594x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8429.139 | 32980.931 | 0.256x | 540.579 +/- 3.230 | 526.510 +/- 6.023 | 1.027x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 274.690 | 22119.152 | 0.012x | 24.846 +/- 0.125 | 13.581 +/- 0.167 | 1.830x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 834.367 | 24104.149 | 0.035x | 32.037 +/- 0.234 | 19.199 +/- 0.152 | 1.669x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 304.781 | 23218.636 | 0.013x | 37.238 +/- 0.154 | 17.770 +/- 0.093 | 2.096x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 268.206 | 23755.302 | 0.011x | 37.441 +/- 0.296 | 18.189 +/- 0.365 | 2.058x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 294.620 | 24158.087 | 0.012x | 41.236 +/- 0.334 | 17.959 +/- 0.270 | 2.296x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 882.656 | 25517.378 | 0.035x | 49.100 +/- 0.424 | 23.042 +/- 0.303 | 2.131x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8517.974 | 32617.573 | 0.261x | 568.476 +/- 6.553 | 537.424 +/- 4.332 | 1.058x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 338.817 | 23621.194 | 0.014x | 42.194 +/- 0.253 | 16.518 +/- 0.118 | 2.554x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 911.809 | 26748.315 | 0.034x | 53.068 +/- 0.305 | 23.865 +/- 0.181 | 2.224x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 341.131 | 25946.151 | 0.013x | 44.221 +/- 0.422 | 16.873 +/- 0.157 | 2.621x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 320.565 | 23539.786 | 0.014x | 39.029 +/- 0.262 | 18.128 +/- 0.185 | 2.153x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 357.897 | 24948.467 | 0.014x | 43.032 +/- 0.181 | 18.269 +/- 0.374 | 2.355x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 990.834 | 26362.991 | 0.038x | 51.998 +/- 0.454 | 23.362 +/- 0.164 | 2.226x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8928.034 | 39448.455 | 0.226x | 593.558 +/- 6.470 | 563.067 +/- 6.315 | 1.054x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 409.870 | 24243.701 | 0.017x | 44.188 +/- 0.467 | 16.835 +/- 0.160 | 2.625x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 992.006 | 25062.329 | 0.040x | 58.929 +/- 0.516 | 24.387 +/- 0.285 | 2.416x | `fedf1f495675c5ac` |
| `cpu_float32_inference_relu_no_grad` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 220.073 | 23619.321 | 0.009x | 19.863 +/- 0.146 | 14.076 +/- 0.237 | 1.411x | `11b2aee46363d5ff` |
| `cpu_float32_inference_relu_no_grad` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 193.954 | 23382.203 | 0.008x | 17.521 +/- 0.116 | 13.982 +/- 0.386 | 1.253x | `292485c676f9433a` |
| `cpu_float32_inference_relu_no_grad` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 209.627 | 23600.753 | 0.009x | 18.867 +/- 0.153 | 13.706 +/- 0.169 | 1.377x | `99fbf7ee8cd20333` |
| `cpu_float32_inference_relu_no_grad` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 548.290 | 25108.169 | 0.022x | 22.774 +/- 0.137 | 16.934 +/- 0.290 | 1.345x | `4295284801db4ec1` |
| `cpu_float32_inference_relu_no_grad` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5174.096 | 28373.298 | 0.182x | 333.150 +/- 3.249 | 324.579 +/- 3.476 | 1.026x | `c459941c9565e750` |
| `cpu_float32_inference_relu_no_grad` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 213.728 | 22446.558 | 0.010x | 19.480 +/- 0.170 | 13.380 +/- 0.079 | 1.456x | `e99a6c9902c3119e` |
| `cpu_float32_inference_relu_no_grad` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 557.749 | 22934.171 | 0.024x | 23.843 +/- 0.211 | 16.826 +/- 0.166 | 1.417x | `b065276a7b7f64c3` |
| `cpu_float32_detach_alias_view` | `case_default` | 1 | 256 | shape (2,), stride (3,), offset 1, torch.float32, cpu, requires_grad=False | 182.221 | 22407.579 | 0.008x | 17.650 +/- 0.087 | 11.382 +/- 0.129 | 1.551x | `5780cfdca8917311` |
| `cpu_float32_detach_alias_view` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 176.101 | 21651.766 | 0.008x | 16.333 +/- 0.067 | 11.235 +/- 0.055 | 1.454x | `e75a1d3233117514` |
| `cpu_float32_detach_alias_view` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 212.006 | 22412.686 | 0.009x | 17.681 +/- 0.100 | 11.319 +/- 0.080 | 1.562x | `4c3dc265c5b9d697` |
| `cpu_float32_detach_alias_view` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 768.127 | 23094.959 | 0.033x | 23.095 +/- 0.183 | 16.055 +/- 0.086 | 1.438x | `5ccc89fb94f689e5` |
| `cpu_float32_detach_alias_view` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8659.678 | 30277.042 | 0.286x | 541.241 +/- 5.177 | 526.996 +/- 2.931 | 1.027x | `91fa5699b26ca1b8` |
| `cpu_float32_detach_alias_view` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 243.764 | 23410.971 | 0.010x | 18.383 +/- 0.089 | 11.314 +/- 0.059 | 1.625x | `e99a6c9902c3119e` |
| `cpu_float32_detach_alias_view` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 799.749 | 26209.219 | 0.031x | 23.024 +/- 0.175 | 16.094 +/- 0.139 | 1.431x | `4ba5419e2e3f2393` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 318.983 | 24111.896 | 0.013x | 38.925 +/- 0.194 | 18.665 +/- 0.153 | 2.085x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 335.828 | 23866.010 | 0.014x | 32.790 +/- 0.232 | 17.125 +/- 0.342 | 1.915x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 314.942 | 24873.489 | 0.013x | 36.314 +/- 0.278 | 17.368 +/- 0.555 | 2.091x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 688.887 | 26887.676 | 0.026x | 42.001 +/- 0.337 | 20.152 +/- 0.198 | 2.084x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5636.936 | 30381.350 | 0.186x | 361.616 +/- 2.972 | 337.808 +/- 3.346 | 1.070x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 345.933 | 22907.160 | 0.015x | 36.869 +/- 0.305 | 15.699 +/- 0.225 | 2.348x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 703.169 | 25771.317 | 0.027x | 47.002 +/- 0.314 | 20.665 +/- 0.151 | 2.274x | `9348bfb9afa1f8c3` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 320.470 | 28394.084 | 0.011x | 43.498 +/- 0.328 | 17.497 +/- 0.217 | 2.486x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 906.647 | 25732.342 | 0.035x | 49.944 +/- 0.617 | 22.401 +/- 0.203 | 2.230x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9182.058 | 34743.312 | 0.264x | 600.195 +/- 9.566 | 560.486 +/- 5.569 | 1.071x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 935.806 | 26003.508 | 0.036x | 48.444 +/- 0.644 | 22.271 +/- 0.178 | 2.175x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 892.575 | 24481.526 | 0.036x | 47.294 +/- 0.341 | 22.325 +/- 0.181 | 2.118x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 295.972 | 23994.088 | 0.012x | 42.835 +/- 0.302 | 15.808 +/- 0.089 | 2.710x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 975.901 | 27323.654 | 0.036x | 71.795 +/- 0.376 | 23.154 +/- 0.164 | 3.101x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 282.297 | 27023.249 | 0.010x | 31.823 +/- 0.230 | 15.514 +/- 0.133 | 2.051x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 853.140 | 24566.068 | 0.035x | 36.674 +/- 0.227 | 19.760 +/- 0.110 | 1.856x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8919.281 | 34036.126 | 0.262x | 570.814 +/- 3.890 | 550.572 +/- 3.006 | 1.037x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 902.330 | 23810.244 | 0.038x | 35.550 +/- 0.228 | 19.648 +/- 0.122 | 1.809x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 832.284 | 22985.958 | 0.036x | 36.686 +/- 0.230 | 19.517 +/- 0.183 | 1.880x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 255.276 | 26827.988 | 0.010x | 31.536 +/- 0.194 | 13.993 +/- 0.053 | 2.254x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 915.675 | 25571.794 | 0.036x | 57.267 +/- 0.388 | 20.152 +/- 0.135 | 2.842x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 241.310 | 22794.548 | 0.011x | 31.915 +/- 0.344 | 15.844 +/- 0.122 | 2.014x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 839.524 | 23582.029 | 0.036x | 37.624 +/- 0.366 | 19.822 +/- 0.144 | 1.898x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9091.967 | 31858.783 | 0.285x | 588.931 +/- 4.505 | 558.272 +/- 2.819 | 1.055x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 885.324 | 23036.860 | 0.038x | 36.886 +/- 0.219 | 19.639 +/- 0.166 | 1.878x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 874.016 | 24205.878 | 0.036x | 37.156 +/- 0.357 | 19.758 +/- 0.168 | 1.881x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 245.642 | 25594.374 | 0.010x | 31.757 +/- 0.207 | 14.078 +/- 0.142 | 2.256x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 868.068 | 26807.779 | 0.032x | 58.050 +/- 0.271 | 20.846 +/- 0.155 | 2.785x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 246.512 | 24062.255 | 0.010x | 31.705 +/- 0.205 | 15.258 +/- 0.179 | 2.078x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 822.058 | 27861.037 | 0.030x | 36.274 +/- 0.252 | 19.082 +/- 0.169 | 1.901x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8382.569 | 33079.440 | 0.253x | 548.082 +/- 4.760 | 531.100 +/- 3.336 | 1.032x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 927.448 | 24283.460 | 0.038x | 35.545 +/- 0.212 | 18.968 +/- 0.176 | 1.874x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 843.391 | 23355.181 | 0.036x | 36.689 +/- 0.275 | 19.000 +/- 0.175 | 1.931x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 239.453 | 24355.559 | 0.010x | 31.624 +/- 0.170 | 13.516 +/- 0.051 | 2.340x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 893.120 | 26309.099 | 0.034x | 57.887 +/- 0.419 | 19.565 +/- 0.117 | 2.959x | `2bd384aefcaaa397` |
| `cpu_float32_tuple_list_output_pytree` | `case_default` | 2 | 256 | tuple[shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True, list[shape (3,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True]] | 403.581 | 26325.364 | 0.015x | 54.426 +/- 0.247 | 19.605 +/- 0.196 | 2.776x | `a62dacb062c1ed92` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1389.492 | 29407.590 | 0.047x | 62.860 +/- 0.396 | 25.817 +/- 0.215 | 2.435x | `3bce94d7e523bafe` |
| `cpu_float32_tuple_list_output_pytree` | `matrix_vector_127x131_by_131` | 2 | 16 | tuple[shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (131,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 14530.188 | 40414.870 | 0.360x | 931.670 +/- 7.905 | 884.971 +/- 9.584 | 1.053x | `022557af0d301f5e` |
| `cpu_float32_tuple_list_output_pytree` | `tensor_scalar_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 1422.837 | 25393.369 | 0.056x | 61.002 +/- 0.265 | 25.471 +/- 0.270 | 2.395x | `f4ff04ee55c4e2cd` |
| `cpu_float32_tuple_list_output_pytree` | `scalar_tensor_31x37` | 2 | 128 | tuple[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False, shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False]] | 1642.424 | 27385.778 | 0.060x | 62.292 +/- 0.617 | 27.097 +/- 0.277 | 2.299x | `f1950b665bfdc9f1` |
| `cpu_float32_tuple_list_output_pytree` | `empty_2x0_by_0` | 2 | 2048 | tuple[shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False, list[shape (0,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False]] | 377.986 | 27713.674 | 0.014x | 53.043 +/- 0.178 | 16.074 +/- 0.091 | 3.300x | `e89cfed7478c41fa` |
| `cpu_float32_tuple_list_output_pytree` | `transpose_31x37_by_37` | 2 | 128 | tuple[shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False, list[shape (37,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False, shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False]] | 1454.264 | 27239.782 | 0.053x | 85.468 +/- 0.751 | 26.605 +/- 0.246 | 3.212x | `776bd23d05673f66` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 279.964 | 22898.981 | 0.012x | 32.393 +/- 0.353 | 16.171 +/- 0.233 | 2.003x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 255.471 | 22688.378 | 0.011x | 27.947 +/- 0.142 | 16.210 +/- 0.566 | 1.724x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 276.543 | 21574.483 | 0.013x | 30.778 +/- 0.136 | 15.567 +/- 0.229 | 1.977x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 651.861 | 22656.590 | 0.029x | 35.828 +/- 0.223 | 18.965 +/- 0.100 | 1.889x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5829.578 | 27129.560 | 0.215x | 355.148 +/- 2.309 | 334.390 +/- 2.410 | 1.062x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 315.848 | 21820.049 | 0.014x | 31.092 +/- 0.126 | 15.046 +/- 0.184 | 2.066x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 685.266 | 26457.299 | 0.026x | 39.669 +/- 0.217 | 19.694 +/- 0.254 | 2.014x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 306.168 | 26192.683 | 0.012x | 37.367 +/- 0.201 | 16.729 +/- 0.340 | 2.234x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 878.554 | 24096.302 | 0.036x | 43.124 +/- 0.395 | 21.357 +/- 0.281 | 2.019x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8951.761 | 31638.711 | 0.283x | 580.902 +/- 4.663 | 554.692 +/- 2.916 | 1.047x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 932.726 | 25062.389 | 0.037x | 41.886 +/- 0.246 | 21.361 +/- 0.186 | 1.961x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 881.583 | 24742.590 | 0.036x | 43.156 +/- 0.193 | 21.046 +/- 0.154 | 2.051x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 295.952 | 26430.483 | 0.011x | 37.407 +/- 0.258 | 15.083 +/- 0.177 | 2.480x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1122.698 | 29362.082 | 0.038x | 78.185 +/- 0.515 | 21.779 +/- 0.251 | 3.590x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 304.991 | 23148.749 | 0.013x | 38.943 +/- 0.187 | 16.157 +/- 0.217 | 2.410x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 313.469 | 24902.527 | 0.013x | 27.895 +/- 0.185 | 16.062 +/- 0.360 | 1.737x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 271.145 | 25290.298 | 0.011x | 37.134 +/- 0.549 | 15.709 +/- 0.247 | 2.364x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 648.160 | 23392.177 | 0.028x | 35.592 +/- 0.141 | 18.848 +/- 0.224 | 1.888x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5326.922 | 30293.576 | 0.176x | 351.058 +/- 1.787 | 334.939 +/- 2.355 | 1.048x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 292.613 | 23203.617 | 0.013x | 31.296 +/- 0.128 | 14.716 +/- 0.097 | 2.127x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 677.154 | 23948.514 | 0.028x | 39.498 +/- 0.240 | 19.452 +/- 0.178 | 2.030x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 874.192 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 748.384 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 563.522 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 161556.648 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 144427.839 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 87703.417 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 119662.559 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 109147.274 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 76284.267 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 959.202 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 834.254 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 582.131 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, inference, training autograd, mutation_aliasing_views, containers and pytrees, and recompilation guards. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_inference_relu_no_grad` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `mutation_aliasing_views` | 8 | Supported and timed public cases: `cpu_float32_detach_alias_view` |
| `containers_pytrees` | 6 | Supported and timed public cases: `cpu_float32_tuple_list_output_pytree` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `python_control_flow` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `decompositions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `custom_functions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dtype_device_transitions` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 52 / 100. Zero-credit unsupported category weight: 48 / 100.
The torch_compile_corpus_v8 corpus also keeps 2 held-out broadcasting programs, 1 held-out containers-pytrees program, 1 held-out inference program, 1 held-out mutation_aliasing_views program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 2 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
