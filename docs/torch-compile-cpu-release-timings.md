# `torch.compile` Eager CPU Release Timings

Date: 2026-09-04

Candidate provenance: source snapshot based on
`23d1f1d6fc3cfe95f603dd12c2cafe8d0b602a9e`, plus the worktree changes that
merge the v5 training-autograd and ReLU no-grad inference compile corpus
expansions.

The timing commands were run from the repository root using the pinned PyTorch
environment at `/data/users/bobren/a/pytorch-rs-burner/.venv` and this
worktree's locally built `torch_rs` wheel installed under `target/test-site`.
The reusable timing driver is checked in as
`scripts/benchmark_compile_cpu.py`; its complete raw JSON output is committed
at `docs/benchmark-data/torch-compile-cpu-v4.json`.

```bash
env CARGO_HOME="$PWD/target/cargo-home" TMPDIR="$PWD/target/tmp" \
  python -m maturin build --release --out target/wheels \
  --interpreter /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python -m pip install \
  --target target/test-site --no-deps --no-cache-dir --upgrade --force-reinstall \
  target/wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/target/test-site" \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python \
  scripts/benchmark_compile_cpu.py \
  --output docs/benchmark-data/torch-compile-cpu-v4.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/target/test-site" \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python \
  scripts/benchmark_compile_cpu.py \
  --render-markdown-summary docs/benchmark-data/torch-compile-cpu-v4.json \
  > target/torch-compile-cpu-v4-summary.md
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/target/test-site" \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python \
  scripts/benchmark_compile_cpu.py --validate-artifact
```

Checks run for this refresh:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/target/test-site" \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python -m unittest \
  tests.test_top_level_compile tests.test_compile_corpus \
  tests.test_torch_compile_coverage_evaluator tests.test_compile_benchmark_artifact
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/target/test-site" \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python \
  scripts/evaluate_torch_compile_coverage.py --subset full
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/target/test-site" \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python \
  scripts/benchmark_compile_cpu.py --validate-artifact
```

Results: the focused compile, corpus, evaluator, and benchmark-artifact tests
passed 92 tests. The full compile coverage evaluator passed all 20 eligible
full-corpus cases with a 38.0/100 score. Rust formatting passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.12.12 from `/data/users/bobren/a/pytorch-rs-burner/.venv/bin/python`
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; `torch.cuda.is_available()` reported `True`,
  although all timed benchmark cases use CPU tensors
- `torch_rs`: 0.1.0 from this worktree's `target/test-site/torch_rs/__init__.py`
- Profile: release wheel built with maturin
- Device/dtype: CPU float32
- CPU affinity: unrestricted process affinity across CPUs 0-383
- Threads: `torch.set_num_threads(1)` and
  `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1; `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  and `NUMEXPR_NUM_THREADS` were unset

The benchmark uses the checked-in `torch_compile_corpus_v5` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, four two-input broadcasting programs, one one-input no-grad inference program, one one-input training-autograd program, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Inference-category cells execute inside `torch.no_grad()`, and the corpus-default ReLU inference input requires grad while every timed inference output records `requires_grad=False`. Grad-enabled training-autograd cells validate forward output metadata and the expected input gradients after backward through a materialized sum. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 98 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass. The evidence is generated by the reusable benchmark driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Raw JSON artifact: `docs/benchmark-data/torch-compile-cpu-v4.json`
- Benchmark/corpus: `torch_compile_cpu_eager_benchmark_v3` / `torch_compile_corpus_v5`
- Cold first compiled call: 0.021x uncapped, 0.110x capped
- Steady-state materialized compiled call: 1.469x uncapped, 1.469x capped
- Timed supported cells: 98 (35 tensor-arithmetic, 28 broadcasting, 7 inference, 7 training-autograd, 21 recompilation-guard)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 38.0% supported by native compile cases, 62% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 311.736 | 183235.105 | 0.002x | 21.546 +/- 0.092 | 15.913 +/- 0.267 | 1.354x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 205.291 | 24435.975 | 0.008x | 18.421 +/- 0.063 | 15.394 +/- 0.130 | 1.197x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 225.246 | 23097.846 | 0.010x | 20.505 +/- 0.082 | 15.848 +/- 0.308 | 1.294x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 684.299 | 28591.476 | 0.024x | 25.210 +/- 0.117 | 19.350 +/- 0.196 | 1.303x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6225.426 | 28356.046 | 0.220x | 390.858 +/- 1.791 | 377.208 +/- 2.375 | 1.036x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 248.776 | 21784.281 | 0.011x | 21.099 +/- 0.052 | 14.946 +/- 0.089 | 1.412x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 632.441 | 23044.201 | 0.027x | 26.451 +/- 0.113 | 20.069 +/- 0.290 | 1.318x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 174.794 | 24589.857 | 0.007x | 17.248 +/- 0.081 | 13.903 +/- 0.096 | 1.241x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 157.683 | 22363.256 | 0.007x | 14.981 +/- 0.076 | 13.834 +/- 0.117 | 1.083x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 188.690 | 21685.942 | 0.009x | 19.085 +/- 0.442 | 13.796 +/- 0.197 | 1.383x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 682.377 | 22878.525 | 0.030x | 21.168 +/- 0.189 | 17.469 +/- 0.269 | 1.212x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6594.113 | 30675.357 | 0.215x | 399.137 +/- 7.612 | 388.439 +/- 6.748 | 1.028x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 193.567 | 24537.278 | 0.008x | 17.035 +/- 0.138 | 13.338 +/- 0.069 | 1.277x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 626.101 | 26023.955 | 0.024x | 22.799 +/- 0.066 | 17.274 +/- 0.097 | 1.320x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 199.682 | 23116.374 | 0.009x | 21.376 +/- 0.105 | 15.960 +/- 0.166 | 1.339x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 182.036 | 22136.042 | 0.008x | 18.413 +/- 0.124 | 15.587 +/- 0.147 | 1.181x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 206.432 | 22361.588 | 0.009x | 20.213 +/- 0.047 | 15.765 +/- 0.300 | 1.282x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 638.946 | 24448.048 | 0.026x | 25.100 +/- 0.078 | 19.038 +/- 0.164 | 1.318x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6120.887 | 30246.974 | 0.202x | 390.830 +/- 1.487 | 379.555 +/- 2.971 | 1.030x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 235.281 | 21396.124 | 0.011x | 20.760 +/- 0.093 | 14.699 +/- 0.072 | 1.412x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 660.263 | 26883.259 | 0.025x | 26.375 +/- 0.095 | 19.873 +/- 0.157 | 1.327x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 252.126 | 24904.002 | 0.010x | 31.288 +/- 0.098 | 19.429 +/- 0.244 | 1.610x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 228.245 | 23604.813 | 0.010x | 31.601 +/- 0.149 | 19.548 +/- 0.310 | 1.617x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 263.208 | 24875.409 | 0.011x | 35.178 +/- 0.113 | 19.194 +/- 0.419 | 1.833x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 694.264 | 24228.531 | 0.029x | 42.108 +/- 0.213 | 23.128 +/- 0.176 | 1.821x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6390.145 | 29066.810 | 0.220x | 419.993 +/- 2.241 | 392.435 +/- 2.562 | 1.070x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 288.346 | 24098.164 | 0.012x | 36.107 +/- 0.187 | 17.558 +/- 0.106 | 2.056x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 742.452 | 25448.891 | 0.029x | 45.340 +/- 0.218 | 24.070 +/- 0.194 | 1.884x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 294.856 | 22829.306 | 0.013x | 38.180 +/- 0.187 | 18.369 +/- 0.110 | 2.079x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 276.904 | 25445.470 | 0.011x | 33.463 +/- 0.242 | 20.288 +/- 0.492 | 1.649x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 306.418 | 23470.856 | 0.013x | 37.461 +/- 0.218 | 20.062 +/- 0.493 | 1.867x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 889.419 | 22734.923 | 0.039x | 44.096 +/- 0.176 | 24.403 +/- 0.206 | 1.807x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6515.880 | 32213.306 | 0.202x | 435.457 +/- 1.312 | 405.692 +/- 4.982 | 1.073x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 307.786 | 22900.208 | 0.013x | 37.848 +/- 0.163 | 18.293 +/- 0.091 | 2.069x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 775.417 | 28320.387 | 0.027x | 50.000 +/- 0.160 | 24.972 +/- 0.167 | 2.002x | `fedf1f495675c5ac` |
| `cpu_float32_training_unary_neg_abs_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 267.875 | 26099.335 | 0.010x | 33.640 +/- 0.102 | 21.229 +/- 0.193 | 1.585x | `9dcffd23ae8a957d` |
| `cpu_float32_training_unary_neg_abs_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 258.286 | 23307.899 | 0.011x | 27.819 +/- 0.094 | 18.709 +/- 0.273 | 1.487x | `5c2ffe407931c8ee` |
| `cpu_float32_training_unary_neg_abs_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 275.491 | 24304.030 | 0.011x | 31.369 +/- 0.101 | 19.393 +/- 0.591 | 1.618x | `d701faefd13d63e3` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 559.962 | 27330.590 | 0.020x | 36.374 +/- 0.108 | 21.645 +/- 0.269 | 1.680x | `fd8f6faa30e6834e` |
| `cpu_float32_training_unary_neg_abs_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4253.239 | 30939.706 | 0.137x | 280.656 +/- 1.606 | 261.590 +/- 1.873 | 1.073x | `89b634c0d077be1b` |
| `cpu_float32_training_unary_neg_abs_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 307.570 | 24410.586 | 0.013x | 31.795 +/- 0.093 | 17.311 +/- 0.153 | 1.837x | `e99a6c9902c3119e` |
| `cpu_float32_training_unary_neg_abs_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 577.178 | 25215.528 | 0.023x | 40.126 +/- 0.143 | 22.232 +/- 0.188 | 1.805x | `9348bfb9afa1f8c3` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 273.048 | 24536.317 | 0.011x | 37.592 +/- 0.049 | 20.288 +/- 0.215 | 1.853x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 721.226 | 31279.300 | 0.023x | 41.865 +/- 0.095 | 23.360 +/- 0.122 | 1.792x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6586.627 | 34505.151 | 0.191x | 437.439 +/- 3.038 | 412.016 +/- 1.587 | 1.062x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 741.977 | 26940.556 | 0.028x | 40.824 +/- 0.127 | 23.417 +/- 0.149 | 1.743x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 712.402 | 27282.628 | 0.026x | 39.903 +/- 0.163 | 23.165 +/- 0.176 | 1.723x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 251.240 | 24158.420 | 0.010x | 36.742 +/- 0.143 | 17.785 +/- 0.085 | 2.066x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 769.813 | 26815.868 | 0.029x | 60.066 +/- 0.587 | 24.244 +/- 0.146 | 2.478x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 243.448 | 22812.391 | 0.011x | 27.244 +/- 0.353 | 17.864 +/- 0.178 | 1.525x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 664.094 | 22999.768 | 0.029x | 31.505 +/- 0.652 | 20.713 +/- 0.123 | 1.521x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6751.882 | 28471.811 | 0.237x | 414.139 +/- 3.506 | 395.008 +/- 1.867 | 1.048x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 690.428 | 22862.261 | 0.030x | 29.309 +/- 0.100 | 20.293 +/- 0.142 | 1.444x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 652.242 | 27382.744 | 0.024x | 30.434 +/- 0.133 | 20.177 +/- 0.100 | 1.508x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 216.282 | 22510.344 | 0.010x | 26.600 +/- 0.068 | 15.556 +/- 0.058 | 1.710x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 688.170 | 24723.284 | 0.028x | 47.192 +/- 0.194 | 20.903 +/- 0.244 | 2.258x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 226.722 | 22838.114 | 0.010x | 27.017 +/- 0.123 | 18.702 +/- 0.273 | 1.445x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 662.572 | 24787.982 | 0.027x | 30.897 +/- 0.152 | 20.687 +/- 0.113 | 1.494x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6517.852 | 30713.629 | 0.212x | 414.885 +/- 2.528 | 403.667 +/- 2.271 | 1.028x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 676.768 | 25046.793 | 0.027x | 30.435 +/- 0.105 | 20.630 +/- 0.302 | 1.475x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 655.226 | 23653.903 | 0.028x | 30.775 +/- 0.160 | 20.278 +/- 0.154 | 1.518x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 213.358 | 22684.522 | 0.009x | 26.878 +/- 0.080 | 15.943 +/- 0.082 | 1.686x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 699.658 | 25197.386 | 0.028x | 48.906 +/- 0.244 | 21.807 +/- 0.176 | 2.243x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 234.515 | 25396.061 | 0.009x | 27.356 +/- 0.085 | 17.143 +/- 0.177 | 1.596x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 639.812 | 25538.997 | 0.025x | 30.699 +/- 0.154 | 19.723 +/- 0.144 | 1.556x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6584.595 | 33531.198 | 0.196x | 397.960 +/- 2.200 | 388.084 +/- 2.561 | 1.025x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 690.464 | 23434.525 | 0.029x | 29.600 +/- 0.191 | 19.505 +/- 0.220 | 1.518x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 646.287 | 25472.466 | 0.025x | 30.584 +/- 0.155 | 19.318 +/- 0.121 | 1.583x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 235.501 | 22647.551 | 0.010x | 26.695 +/- 0.067 | 14.970 +/- 0.092 | 1.783x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 712.087 | 26762.436 | 0.027x | 48.494 +/- 0.197 | 19.849 +/- 0.175 | 2.443x | `2bd384aefcaaa397` |
| `cpu_float32_relu_no_grad_inference` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 180.748 | 22659.338 | 0.008x | 16.145 +/- 0.069 | 13.923 +/- 0.209 | 1.160x | `4c4863775b297fa0` |
| `cpu_float32_relu_no_grad_inference` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 159.957 | 25213.279 | 0.006x | 14.021 +/- 0.038 | 13.375 +/- 0.117 | 1.048x | `292485c676f9433a` |
| `cpu_float32_relu_no_grad_inference` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 180.913 | 22582.192 | 0.008x | 15.399 +/- 0.036 | 13.671 +/- 0.124 | 1.126x | `99fbf7ee8cd20333` |
| `cpu_float32_relu_no_grad_inference` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 435.214 | 22000.417 | 0.020x | 18.544 +/- 0.088 | 16.052 +/- 0.106 | 1.155x | `4295284801db4ec1` |
| `cpu_float32_relu_no_grad_inference` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 3847.932 | 26317.209 | 0.146x | 246.985 +/- 1.632 | 248.851 +/- 1.865 | 0.992x | `c459941c9565e750` |
| `cpu_float32_relu_no_grad_inference` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 184.013 | 22025.836 | 0.008x | 15.885 +/- 0.038 | 13.177 +/- 0.051 | 1.206x | `e99a6c9902c3119e` |
| `cpu_float32_relu_no_grad_inference` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 446.490 | 23700.978 | 0.019x | 19.246 +/- 0.090 | 16.046 +/- 0.103 | 1.199x | `b065276a7b7f64c3` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 235.897 | 21930.417 | 0.011x | 27.573 +/- 0.066 | 18.145 +/- 0.150 | 1.520x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 230.614 | 21329.564 | 0.011x | 23.365 +/- 0.101 | 17.802 +/- 0.444 | 1.312x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 247.830 | 23870.003 | 0.010x | 26.013 +/- 0.083 | 17.899 +/- 0.303 | 1.453x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 514.918 | 23528.388 | 0.022x | 30.450 +/- 0.164 | 20.388 +/- 0.220 | 1.493x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4203.059 | 26539.299 | 0.158x | 264.391 +/- 1.985 | 259.688 +/- 2.123 | 1.018x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 269.893 | 25847.088 | 0.010x | 26.787 +/- 0.096 | 16.355 +/- 0.148 | 1.638x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 540.452 | 26814.951 | 0.020x | 33.549 +/- 0.217 | 20.914 +/- 0.152 | 1.604x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 260.404 | 25768.770 | 0.010x | 32.175 +/- 0.202 | 18.872 +/- 0.314 | 1.705x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 684.184 | 23313.588 | 0.029x | 35.752 +/- 0.277 | 22.208 +/- 0.213 | 1.610x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 6524.067 | 36040.531 | 0.181x | 416.188 +/- 1.912 | 404.815 +/- 4.224 | 1.028x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 724.255 | 26369.363 | 0.027x | 34.779 +/- 0.156 | 22.164 +/- 0.091 | 1.569x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 667.534 | 24920.196 | 0.027x | 35.855 +/- 0.195 | 21.551 +/- 0.076 | 1.664x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 241.375 | 24661.415 | 0.010x | 31.478 +/- 0.153 | 16.838 +/- 0.169 | 1.869x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 718.036 | 29463.649 | 0.024x | 52.821 +/- 0.191 | 22.519 +/- 0.180 | 2.346x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 230.709 | 23974.006 | 0.010x | 27.346 +/- 0.128 | 17.893 +/- 0.173 | 1.528x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 225.106 | 24326.298 | 0.009x | 23.368 +/- 0.335 | 17.784 +/- 0.313 | 1.314x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 236.752 | 23085.593 | 0.010x | 25.703 +/- 0.080 | 17.904 +/- 0.373 | 1.436x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 515.214 | 22528.756 | 0.023x | 30.106 +/- 0.119 | 20.424 +/- 0.298 | 1.474x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 4021.094 | 24483.682 | 0.164x | 262.253 +/- 1.526 | 266.523 +/- 2.687 | 0.984x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 265.702 | 20335.336 | 0.013x | 26.493 +/- 0.070 | 16.259 +/- 0.113 | 1.629x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 532.811 | 24372.474 | 0.022x | 33.688 +/- 0.365 | 20.738 +/- 0.135 | 1.624x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 822.213 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 661.801 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 474.677 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 156460.811 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 133494.442 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 82523.961 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 114463.563 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 115148.971 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 80389.494 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 1082.165 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 842.823 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 611.494 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, inference, training autograd, and recompilation guards. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `inference` | 6 | Supported and timed public cases: `cpu_float32_relu_no_grad_inference` |
| `training_autograd` | 8 | Supported and timed public cases: `cpu_float32_training_unary_neg_abs_add` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `python_control_flow` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `mutation_aliasing_views` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `containers_pytrees` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `decompositions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `custom_functions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dtype_device_transitions` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 38 / 100. Zero-credit unsupported category weight: 62 / 100.
The torch_compile_corpus_v5 corpus also keeps 2 held-out broadcasting programs, 1 held-out inference program, 2 held-out recompilation-guard programs, 1 held-out training-autograd program, and 2 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
