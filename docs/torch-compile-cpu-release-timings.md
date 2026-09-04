# `torch.compile` Eager CPU Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on
`d0a0540fe204dcc5245e398f4ab9736eedceb945`, plus the worktree changes that
add the `torch_compile_corpus_v4` recompilation-guard corpus coverage and keep
the fixed-affinity benchmark driver reusable for category-specific timing
variants.

Exact setup, build, check, and timing commands were run from the repository
root. The reusable timing driver is checked in as
`scripts/benchmark_compile_cpu.py` and emitted JSON under
`target/compile-cpu-release-timings.json`. The active Conda environment held
ambient PyTorch 2.14, so the PyTorch 2.13 reference evidence used a
worktree-local `.venv`; uv and Cargo state were redirected under `target/`.
The local Cargo cache was populated from the existing read-only user cache
because the crates.io proxy rejected direct unauthenticated fetches during
this run.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12 .venv
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --python "$PWD/.venv/bin/python" \
  --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/cache \
  /home/bobren/.cargo/registry/index \
  /home/bobren/.cargo/registry/src target/cargo-home/registry/
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked --offline
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_top_level_compile tests.test_compile_corpus tests.test_readme_quickstart
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python scripts/benchmark_compile_cpu.py \
  --require-single-cpu-affinity \
  --output target/compile-cpu-release-timings.json
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  cargo clippy --locked --offline --all-targets -- -D warnings
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  cargo test --locked --offline --all-targets
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  cargo test --locked --offline --doc
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --locked --offline --all-targets --features python-bindings -- \
  -D warnings
mkdir -p target/pyo3
printf "%s\n" implementation=CPython version=3.12 shared=true \
  abi3=true lib_name=python3.12 \
  lib_dir=/usr/local/fbcode/platform010/lib \
  executable="$PWD/.venv/bin/python" pointer_width=64 build_flags= \
  suppress_build_script_link_lines=false > target/pyo3/config.txt
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true VIRTUAL_ENV="$PWD/.venv" \
  PYO3_CONFIG_FILE="$PWD/target/pyo3/config.txt" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --locked --offline --all-targets --features python-bindings
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  ./scripts/test-python.sh
```

Checks run for this evidence:

```bash
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_top_level_compile tests.test_compile_corpus tests.test_readme_quickstart
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  cargo clippy --locked --offline --all-targets -- -D warnings
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  cargo test --locked --offline --all-targets
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  cargo test --locked --offline --doc
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --locked --offline --all-targets --features python-bindings -- \
  -D warnings
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true VIRTUAL_ENV="$PWD/.venv" \
  PYO3_CONFIG_FILE="$PWD/target/pyo3/config.txt" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo test --locked --offline --all-targets --features python-bindings
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  CARGO_NET_OFFLINE=true \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  ./scripts/test-python.sh
```

Results: the focused public compile, compile corpus, PyTorch 2.13
differential, and docs smoke tests passed 74 tests. The repository-managed
pinned PyTorch 2.13 full Python suite passed 4708 tests with 3 skips. The
default Rust suite passed 311 tests across unit and integration targets plus
0 doctests; the Python-bindings Rust suite passed 322 tests. Default Clippy
and Python-bindings Clippy both passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_73f41599/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_73f41599/.venv/lib/python3.12/site-packages/torch_rs/__init__.py`
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
- Dependency installation: locked `uv sync` resolved in 31 ms, prepared
  packages in 17.05s, and installed in 2.20s
- Build time: first successful Python 3.12 release extension build completed
  in 38.00s; the full-suite wheel build reused release artifacts and
  completed in 0.02s

The benchmark uses the checked-in `torch_compile_corpus_v4` programs. The
timed supported set contains every public native compile case: five one-input
tensor-arithmetic programs, four two-input broadcasting programs, and four
recompilation-guard programs. One-input arithmetic programs run across the
corpus default input plus scalar, vector, row-major matrix, larger row-major
matrix, empty, and non-contiguous transpose inputs. Two-input broadcasting
programs run across the corpus default input plus row-major matrix/vector,
larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast,
and non-contiguous matrix/vector broadcast inputs. Recompilation-guard programs
are timed on their public default input and covered semantically by separate
guard sequences for shape, stride, `requires_grad`, two-input metadata,
bounded `recompile_limit`, and `torch.compiler.reset()`.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock
`torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the
native guarded eager/fullgraph path. Both implementations run in both orders:
`torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant
compiler state for cold timing, measures the first materialized compiled call
separately, then runs 7 untimed warmup blocks and 31 measured blocks. A
measured block repeats the operation according to the table's `Repeats`
column; medians below are microseconds per compiled call. The CPU workload has
no asynchronous device queue, but the driver still calls synchronization hooks
when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, shape, stride,
storage offset, contiguity, dtype, device, and `requires_grad` against the
same eager program. The `torch_rs` result is also checked against the PyTorch
result. After every warmup and measured block, the driver materializes the
last output and records a 64-bit BLAKE2b checksum over values and metadata.
All 67 timed cells had matching `torch_rs` and PyTorch checksums. Benchmark
integrity controls retained fixed affinity, both implementation orders,
strict materialization, pinned reference PyTorch, exact native CPU float32
execution, and explicit zero-credit denominator accounting; this run satisfies
the >=99 integrity requirement for this evidence update.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Cold first compiled call: 0.024x uncapped, 0.114x capped
- Steady-state materialized compiled call: 1.724x uncapped, 1.724x capped
- Timed supported cells: 67 (35 one-input tensor arithmetic, 28 two-input broadcasting, 4 recompilation-guard)
- Versioned denominator coverage: 24.0% supported by native compile cases, 76% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 332.869 | 81638.826 | 0.004x | 23.121 +/- 0.177 | 13.516 +/- 0.189 | 1.711x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 217.503 | 26722.923 | 0.008x | 19.615 +/- 0.095 | 13.520 +/- 0.352 | 1.451x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 228.716 | 23994.978 | 0.010x | 21.770 +/- 0.172 | 13.266 +/- 0.270 | 1.641x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 997.433 | 24124.303 | 0.041x | 27.795 +/- 0.253 | 18.319 +/- 0.299 | 1.517x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8482.943 | 35313.297 | 0.240x | 534.910 +/- 3.900 | 549.563 +/- 7.164 | 0.973x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 243.483 | 23312.045 | 0.010x | 22.026 +/- 0.095 | 12.645 +/- 0.077 | 1.742x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 824.591 | 25449.582 | 0.032x | 29.398 +/- 0.181 | 19.016 +/- 0.212 | 1.546x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 189.637 | 23635.429 | 0.008x | 18.176 +/- 0.095 | 12.164 +/- 0.159 | 1.494x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 168.480 | 25163.605 | 0.007x | 15.459 +/- 0.044 | 12.046 +/- 0.192 | 1.283x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 197.779 | 23807.825 | 0.008x | 17.211 +/- 0.130 | 11.908 +/- 0.178 | 1.445x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 954.348 | 25010.654 | 0.038x | 23.093 +/- 0.176 | 16.809 +/- 0.170 | 1.374x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8658.076 | 31575.671 | 0.274x | 552.797 +/- 2.859 | 572.279 +/- 11.379 | 0.966x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 246.127 | 23473.789 | 0.010x | 18.066 +/- 0.150 | 11.666 +/- 0.088 | 1.549x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 810.786 | 25069.888 | 0.032x | 25.955 +/- 0.297 | 17.451 +/- 0.364 | 1.487x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 247.229 | 22073.808 | 0.011x | 22.787 +/- 0.188 | 13.553 +/- 0.127 | 1.681x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 200.398 | 29158.886 | 0.007x | 19.463 +/- 0.143 | 15.167 +/- 0.622 | 1.283x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 217.584 | 25857.769 | 0.008x | 21.384 +/- 0.119 | 15.686 +/- 0.225 | 1.363x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 813.500 | 25659.179 | 0.032x | 27.571 +/- 0.204 | 18.014 +/- 0.146 | 1.531x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8443.658 | 30835.866 | 0.274x | 543.114 +/- 6.517 | 534.797 +/- 3.464 | 1.016x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 234.079 | 27374.914 | 0.009x | 22.279 +/- 0.132 | 12.712 +/- 0.073 | 1.753x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 829.198 | 24753.189 | 0.033x | 29.520 +/- 0.227 | 19.007 +/- 0.382 | 1.553x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 266.227 | 23108.162 | 0.012x | 33.045 +/- 0.213 | 17.369 +/- 0.264 | 1.902x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 262.401 | 24135.175 | 0.011x | 33.039 +/- 0.112 | 17.157 +/- 0.285 | 1.926x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 299.077 | 26007.947 | 0.011x | 36.794 +/- 0.108 | 17.125 +/- 0.407 | 2.149x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 861.493 | 27335.285 | 0.032x | 44.832 +/- 0.244 | 22.483 +/- 0.286 | 1.994x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8548.978 | 35478.514 | 0.241x | 560.596 +/- 2.360 | 539.083 +/- 5.706 | 1.040x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 322.608 | 24804.834 | 0.013x | 37.799 +/- 0.209 | 15.570 +/- 0.092 | 2.428x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 922.119 | 26418.378 | 0.035x | 48.986 +/- 0.325 | 22.972 +/- 0.274 | 2.132x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 329.138 | 25453.328 | 0.013x | 39.185 +/- 0.295 | 17.152 +/- 0.466 | 2.285x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 305.382 | 26141.940 | 0.012x | 34.082 +/- 0.112 | 17.808 +/- 0.490 | 1.914x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 324.701 | 24579.312 | 0.013x | 38.386 +/- 0.231 | 17.914 +/- 0.533 | 2.143x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 947.142 | 24808.478 | 0.038x | 47.110 +/- 0.351 | 22.774 +/- 0.456 | 2.069x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9384.786 | 32744.189 | 0.287x | 586.900 +/- 5.152 | 571.221 +/- 4.478 | 1.027x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 371.736 | 26145.301 | 0.014x | 39.342 +/- 0.142 | 16.036 +/- 0.134 | 2.453x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 969.216 | 25844.576 | 0.038x | 54.018 +/- 0.373 | 23.570 +/- 0.285 | 2.292x | `fedf1f495675c5ac` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 310.224 | 26708.157 | 0.012x | 39.342 +/- 0.361 | 16.860 +/- 0.257 | 2.333x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 889.870 | 26295.653 | 0.034x | 44.746 +/- 0.235 | 21.947 +/- 0.402 | 2.039x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9147.136 | 35220.814 | 0.260x | 587.437 +/- 3.340 | 591.557 +/- 8.859 | 0.993x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 947.593 | 30867.891 | 0.031x | 44.139 +/- 0.427 | 22.502 +/- 0.323 | 1.962x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 891.403 | 29112.312 | 0.031x | 42.704 +/- 0.202 | 22.941 +/- 0.519 | 1.861x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 280.389 | 28306.113 | 0.010x | 38.930 +/- 0.223 | 15.374 +/- 0.230 | 2.532x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 1038.640 | 29662.604 | 0.035x | 63.930 +/- 0.299 | 22.770 +/- 0.422 | 2.808x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 243.814 | 32912.273 | 0.007x | 28.426 +/- 0.179 | 14.781 +/- 0.170 | 1.923x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 904.764 | 28441.938 | 0.032x | 33.411 +/- 0.202 | 19.140 +/- 0.195 | 1.746x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9231.118 | 33743.520 | 0.274x | 573.545 +/- 3.978 | 566.833 +/- 5.809 | 1.012x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 885.755 | 27584.888 | 0.032x | 32.351 +/- 0.227 | 19.336 +/- 0.293 | 1.673x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 828.754 | 30915.870 | 0.027x | 33.479 +/- 0.201 | 19.147 +/- 0.312 | 1.749x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 241.815 | 26722.163 | 0.009x | 28.792 +/- 0.230 | 13.451 +/- 0.238 | 2.141x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 957.974 | 30145.252 | 0.032x | 51.088 +/- 0.606 | 20.221 +/- 0.651 | 2.527x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 234.745 | 26268.926 | 0.009x | 28.547 +/- 0.188 | 15.194 +/- 0.156 | 1.879x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 843.115 | 27167.609 | 0.031x | 34.125 +/- 0.388 | 19.604 +/- 0.352 | 1.741x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9110.476 | 37941.095 | 0.240x | 586.078 +/- 4.269 | 571.550 +/- 6.174 | 1.025x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 900.442 | 27549.102 | 0.033x | 33.480 +/- 0.395 | 19.400 +/- 0.495 | 1.726x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 864.322 | 24194.804 | 0.036x | 33.736 +/- 0.453 | 19.559 +/- 0.549 | 1.725x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 232.427 | 25239.480 | 0.009x | 28.548 +/- 0.121 | 13.275 +/- 0.084 | 2.150x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 893.997 | 24485.655 | 0.037x | 52.091 +/- 0.195 | 20.280 +/- 0.333 | 2.569x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 250.538 | 27373.771 | 0.009x | 28.634 +/- 0.178 | 14.461 +/- 0.200 | 1.980x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 839.274 | 24939.336 | 0.034x | 33.651 +/- 0.258 | 18.572 +/- 0.319 | 1.812x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8576.758 | 34233.690 | 0.251x | 573.336 +/- 12.604 | 532.245 +/- 3.708 | 1.077x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 977.403 | 24782.343 | 0.039x | 34.763 +/- 1.854 | 18.157 +/- 0.158 | 1.915x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 1160.234 | 27187.364 | 0.043x | 40.851 +/- 3.320 | 18.242 +/- 0.095 | 2.239x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 294.500 | 24103.482 | 0.012x | 28.471 +/- 0.201 | 12.993 +/- 0.151 | 2.191x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 893.661 | 29026.916 | 0.031x | 51.022 +/- 0.466 | 18.898 +/- 0.330 | 2.700x | `2bd384aefcaaa397` |
| `cpu_float32_guard_shape_change` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 246.938 | 23756.602 | 0.010x | 29.111 +/- 0.212 | 15.068 +/- 0.160 | 1.932x | `407591794ffc1f3c` |
| `cpu_float32_guard_stride_change` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 258.010 | 23613.421 | 0.011x | 29.110 +/- 0.146 | 15.236 +/- 0.126 | 1.911x | `72013c6beee3a86d` |
| `cpu_float32_guard_requires_grad_change` | `case_default` | 1 | 256 | shape (2, 2), stride (2, 1), offset 0, torch.float32, cpu, requires_grad=False | 210.388 | 27552.626 | 0.008x | 23.729 +/- 0.150 | 14.031 +/- 0.237 | 1.691x | `e2672b7a55a70931` |
| `cpu_float32_guard_two_input_metadata_change` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 250.815 | 25659.224 | 0.010x | 28.312 +/- 0.223 | 14.347 +/- 0.214 | 1.973x | `80e86bb0726f7270` |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native
`torch_rs` path currently has executable public cases for tensor arithmetic,
broadcasting, and recompilation guards. Every remaining category below stays
in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_guard_shape_change`, `cpu_float32_guard_stride_change`, `cpu_float32_guard_requires_grad_change`, `cpu_float32_guard_two_input_metadata_change` |
| `modules_parameters_buffers` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `inference` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `training_autograd` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `python_control_flow` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `graph_breaks_fullgraph` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dynamic_shapes_symbolics` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `mutation_aliasing_views` | 8 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `containers_pytrees` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `decompositions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `custom_functions` | 6 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dtype_device_transitions` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 24 / 100. Zero-credit unsupported category weight: 76 / 100.
The v4 corpus also keeps 2 held-out broadcasting programs and 2 held-out recompilation-guard metadata scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
