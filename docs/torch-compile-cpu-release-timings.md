# `torch.compile` Eager CPU Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on
`c6d27187fd0c621d4cf5fff42bf60419dabbe311`, plus the worktree changes that port the reviewed
two-input broadcasting compile implementation from PR #1799 / `d836383` and
refresh this benchmark driver/report for corpus v3.

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
cp -a /home/bobren/.cargo/registry/cache \
  /home/bobren/.cargo/registry/index \
  /home/bobren/.cargo/registry/src target/cargo-home/registry/
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked --offline
env -u CONDA_PREFIX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_top_level_compile tests.test_compile_corpus tests.test_readme_quickstart
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo clippy --locked --offline --all-targets -- -D warnings
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --doc
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
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
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python scripts/benchmark_compile_cpu.py \
  --require-single-cpu-affinity \
  --output target/compile-cpu-release-timings.json
```

Checks run for this evidence:

```bash
env -u CONDA_PREFIX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_top_level_compile tests.test_compile_corpus tests.test_readme_quickstart
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo clippy --locked --offline --all-targets -- -D warnings
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --all-targets
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --doc
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
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
differential, and docs smoke tests passed 71 tests. The repository-managed
pinned PyTorch 2.13 full Python suite passed 4703 tests with 3 skips. The
default Rust suite passed 310 tests across unit and integration targets plus
0 doctests; the Python-bindings Rust suite passed 321 tests. Default Clippy
and Python-bindings Clippy both passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_9f374bb9/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_9f374bb9/.venv/lib/python3.12/site-packages/torch_rs/__init__.py`
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
- Dependency installation: locked `uv sync` resolved in 30 ms, prepared
  packages in 16.43s, and installed in 1.29s
- Build time: first successful Python 3.12 release extension build completed
  in 37.63s; the full-suite wheel build reused release artifacts and
  completed in 0.02s

The benchmark uses the checked-in `torch_compile_corpus_v3` programs. The
timed supported set contains every public native compile case: five
one-input tensor-arithmetic programs and four two-input broadcasting
programs. One-input programs run across the corpus default input plus scalar,
vector, row-major matrix, larger row-major matrix, empty, and non-contiguous
transpose inputs. Two-input programs run across the corpus default input plus
row-major matrix/vector, larger row-major matrix/vector, tensor/scalar,
scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast
inputs. Inputs are created outside timed regions from deterministic values.
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
All 63 timed cells had matching `torch_rs` and PyTorch checksums.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Cold first compiled call: 0.024x uncapped, 0.114x capped
- Steady-state materialized compiled call: 0.719x uncapped, 0.719x capped
- Timed supported cells: 63 (35 one-input, 28 two-input broadcasting)
- Versioned denominator coverage: 20.0% supported by native compile cases, 80% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 277.800 | 88648.233 | 0.003x | 7.428 +/- 0.039 | 13.736 +/- 0.128 | 0.541x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 203.648 | 25246.007 | 0.008x | 6.828 +/- 0.035 | 13.330 +/- 0.146 | 0.512x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 220.989 | 26075.156 | 0.008x | 7.186 +/- 0.043 | 13.454 +/- 0.225 | 0.534x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 797.280 | 25924.894 | 0.031x | 12.232 +/- 0.156 | 18.179 +/- 0.146 | 0.673x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8629.785 | 32784.381 | 0.263x | 527.182 +/- 2.533 | 537.349 +/- 4.274 | 0.981x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 240.699 | 29132.351 | 0.008x | 7.338 +/- 0.120 | 12.774 +/- 0.052 | 0.574x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 801.878 | 26418.160 | 0.030x | 13.750 +/- 0.298 | 18.878 +/- 0.157 | 0.728x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 178.885 | 26025.450 | 0.007x | 7.791 +/- 0.180 | 12.147 +/- 0.134 | 0.641x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 161.505 | 24993.100 | 0.006x | 6.436 +/- 0.042 | 11.917 +/- 0.085 | 0.540x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 177.348 | 25852.024 | 0.007x | 6.744 +/- 0.034 | 11.821 +/- 0.188 | 0.570x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 793.726 | 26440.484 | 0.030x | 11.955 +/- 0.078 | 16.854 +/- 0.166 | 0.709x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8794.524 | 32369.393 | 0.272x | 550.783 +/- 4.788 | 549.069 +/- 6.749 | 1.003x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 216.532 | 24628.487 | 0.009x | 6.932 +/- 0.036 | 11.642 +/- 0.079 | 0.595x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 788.793 | 26876.258 | 0.029x | 12.693 +/- 0.099 | 16.971 +/- 0.142 | 0.748x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 202.311 | 25285.176 | 0.008x | 7.341 +/- 0.046 | 13.856 +/- 0.229 | 0.530x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 180.588 | 23265.601 | 0.008x | 7.062 +/- 0.108 | 13.408 +/- 0.112 | 0.527x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 215.586 | 24776.177 | 0.009x | 7.066 +/- 0.032 | 13.521 +/- 0.176 | 0.523x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 795.268 | 27694.771 | 0.029x | 12.185 +/- 0.087 | 18.616 +/- 0.371 | 0.655x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8453.343 | 34762.193 | 0.243x | 593.973 +/- 17.300 | 554.239 +/- 8.875 | 1.072x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 266.593 | 25611.020 | 0.010x | 7.610 +/- 0.044 | 12.967 +/- 0.078 | 0.587x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 899.651 | 24364.709 | 0.037x | 12.861 +/- 0.218 | 18.779 +/- 0.180 | 0.685x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 241.946 | 27370.891 | 0.009x | 8.131 +/- 0.276 | 17.254 +/- 0.141 | 0.471x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 257.405 | 24994.221 | 0.010x | 8.073 +/- 0.199 | 17.490 +/- 0.386 | 0.462x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 267.173 | 25296.357 | 0.011x | 8.696 +/- 0.116 | 17.208 +/- 0.279 | 0.505x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 826.660 | 28109.772 | 0.029x | 14.552 +/- 0.558 | 22.323 +/- 0.193 | 0.652x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8507.560 | 37334.553 | 0.228x | 531.368 +/- 5.311 | 539.823 +/- 3.536 | 0.984x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 289.938 | 25042.908 | 0.012x | 10.422 +/- 0.854 | 15.772 +/- 0.097 | 0.661x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 854.608 | 25527.351 | 0.033x | 14.006 +/- 0.092 | 22.925 +/- 0.121 | 0.611x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 303.709 | 24005.709 | 0.013x | 8.655 +/- 0.165 | 15.950 +/- 0.101 | 0.543x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 286.624 | 29023.635 | 0.010x | 7.951 +/- 0.031 | 17.324 +/- 0.211 | 0.459x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 325.653 | 27489.410 | 0.012x | 8.671 +/- 0.093 | 16.892 +/- 0.199 | 0.513x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 918.810 | 25634.629 | 0.036x | 14.513 +/- 0.184 | 22.639 +/- 0.180 | 0.641x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8884.707 | 34344.495 | 0.259x | 558.421 +/- 7.659 | 564.995 +/- 5.695 | 0.988x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 332.769 | 24374.632 | 0.014x | 8.585 +/- 0.058 | 15.852 +/- 0.092 | 0.542x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 912.505 | 25969.335 | 0.035x | 15.907 +/- 0.112 | 23.414 +/- 0.253 | 0.679x | `fedf1f495675c5ac` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 256.643 | 25683.117 | 0.010x | 10.615 +/- 0.085 | 17.052 +/- 0.189 | 0.623x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 882.760 | 26821.459 | 0.033x | 16.077 +/- 0.151 | 21.806 +/- 0.238 | 0.737x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8935.683 | 35928.401 | 0.249x | 558.363 +/- 5.243 | 562.998 +/- 3.831 | 0.992x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 889.370 | 27172.601 | 0.033x | 15.577 +/- 0.195 | 21.683 +/- 0.250 | 0.718x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 920.067 | 26437.514 | 0.035x | 15.440 +/- 0.152 | 21.589 +/- 0.116 | 0.715x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 241.276 | 26132.622 | 0.009x | 10.331 +/- 0.055 | 15.226 +/- 0.061 | 0.679x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 904.143 | 29576.677 | 0.031x | 33.103 +/- 0.351 | 22.596 +/- 0.154 | 1.465x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 227.049 | 24943.388 | 0.009x | 9.926 +/- 0.092 | 14.943 +/- 0.117 | 0.664x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 827.992 | 24948.426 | 0.033x | 14.673 +/- 0.104 | 19.229 +/- 0.173 | 0.763x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8917.225 | 35470.859 | 0.251x | 551.498 +/- 2.744 | 555.576 +/- 2.287 | 0.993x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 878.083 | 25897.812 | 0.034x | 14.415 +/- 0.152 | 19.293 +/- 0.204 | 0.747x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 829.971 | 26998.186 | 0.031x | 14.378 +/- 0.070 | 19.181 +/- 0.194 | 0.750x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 213.829 | 26649.809 | 0.008x | 9.677 +/- 0.105 | 13.350 +/- 0.109 | 0.725x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 877.493 | 27766.579 | 0.032x | 33.608 +/- 0.490 | 19.618 +/- 0.162 | 1.713x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 225.361 | 27335.492 | 0.008x | 9.786 +/- 0.060 | 15.345 +/- 0.140 | 0.638x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 856.000 | 28276.711 | 0.030x | 14.960 +/- 0.081 | 19.276 +/- 0.124 | 0.776x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8931.336 | 35358.113 | 0.253x | 566.086 +/- 5.941 | 573.977 +/- 3.921 | 0.986x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 869.795 | 27263.688 | 0.032x | 14.461 +/- 0.100 | 19.501 +/- 0.335 | 0.742x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 816.355 | 25574.323 | 0.032x | 14.504 +/- 0.108 | 19.458 +/- 0.251 | 0.745x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 213.558 | 28192.834 | 0.008x | 9.676 +/- 0.076 | 13.528 +/- 0.118 | 0.715x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 865.604 | 26936.984 | 0.032x | 31.945 +/- 0.287 | 20.125 +/- 0.148 | 1.587x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 225.115 | 28729.054 | 0.008x | 9.810 +/- 0.079 | 14.403 +/- 0.167 | 0.681x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 793.190 | 26169.238 | 0.030x | 14.713 +/- 0.099 | 18.378 +/- 0.197 | 0.801x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8441.540 | 35883.533 | 0.235x | 534.388 +/- 5.135 | 534.746 +/- 3.119 | 0.999x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 859.240 | 25617.828 | 0.034x | 14.397 +/- 0.080 | 18.403 +/- 0.153 | 0.782x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 834.998 | 23706.873 | 0.035x | 14.709 +/- 0.200 | 18.601 +/- 0.171 | 0.791x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 224.239 | 23808.712 | 0.009x | 9.834 +/- 0.100 | 12.965 +/- 0.080 | 0.759x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 886.667 | 27388.016 | 0.032x | 32.325 +/- 0.349 | 19.072 +/- 0.164 | 1.695x | `2bd384aefcaaa397` |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native
`torch_rs` path currently has executable public cases only for tensor
arithmetic and broadcasting. Every remaining category below stays in the
denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
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
| `recompilation_guards` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |
| `dtype_device_transitions` | 4 | Zero credit: no native torch_rs eager/fullgraph compile cases are implemented for this category in the checked-in corpus |

Supported category weight: 20 / 100. Zero-credit unsupported category weight: 80 / 100.
The v3 corpus also keeps 2 held-out broadcasting programs in tests to guard against case-specific specialization; they are not included in the public timing table.
