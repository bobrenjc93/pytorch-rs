# `torch.compile` Eager CPU Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on `a8b28c9235d03a14be1498c787d00f65712faaeb`, plus the worktree changes that add recompilation-guard corpus coverage and refresh this benchmark driver/report for corpus v4.

Exact setup, build, check, and timing commands were run from the repository root. The reusable timing driver is checked in as `scripts/benchmark_compile_cpu.py` and emitted JSON under `target/compile-cpu-release-timings.json`. The active Conda environment held ambient PyTorch 2.14, so the PyTorch 2.13 reference evidence used this worktree's local `.venv`; uv and Cargo state were redirected under `target/`. The local Cargo cache was populated from the existing read-only user cache because the crates.io proxy rejected direct unauthenticated fetches during this run.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" uv venv --clear --python 3.12 .venv
env UV_CACHE_DIR="$PWD/target/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" uv sync --locked --python "$PWD/.venv/bin/python" --no-install-project --group dev --group reference
mkdir -p target/cargo-home/registry
cp -a /home/bobren/.cargo/registry/cache /home/bobren/.cargo/registry/index /home/bobren/.cargo/registry/src target/cargo-home/registry/
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true TMPDIR="$PWD/target" VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" .venv/bin/maturin develop --release --locked
env -u CONDA_PREFIX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest tests.test_compile_corpus tests.test_top_level_compile tests.test_readme_quickstart
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true cargo clippy --locked --all-targets -- -D warnings
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true cargo test --locked --all-targets
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true cargo test --locked --doc
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --locked --all-targets --features python-bindings -- -D warnings
mkdir -p target/pyo3
printf "%s\n" implementation=CPython version=3.12 shared=true abi3=true lib_name=python3.12 lib_dir=/usr/local/fbcode/platform010/lib executable="$PWD/.venv/bin/python" pointer_width=64 build_flags= suppress_build_script_link_lines=false > target/pyo3/config.txt
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true VIRTUAL_ENV="$PWD/.venv" PYO3_CONFIG_FILE="$PWD/target/pyo3/config.txt" PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --locked --all-targets --features python-bindings
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true UV_CACHE_DIR="$PWD/target/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" ./scripts/test-python.sh
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= taskset -c 24 .venv/bin/python scripts/benchmark_compile_cpu.py --require-single-cpu-affinity --output target/compile-cpu-release-timings.json
```

Checks run for this evidence:

```bash
env -u CONDA_PREFIX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest tests.test_compile_corpus tests.test_top_level_compile tests.test_readme_quickstart
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true cargo fmt --check
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true cargo clippy --locked --all-targets -- -D warnings
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true cargo test --locked --all-targets
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true cargo test --locked --doc
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --locked --all-targets --features python-bindings -- -D warnings
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true VIRTUAL_ENV="$PWD/.venv" PYO3_CONFIG_FILE="$PWD/target/pyo3/config.txt" PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --locked --all-targets --features python-bindings
env CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target" CARGO_NET_OFFLINE=true UV_CACHE_DIR="$PWD/target/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" ./scripts/test-python.sh
```

Results: the focused public compile, compile corpus, PyTorch 2.13 differential, guard-sequence, and docs smoke tests passed 75 tests. The repository-managed pinned PyTorch 2.13 full Python suite passed 4707 tests with 3 skips. The default Rust suite passed 310 tests across unit and integration targets plus 0 doctests; the Python-bindings Rust suite passed 321 tests. Default Clippy and Python-bindings Clippy both passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux-6.13.2-0_fbk12_0_g0b66b3635210-x86_64-with-glibc2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_05f9d790/.venv/lib/python3.12/site-packages/torch/__init__.py`
- PyTorch CUDA runtime: 13.0; CUDA availability disabled for CPU timing with `CUDA_VISIBLE_DEVICES=`
- `torch_rs`: 0.1.0 from `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_05f9d790/python/torch_rs/__init__.py`
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
- Build time: initial release extension build completed in 37.49s; the
  full-suite release wheel build completed in 38.47s; the final editable
  release build reused local artifacts and completed in 0.02s

The benchmark uses the checked-in `torch_compile_corpus_v4` programs. The timed supported set contains every public native compile case: five one-input tensor-arithmetic programs, four two-input broadcasting programs, and three recompilation-guard programs. One-input programs run across the corpus default input plus scalar, vector, row-major matrix, larger row-major matrix, empty, and non-contiguous transpose inputs. Two-input programs run across the corpus default input plus row-major matrix/vector, larger row-major matrix/vector, tensor/scalar, scalar/tensor, empty broadcast, and non-contiguous matrix/vector broadcast inputs. Recompilation-guard programs run across shape, stride, and `requires_grad` metadata variants; separate guard-sequence rows exercise cache reuse, bounded `recompile_limit` behavior, `torch.compiler.reset()` semantics, and both implementation orders. Inputs are created outside timed regions from deterministic values.

For PyTorch, the driver requires pinned PyTorch 2.13 and uses stock `torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the native guarded eager/fullgraph path. Both implementations run in both orders: `torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant compiler state for cold timing, measures the first materialized compiled call separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured block repeats the operation according to the table's `Repeats` column; medians below are microseconds per compiled call. The CPU workload has no asynchronous device queue, but the driver still calls synchronization hooks when an implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, shape, stride, storage offset, contiguity, dtype, device, and `requires_grad` against the same eager program. The `torch_rs` result is also checked against the PyTorch result. After every warmup and measured block, the driver materializes the last output and records a 64-bit BLAKE2b checksum over values and metadata. All 84 timed cells had matching `torch_rs` and PyTorch checksums.

Benchmark integrity gate: pass for the >=99 requirement. The evidence is generated by the reusable fixed-affinity driver, uses equivalent work in both implementation orders, pins the reference version, materializes and checks outputs instead of timing dead code, keeps held-out corpus cases in differential tests, validates guard sequences separately from timed cells, and retains every unsupported category in the explicit zero-credit denominator.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Cold first compiled call: 0.026x uncapped, 0.115x capped
- Steady-state materialized compiled call: 1.704x uncapped, 1.704x capped
- Timed supported cells: 84 (35 tensor-arithmetic, 28 broadcasting, 21 recompilation-guard)
- Recompilation guard sequences: 12 rows, 60 checked steps, statuses expected_error, ok
- Versioned denominator coverage: 24.0% supported by native compile cases, 76% zero-credit unsupported category weight

## Supported Timed Cells

| Program | Input variant | Inputs | Repeats | Output metadata | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 349.794 | 78836.061 | 0.004x | 22.621 +/- 0.214 | 13.544 +/- 0.109 | 1.670x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 206.632 | 26444.816 | 0.008x | 19.501 +/- 0.110 | 13.535 +/- 0.132 | 1.441x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 233.303 | 24834.285 | 0.009x | 22.271 +/- 0.873 | 13.364 +/- 0.111 | 1.667x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 813.035 | 23464.913 | 0.035x | 27.833 +/- 0.564 | 18.653 +/- 0.209 | 1.492x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9853.735 | 31006.032 | 0.318x | 609.221 +/- 4.942 | 683.308 +/- 9.773 | 0.892x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 274.104 | 25966.467 | 0.011x | 21.891 +/- 0.155 | 12.921 +/- 0.078 | 1.694x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 889.380 | 24731.224 | 0.036x | 29.105 +/- 0.162 | 19.143 +/- 0.129 | 1.520x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 222.111 | 22157.446 | 0.010x | 18.058 +/- 0.164 | 12.388 +/- 0.146 | 1.458x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 167.624 | 22523.930 | 0.007x | 15.336 +/- 0.070 | 12.174 +/- 0.122 | 1.260x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 193.121 | 22425.582 | 0.009x | 17.032 +/- 0.092 | 11.956 +/- 0.184 | 1.425x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 810.891 | 23165.415 | 0.035x | 23.078 +/- 0.166 | 17.104 +/- 0.235 | 1.349x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8948.746 | 31480.900 | 0.284x | 562.844 +/- 3.301 | 561.927 +/- 11.113 | 1.002x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 206.498 | 23078.459 | 0.009x | 17.639 +/- 0.059 | 11.732 +/- 0.064 | 1.504x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 824.707 | 27940.305 | 0.030x | 25.139 +/- 0.124 | 17.199 +/- 0.145 | 1.462x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 226.002 | 24692.806 | 0.009x | 22.207 +/- 0.088 | 13.627 +/- 0.120 | 1.630x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 201.410 | 22578.838 | 0.009x | 19.151 +/- 0.056 | 13.468 +/- 0.120 | 1.422x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 217.770 | 25174.841 | 0.009x | 21.348 +/- 0.088 | 13.759 +/- 0.112 | 1.552x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 795.308 | 26317.664 | 0.030x | 27.450 +/- 0.166 | 18.278 +/- 0.149 | 1.502x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8451.669 | 33555.810 | 0.252x | 538.581 +/- 2.554 | 533.316 +/- 4.908 | 1.010x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 235.842 | 25281.643 | 0.009x | 21.858 +/- 0.079 | 12.907 +/- 0.089 | 1.693x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 817.396 | 23528.765 | 0.035x | 29.088 +/- 0.213 | 19.019 +/- 0.133 | 1.529x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 1 | 256 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 267.704 | 22460.789 | 0.012x | 32.749 +/- 0.192 | 16.880 +/- 0.139 | 1.940x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 260.539 | 25844.014 | 0.010x | 32.868 +/- 0.166 | 17.183 +/- 0.208 | 1.913x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 281.961 | 26269.771 | 0.011x | 37.410 +/- 1.028 | 17.631 +/- 0.294 | 2.122x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 905.926 | 26297.885 | 0.034x | 44.779 +/- 0.547 | 22.419 +/- 0.154 | 1.997x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8580.204 | 32105.551 | 0.267x | 567.894 +/- 6.573 | 539.578 +/- 3.603 | 1.052x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 317.090 | 24205.088 | 0.013x | 37.139 +/- 0.102 | 15.945 +/- 0.065 | 2.329x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 911.909 | 24836.660 | 0.037x | 48.232 +/- 0.231 | 23.139 +/- 0.257 | 2.084x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 1 | 256 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 313.745 | 23067.688 | 0.014x | 38.895 +/- 0.168 | 16.094 +/- 0.094 | 2.417x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 342.954 | 23258.111 | 0.015x | 34.117 +/- 0.140 | 17.677 +/- 0.208 | 1.930x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 332.338 | 24744.835 | 0.013x | 38.202 +/- 0.201 | 17.407 +/- 0.269 | 2.195x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 940.623 | 26026.649 | 0.036x | 46.614 +/- 0.373 | 22.976 +/- 0.200 | 2.029x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8870.529 | 33045.217 | 0.268x | 592.600 +/- 9.609 | 568.385 +/- 4.893 | 1.043x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 348.527 | 26167.654 | 0.013x | 38.951 +/- 0.095 | 16.030 +/- 0.061 | 2.430x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 959.542 | 25276.239 | 0.038x | 52.956 +/- 0.304 | 23.634 +/- 0.157 | 2.241x | `fedf1f495675c5ac` |
| `cpu_float32_matrix_vector_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 299.568 | 25273.120 | 0.012x | 38.938 +/- 0.186 | 16.817 +/- 0.180 | 2.315x | `98a179ecb42242f2` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 906.742 | 24896.109 | 0.036x | 44.629 +/- 0.291 | 22.246 +/- 0.154 | 2.006x | `ad5274b06474f25a` |
| `cpu_float32_matrix_vector_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9012.439 | 33536.485 | 0.269x | 588.974 +/- 3.004 | 566.829 +/- 4.764 | 1.039x | `2d29b8c5db7cf3a3` |
| `cpu_float32_matrix_vector_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 950.663 | 25179.758 | 0.038x | 43.135 +/- 0.251 | 21.920 +/- 0.190 | 1.968x | `789e567fe16ee50d` |
| `cpu_float32_matrix_vector_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 886.406 | 24504.471 | 0.036x | 42.466 +/- 0.217 | 21.890 +/- 0.212 | 1.940x | `fd2a8cc8274a95a3` |
| `cpu_float32_matrix_vector_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 283.955 | 25455.109 | 0.011x | 38.466 +/- 0.170 | 15.002 +/- 0.082 | 2.564x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 993.548 | 28125.514 | 0.035x | 66.994 +/- 0.455 | 22.784 +/- 0.195 | 2.940x | `dba903ec40510312` |
| `cpu_float32_matrix_vector_add_method` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 252.136 | 24563.080 | 0.010x | 28.614 +/- 0.253 | 14.775 +/- 0.151 | 1.937x | `0d899ef0331555c3` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 861.383 | 23843.612 | 0.036x | 33.297 +/- 0.304 | 19.391 +/- 0.167 | 1.717x | `a50cc7734a507f4b` |
| `cpu_float32_matrix_vector_add_method` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9068.128 | 32014.838 | 0.283x | 574.150 +/- 4.667 | 560.489 +/- 6.395 | 1.024x | `7f09321c9dd8f431` |
| `cpu_float32_matrix_vector_add_method` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 870.842 | 24145.293 | 0.036x | 31.890 +/- 0.191 | 19.432 +/- 0.195 | 1.641x | `d14229933b8a4e37` |
| `cpu_float32_matrix_vector_add_method` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 824.863 | 23641.060 | 0.035x | 33.105 +/- 0.130 | 19.082 +/- 0.147 | 1.735x | `5bf5343414da1f5c` |
| `cpu_float32_matrix_vector_add_method` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 264.294 | 22941.752 | 0.012x | 28.142 +/- 0.154 | 13.392 +/- 0.093 | 2.101x | `e99a6c9902c3119e` |
| `cpu_float32_matrix_vector_add_method` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 902.080 | 24475.427 | 0.037x | 53.524 +/- 0.282 | 19.951 +/- 0.174 | 2.683x | `ea3197d484cde28e` |
| `cpu_float32_tensor_scalar_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 271.471 | 23308.853 | 0.012x | 28.430 +/- 0.187 | 15.316 +/- 0.210 | 1.856x | `5b94f7e5a6a718c6` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 843.877 | 24160.787 | 0.035x | 33.820 +/- 0.216 | 19.807 +/- 0.345 | 1.707x | `82c540110f39c215` |
| `cpu_float32_tensor_scalar_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9054.698 | 32447.908 | 0.279x | 584.489 +/- 4.236 | 572.434 +/- 6.976 | 1.021x | `689c76d673bbbf07` |
| `cpu_float32_tensor_scalar_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 877.938 | 25517.338 | 0.034x | 32.940 +/- 0.144 | 19.549 +/- 0.173 | 1.685x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 815.904 | 26406.158 | 0.031x | 33.640 +/- 0.280 | 19.645 +/- 0.362 | 1.712x | `fd2a8cc8274a95a3` |
| `cpu_float32_tensor_scalar_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 222.117 | 24040.535 | 0.009x | 28.584 +/- 0.105 | 13.876 +/- 0.212 | 2.060x | `e99a6c9902c3119e` |
| `cpu_float32_tensor_scalar_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 915.079 | 25717.727 | 0.036x | 54.692 +/- 0.383 | 20.385 +/- 0.140 | 2.683x | `79703a9e62d5f513` |
| `cpu_float32_scalar_tensor_add` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=True | 248.912 | 23162.171 | 0.011x | 28.456 +/- 0.182 | 14.924 +/- 0.153 | 1.907x | `48c8ec8bd2aa6e72` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 833.567 | 23037.712 | 0.036x | 33.101 +/- 0.239 | 18.533 +/- 0.182 | 1.786x | `32e11c81cc753c53` |
| `cpu_float32_scalar_tensor_add` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 8633.461 | 31084.431 | 0.278x | 554.144 +/- 2.508 | 542.404 +/- 3.165 | 1.022x | `2833a8dd1f6e9453` |
| `cpu_float32_scalar_tensor_add` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 894.913 | 24418.852 | 0.037x | 32.061 +/- 0.209 | 18.556 +/- 0.138 | 1.728x | `d14229933b8a4e37` |
| `cpu_float32_scalar_tensor_add` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 836.255 | 24175.378 | 0.035x | 33.300 +/- 0.318 | 18.576 +/- 0.146 | 1.793x | `c86610390c9eadb5` |
| `cpu_float32_scalar_tensor_add` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 248.376 | 23804.362 | 0.010x | 28.068 +/- 0.119 | 13.202 +/- 0.143 | 2.126x | `e99a6c9902c3119e` |
| `cpu_float32_scalar_tensor_add` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 939.005 | 28226.082 | 0.033x | 53.463 +/- 0.340 | 21.762 +/- 0.243 | 2.457x | `2bd384aefcaaa397` |
| `cpu_float32_recompile_guard_unary_metadata` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 252.752 | 25668.256 | 0.010x | 28.651 +/- 0.148 | 17.233 +/- 0.152 | 1.663x | `0e17c6493745a257` |
| `cpu_float32_recompile_guard_unary_metadata` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 243.714 | 24698.970 | 0.010x | 24.387 +/- 0.106 | 17.192 +/- 0.775 | 1.418x | `292485c676f9433a` |
| `cpu_float32_recompile_guard_unary_metadata` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 260.394 | 23053.971 | 0.011x | 27.121 +/- 0.105 | 15.061 +/- 0.251 | 1.801x | `62c3654eb7d82d74` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 639.382 | 30216.552 | 0.021x | 32.320 +/- 0.201 | 18.484 +/- 0.141 | 1.749x | `5d7b4862cd84174c` |
| `cpu_float32_recompile_guard_unary_metadata` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5412.458 | 28246.297 | 0.192x | 358.348 +/- 2.925 | 340.503 +/- 3.698 | 1.052x | `69ce9a45017fa7db` |
| `cpu_float32_recompile_guard_unary_metadata` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 310.710 | 24660.641 | 0.013x | 27.924 +/- 0.120 | 14.180 +/- 0.080 | 1.969x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_unary_metadata` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 669.283 | 28764.356 | 0.023x | 36.259 +/- 0.461 | 18.977 +/- 0.151 | 1.911x | `7af03502688e9f8f` |
| `cpu_float32_recompile_guard_binary_metadata` | `case_default` | 2 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 282.868 | 26107.081 | 0.011x | 33.575 +/- 0.207 | 15.921 +/- 0.085 | 2.109x | `3ee8bcca8b6a65b6` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_31x37_by_37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 875.494 | 28164.266 | 0.031x | 39.640 +/- 0.581 | 20.822 +/- 0.213 | 1.904x | `c92ef12c0bea0b39` |
| `cpu_float32_recompile_guard_binary_metadata` | `matrix_vector_127x131_by_131` | 2 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 9113.426 | 33062.473 | 0.276x | 590.356 +/- 5.009 | 563.944 +/- 4.056 | 1.047x | `5fe26f494117f54c` |
| `cpu_float32_recompile_guard_binary_metadata` | `tensor_scalar_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 942.962 | 24835.342 | 0.038x | 37.839 +/- 0.275 | 20.899 +/- 0.163 | 1.811x | `53f7a4127e94cf26` |
| `cpu_float32_recompile_guard_binary_metadata` | `scalar_tensor_31x37` | 2 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 869.054 | 24580.641 | 0.035x | 39.199 +/- 0.357 | 20.309 +/- 0.127 | 1.930x | `bc7dbda4eb0dc81a` |
| `cpu_float32_recompile_guard_binary_metadata` | `empty_2x0_by_0` | 2 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 290.214 | 23908.944 | 0.012x | 33.060 +/- 0.136 | 14.219 +/- 0.121 | 2.325x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_guard_binary_metadata` | `transpose_31x37_by_37` | 2 | 128 | shape (31, 37), stride (1, 31), offset 0, torch.float32, cpu, requires_grad=False | 946.241 | 28179.971 | 0.034x | 59.837 +/- 0.328 | 21.160 +/- 0.144 | 2.828x | `256365df8d5f4628` |
| `cpu_float32_recompile_limit_reset` | `case_default` | 1 | 256 | shape (2, 3), stride (3, 1), offset 0, torch.float32, cpu, requires_grad=False | 263.809 | 25875.705 | 0.010x | 28.567 +/- 0.181 | 15.077 +/- 0.179 | 1.895x | `9b27d4997fd00973` |
| `cpu_float32_recompile_limit_reset` | `scalar` | 1 | 2048 | shape (), stride (), offset 0, torch.float32, cpu, requires_grad=False | 247.190 | 26370.760 | 0.009x | 24.371 +/- 0.124 | 15.060 +/- 0.184 | 1.618x | `5c2ffe407931c8ee` |
| `cpu_float32_recompile_limit_reset` | `vector_17` | 1 | 1024 | shape (17,), stride (1,), offset 0, torch.float32, cpu, requires_grad=False | 266.909 | 23267.790 | 0.011x | 27.320 +/- 0.134 | 14.769 +/- 0.188 | 1.850x | `d701faefd13d63e3` |
| `cpu_float32_recompile_limit_reset` | `matrix_31x37` | 1 | 128 | shape (31, 37), stride (37, 1), offset 0, torch.float32, cpu, requires_grad=False | 638.886 | 23910.267 | 0.027x | 32.276 +/- 0.122 | 18.320 +/- 0.152 | 1.762x | `fd8f6faa30e6834e` |
| `cpu_float32_recompile_limit_reset` | `matrix_127x131` | 1 | 16 | shape (127, 131), stride (131, 1), offset 0, torch.float32, cpu, requires_grad=False | 5529.569 | 31956.068 | 0.173x | 357.210 +/- 2.101 | 437.820 +/- 13.427 | 0.816x | `89b634c0d077be1b` |
| `cpu_float32_recompile_limit_reset` | `empty_2x0` | 1 | 2048 | shape (2, 0), stride (1, 1), offset 0, torch.float32, cpu, requires_grad=False | 302.237 | 26478.037 | 0.011x | 27.820 +/- 0.091 | 14.092 +/- 0.089 | 1.974x | `e99a6c9902c3119e` |
| `cpu_float32_recompile_limit_reset` | `transpose_37x31` | 1 | 128 | shape (37, 31), stride (1, 37), offset 0, torch.float32, cpu, requires_grad=False | 663.573 | 24792.532 | 0.027x | 35.966 +/- 0.202 | 18.992 +/- 0.115 | 1.894x | `9348bfb9afa1f8c3` |

## Recompilation Guard Sequences

These rows are behavioral evidence, not throughput cells. Each scenario runs once per implementation and once per implementation order. Steps marked `expected_error` are required fullgraph `recompile_limit` failures; the following cached call and reset call verify bounded-cache and reset semantics.

| Scenario | Order | Implementation | Limit | Steps | Total us |
| --- | --- | --- | ---: | --- | ---: |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 828.494 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 739.500 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 525.454 |
| `unary_shape_stride_requires_grad_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 182578.057 |
| `binary_argument_metadata_guards` | `torch_rs,pytorch` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 139321.656 |
| `bounded_limit_then_reset` | `torch_rs,pytorch` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 81081.931 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 122231.323 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `pytorch` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 131108.863 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `pytorch` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: FailOnRecompileLimitHit); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 84136.831 |
| `unary_shape_stride_requires_grad_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); shape_change ok(shape); stride_change ok(stride); requires_grad_change ok(requires_grad) | 989.157 |
| `binary_argument_metadata_guards` | `pytorch,torch_rs` | `torch_rs` | 4 | base ok(initial); same_metadata ok(same_metadata); left_stride_change ok(stride); right_shape_change ok(shape); right_requires_grad_change ok(requires_grad) | 745.888 |
| `bounded_limit_then_reset` | `pytorch,torch_rs` | `torch_rs` | 2 | base ok(initial); shape_change ok(shape); limit_rejects_stride_change expected_error(recompile_limit: CompileTraceUnsupportedError); cached_base_after_limit ok(same_metadata); reset_allows_stride_change ok(reset) | 544.045 |

## Zero-Credit Unsupported Denominator

The compile corpus keeps the full 100-point category denominator. The native `torch_rs` path currently has executable public cases for tensor arithmetic, broadcasting, and recompilation guards. Every remaining category below stays in the denominator as zero credit instead of being dropped from the report.

| Category | Weight | Accounting |
| --- | ---: | --- |
| `tensor_arithmetic` | 12 | Supported and timed public cases: `cpu_float32_unary_abs_neg`, `cpu_float32_self_add`, `cpu_float32_abs_neg_reordered`, `cpu_float32_repeated_unary_chain`, `cpu_float32_add_unary_composition` |
| `broadcasting` | 8 | Supported and timed public cases: `cpu_float32_matrix_vector_add`, `cpu_float32_matrix_vector_add_method`, `cpu_float32_tensor_scalar_add`, `cpu_float32_scalar_tensor_add` |
| `recompilation_guards` | 4 | Supported and timed public cases: `cpu_float32_recompile_guard_unary_metadata`, `cpu_float32_recompile_guard_binary_metadata`, `cpu_float32_recompile_limit_reset` |
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
The v4 corpus also keeps 2 held-out broadcasting programs and 2 held-out recompilation-guard scenarios in tests to guard against case-specific specialization; they are not included in the public timing table.
