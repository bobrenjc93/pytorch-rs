# `torch.compile` Eager CPU Release Timings

Date: 2026-09-03

Candidate provenance: source snapshot based on
`da5538ee42b013586e602c3ae93e23ecd8ab9d2f`, plus the worktree changes that
port the guarded bytecode frontend from PR #1796 / `c47c97c`, add per-wrapper
compile graph caching, and add this benchmark driver/report.

Exact setup, build, check, and timing commands were run from the repository
root. The reusable timing driver is checked in as
`scripts/benchmark_compile_cpu.py` and emitted JSON under
`target/compile-cpu-release-timings.json`. The active Conda environment held
ambient PyTorch 2.14, so the PyTorch 2.13 reference evidence used a
worktree-local `.venv`; uv and Cargo state were redirected under `target/`.

```bash
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv venv --clear --python 3.12 .venv
env UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  uv sync --locked --python "$PWD/.venv/bin/python" \
  --no-install-project --group dev --group reference
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  cargo fetch --locked
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  TMPDIR="$PWD/target" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
env -u CONDA_PREFIX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m unittest \
  tests.test_top_level_compile tests.test_compile_corpus tests.test_readme_quickstart
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  ./scripts/test-python.sh
env -u CONDA_PREFIX PATH="$PWD/.venv/bin:$PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  taskset -c 24 .venv/bin/python scripts/benchmark_compile_cpu.py \
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
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python" \
  ./scripts/test-python.sh
```

Results: the focused public compile, compile corpus, PyTorch 2.13
differential, and docs smoke tests passed 58 tests. The repository-managed
pinned PyTorch 2.13 full Python suite passed 4686 tests with 3 skips. The
default Rust suite passed 308 tests across unit and integration targets plus
0 doctests, default Clippy passed, and Python-bindings Clippy passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- Maturin: 1.14.1
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from the wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run
- CPU affinity: `taskset -c 24`
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 33 ms, prepared
  packages in 8.74s, and installed in 1.82s
- Build time: first successful Python 3.12 release extension build completed in
  34.10s; the full-suite wheel build reused release artifacts and completed in
  0.02s

The benchmark uses the checked-in `torch_compile_corpus_v2` programs and runs
each one across the corpus default input plus scalar, vector, row-major matrix,
larger row-major matrix, empty, and non-contiguous transpose inputs. Inputs are
created outside timed regions from deterministic values. For PyTorch, the
driver requires pinned PyTorch 2.13 and uses stock
`torch.compile(backend="eager", fullgraph=True)`. For `torch_rs`, it uses the
native guarded eager/fullgraph path. Both implementations run in both orders:
`torch_rs,pytorch` and `pytorch,torch_rs`. Each order pass resets the relevant
compiler state for cold timing, measures the first materialized compiled call
separately, then runs 7 untimed warmup blocks and 31 measured blocks. A measured
block repeats the operation according to the table's `Repeats` column; medians
below are microseconds per compiled call. The CPU workload has no asynchronous
device queue, but the driver still calls synchronization hooks when an
implementation exposes an available CUDA runtime.

Before timing each cell, the driver checks exact output values, shape, stride,
storage offset, contiguity, dtype, device, and `requires_grad` against the
same eager program. The `torch_rs` result is also checked against the PyTorch
result. After every warmup and measured block, the driver materializes the last
output and records a 64-bit BLAKE2b checksum over values and metadata. All 35
timed cells had matching `torch_rs` and PyTorch checksums.

`torch_rs / PyTorch` is a slowdown ratio, so lower is better and 1.00x is
parity. Capped geomeans clamp each per-cell ratio to `[0.10x, 10.00x]`.

## Aggregate

- Cold first compiled call: 0.021x uncapped, 0.114x capped
- Steady-state materialized compiled call: 0.584x uncapped, 0.584x capped

## Supported Timed Cells

| Program | Input variant | Repeats | `torch_rs` cold us | PyTorch cold us | Cold ratio | `torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | Steady ratio | Checksum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu_float32_unary_abs_neg` | `case_default` | 256 | 308.4 | 78398.9 | 0.004x | 6.989 +/- 0.050 | 13.611 +/- 0.151 | 0.513x | `e7effd8599e8fd3e` |
| `cpu_float32_unary_abs_neg` | `scalar` | 2048 | 219.0 | 26368.1 | 0.008x | 6.474 +/- 0.130 | 13.330 +/- 0.109 | 0.486x | `96474978e4b2c20f` |
| `cpu_float32_unary_abs_neg` | `vector_17` | 1024 | 248.9 | 26017.9 | 0.010x | 6.721 +/- 0.029 | 13.587 +/- 0.225 | 0.495x | `df430381d21069c0` |
| `cpu_float32_unary_abs_neg` | `matrix_31x37` | 128 | 804.0 | 28263.7 | 0.028x | 11.790 +/- 0.076 | 18.454 +/- 0.225 | 0.639x | `a6615e9dbd215dce` |
| `cpu_float32_unary_abs_neg` | `matrix_127x131` | 16 | 8327.0 | 33745.4 | 0.247x | 514.480 +/- 2.326 | 526.226 +/- 5.905 | 0.978x | `4bb9338c2bde3594` |
| `cpu_float32_unary_abs_neg` | `empty_2x0` | 2048 | 229.9 | 23514.9 | 0.010x | 6.823 +/- 0.039 | 12.781 +/- 0.056 | 0.534x | `e99a6c9902c3119e` |
| `cpu_float32_unary_abs_neg` | `transpose_37x31` | 128 | 805.4 | 26824.4 | 0.030x | 11.777 +/- 0.062 | 19.039 +/- 0.118 | 0.619x | `3083af797face788` |
| `cpu_float32_self_add` | `case_default` | 256 | 196.0 | 22218.8 | 0.009x | 6.542 +/- 0.036 | 12.092 +/- 0.099 | 0.541x | `cf580eb9d53f4ab8` |
| `cpu_float32_self_add` | `scalar` | 2048 | 175.1 | 22587.4 | 0.008x | 6.022 +/- 0.039 | 11.959 +/- 0.065 | 0.504x | `2893378e1c7355c5` |
| `cpu_float32_self_add` | `vector_17` | 1024 | 211.3 | 22155.6 | 0.010x | 6.361 +/- 0.027 | 11.812 +/- 0.131 | 0.538x | `8f9b9bdd6cd9bd2a` |
| `cpu_float32_self_add` | `matrix_31x37` | 128 | 786.5 | 24083.9 | 0.033x | 11.447 +/- 0.084 | 16.874 +/- 0.153 | 0.678x | `6f4a9fa909165974` |
| `cpu_float32_self_add` | `matrix_127x131` | 16 | 8707.9 | 37483.7 | 0.232x | 537.296 +/- 2.734 | 542.789 +/- 3.655 | 0.990x | `831f2172069daaaf` |
| `cpu_float32_self_add` | `empty_2x0` | 2048 | 229.8 | 23414.3 | 0.010x | 6.501 +/- 0.037 | 11.594 +/- 0.055 | 0.561x | `e99a6c9902c3119e` |
| `cpu_float32_self_add` | `transpose_37x31` | 128 | 798.7 | 27944.7 | 0.029x | 12.184 +/- 0.107 | 16.981 +/- 0.151 | 0.718x | `cb2131b53d3b05d5` |
| `cpu_float32_abs_neg_reordered` | `case_default` | 256 | 210.5 | 25276.8 | 0.008x | 6.861 +/- 0.032 | 13.610 +/- 0.105 | 0.504x | `abbc312073a422dc` |
| `cpu_float32_abs_neg_reordered` | `scalar` | 2048 | 176.5 | 24574.3 | 0.007x | 6.300 +/- 0.029 | 13.364 +/- 0.090 | 0.471x | `e75a1d3233117514` |
| `cpu_float32_abs_neg_reordered` | `vector_17` | 1024 | 229.4 | 25486.7 | 0.009x | 6.677 +/- 0.042 | 13.552 +/- 0.204 | 0.493x | `ba2eaa9e2ad0830d` |
| `cpu_float32_abs_neg_reordered` | `matrix_31x37` | 128 | 811.6 | 25673.7 | 0.032x | 11.840 +/- 0.146 | 18.211 +/- 0.157 | 0.650x | `323b11b354c9b7a8` |
| `cpu_float32_abs_neg_reordered` | `matrix_127x131` | 16 | 9056.0 | 31346.8 | 0.289x | 516.325 +/- 3.161 | 519.998 +/- 3.088 | 0.993x | `f9feb1c7c3003aea` |
| `cpu_float32_abs_neg_reordered` | `empty_2x0` | 2048 | 246.7 | 23655.8 | 0.010x | 6.793 +/- 0.029 | 12.847 +/- 0.059 | 0.529x | `e99a6c9902c3119e` |
| `cpu_float32_abs_neg_reordered` | `transpose_37x31` | 128 | 821.9 | 23754.4 | 0.035x | 11.797 +/- 0.105 | 19.060 +/- 0.191 | 0.619x | `013ec8b4a8ced6ed` |
| `cpu_float32_repeated_unary_chain` | `case_default` | 256 | 250.5 | 27140.4 | 0.009x | 7.307 +/- 0.040 | 17.163 +/- 0.201 | 0.426x | `e23ed4736483131b` |
| `cpu_float32_repeated_unary_chain` | `scalar` | 2048 | 254.1 | 29274.7 | 0.009x | 7.280 +/- 0.026 | 17.479 +/- 0.390 | 0.416x | `e75a1d3233117514` |
| `cpu_float32_repeated_unary_chain` | `vector_17` | 1024 | 287.4 | 24154.5 | 0.012x | 7.878 +/- 0.050 | 17.230 +/- 0.387 | 0.457x | `ba2eaa9e2ad0830d` |
| `cpu_float32_repeated_unary_chain` | `matrix_31x37` | 128 | 855.9 | 25364.9 | 0.034x | 13.566 +/- 0.162 | 22.230 +/- 0.167 | 0.610x | `323b11b354c9b7a8` |
| `cpu_float32_repeated_unary_chain` | `matrix_127x131` | 16 | 8621.2 | 35611.9 | 0.242x | 522.433 +/- 2.587 | 532.092 +/- 5.265 | 0.982x | `f9feb1c7c3003aea` |
| `cpu_float32_repeated_unary_chain` | `empty_2x0` | 2048 | 284.0 | 25778.9 | 0.011x | 7.913 +/- 0.031 | 15.889 +/- 0.108 | 0.498x | `e99a6c9902c3119e` |
| `cpu_float32_repeated_unary_chain` | `transpose_37x31` | 128 | 867.3 | 28400.4 | 0.031x | 13.507 +/- 0.073 | 23.138 +/- 0.250 | 0.584x | `013ec8b4a8ced6ed` |
| `cpu_float32_add_unary_composition` | `case_default` | 256 | 306.0 | 28428.1 | 0.011x | 8.079 +/- 0.032 | 16.039 +/- 0.131 | 0.504x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `scalar` | 2048 | 284.3 | 29476.9 | 0.010x | 7.480 +/- 0.024 | 17.587 +/- 0.269 | 0.425x | `72f27995b7dd0815` |
| `cpu_float32_add_unary_composition` | `vector_17` | 1024 | 315.7 | 28508.1 | 0.011x | 8.049 +/- 0.041 | 17.362 +/- 0.416 | 0.464x | `e33edbb6040ef154` |
| `cpu_float32_add_unary_composition` | `matrix_31x37` | 128 | 909.8 | 28521.3 | 0.032x | 13.815 +/- 0.078 | 22.950 +/- 0.148 | 0.602x | `8b4cf5faabeff82f` |
| `cpu_float32_add_unary_composition` | `matrix_127x131` | 16 | 8729.3 | 35896.5 | 0.243x | 543.406 +/- 2.976 | 554.966 +/- 4.657 | 0.979x | `2cab6c3527a20afd` |
| `cpu_float32_add_unary_composition` | `empty_2x0` | 2048 | 319.9 | 27947.5 | 0.011x | 8.150 +/- 0.044 | 16.061 +/- 0.088 | 0.507x | `e99a6c9902c3119e` |
| `cpu_float32_add_unary_composition` | `transpose_37x31` | 128 | 933.1 | 27756.2 | 0.034x | 15.557 +/- 0.154 | 23.700 +/- 0.145 | 0.656x | `fedf1f495675c5ac` |
