# `torch.empty` Release Timings

Date: 2026-08-31

Candidate provenance: source snapshot based on
`8770edb9b54a2fa73f30af61d9d4e84e69278654`, plus the worktree changes that add
lazy safe storage initialization for CPU float32 `torch.empty`.

Command shape: from the repository root, a release wheel was built with
`maturin build --release --locked --out target/python-wheels` using the
worktree `.venv` Python interpreter, then force-installed with
`uv pip install --force-reinstall --no-deps`. The timing driver ran against the
installed wheel after imports, pinned with `taskset -c 24`, with
`CUDA_VISIBLE_DEVICES=` and one runtime thread. Each implementation ran 15
warmup blocks and 81 measured blocks.

The timings below measure eager `torch.empty` allocation plus metadata and
pointer access for supported CPU float32 shapes. The driver checked PyTorch
2.13 metadata parity before timing. Each timed block consumed shape, stride,
numel, data pointer, contiguity, and pinning metadata as a dead-code and
deferred-work guard. It intentionally did not read tensor values: `empty`
contents have no value contract, and reading values would measure value
materialization rather than allocation behavior. Pointer values are
implementation-specific and were consumed, not compared.

Checks run for this candidate before the final timing pass:

```bash
cargo fmt --check
git diff --check
cargo test --locked --offline storage::tests:: --lib
cargo test --locked --offline --all-targets
PYO3_PYTHON="$PWD/.venv/bin/python" \
  cargo clippy --locked --offline --all-targets --features python-bindings -- \
  -D warnings
cargo test --locked --offline --doc
PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --out target/python-wheels
UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/python-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
.venv/bin/python .github/scripts/verify_native_extension.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest tests.test_empty tests.test_empty_reference
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 24 .venv/bin/python target/empty_benchmark.py
```

Results: the focused `empty` Python tests passed 16 tests. The all-target Rust
suite passed 112 library tests, 79 autograd tests, and 103 tensor-baseline
tests. Doc tests had no doctests to run. The native extension provenance smoke
check passed.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from
  `.venv/lib/python3.12/site-packages/torch`
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
- Dependency installation: no dependency changes; wheel reinstall resolved in
  2 ms, prepared in 42 ms, uninstalled the prior local wheel in 4 ms, and
  installed in 84 ms
- Build time: final release wheel build completed in 33.43s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Allocation Metadata

Geometric mean `torch_rs / PyTorch` slowdown across the supported `empty`
allocation cells:

- Uncapped: 0.87x
- Capped to `[0.10x, 10.00x]` per cell: 0.87x

Deterministic-algorithm setter and getter helpers are intentionally excluded
from this allocation performance report because they do not affect hot tensor
execution paths.

| Workload | Size | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | ---: | ---: | ---: | ---: |
| `scalar_empty` | `()` | 5000 | 1.498 us +/- 0.010, var 0.002 | 2.563 us +/- 0.019, var 0.004 | 0.58x |
| `small_2x3` | `(2, 3)` | 5000 | 2.702 us +/- 0.020, var 0.003 | 2.952 us +/- 0.029, var 0.407 | 0.92x |
| `large_2048x2048` | `(2048, 2048)` | 8 | 2.825 us +/- 0.024, var 0.140 | 2.938 us +/- 0.021, var 0.091 | 0.96x |
| `empty_dimension_2x0x4096` | `(2, 0, 4096)` | 5000 | 3.108 us +/- 0.015, var 0.002 | 2.924 us +/- 0.021, var 0.019 | 1.06x |
| `repeated_allocation_1024x16x16` | `(16, 16)` | 8 x 512 | 2.586 us +/- 0.015, var 0.002 | 2.850 us +/- 0.014, var 0.001 | 0.91x |
