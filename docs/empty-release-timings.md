# `torch.empty` Release Timings

Date: 2026-08-31

Candidate provenance: source snapshot based on
`8770edb9b54a2fa73f30af61d9d4e84e69278654`, plus the worktree changes that add
lazy safe storage initialization for CPU float32 `torch.empty`.

Command shape: from the repository root, a release wheel was rebuilt with
`maturin build --release --locked --out target/python-wheels` using the
worktree `.venv` Python interpreter, then force-installed with
`uv pip install --force-reinstall --no-deps`. The timing driver ran against the
installed wheel after imports, pinned with `taskset -c 24`, with
`CUDA_VISIBLE_DEVICES=` and one runtime thread. Each implementation ran 15
warmup blocks and 81 measured blocks.

The primary timings below measure eager `torch.empty` allocation plus metadata
access for supported CPU float32 shapes, without calling `data_ptr()`. The
driver checked PyTorch 2.13 metadata parity before timing. Each timed block
consumed shape, stride, numel, contiguity, and pinning metadata as a dead-code
and deferred-work guard. It intentionally did not read tensor values: `empty`
contents have no value contract, and reading values would measure value
materialization rather than allocation behavior.

A second table measures public pointer exposure. `Tensor.data_ptr()` and
`Tensor.const_data_ptr()` now materialize deferred `empty` storage before
returning an address so ctypes writes through that address become the tensor's
real contents and raw pointer reads never observe uninitialized spare capacity.
Pointer values are implementation-specific and were consumed, not compared.

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
  .venv/bin/python -m unittest \
  tests.test_empty tests.test_empty_reference \
  tests.test_use_deterministic_algorithms \
  tests.test_use_deterministic_algorithms_reference \
  tests.test_are_deterministic_algorithms_enabled \
  tests.test_are_deterministic_algorithms_enabled_reference \
  tests.test_get_deterministic_debug_mode \
  tests.test_get_deterministic_debug_mode_reference \
  tests.test_is_deterministic_algorithms_warn_only_enabled \
  tests.test_is_deterministic_algorithms_warn_only_enabled_reference \
  tests.test_set_deterministic_debug_mode \
  tests.test_set_deterministic_debug_mode_reference
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 24 .venv/bin/python target/empty_benchmark.py
```

Results: the focused Python suites passed 91 tests, including the `empty`
pointer-write regression case and the deterministic API checks. The all-target
Rust suite passed 112 library tests, 79 autograd tests, and 103 tensor-baseline
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
  1 ms, prepared in 40 ms, uninstalled the prior local wheel in 3 ms, and
  installed in 13 ms
- Build time: final release wheel rebuild completed in 25.96s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Allocation Metadata

Geometric mean `torch_rs / PyTorch` slowdown across the supported `empty`
allocation metadata cells:

- Uncapped: 0.84x
- Capped to `[0.10x, 10.00x]` per cell: 0.84x

Deterministic-algorithm setter and getter helpers are intentionally excluded
from this allocation performance report because they do not affect hot tensor
execution paths.

| Workload | Size | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | ---: | ---: | ---: | ---: |
| `scalar_empty` | `()` | 5000 | 1.396 us +/- 0.007, var 0.003 | 2.572 us +/- 0.027, var 0.044 | 0.54x |
| `small_2x3` | `(2, 3)` | 5000 | 2.558 us +/- 0.009, var 0.005 | 2.835 us +/- 0.011, var 0.044 | 0.90x |
| `large_2048x2048` | `(2048, 2048)` | 8 | 2.738 us +/- 0.024, var 0.265 | 3.010 us +/- 0.015, var 0.237 | 0.91x |
| `empty_dimension_2x0x4096` | `(2, 0, 4096)` | 5000 | 3.032 us +/- 0.010, var 0.001 | 2.901 us +/- 0.013, var 0.012 | 1.04x |
| `repeated_allocation_1024x16x16` | `(16, 16)` | 8 x 512 | 2.527 us +/- 0.018, var 0.003 | 2.868 us +/- 0.016, var 0.046 | 0.88x |

## Public Pointer Exposure

Geometric mean `torch_rs / PyTorch` slowdown across the public pointer exposure
cells:

- Uncapped: 1.93x
- Capped to `[0.10x, 10.00x]` per cell: 1.36x

The large pointer-exposure cell includes required `torch_rs` materialization of
the 16 MiB float32 storage before the pointer is returned. This is intentionally
reported separately from allocation metadata parity because public pointer
exposure must preserve external write-through and raw-read safety.

| Workload | Size | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch |
| --- | --- | ---: | ---: | ---: | ---: |
| `scalar_empty_pointer` | `()` | 5000 | 1.459 us +/- 0.008, var 0.000 | 2.617 us +/- 0.014, var 0.046 | 0.56x |
| `small_2x3_pointer` | `(2, 3)` | 5000 | 2.690 us +/- 0.050, var 0.006 | 2.949 us +/- 0.010, var 0.041 | 0.91x |
| `large_2048x2048_pointer` | `(2048, 2048)` | 2 | 188.906 us +/- 4.797, var 54.058 | 3.230 us +/- 0.030, var 0.960 | 58.48x |
| `empty_dimension_2x0x4096_pointer` | `(2, 0, 4096)` | 5000 | 3.071 us +/- 0.009, var 0.004 | 3.007 us +/- 0.011, var 0.004 | 1.02x |
| `repeated_allocation_256x16x16_pointer` | `(16, 16)` | 4 x 256 | 2.602 us +/- 0.018, var 0.035 | 2.914 us +/- 0.014, var 0.005 | 0.89x |
