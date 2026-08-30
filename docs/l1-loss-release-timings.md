# `torch.nn.functional.l1_loss(reduction="none")` Release Timings

Date: 2026-08-30

Revision under test: uncommitted worktree based on
`039b1603b75ab922e7bc37caa0d032cd680e926f`.

Command shape: worktree-local `uv venv --clear --python 3.12`, locked
`uv sync --locked --no-install-project --group dev --group reference`, then
release wheel builds through `maturin build --release --locked` and
installation with `uv pip install --force-reinstall --no-deps`. The clean base
wheel was built from a `git archive HEAD` snapshot under
`target/l1-transposed-bench/run/base-src`; the candidate wheel was built from
this worktree. The timing driver ran from
`target/l1-transposed-bench/l1_transposed_driver.py` after imports and input
construction, with 15 warmup blocks and 81 measured blocks per implementation.
Inputs were CPU `float32` tensors. Outputs were bit-compared against PyTorch
before timing, and the last output from every warmup and measured block was
consumed as a dead-code and deferred-work guard. Size-mismatch warnings were
ignored symmetrically inside timing.

Checks run for this change:

```bash
/home/bobren/.cargo/bin/cargo fmt --check
git diff --check
/home/bobren/.cargo/bin/cargo test absolute_difference --all-targets
/home/bobren/.cargo/bin/cargo test --all-targets
/home/bobren/.cargo/bin/cargo clippy --all-targets -- -D warnings
PATH="/home/bobren/.cargo/bin:$PATH" PYO3_PYTHON="/usr/bin/python3.12" \
  /home/bobren/.cargo/bin/cargo clippy --all-targets --features python-bindings -- -D warnings
PATH="/home/bobren/.cargo/bin:$PATH" PYO3_PYTHON="/usr/bin/python3.12" \
  /home/bobren/.cargo/bin/cargo test --all-targets --features python-bindings
PATH="/home/bobren/.cargo/bin:$PATH" VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_nn_functional_l1_loss \
  tests.test_nn_functional_l1_loss_reference
```

Results: the focused L1 Python tests passed 30 tests.
After timing, `./scripts/test-python.sh` rebuilt the current release wheel and
passed the full Python suite: 4215 tests, 3 skips.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor, 2 sockets, 96 cores/socket,
  2 threads/core
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- NumPy: 2.5.1
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130 from `.venv/lib/python3.12/site-packages/torch`
- `torch_rs`: 0.1.0 from wheel-installed
  `.venv/lib/python3.12/site-packages/torch_rs`
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen
  unit
- Device/dtype: CPU float32; `CUDA_VISIBLE_DEVICES=` for the timing run
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`;
  `torch_rs.get_num_threads()` and `torch_rs.get_num_interop_threads()` both
  reported 1
- Dependency installation: locked `uv sync` resolved in 28 ms, prepared
  31 packages in 15.78s, and installed in 1.93s
- Build time: clean `HEAD` base release wheel build completed in 30.94s; the
  final candidate release wheel rebuild completed in 24.00s

Times are median microseconds per call. MAD is median absolute deviation in
microseconds, and variance is sample variance of per-call sample timings in
microseconds squared. `torch_rs / PyTorch` is a slowdown ratio, so lower is
better and 1.00x is parity. Capped geomeans clamp each per-cell ratio to
`[0.10x, 10.00x]`.

## Rank-2 Transposed Dense Held-Out

Relative to the clean `HEAD` base, held-out rank-2 transposed L1 cells improved
by a geometric mean of 69.1%. All cells use same-shape noncontiguous dense
transpose operands with identical strides; `offset_transpose_997x257_heldout`
uses nonzero storage offsets.

Geometric mean `torch_rs / PyTorch` slowdown for held-out transposed cells:

- Uncapped: 0.70x
- Capped to `[0.10x, 10.00x]` per cell: 0.70x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transpose_769x641_heldout` | `(769, 641)` / `(769, 641)`, stride `(1, 769)` | `(769, 641)`, stride `(1, 769)` | 5 | 103.469 us +/- 2.678, var 36.318 | 117.628 us +/- 2.181, var 23.867 | 0.88x | 318.143 us | -67.5% |
| `offset_transpose_997x257_heldout` | `(997, 257)` / `(997, 257)`, stride `(1, 997)`, nonzero offsets | `(997, 257)`, stride `(1, 997)` | 8 | 45.249 us +/- 0.968, var 4.594 | 64.241 us +/- 0.531, var 0.636 | 0.70x | 153.705 us | -70.6% |
| `transpose_2003x37_heldout` | `(2003, 37)` / `(2003, 37)`, stride `(1, 2003)` | `(2003, 37)`, stride `(1, 2003)` | 24 | 13.512 us +/- 0.170, var 0.315 | 24.153 us +/- 0.196, var 0.102 | 0.56x | 43.654 us | -69.0% |

## Regression Controls

Relative to the clean `HEAD` base, no current contiguous, scalar-broadcast, or
mixed-layout L1 control regressed by more than 5%. The largest positive movement
was `mixed_transpose_contiguous_control`, from 5860.192 us to 6022.530 us
(+2.8%).

Geometric mean `torch_rs / PyTorch` slowdown for contiguous controls:

- Uncapped: 0.27x
- Capped to `[0.10x, 10.00x]` per cell: 0.27x

Geometric mean `torch_rs / PyTorch` slowdown for scalar-broadcast controls:

- Uncapped: 0.74x
- Capped to `[0.10x, 10.00x]` per cell: 0.74x

| Workload | Input / target | Output | Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, variance | `torch_rs` / PyTorch | Base median | Current vs base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_contiguous_prime_control` | `(257, 263)` / `(257, 263)` | `(257, 263)`, stride `(263, 1)` | 32 | 11.507 us +/- 0.095, var 0.449 | 21.969 us +/- 0.144, var 0.084 | 0.52x | 11.946 us | -3.7% |
| `same_contiguous_bandwidth_control` | `(2048, 2048)` / `(2048, 2048)` | `(2048, 2048)`, stride `(2048, 1)` | 1 | 1470.985 us +/- 18.758, var 6071.925 | 10220.889 us +/- 200.874, var 85212.508 | 0.14x | 1469.022 us | +0.1% |
| `scalar_input_2d_control` | `()` / `(640, 768)` | `(640, 768)`, stride `(768, 1)` | 10 | 90.871 us +/- 0.932, var 6.177 | 96.025 us +/- 0.329, var 3.909 | 0.95x | 91.833 us | -1.0% |
| `scalar_target_2d_control` | `(640, 768)` / `()` | `(640, 768)`, stride `(768, 1)` | 10 | 56.503 us +/- 1.403, var 42.244 | 97.950 us +/- 0.906, var 18.083 | 0.58x | 58.614 us | -3.6% |
| `mixed_transpose_contiguous_control` | `(509, 521)` transposed / `(509, 521)` contiguous | `(509, 521)`, stride `(1, 509)` | 8 | 6022.530 us +/- 99.851, var 67379.185 | 179.075 us +/- 1.521, var 36.781 | 33.63x | 5860.192 us | +2.8% |
