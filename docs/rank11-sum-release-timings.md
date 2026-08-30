# Rank-11 `Tensor.sum` Release Timings

Date: 2026-08-30

Base revision: `977ed05`

Command shape: release `maturin build --release --locked` wheels for both the
current worktree and a clean `HEAD` export under `target/rank11-sum-run-*`.
Each wheel was force-installed into `target/rank11-sum-run-*/benchmark-env`
before its timing run. The timing driver ran inside each process after imports
and input construction, pinned with `taskset -c 0`, with 9 warmup blocks and 51
measured blocks. Each block repeated the eager `tensor.sum().item()` call enough
times to materialize the scalar result and avoid deferred work. Before timing,
each `torch_rs` scalar result was checked bitwise against the equivalent PyTorch
result.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2 x86_64, glibc 2.34
- Python: 3.12.12
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130
- `torch_rs`: 0.1.0
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- Threads: `taskset -c 0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`
- Dependency installation: 1.44s wall time for the benchmark environment from
  the locked dependency graph and local uv cache
- Compile time: current wheel 31.47s wall time; clean-base wheel 31.62s wall
  time

The rank-11 workloads are held-out offset, non-contiguous, permuted full
reductions. Rank-9 and rank-10 workloads are non-contiguous controls. Times are
median microseconds per `sum().item()` call; MAD is median absolute deviation
in microseconds.

| Workload | Shape | Permutation | Elements | Current `torch_rs` | PyTorch | Current / PyTorch | Base `torch_rs` | Current vs Base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank9_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2)` | `(8, 7, 6, 5, 4, 3, 2, 1, 0)` | 10,080 | 14.282 us +/- 0.334 | 2.839 us +/- 0.025 | 5.03x | 13.818 us +/- 0.105 | +3.4% |
| `rank9_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2)` | `(2, 0, 6, 8, 4, 1, 7, 3, 5)` | 10,080 | 17.400 us +/- 0.101 | 2.744 us +/- 0.019 | 6.34x | 17.348 us +/- 0.059 | +0.3% |
| `rank10_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2)` | `(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 20,160 | 26.288 us +/- 0.652 | 3.213 us +/- 0.018 | 8.18x | 27.897 us +/- 0.446 | -5.8% |
| `rank10_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2)` | `(2, 0, 6, 9, 4, 1, 8, 3, 7, 5)` | 20,160 | 33.887 us +/- 0.121 | 3.560 us +/- 0.042 | 9.52x | 33.732 us +/- 0.099 | +0.5% |
| `rank11_heldout_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2)` | `(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 40,320 | 55.742 us +/- 1.643 | 3.878 us +/- 0.034 | 14.38x | 1770.157 us +/- 1.652 | -96.9% |
| `rank11_heldout_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2)` | `(2, 0, 6, 10, 4, 1, 9, 3, 8, 7, 5)` | 40,320 | 78.002 us +/- 1.130 | 3.779 us +/- 0.040 | 20.64x | 1656.988 us +/- 1.704 | -95.3% |
| `rank11_heldout_singleton` | `(3, 1, 5, 2, 1, 7, 2, 2, 3, 2, 2)` | `(5, 0, 10, 2, 6, 3, 9, 8, 7, 4, 1)` | 10,080 | 35.482 us +/- 0.149 | 2.845 us +/- 0.016 | 12.47x | 415.461 us +/- 0.992 | -91.5% |

Rank-9 and rank-10 controls remained within the no-greater-than-5% regression
guardrail in this run. The largest control regression was +3.4% on
`rank9_control_reverse`; both rank-11 dense held-out permutations moved from
the generic strided fallback to the fixed-rank odometer path.
