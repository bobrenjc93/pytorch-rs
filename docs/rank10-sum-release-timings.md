# Rank-10 `Tensor.sum` Release Timings

Date: 2026-08-29

Base revision: `280961a`

Command shape: release `maturin build --release --locked` wheels for both the
current worktree and a clean `HEAD` export under `target/rank10-sum-base`.
Each wheel was force-installed into `target/benchmark-env` before its timing
run. The timing driver ran inside each process after imports and input
construction, with 9 warmup blocks and 51 measured blocks. Each block repeated
the eager `tensor.sum().item()` call enough times to materialize the scalar
result and avoid deferred work. Before timing, each `torch_rs` scalar result
was checked bitwise against the equivalent PyTorch result.

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
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`

The rank-10 workloads are held-out offset, non-contiguous, permuted full
reductions. Rank-8 and rank-9 workloads are non-contiguous controls. Times are
median microseconds per `sum().item()` call; MAD is median absolute deviation
in microseconds.

| Workload | Shape | Permutation | Elements | Current `torch_rs` | PyTorch | Current / PyTorch | Base `torch_rs` | Current vs Base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank8_control_reverse` | `(3, 5, 7, 2, 3, 2, 2, 2)` | `(7, 6, 5, 4, 3, 2, 1, 0)` | 5,040 | 7.043 us +/- 0.027 | 2.442 us +/- 0.008 | 2.88x | 7.365 us +/- 0.120 | -4.4% |
| `rank8_control_mixed` | `(3, 5, 7, 2, 3, 2, 2, 2)` | `(2, 0, 4, 6, 1, 7, 3, 5)` | 5,040 | 8.794 us +/- 0.047 | 2.394 us +/- 0.006 | 3.67x | 8.387 us +/- 0.103 | +4.9% |
| `rank9_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2)` | `(8, 7, 6, 5, 4, 3, 2, 1, 0)` | 10,080 | 14.159 us +/- 0.129 | 2.705 us +/- 0.016 | 5.23x | 14.039 us +/- 0.044 | +0.9% |
| `rank9_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2)` | `(2, 0, 6, 8, 4, 1, 7, 3, 5)` | 10,080 | 17.723 us +/- 0.127 | 2.646 us +/- 0.007 | 6.70x | 16.998 us +/- 0.042 | +4.3% |
| `rank10_heldout_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2)` | `(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 20,160 | 30.962 us +/- 0.073 | 3.088 us +/- 0.010 | 10.03x | 741.089 us +/- 3.315 | -95.8% |
| `rank10_heldout_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2)` | `(2, 0, 6, 9, 4, 1, 8, 3, 7, 5)` | 20,160 | 35.781 us +/- 0.163 | 2.994 us +/- 0.013 | 11.95x | 736.834 us +/- 3.151 | -95.1% |
| `rank10_heldout_singleton` | `(3, 1, 5, 2, 1, 7, 2, 2, 3, 2)` | `(5, 0, 9, 2, 6, 3, 8, 7, 4, 1)` | 5,040 | 14.587 us +/- 0.055 | 2.461 us +/- 0.009 | 5.93x | 185.355 us +/- 1.083 | -92.1% |

Rank-8 and rank-9 controls remained within the no-greater-than-5% regression
guardrail in this run. The largest control movement was +4.9% on
`rank8_control_mixed`.
