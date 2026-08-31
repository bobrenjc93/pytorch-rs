# Rank-12 `Tensor.sum` Release Timings

Date: 2026-08-30

Base revision: `73ff352`

Command shape: release `maturin build --release --locked` wheels for both the
current worktree and a clean `HEAD` export under `target/rank12-sum-run-*`.
The current wheel was built under `target/rank12-sum-run-current-final` and
force-installed into `target/rank12-sum-run-current/benchmark-env`; the base
wheel was built and installed under `target/rank12-sum-run-base`. The timing
driver ran inside each process after imports and input construction, pinned
with `taskset -c 0`, with 9 warmup blocks and 51 measured blocks. Each block
repeated the eager `tensor.sum().item()` call enough times to materialize the
scalar result and avoid deferred work. Before timing, each `torch_rs` scalar
result and `torch_rs.sum(input)` result were checked bitwise against the
equivalent PyTorch result.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.14+meta
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
  `cargo 1.92.0 (344c4567c 2025-10-21)`
- PyTorch: 2.13.0+cu130
- NumPy: 2.5.1
- `torch_rs`: 0.1.0
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32; CUDA 13.0 runtime was importable but not selected
- Threads: `taskset -c 0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`
- Dependency installation: 0.98s wall time for the current benchmark
  environment and 0.99s for the clean-base benchmark environment, from the
  locked dependency graph and local uv cache
- Compile time: current wheel 31.85s wall time; clean-base wheel 31.19s wall
  time

The rank-12 workloads are held-out offset, non-contiguous, permuted full
reductions. Rank-10 and rank-11 workloads are non-contiguous controls. Times
are median microseconds per `sum().item()` call; MAD is median absolute
deviation in microseconds.

| Workload | Shape | Permutation | Elements | Current `torch_rs` | PyTorch | Current / PyTorch | Base `torch_rs` | Current vs Base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank10_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2)` | `(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 20,160 | 31.799 us +/- 0.282 | 3.238 us +/- 0.011 | 9.82x | 30.946 us +/- 0.075 | +2.8% |
| `rank10_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2)` | `(2, 0, 6, 9, 4, 1, 8, 3, 7, 5)` | 20,160 | 38.355 us +/- 0.080 | 3.149 us +/- 0.013 | 12.18x | 36.623 us +/- 0.134 | +4.7% |
| `rank11_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2)` | `(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 40,320 | 56.409 us +/- 0.436 | 4.002 us +/- 0.018 | 14.10x | 57.293 us +/- 1.354 | -1.5% |
| `rank11_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2)` | `(2, 0, 6, 10, 4, 1, 9, 3, 8, 7, 5)` | 40,320 | 72.469 us +/- 0.496 | 3.902 us +/- 0.021 | 18.57x | 80.351 us +/- 1.255 | -9.8% |
| `rank12_heldout_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2)` | `(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 80,640 | 122.242 us +/- 0.233 | 5.778 us +/- 0.030 | 21.16x | 3622.934 us +/- 41.454 | -96.6% |
| `rank12_heldout_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2)` | `(2, 0, 6, 11, 4, 1, 10, 3, 9, 8, 7, 5)` | 80,640 | 160.600 us +/- 0.567 | 5.621 us +/- 0.011 | 28.57x | 3578.420 us +/- 14.423 | -95.5% |
| `rank12_heldout_singleton` | `(3, 1, 5, 2, 1, 7, 2, 2, 3, 2, 2, 2)` | `(5, 0, 11, 2, 6, 3, 10, 9, 8, 7, 4, 1)` | 20,160 | 71.272 us +/- 0.180 | 3.505 us +/- 0.065 | 20.34x | 901.727 us +/- 4.107 | -92.1% |

Rank-10 and rank-11 controls stayed within the no-greater-than-5% regression
guardrail in this run. The largest control regression was +4.7% on
`rank10_control_mixed`; all other controls improved or moved less. The rank-12
held-out permutations moved from the generic strided fallback to the fixed-rank
odometer path.
