# Rank-13 `Tensor.sum` Release Timings

Date: 2026-08-30

Base revision: `2838cee`

Command shape: release `maturin build --release --locked` wheels for both the
current worktree and a clean `HEAD` export under `target/rank13-sum-run-*`.
The final current wheel was built under
`target/rank13-sum-run-current-sumhelper`; the base wheel was built under
`target/rank13-sum-run-base`. Each wheel was force-installed into its own
`target/rank13-sum-run-*/benchmark-env` before timing. The timing driver ran
inside each process after imports and input construction, pinned with
`taskset -c 0`, with 9 warmup blocks and 51 measured blocks. Each block
repeated the eager `tensor.sum().item()` call enough times to materialize the
scalar result and avoid deferred work. Before timing, each `torch_rs`
`Tensor.sum` result and top-level `torch_rs.sum(input)` result was checked
bitwise against the equivalent PyTorch result.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.12
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
- Dependency installation: 1.19s wall time for the current benchmark
  environment and 1.07s for the clean-base benchmark environment, from pinned
  package versions and the local uv cache
- Compile time: current wheel 32.44s wall time; clean-base wheel 32.97s wall
  time
- Workload repeats per measured block: 64 for rank-11 controls, 32 for rank-12
  controls and the rank-13 singleton case, 16 for dense rank-13 held-out cases,
  and 4 for rank-14 controls

The rank-13 workloads are held-out offset, non-contiguous, permuted full
reductions. Rank-11, rank-12, and rank-14 workloads are non-contiguous controls.
Times are median microseconds per `sum().item()` call; MAD is median absolute
deviation in microseconds.

| Workload | Shape | Permutation | Elements | Current `Tensor.sum` | Current `torch.sum` | PyTorch | Current `Tensor.sum` / PyTorch | Base `Tensor.sum` | Current vs Base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank11_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2)` | `(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 40,320 | 63.161 us +/- 0.236 | 63.485 us +/- 0.265 | 4.934 us +/- 0.023 | 12.80x | 63.284 us +/- 0.307 | -0.2% |
| `rank11_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2)` | `(2, 0, 6, 10, 4, 1, 9, 3, 8, 7, 5)` | 40,320 | 80.215 us +/- 0.567 | 80.596 us +/- 0.487 | 4.832 us +/- 0.019 | 16.60x | 80.041 us +/- 0.434 | +0.2% |
| `rank12_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2)` | `(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 80,640 | 125.548 us +/- 2.184 | 124.830 us +/- 0.718 | 6.910 us +/- 0.045 | 18.17x | 126.446 us +/- 0.759 | -0.7% |
| `rank12_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2)` | `(2, 0, 6, 11, 4, 1, 10, 3, 9, 8, 7, 5)` | 80,640 | 151.621 us +/- 1.037 | 147.648 us +/- 1.270 | 6.603 us +/- 0.028 | 22.96x | 144.716 us +/- 0.729 | +4.8% |
| `rank13_heldout_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2)` | `(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 161,280 | 265.236 us +/- 2.074 | 264.814 us +/- 1.803 | 11.216 us +/- 0.037 | 23.65x | 7230.560 us +/- 21.391 | -96.3% |
| `rank13_heldout_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2)` | `(2, 0, 6, 12, 4, 1, 11, 3, 10, 9, 8, 7, 5)` | 161,280 | 296.335 us +/- 3.714 | 297.340 us +/- 4.789 | 11.016 us +/- 0.083 | 26.90x | 7198.180 us +/- 17.414 | -95.9% |
| `rank13_heldout_singleton` | `(3, 1, 5, 2, 1, 7, 2, 2, 3, 2, 2, 2, 2)` | `(5, 0, 12, 2, 6, 3, 11, 10, 9, 8, 7, 4, 1)` | 40,320 | 123.103 us +/- 1.676 | 127.021 us +/- 0.834 | 5.153 us +/- 0.032 | 23.89x | 2055.633 us +/- 4.561 | -94.0% |
| `rank14_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2)` | `(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 322,560 | 15651.995 us +/- 40.649 | 15679.004 us +/- 56.025 | 20.631 us +/- 0.123 | 758.65x | 15591.711 us +/- 62.853 | +0.4% |
| `rank14_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2)` | `(2, 0, 6, 13, 4, 1, 12, 3, 11, 10, 9, 8, 7, 5)` | 322,560 | 15571.832 us +/- 123.832 | 15567.916 us +/- 113.600 | 19.509 us +/- 0.068 | 798.17x | 15504.526 us +/- 102.588 | +0.4% |

Rank-11, rank-12, and rank-14 controls stayed within the no-greater-than-5%
regression guardrail in this run. The largest control regression was +4.8% on
`rank12_control_mixed`; all other controls moved by less than 1%. The rank-13
held-out permutations moved from the generic strided fallback to the fixed-rank
odometer path.
