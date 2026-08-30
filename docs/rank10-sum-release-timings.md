# Rank-10 `Tensor.sum` Release Timings

Date: 2026-08-29

Base revision: `70b3fab`

Command shape: release `maturin develop --release --locked` builds for both the
current worktree and a clean `HEAD` copy under
`target/rank10-sum-base-70b3fab-1`. The timing driver ran inside each process
after imports and input construction, with 7 warmup blocks and 31 measured
blocks. Each block repeated 512 eager `tensor.sum().item()` calls to materialize
the scalar result and avoid deferred work. Before timing, each `torch_rs` scalar
result was checked bitwise against the equivalent PyTorch result.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2 x86_64, glibc 2.34
- Python: 3.12.12
- PyTorch: 2.13.0+cu130
- `torch_rs`: 0.1.0
- Rust: 1.92.0
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32
- Threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`

All workloads are held-out offset, non-contiguous, permuted full reductions.
Times are median microseconds per `sum().item()` call; MAD is median absolute
deviation in microseconds.

| Workload | Shape | Permutation | Elements | Current `torch_rs` | PyTorch | Current / PyTorch | Base `torch_rs` | Current vs Base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank8_control_reverse` | `(3, 5, 7, 2, 3, 2, 2, 2)` | `(7, 6, 5, 4, 3, 2, 1, 0)` | 5,040 | 6.274 us +/- 0.039 | 2.476 us +/- 0.014 | 2.53x | 7.053 us +/- 0.063 | -11.1% |
| `rank8_control_mixed` | `(3, 5, 7, 2, 3, 2, 2, 2)` | `(2, 0, 4, 6, 1, 7, 3, 5)` | 5,040 | 8.531 us +/- 0.057 | 2.429 us +/- 0.023 | 3.51x | 8.795 us +/- 0.058 | -3.0% |
| `rank9_control_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2)` | `(8, 7, 6, 5, 4, 3, 2, 1, 0)` | 10,080 | 13.663 us +/- 0.584 | 2.727 us +/- 0.021 | 5.01x | 14.492 us +/- 0.473 | -5.7% |
| `rank9_control_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2)` | `(2, 0, 6, 8, 4, 1, 7, 3, 5)` | 10,080 | 17.512 us +/- 0.359 | 2.653 us +/- 0.013 | 6.60x | 17.221 us +/- 0.129 | +1.7% |
| `rank10_heldout_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2)` | `(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)` | 20,160 | 27.991 us +/- 1.352 | 3.136 us +/- 0.021 | 8.93x | 699.941 us +/- 9.266 | -96.0% |
| `rank10_heldout_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2, 2)` | `(2, 0, 6, 9, 4, 1, 8, 3, 7, 5)` | 20,160 | 34.282 us +/- 0.426 | 3.086 us +/- 0.052 | 11.11x | 871.137 us +/- 86.201 | -96.1% |
| `rank10_heldout_singleton` | `(3, 1, 5, 2, 1, 7, 2, 2, 3, 2)` | `(5, 0, 8, 2, 9, 6, 3, 7, 4, 1)` | 5,040 | 18.631 us +/- 1.060 | 2.532 us +/- 0.012 | 7.36x | 183.911 us +/- 4.202 | -89.9% |

Rank-8 and rank-9 current timings show no greater than 5% regression against
the clean base revision in this run; three controls are faster than base and the
slowest control is +1.7%.
