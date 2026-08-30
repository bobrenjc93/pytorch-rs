# Rank-9 `Tensor.sum` Release Timings

Date: 2026-08-29

Base revision: `567ab4c`

Command shape: release `maturin develop --release --locked` builds for both the
current worktree and a clean `HEAD` copy under `target/rank9-sum-base-567ab4c`.
The timing driver ran inside each process after imports and input construction,
with 7 warmup blocks and 31 measured blocks. Each block repeated the eager
`tensor.sum().item()` call enough times to materialize the scalar result and
avoid deferred work. Before timing, each `torch_rs` scalar result was checked
bitwise against the equivalent PyTorch result.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2 x86_64, glibc 2.34
- Python: 3.12.12
- PyTorch: 2.13.0+cu130
- `torch_rs`: 0.1.0
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
| `rank7_offset_reverse` | `(3, 5, 7, 2, 3, 2, 2)` | `(6, 5, 4, 3, 2, 1, 0)` | 2,520 | 3.541 us +/- 0.037 | 2.309 us +/- 0.008 | 1.53x | 3.942 us +/- 0.155 | -10.2% |
| `rank7_offset_mixed` | `(3, 5, 7, 2, 3, 2, 2)` | `(2, 0, 4, 6, 1, 3, 5)` | 2,520 | 4.288 us +/- 0.123 | 2.279 us +/- 0.017 | 1.88x | 4.323 us +/- 0.091 | -0.8% |
| `rank8_offset_reverse` | `(3, 5, 7, 2, 3, 2, 2, 2)` | `(7, 6, 5, 4, 3, 2, 1, 0)` | 5,040 | 7.227 us +/- 0.135 | 2.452 us +/- 0.015 | 2.95x | 7.773 us +/- 0.063 | -7.0% |
| `rank8_offset_mixed` | `(3, 5, 7, 2, 3, 2, 2, 2)` | `(2, 0, 4, 6, 1, 7, 3, 5)` | 5,040 | 8.772 us +/- 0.057 | 2.384 us +/- 0.008 | 3.68x | 9.661 us +/- 0.083 | -9.2% |
| `rank9_heldout_reverse` | `(3, 5, 7, 2, 2, 2, 3, 2, 2)` | `(8, 7, 6, 5, 4, 3, 2, 1, 0)` | 10,080 | 14.075 us +/- 0.348 | 2.748 us +/- 0.036 | 5.12x | 349.249 us +/- 6.276 | -96.0% |
| `rank9_heldout_mixed` | `(3, 5, 7, 2, 2, 2, 3, 2, 2)` | `(2, 0, 6, 8, 4, 1, 7, 3, 5)` | 10,080 | 17.176 us +/- 0.371 | 2.697 us +/- 0.025 | 6.37x | 337.194 us +/- 2.856 | -94.9% |
| `rank9_heldout_singleton` | `(3, 1, 5, 2, 1, 7, 2, 2, 3)` | `(5, 0, 8, 2, 6, 3, 7, 4, 1)` | 2,520 | 8.351 us +/- 0.083 | 2.354 us +/- 0.027 | 3.55x | 84.545 us +/- 0.214 | -90.1% |

Rank-7 and rank-8 current timings show no regression against the clean base
revision in this run; all four measured cases are faster than base by at least
0.8%.
