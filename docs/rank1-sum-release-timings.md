# Rank-1 `Tensor.sum` Release Timings

Date: 2026-08-30

Base revision: `4945955`

Command shape: release `maturin build --release --locked` wheels for both the
current worktree and a clean `HEAD` export under `target/rank1-sum-run-*`.
Each wheel was force-installed into its own `target/rank1-sum-run-*/benchmark-env`
environment before timing. The timing driver ran inside each process after
imports and input construction, pinned with `taskset -c 0`, with 9 warmup
blocks and 51 measured blocks. Each block repeated the eager
`tensor.sum().item()` call enough times to materialize the scalar result and
avoid deferred work. Before timing, each `torch_rs` scalar result and
`torch_rs.sum(input)` result were checked bitwise against the equivalent PyTorch
result.

Environment:

- CPU: AMD EPYC 9654 96-Core Processor
- OS: Linux 6.13.2-0_fbk12_0_g0b66b3635210 x86_64, glibc 2.34
- Python: 3.12.12
- Rust: `rustc 1.92.0-nightly (2300c2aef 2025-10-12)`,
  `cargo 1.92.0-nightly (81c3f77a4 2025-10-10)`
- PyTorch: 2.13.0+cu130
- NumPy: 2.5.2
- `torch_rs`: 0.1.0
- Profile: release, Cargo `[profile.release]` with thin LTO and one codegen unit
- Device/dtype: CPU float32; CUDA was hidden with `CUDA_VISIBLE_DEVICES=`
- Threads: `taskset -c 0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`
- Dependency installation: 1.96s wall time for the current benchmark
  environment and 1.98s for the clean-base benchmark environment, using the
  worktree-local uv cache
- Compile time: current wheel 31.24s wall time; clean-base wheel 32.11s wall
  time

The rank-1 held-out workloads are offset, non-contiguous vectors selected from
transposed matrices. Contiguous rank-1 and rank-2 non-contiguous workloads are
controls. Times are median microseconds per `sum().item()` call; MAD is median
absolute deviation in microseconds.

| Workload | Shape | Stride | Elements | Current `torch_rs` | PyTorch | Current / PyTorch | Base `torch_rs` | Current vs Base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank1_heldout_stride3_prime` | `(65521, 3)` selected column 1 | `(3,)` | 65,521 | 55.071 us +/- 0.185 | 14.322 us +/- 0.133 | 3.85x | 477.704 us +/- 1.281 | -88.5% |
| `rank1_heldout_stride17_offset5` | `(32749, 17)` selected column 5 | `(17,)` | 32,749 | 28.579 us +/- 0.142 | 27.167 us +/- 0.144 | 1.05x | 235.831 us +/- 6.046 | -87.9% |
| `rank1_heldout_stride33_tail` | `(8191, 33)` selected column 23 | `(33,)` | 8,191 | 7.155 us +/- 0.019 | 7.235 us +/- 0.034 | 0.99x | 56.713 us +/- 0.135 | -87.4% |
| `rank1_contiguous_control_prime` | `(131071,)` | `(1,)` | 131,071 | 109.776 us +/- 0.221 | 7.092 us +/- 0.218 | 15.48x | 109.865 us +/- 0.172 | -0.1% |
| `rank1_contiguous_control_tail` | `(4099,)` | `(1,)` | 4,099 | 3.586 us +/- 0.009 | 1.938 us +/- 0.009 | 1.85x | 3.553 us +/- 0.006 | +0.9% |
| `rank2_noncontiguous_control_transpose` | `(257, 509)` transposed | `(1, 509)` | 130,813 | 110.437 us +/- 0.254 | 7.116 us +/- 0.250 | 15.52x | 109.624 us +/- 0.186 | +0.7% |
| `rank2_noncontiguous_control_offset` | `(251, 503)` offset transposed | `(1, 503)` | 126,253 | 106.580 us +/- 0.257 | 6.807 us +/- 0.233 | 15.66x | 106.028 us +/- 0.246 | +0.5% |

The contiguous rank-1 and rank-2 non-contiguous controls stayed within the
no-greater-than-5% regression guardrail in this run. The held-out rank-1
strided views moved from the generic strided fallback to the owned rank-1 fold
path and improved by 87.4% to 88.5% versus the clean base.
