# H100 CUDA `torch.compile` Runtime-Ownership Timings

Date: 2026-09-06

Candidate provenance: clean worktree at
`01655e1b8b252e2057b2371e43f00241eea90eab`. The JSON artifact records
empty `git.status_short` and `git.diff_stat` before writing the refreshed
output.

Measurement command, written to an ignored `target/` path before copying the
refreshed JSON into the checked-in artifact path:

```bash
CUDA_VISIBLE_DEVICES=0 target/torch-compile-coverage/venv/bin/python \
  scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison \
  --output target/regenerated-benchmarks/torch-compile-cuda-h100-runtime-ownership-v10.json
```

Additional checks run after refreshing this evidence:

```bash
CUDA_VISIBLE_DEVICES=0 target/torch-compile-coverage/venv/bin/python -m unittest tests.test_compile_cuda_benchmark
target/torch-compile-coverage/venv/bin/python -m unittest tests.test_compile_benchmark_artifact
target/torch-compile-coverage/venv/bin/python -m unittest tests.test_top_level_compile
target/torch-compile-coverage/venv/bin/python -m unittest tests.test_torch_compile_coverage_evaluator
target/torch-compile-coverage/venv/bin/python -m compileall -q \
  python/torch_rs scripts/benchmark_compile_cuda.py tests/test_compile_cuda_benchmark.py
cargo fmt --check
```

Environment recorded by the JSON artifact:

- GPU: NVIDIA H100, compute capability 9.0, driver 580.82.07
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0
- `nvcc`: CUDA compilation tools 12.6, V12.6.85
- Benchmark: `torch_compile_cuda_h100_reference_benchmark_v10`
- Workload: `h100_cuda_pointwise_reduce_float32_v1`, shape `(1024, 1024)`,
  dtype `torch.float32`, seed `20260904`
- Native CUDA source checksum: `db19ead9f1bddd35e91e1de643e0b810`
- Timing: 5 warmups, 17 samples, 3 repeated calls per sample
- Timing boundary: both PyTorch and `torch_rs` synchronize before and after
  timed compiled calls; output checksum materialization occurs after the timed
  region for both paths

Results:

| Measurement | Steady median us | MAD us | Notes |
| --- | ---: | ---: | --- |
| PyTorch 2.13 `torch.compile(..., backend="inductor")` | 48.634 | 2.200 | Reference workload on the same visible H100 |
| `torch_rs` prepared compile wrapper | 53.517 | 0.948 | Eligible native CUDA compile evidence using pooled outputs |
| `torch_rs` unprepared compatibility call | 11757.638 | 522.090 | Non-scoring comparison that prepares on every invocation |

The prepared wrapper recorded executor invocation count 0 before the first
call and 67 after timing; the last steady-state call used the same preparation
id `01ad652f08d9d291`. Cold compile wrapper creation took 13080.746 us, and
the first compiled call took 192.391 us. The output pool allocated two buffers,
released 67 leases, had zero live buffers after timing, and reused a released
buffer for the steady-state path. The measured prepared-vs-unprepared
steady-state speedup was 219.70x. The candidate stayed within roughly ten percent of
the PyTorch reference, with a CUDA compile score of 90.87% for this single
workload.

This release artifact is intentionally narrow: it proves one fixed H100
CUDA pointwise-plus-row-reduction workload and should not be treated as broad
CUDA compile coverage until paired with held-out shapes, more operators,
dynamic/fullgraph variants, backward cases where applicable, and explicit
unsupported zero-credit categories.

Correctness evidence remained fail-closed: the candidate ran on CUDA device 0,
used `backend="inductor"`, `fullgraph=True`, `dynamic=False`, reported
`native_cuda_compile=True`, rejected eager/PyTorch forwarding, synchronized
around the timed launch, deferred readback until after timing, and matched the
PyTorch output checksum
`72f74be90b99aea7`.
