# H100 CUDA `torch.compile` Prepared-Executor Timings

Date: 2026-09-06

Candidate provenance: composite worktree at
`92f7fb6278b5a6415f2b4cbba84d30e0ece7c640` with the review-response
architecture note present in the artifact git status.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison \
  --output docs/benchmark-data/torch-compile-cuda-h100-prepared-executor-v8.json
```

Checks run for this evidence:

```bash
.venv/bin/python -m py_compile \
  python/torch_rs/_cuda_pointwise_reduce_workload.py \
  python/torch_rs/__init__.py \
  scripts/benchmark_compile_cuda.py \
  tests/test_compile_cuda_benchmark.py
git diff --check -- \
  python/torch_rs/_cuda_pointwise_reduce_workload.py \
  python/torch_rs/__init__.py \
  scripts/benchmark_compile_cuda.py \
  tests/test_compile_cuda_benchmark.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest tests.test_compile_cuda_benchmark
```

Environment recorded by the JSON artifact:

- GPU: NVIDIA H100, compute capability 9.0, driver 580.82.07
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0
- `nvcc`: CUDA compilation tools 12.6, V12.6.85
- Benchmark: `torch_compile_cuda_h100_reference_benchmark_v8`
- Workload: `h100_cuda_pointwise_reduce_float32_v1`, shape `(1024, 1024)`,
  dtype `torch.float32`, seed `20260904`
- Timing: 5 warmups, 17 samples, 3 repeated calls per sample

Results:

| Measurement | Steady median us | MAD us | Notes |
| --- | ---: | ---: | --- |
| PyTorch 2.13 `torch.compile(..., backend="inductor")` | 238.688 | 5.051 | Reference workload on the same visible H100 |
| `torch_rs` prepared compile wrapper | 1115.080 | 15.370 | Eligible native CUDA compile evidence |
| `torch_rs` unprepared compatibility call | 12825.971 | 575.734 | Non-scoring comparison that prepares on every invocation |

The prepared wrapper recorded executor invocation count 0 before the first
call and 67 after timing; the last steady-state call used the same preparation
id `de8a8200afa4686b`. Cold compile wrapper creation took 13530.280 us, and
the first compiled call took 1445.466 us. The measured prepared-vs-unprepared
steady-state speedup was 11.50x. The candidate remains slower than the PyTorch
reference, with a coverage-adjusted CUDA compile score of 21.41% for this
single workload.

Correctness evidence remained fail-closed: the candidate ran on CUDA device 0,
used `backend="inductor"`, `fullgraph=True`, `dynamic=False`, reported
`native_cuda_compile=True`, rejected eager/PyTorch forwarding, synchronized
the launch/readback path, and matched the PyTorch output checksum
`72f74be90b99aea7`.
