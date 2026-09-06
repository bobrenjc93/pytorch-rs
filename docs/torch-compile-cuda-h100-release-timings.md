# H100 CUDA `torch.compile` Runtime-Ownership Timings

Date: 2026-09-06

Candidate provenance: composite worktree at
`18aa88a6ba9153d840c1f7e9ba9b5f54cba149ff` with the runtime-ownership
implementation changes listed in the artifact git status.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=target/test-python-site-2 \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python \
  scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison \
  --output docs/benchmark-data/torch-compile-cuda-h100-runtime-ownership-v9.json
```

Checks run for this evidence:

```bash
python -m py_compile \
  python/torch_rs/_cuda_runtime_ownership.py \
  python/torch_rs/_cuda_benchmark_tensor.py \
  python/torch_rs/_cuda_pointwise_reduce_workload.py \
  python/torch_rs/_compiler_state.py \
  python/torch_rs/__init__.py \
  scripts/benchmark_compile_cuda.py \
  tests/test_compile_cuda_benchmark.py
PYTHONPATH=target/test-python-site-2 python -m unittest \
  tests.test_compile_cuda_benchmark
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=target/test-python-site-2 \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python -m unittest \
  tests.test_compile_cuda_benchmark
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=target/test-python-site-2 \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python -m unittest \
  tests.test_compile_cuda_benchmark.CompileCudaBenchmarkTests.\
test_torch_compile_inductor_pointwise_reduce_restores_device0_before_launch_on_h100
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=target/test-python-site-2 \
  /data/users/bobren/a/pytorch-rs-burner/.venv/bin/python -m unittest \
  discover -s tests -p 'test_*.py'
env CARGO_HOME="$PWD/target/cargo-home" \
  CARGO_TARGET_DIR="$PWD/target/cargo-build" cargo test --locked
```

Environment recorded by the JSON artifact:

- GPU: NVIDIA H100, compute capability 9.0, driver 580.82.07
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0
- `nvcc`: CUDA compilation tools 12.6, V12.6.85
- Benchmark: `torch_compile_cuda_h100_reference_benchmark_v9`
- Workload: `h100_cuda_pointwise_reduce_float32_v1`, shape `(1024, 1024)`,
  dtype `torch.float32`, seed `20260904`
- Timing: 5 warmups, 17 samples, 3 repeated calls per sample
- Timing boundary: explicit CUDA synchronization before and after timed
  compiled calls; output checksum materialization occurs after the timed region

Results:

| Measurement | Steady median us | MAD us | Notes |
| --- | ---: | ---: | --- |
| PyTorch 2.13 `torch.compile(..., backend="inductor")` | 187.053 | 4.183 | Reference workload on the same visible H100 |
| `torch_rs` prepared compile wrapper | 290.863 | 2.340 | Eligible native CUDA compile evidence using pooled outputs |
| `torch_rs` unprepared compatibility call | 10650.330 | 345.022 | Non-scoring comparison that prepares on every invocation |

The prepared wrapper recorded executor invocation count 0 before the first
call and 67 after timing; the last steady-state call used the same preparation
id `8b49f61b273d8214`. Cold compile wrapper creation took 11812.347 us, and
the first compiled call took 448.889 us. The output pool allocated two buffers,
released 67 leases, had zero live buffers after timing, and reused a released
buffer for the steady-state path. The measured prepared-vs-unprepared
steady-state speedup was 36.62x. The candidate remains slower than the PyTorch
reference, with a coverage-adjusted CUDA compile score of 64.31% for this
single workload.

Correctness evidence remained fail-closed: the candidate ran on CUDA device 0,
used `backend="inductor"`, `fullgraph=True`, `dynamic=False`, reported
`native_cuda_compile=True`, rejected eager/PyTorch forwarding, synchronized
around the timed launch, deferred readback until after timing, and matched the
PyTorch output checksum
`72f74be90b99aea7`.
