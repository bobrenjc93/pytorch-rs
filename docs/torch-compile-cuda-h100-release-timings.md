# H100 CUDA `torch.compile` Shape-Matrix Timings

Date: 2026-09-07

Candidate provenance: current worktree at
`0bb026bf4aa200cf197e37f29a6964cd070721c6`; the raw JSON records the dirty
implementation and benchmark-harness diff used for this uncommitted review
candidate before writing the refreshed artifact.

Measurement command:

```bash
CUDA_VISIBLE_DEVICES=0 \
TMPDIR="$PWD/target/tmp" \
TORCHINDUCTOR_CACHE_DIR="$PWD/target/torchinductor-cache312" \
TRITON_CACHE_DIR="$PWD/target/triton-cache312" \
XDG_CACHE_HOME="$PWD/target/xdg-cache312" \
.venv312/bin/python scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison \
  --output target/clean-evidence/torch-compile-cuda-h100-shape-matrix-v11.json
```

The generated target JSON was copied verbatim to
`docs/benchmark-data/torch-compile-cuda-h100-shape-matrix-v11.json`.

Additional checks run after refreshing this evidence:

```bash
CUDA_VISIBLE_DEVICES=0 .venv312/bin/python -m unittest tests.test_compile_cuda_benchmark
CUDA_VISIBLE_DEVICES= .venv312/bin/python -m unittest tests.test_top_level_compile tests.test_compile_benchmark_artifact tests.test_readme_quickstart tests.test_torch_compile_coverage_evaluator tests.test_allclose tests.test_allclose_reference
CUDA_VISIBLE_DEVICES= .venv312/bin/python scripts/benchmark_compile_cpu.py --validate-artifact
.venv312/bin/python -m compileall -q python/torch_rs/__init__.py python/torch_rs/_cuda_pointwise_reduce_workload.py scripts/benchmark_compile_cuda.py scripts/benchmark_compile_cpu.py tests/test_compile_cuda_benchmark.py tests/test_compile_benchmark_artifact.py tests/test_allclose.py tests/test_allclose_reference.py
cargo test --lib
cargo fmt --check
```

Environment recorded by the JSON artifact:

- Python: 3.12.14+meta
- GPU: NVIDIA H100, compute capability 9.0, driver 580.82.07
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0
- `nvcc`: Cuda compilation tools, release 12.6, V12.6.85
- Benchmark: `torch_compile_cuda_h100_reference_benchmark_v11`
- Report schema: `torch_compile_cuda_h100_shape_matrix_report_v1`
- Workload expression: `h100_cuda_pointwise_reduce_float32_v1`
- Shape matrix: `h100_cuda_pointwise_reduce_float32_shape_matrix_v1`
- Dtype: `torch.float32`
- Timing: 5 warmups, 17 samples, 3 repeated calls per sample
- Timing boundary: both PyTorch and `torch_rs` synchronize before and after
  timed compiled calls; output checksum materialization occurs after the timed
  region for both paths. `torch_rs` also releases intermediate repeated-call
  outputs after the timed region, stops the clock immediately after the timed
  post-call `cudaDeviceSynchronize`, and hoists CUDA device selection plus
  prepared executor setup into `factory_us`.

Cold-call accounting is intentionally explicit in the JSON. PyTorch
`cold_first_call_us` times the first compiled invocation after
`torch.compile(...)` wrapper construction, so deferred Inductor graph/code
generation is included there. `torch_rs` `factory_us` includes wrapper creation,
CUDA executor preparation, kernel build/load reuse, device selection, and output
pool preallocation; its `cold_first_call_us` begins after that preparation. The
score therefore uses steady-state latency only.

The fixed matrix uses equal documented weights:

| Workload | Shape | Seed | Weight |
| --- | ---: | ---: | ---: |
| `square_256x256` | `(256, 256)` | 20260904 | 0.25 |
| `square_1024x1024` | `(1024, 1024)` | 20260905 | 0.25 |
| `tall_4096x256` | `(4096, 256)` | 20260906 | 0.25 |
| `wide_256x4096` | `(256, 4096)` | 20260907 | 0.25 |

Results:

| Workload | Shape | Weight | PyTorch cold us | PyTorch steady median us | `torch_rs` cold us | `torch_rs` steady median us | Ratio | Score contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `square_256x256` | `(256, 256)` | 0.25 | 1251853.947 | 44.978 | 203.268 | 38.802 | 1.159x | 25.00 |
| `square_1024x1024` | `(1024, 1024)` | 0.25 | 43367.941 | 46.440 | 144.749 | 39.887 | 1.164x | 25.00 |
| `tall_4096x256` | `(4096, 256)` | 0.25 | 42679.609 | 52.970 | 167.954 | 43.396 | 1.221x | 25.00 |
| `wide_256x4096` | `(256, 4096)` | 0.25 | 41026.269 | 48.016 | 204.590 | 40.324 | 1.191x | 25.00 |

Aggregate:

- Common-success geometric-mean speed ratio: 1.1835x across 4/4 shapes.
- Coverage-adjusted capped ratio: 1.0000.
- CUDA compile score: 100.00%.
- Zero-credit cells retained in denominator: 0.

Correctness evidence remained fail-closed for every shape: the candidate ran on
CUDA device 0, used `backend="inductor"`, `fullgraph=True`, `dynamic=False`,
reported `native_cuda_compile=True`, rejected eager/PyTorch forwarding,
synchronized around timed launches, deferred readback until after timing, and
matched the PyTorch output checksum. This matrix is forward-output parity
evidence only; it does not claim CUDA training compile parity because it does
not include gradient-bearing compiled CUDA workloads. The optional `--quick`
mode is a strict subset containing only `square_1024x1024`; final evidence uses
the full matrix.
