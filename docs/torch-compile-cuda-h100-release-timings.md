# H100 CUDA `torch.compile` Shape-Matrix Timings

Date: 2026-09-06

Candidate provenance: clean current worktree
`18519b830bc5deb28bac363197b54502a251148b`; the raw JSON records empty
`git.status_short` and `git.diff_stat` before writing the refreshed artifact.

Measurement command:

```bash
CUDA_VISIBLE_DEVICES=0 \
TMPDIR="$PWD/target/tmp" \
TORCHINDUCTOR_CACHE_DIR="$PWD/target/torchinductor-cache" \
TRITON_CACHE_DIR="$PWD/target/triton-cache" \
XDG_CACHE_HOME="$PWD/target/xdg-cache" \
.venv/bin/python scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison \
  --output target/clean-evidence/torch-compile-cuda-h100-shape-matrix-v11.json
```

The generated target JSON was copied verbatim to
`docs/benchmark-data/torch-compile-cuda-h100-shape-matrix-v11.json`.

Additional checks run after refreshing this evidence:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest tests.test_compile_cuda_benchmark
.venv/bin/python -m unittest tests.test_readme_quickstart
.venv/bin/python -m unittest tests.test_torch_compile_coverage_evaluator
.venv/bin/python -m unittest tests.test_compile_benchmark_artifact
.venv/bin/python -m compileall -q python/torch_rs/_cuda_pointwise_reduce_workload.py scripts/benchmark_compile_cuda.py tests/test_compile_cuda_benchmark.py tests/test_readme_quickstart.py
cargo test --lib
cargo fmt --check
```

Environment recorded by the JSON artifact:

- GPU: NVIDIA H100, compute capability 9.0, driver 580.82.07
- PyTorch: 2.13.0+cu130, CUDA runtime 13.0
- `nvcc`: CUDA compilation tools 12.6, V12.6.85
- Benchmark: `torch_compile_cuda_h100_reference_benchmark_v11`
- Report schema: `torch_compile_cuda_h100_shape_matrix_report_v1`
- Workload expression: `h100_cuda_pointwise_reduce_float32_v1`
- Shape matrix: `h100_cuda_pointwise_reduce_float32_shape_matrix_v1`
- Dtype: `torch.float32`
- Timing: 5 warmups, 17 samples, 3 repeated calls per sample
- Timing boundary: both PyTorch and `torch_rs` synchronize before and after
  timed compiled calls; output checksum materialization occurs after the timed
  region for both paths. `torch_rs` also releases intermediate repeated-call
  outputs after the timed region and hoists CUDA device selection plus prepared
  executor setup into `factory_us`.

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
| `square_256x256` | `(256, 256)` | 0.25 | 1153708.951 | 40.571 | 216.418 | 39.867 | 1.018x | 25.00 |
| `square_1024x1024` | `(1024, 1024)` | 0.25 | 39109.056 | 44.080 | 137.698 | 40.895 | 1.078x | 25.00 |
| `tall_4096x256` | `(4096, 256)` | 0.25 | 40097.892 | 45.779 | 147.763 | 43.800 | 1.045x | 25.00 |
| `wide_256x4096` | `(256, 4096)` | 0.25 | 38987.252 | 38.241 | 174.064 | 40.204 | 0.951x | 23.78 |

Aggregate:

- Common-success geometric-mean speed ratio: 1.0219x across 4/4 shapes.
- Coverage-adjusted capped ratio: 0.9878.
- CUDA compile score: 98.78%.
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
