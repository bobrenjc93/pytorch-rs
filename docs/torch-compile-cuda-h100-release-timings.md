# H100 CUDA `torch.compile` Shape-Matrix Timings

Date: 2026-09-06

Candidate provenance: worktree at
`8e4a9f4cc9026534a39ca996696d300e418f351d`. The JSON artifact records
the git status and diff stat observed before writing the refreshed output.

Measurement command, written to an ignored `target/` path before copying the
refreshed JSON into the checked-in artifact path:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison \
  --output target/regenerated-benchmarks/torch-compile-cuda-h100-shape-matrix-v11.json
```

Additional checks run after refreshing this evidence:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest tests.test_compile_cuda_benchmark
.venv/bin/python -m unittest tests.test_readme_quickstart
.venv/bin/python -m unittest tests.test_torch_compile_coverage_evaluator
.venv/bin/python -m unittest tests.test_compile_benchmark_artifact
.venv/bin/python -m compileall -q scripts/benchmark_compile_cuda.py tests/test_compile_cuda_benchmark.py
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
  region for both paths

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
| `square_256x256` | `(256, 256)` | 0.25 | 1155075.001 | 45.809 | 197.639 | 45.385 | 1.009x | 25.00 |
| `square_1024x1024` | `(1024, 1024)` | 0.25 | 43592.623 | 43.369 | 155.345 | 44.254 | 0.980x | 24.50 |
| `tall_4096x256` | `(4096, 256)` | 0.25 | 42833.735 | 55.531 | 153.121 | 46.484 | 1.195x | 25.00 |
| `wide_256x4096` | `(256, 4096)` | 0.25 | 37173.773 | 39.016 | 170.869 | 43.352 | 0.900x | 22.50 |

Aggregate:

- Common-success geometric-mean speed ratio: 1.0155x across 4/4 shapes.
- Coverage-adjusted capped ratio: 0.9700.
- CUDA compile score: 97.00%.
- Zero-credit cells retained in denominator: 0.

Correctness evidence remained fail-closed for every shape: the candidate ran on
CUDA device 0, used `backend="inductor"`, `fullgraph=True`, `dynamic=False`,
reported `native_cuda_compile=True`, rejected eager/PyTorch forwarding,
synchronized around timed launches, deferred readback until after timing, and
matched the PyTorch output checksum. The optional `--quick` mode is a strict
subset containing only `square_1024x1024`; final evidence uses the full matrix.
