# H100 CUDA `torch.compile` Shape-Matrix Timings

Date: 2026-09-09 (UTC)

Measured composite commit: `e2f40ff16f8aba5216bc51699b56571e7e82a3e4`.
The raw JSON records empty `git.status_short` and `git.diff_stat` at capture.
Interpreter, imports, and generated kernel paths belong to this composite
worktree. Only benchmark evidence and reports were updated after measurement.

Setup costs: [same-code setup rerun](benchmark-setup/2026-09-09-e2f40ff/setup.json)
([timings and limitations](benchmark-setup.md)); original setup timings are unavailable.

Build and cache configuration: [shared measurement setup](top-level-stack-release-timings.md).

Measurement command, using the shared environment:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_compile_cuda.py --include-unprepared-comparison --output target/evidence/torch-compile-cuda-h100-shape-matrix-v11.json
```

The full four-shape matrix ran on GPU 0: NVIDIA H100 (compute capability 9.0),
driver 580.82.07, PyTorch 2.13.0+cu130 with CUDA runtime 13.0, and nvcc 12.6.85.
The JSON includes the selected CUDA libraries and kernel compiler commands.
Timing uses five warmups, 17 samples, and three calls per sample, with CUDA
synchronization around timing and output checksums materialized afterward.

Cold-call accounting differs between implementations: PyTorch's first call
includes deferred Inductor compilation, while the candidate prepares its
executor and output pool in `factory_us`. The score uses steady-state latency.
This is forward-output evidence; it does not establish CUDA training parity.
The optional unprepared comparison remains non-scoring evidence.

CUDA speed ratios are PyTorch latency divided by `torch_rs` latency, so higher
is better. The coverage-adjusted score caps each shape's ratio at parity.

## Results

| Workload | Shape | Weight | PyTorch cold us | PyTorch steady median us | `torch_rs` cold us | `torch_rs` steady median us | Ratio | Score contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `square_256x256` | `(256, 256)` | 0.25 | 3307152.460 | 46.934 | 180.513 | 39.553 | 1.187x | 25.00 |
| `square_1024x1024` | `(1024, 1024)` | 0.25 | 913568.239 | 49.408 | 171.019 | 40.465 | 1.221x | 25.00 |
| `tall_4096x256` | `(4096, 256)` | 0.25 | 898734.419 | 62.708 | 161.004 | 42.494 | 1.476x | 25.00 |
| `wide_256x4096` | `(256, 4096)` | 0.25 | 358766.431 | 46.187 | 175.976 | 39.710 | 1.163x | 25.00 |

- Common-success geometric-mean speed ratio: 1.2558x across 4/4 shapes.
- Coverage-adjusted capped ratio: 1.0000.
- CUDA compile score: 100.00%.
- Zero-credit cells retained in denominator: 0.
