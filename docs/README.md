# Documentation Index

Use this page to find the durable project contracts, runnable examples,
contributor guides, and historical validation and timing evidence. Burner-managed progress
artifacts are generated at merge time and are not source documentation.

## Current Contracts

- [Supported surface](supported-surface.md): Exhaustive Python API coverage and unsupported boundary contract.
- [Feature coverage contract](../FEATURES.md): Weighted feature areas and what counts toward coverage.
- [Benchmark policy](../BENCHMARKING.md): Correctness gates, measurement rules, provenance, and anti-gaming policy.
- [Compile coverage evaluator](torch-compile-coverage-evaluator.md): Command-backed Burner evaluator for the v4 reference-eligible `torch.compile` corpus.
- [Hardware heterogeneity evaluator](hardware-heterogeneity-evaluator.md): Fixed accelerator-family and feature-depth matrix, evidence rules, and real-hardware scoring policy.
- [Generated creation validator](../scripts/validate_creation_factory_benchmark.py): Held-out seeded shape path for creation-factory benchmark review.

## Examples

- [First-success example](../examples/first_success.py): Runnable version of the README first-success assertions.

## Contributor Guides

- [Contributing guide](../CONTRIBUTING.md): Locked setup, environment expectations, test selection, draft workflow, and documentation ownership.
- [Setup troubleshooting](troubleshooting.md): Short fixes for common environment, import, reference dependency, and stale wheel failures.
- [Repository README](../README.md): Install commands, first-success example, scope summary, and validation entry points.
- [Architecture map](../ARCHITECTURE.md): Source map for the Rust core, Python bindings, wrappers, and test layout.
- [Compiled CUDA reshape](compile-cuda-reshape.md): Constant rank-0/1/2 shapes, native alias-or-pack semantics and H100 validation.
- [Compiled CUDA transpose views](compile-cuda-t.md): Bounded `t()`/constant-axis `transpose()` capture, packing composition and H100 validation.
- [Compiled CUDA contiguous](compile-cuda-contiguous.md): Parameterless native view packing, alias semantics and clean-commit H100 validation including review regressions.
- [Compiled CUDA matmul](compile-cuda-matmul.md): Exact capture scope, runnable example, and separate `torch.compile` timing reproduction.
- [Compiled CUDA row sums](compile-cuda-sum-rows.md): Runnable example, constant reduction options, layout/device boundaries, and focused checks.
- [CUDA arithmetic and ReLU graph capture](compile-cuda-add.md): Public bounded eager compilation, device/cache guards, unsupported cases, and reproduction.
- [Compiled CUDA scalar multiplication validation](compile-cuda-mul-scalar-validation.md): Guarded scalar grammar, independent diagnostics, and H100 checks.
- [Compiled CUDA negation validation](compile-cuda-neg-validation.md): Integrated-commit results, provenance, and current neg/add diagnostic commands.
- [CUDA scalar multiplication validation](cuda-mul-scalar-validation.md): General eager float32 kernel, conversion/layout boundaries, and H100 evidence.
- [Native and compiled CUDA ReLU](cuda-relu.md): Method capture, IEEE bit semantics, layout bounds and H100 development checks.
- [CUDA negation validation](cuda-neg-validation.md): Contiguous float32 eager scope, H100 checks, and retained evaluation evidence.
- [CUDA view packing validation](cuda-contiguous-validation.md): Bounded contiguous/reshape materialization, aliasing contracts and clean-commit H100 correctness evidence.
- [Text collation diagnostics](text-collation-diagnostics.md): Public-call timings and identity checks for text-led metadata passthrough.
- [CUDA-add diagnostics](cuda-add-diagnostics.md): Latency, sustained throughput, cache saturation, and release provenance.
- [CUDA matrix/vector addition](cuda-add-trailing-vector.md): Bounded eager broadcast support, H100 differentials, and six-case math validation.
- [CUDA vector addition and depth-stack integration](composite-cuda-vector-dstack.md): Records composite repairs, diagnostics, and the clean-commit delivery gate.
- [Native CUDA matrix multiplication](cuda-matmul.md): Contiguous same-device rank-2 float32 matmul, native cuBLAS setup, numerical regressions, and capture commands.
- [CUDA matrix row sums](cuda-sum-rows.md): Contiguous rank-2 float32 dim=1/-1 reduction, numerical behavior, and validation.
- [Rank-2 mean diagnostics](rank2-mean-diagnostics.md): Both reduction axes and layouts at matched CPU worker budgets.

## Historical Validation Evidence

- [Compiler analysis and CPU tanh composite validation](diagnostics/compile-cuda-graph/composite-postcommit-d94daecd/README.md): Historical measurements of implementation `d94daecd`, with
  [source clean](diagnostics/compile-cuda-graph/postcommit-4cb0432/README.md) (`4cb0432`) and
  [source development](diagnostics/compile-cuda-graph/static-analysis-b7936239-development/README.md) predecessors.
  Measured revisions are distinct from evidence publication commits; each report retains its provenance and performance caveats.
- [Compiled CUDA row-sum validation](compile-cuda-sum-rows-validation.md): Commit-bound source captures, retained development attempts, and timing-artifact limitations.
- [Matmul diagnostic index](diagnostics/composite-matmul-unflatten-l1/README.md): Current evidence and earlier source/composite captures, with their original identities and raw results.
- [Matmul, unflatten and L1 integration history](composite-matmul-unflatten-l1-validation.md): Repair history, development validation and managed handoff notes.
- [CUDA addition capture validation](compile-cuda-add-validation.md): Source validation history, baseline caveats, and composite correctness evidence; no performance score.

## Historical Timing Evidence

These reports are historical release evidence snapshots, not live benchmark
gates.

### Reductions

- [Rank-1 sum timings](rank1-sum-release-timings.md): Rank-1 `Tensor.sum` release evidence.
- [Rank-9 sum timings](rank9-sum-release-timings.md): Rank-9 `Tensor.sum` release evidence.
- [Rank-10 sum timings](rank10-sum-release-timings.md): Rank-10 `Tensor.sum` release evidence.
- [Rank-11 sum timings](rank11-sum-release-timings.md): Rank-11 `Tensor.sum` release evidence.
- [Rank-12 sum timings](rank12-sum-release-timings.md): Rank-12 `Tensor.sum` release evidence.
- [Mean timings](tensor-mean-release-timings.md): Full-reduction `Tensor.mean` and `torch.mean` release evidence.

### Creation

- [Creation factory timings](creation-factory-release-timings.md): `torch.empty`, `torch.zeros`, and `torch.ones` eager CPU factory benchmark coverage.

### Elementwise ops

- [Addition timings](tensor-add-release-timings.md): `+` and `Tensor.add` release evidence.
- [Subtraction timings](top-level-subtract-release-timings.md): `torch.sub` and `torch.subtract` release evidence with retained raw JSON in [benchmark-data/top-level-subtract-release-timings.json](benchmark-data/top-level-subtract-release-timings.json).
- [Stack timings](top-level-stack-release-timings.md): `torch.stack` same-shape release evidence with retained raw JSON in [benchmark-data/top-level-stack-release-timings.json](benchmark-data/top-level-stack-release-timings.json).
- [Multiplication timings](tensor-mul-release-timings.md): `*`, `Tensor.mul`/`Tensor.multiply`, and top-level multiplication release evidence.
- [Division timings](top-level-division-release-timings.md): `torch.div` and `torch.divide` release evidence.
- [Absolute value timings](tensor-abs-release-timings.md): `Tensor.abs` and `torch.abs` release evidence.
- [Square-root timings](tensor-sqrt-release-timings.md): `Tensor.sqrt` and `torch.sqrt` release evidence.
- [Reciprocal timings](tensor-reciprocal-release-timings.md): `Tensor.reciprocal` and `torch.reciprocal` release evidence.
- [Softsign timings](softsign-release-timings.md): `torch.nn.functional.softsign` release evidence.

### Compilation

- [Compile CPU timings](torch-compile-cpu-release-timings.md): `torch.compile(..., backend="eager")` fullgraph and no-break `fullgraph=False` CPU release evidence with retained raw JSON in [benchmark-data/torch-compile-cpu-v4.json](benchmark-data/torch-compile-cpu-v4.json).
- [Compile H100 CUDA prepared-executor timings](torch-compile-cuda-h100-release-timings.md): `torch.compile(..., backend="inductor")` H100 CUDA four-shape forward-output prepared-executor release evidence with retained raw JSON in [benchmark-data/torch-compile-cuda-h100-shape-matrix-v11.json](benchmark-data/torch-compile-cuda-h100-shape-matrix-v11.json).

### Layout/view ops

- [View and reshape timings](tensor-view-release-timings.md): View, reshape, flatten, ravel, unbind, and edge-unsqueeze release evidence.

### Linear algebra

- [Rank-2 matmul timings](rank2-matmul-release-timings.md): Rank-2 `@`, `Tensor.matmul`, and `torch.matmul` release evidence.

### NN losses

- [MSE loss timings](mse-loss-release-timings.md): `torch.nn.functional.mse_loss` release evidence.
- [L1 loss timings](l1-loss-release-timings.md): `torch.nn.functional.l1_loss(reduction="none")` release evidence.
- [L1 loss sum timings](l1-loss-sum-release-timings.md): `torch.nn.functional.l1_loss(reduction="sum")` release evidence.

[Compiled CUDA matrix/vector addition](compile-cuda-trailing-vector-validation.md)
records the bounded shape extension and non-scoring diagnostic commands.
