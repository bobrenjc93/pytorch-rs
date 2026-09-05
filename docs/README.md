# Documentation Index

Use this page to find the durable project contracts, runnable examples,
contributor guides, and historical timing evidence. Burner-managed progress
artifacts are generated at merge time and are not source documentation.

## Current Contracts

- [Supported surface](supported-surface.md): Exhaustive Python API coverage and unsupported boundary contract.
- [Feature coverage contract](../FEATURES.md): Weighted feature areas and what counts toward coverage.
- [Benchmark policy](../BENCHMARKING.md): Correctness gates, measurement rules, provenance, and anti-gaming policy.
- [Compile coverage evaluator](torch-compile-coverage-evaluator.md): Command-backed Burner evaluator for the v4 reference-eligible `torch.compile` corpus.
- [Generated creation validator](../scripts/validate_creation_factory_benchmark.py): Held-out seeded shape path for creation-factory benchmark review.

## Examples

- [First-success example](../examples/first_success.py): Runnable version of the README first-success assertions.

## Contributor Guides

- [Repository README](../README.md): Install commands, first-success example, scope summary, and validation entry points.
- [Contributing guide](../CONTRIBUTING.md): Locked setup, environment expectations, test selection, draft workflow, and documentation ownership.
- [Setup troubleshooting](troubleshooting.md): Short fixes for common environment, import, reference dependency, and stale wheel failures.
- [Architecture map](../ARCHITECTURE.md): Source map for the Rust core, Python bindings, wrappers, and test layout.

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
- [Multiplication timings](tensor-mul-release-timings.md): `*`, `Tensor.mul`/`Tensor.multiply`, and top-level multiplication release evidence.
- [Division timings](top-level-division-release-timings.md): `torch.div` and `torch.divide` release evidence.
- [Absolute value timings](tensor-abs-release-timings.md): `Tensor.abs` and `torch.abs` release evidence.
- [Square-root timings](tensor-sqrt-release-timings.md): `Tensor.sqrt` and `torch.sqrt` release evidence.
- [Reciprocal timings](tensor-reciprocal-release-timings.md): `Tensor.reciprocal` and `torch.reciprocal` release evidence.
- [Softsign timings](softsign-release-timings.md): `torch.nn.functional.softsign` release evidence.

### Compilation

- [Compile CPU timings](torch-compile-cpu-release-timings.md): `torch.compile(..., backend="eager", fullgraph=True)` CPU release evidence with retained raw JSON in [benchmark-data/torch-compile-cpu-v4.json](benchmark-data/torch-compile-cpu-v4.json).

### Layout/view ops

- [View and reshape timings](tensor-view-release-timings.md): View, reshape, flatten, ravel, unbind, and edge-unsqueeze release evidence.

### Linear algebra

- [Rank-2 matmul timings](rank2-matmul-release-timings.md): Rank-2 `@`, `Tensor.matmul`, and `torch.matmul` release evidence.

### NN losses

- [MSE loss timings](mse-loss-release-timings.md): `torch.nn.functional.mse_loss` release evidence.
- [L1 loss timings](l1-loss-release-timings.md): `torch.nn.functional.l1_loss(reduction="none")` release evidence.
- [L1 loss sum timings](l1-loss-sum-release-timings.md): `torch.nn.functional.l1_loss(reduction="sum")` release evidence.
