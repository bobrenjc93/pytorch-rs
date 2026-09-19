# Validation and timing history

Return to the [task-oriented documentation index](README.md) for current contracts,
examples and live operation guides. [BENCHMARKING.md](../BENCHMARKING.md) owns
reproduction commands and measurement policy. Historical records retain their original
source/build identities; they do not measure later revisions or expand current support.
Legacy 100 scores from eager/custom-backend coverage or private CUDA kernels are not
comparable to the current public-default compiler gates.

For measurement tooling and its limits, see the
[full public-call diagnostic](diagnostics/default-compile-full-call-20260918.md),
its [offline replay guide](diagnostics/default-compile-full-call-20260918-replay.md),
the [profiled guard investigation](diagnostics/profiled-full-call-20260918.md),
and its [clean-commit three-way capture](diagnostics/default-compile-full-call-postcommit-ed614665.md).
The [guard allocation investigation](diagnostics/guard-allocation-20260918.md)
records discarded author prototypes and their replayable evidence.
The [direct executable resource experiment](diagnostics/direct-resources-20260918.md)
records donor integration, direct-only resource removal and their separate author captures.

## Historical Validation Evidence

- [Legacy eager compile evaluator](torch-compile-coverage-evaluator.md): Historical eager/custom-backend diagnostic, no longer the default compiler scoring gate.
- [Compiler analysis and CPU tanh composite validation](diagnostics/compile-cuda-graph/composite-postcommit-d94daecd/README.md): Historical measurements of implementation `d94daecd`, with
  [source clean](diagnostics/compile-cuda-graph/postcommit-4cb0432/README.md) (`4cb0432`) and
  [source development](diagnostics/compile-cuda-graph/static-analysis-b7936239-development/README.md) predecessors.
  Measured revisions are distinct from evidence publication commits; each report retains its provenance and performance caveats.
- [Compiled CUDA row-sum validation](compile-cuda-sum-rows-validation.md): Commit-bound source captures, retained development attempts, and timing-artifact limitations.
- [Matmul diagnostic index](diagnostics/composite-matmul-unflatten-l1/README.md): Current evidence and earlier source/composite captures, with their original identities and raw results.
- [Matmul, unflatten and L1 integration history](composite-matmul-unflatten-l1-validation.md): Repair history, development validation and managed handoff notes.
- [CUDA addition capture validation](compile-cuda-add-validation.md): Source validation history, baseline caveats, and composite correctness evidence; no performance score.

- [CUDA view packing validation](cuda-contiguous-validation.md): Bounded contiguous/reshape materialization, aliasing contracts and clean-commit H100 correctness evidence.
- [CUDA vector addition and depth-stack integration](composite-cuda-vector-dstack.md): Records composite repairs, diagnostics, and the clean-commit delivery gate.

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
