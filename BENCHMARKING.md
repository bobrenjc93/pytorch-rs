# Benchmark policy

The benchmark program measures useful PyTorch-like behavior, not isolated loops
chosen because this implementation happens to win them.

The workloads are bounded by the [supported-surface contract](docs/supported-surface.md)
and the weighted coverage areas in [FEATURES.md](FEATURES.md).

## Gates and aggregation

1. Run formatting, linting, unit, integration, documentation, and differential
   correctness checks. A workload earns performance credit only when outputs,
   shapes, dtypes, errors, aliasing, and edge cases match the reference.
2. Validate benchmark provenance and verify that the candidate did not alter
   campaign-owned evaluators.
3. Measure equivalent `torch_rs` and `torch` Python calls on the same machine,
   power state, thread count, dtype, shape, and layout, with inputs created
   outside the timed region. Native execution is timed inside each process so
   interpreter startup, imports, builds, and dependency installation are
   excluded. Warmup, synchronization, sampling, checksums or materialized
   outputs, and thread counts must be symmetric.
4. Assign zero performance credit to incorrect, missing, and unsupported
   workload cells; they are never removed from the denominator.
5. Use fixed seeds for reproducible failures and generated or held-out shapes to
   prevent implementations from specializing for a short public list.
6. Aggregate capped per-cell parity ratios geometrically within categories,
   then combine categories at fixed weights. Report uncapped ratios too, but do
   not let a 10x win in one microkernel offset a 2x loss elsewhere.

Every report includes median, dispersion, samples, warmups, compiler profile,
Rust/Python/PyTorch versions, OS, CPU/GPU, thread settings, compile time, and
dependency-installation time. No result may silently fall back to a different
device or dtype.

## Historical release timing reports

These reports are historical release evidence snapshots: they record the code,
environment, checks, and timings from specific runs. They are not live gates and
do not replace the benchmark policy or Burner-managed evaluation progress.

- [Rank-1 `Tensor.sum` release timings](docs/rank1-sum-release-timings.md)
- [Rank-9 `Tensor.sum` release timings](docs/rank9-sum-release-timings.md)
- [Rank-10 `Tensor.sum` release timings](docs/rank10-sum-release-timings.md)
- [Rank-11 `Tensor.sum` release timings](docs/rank11-sum-release-timings.md)
- [Rank-12 `Tensor.sum` release timings](docs/rank12-sum-release-timings.md)
- [Tensor true-division release timings](docs/tensor-div-release-timings.md)
- [Tensor sign release timings](docs/sign-release-timings.md)
- [`torch.nn.functional.mse_loss(reduction="none")` release timings](docs/mse-loss-release-timings.md)
- [`torch.nn.functional.l1_loss(reduction="none")` release timings](docs/l1-loss-release-timings.md)

## Workload matrix

The durable full suite grows toward all of these categories while keeping prior
cells:

| Category | Representative behavior |
| --- | --- |
| creation and indexing | empty/full/range/random, slicing, masks, gather/scatter |
| elementwise | unary, binary, scalar, broadcasting, promotion, non-contiguous inputs |
| reductions | sum/mean/min/max/arg*, dimensions, keepdim, empty and NaN behavior |
| linear algebra | vector, matrix, batched matmul, decompositions, mixed aspect ratios |
| shape and layout | reshape/view, transpose, permute, contiguous, expand, concatenate |
| neural network | activations, normalization, convolution, pooling, losses, embeddings |
| autograd | forward/backward, accumulation, broadcasting, views, no-grad, finite differences |
| modules/optimizers | parameters, state dicts, SGD/Adam, train/eval, serialization |
| devices/dtypes | CPU and available accelerators; bool, integers, f16/bf16/f32/f64 |
| end-to-end | MLP, CNN, transformer block training and inference |

Shapes include scalars, empty dimensions, primes, powers of two, awkward tails,
small latency cases, cache-sized cases, and memory-bandwidth cases. The suite
measures one thread and representative multithreaded settings. Quick screening
is a strict subset of the same workload definitions; final merges run the full
matrix.

## Implementation boundaries

Native platform libraries and explicit hardware backends are valid
implementation techniques. Forwarding production tensor operations to Python or
PyTorch is not.

## Anti-gaming review

A change is rejected if it recognizes benchmark inputs, caches outputs across
independent eager calls, skips synchronization or materialization, uses looser
numerical semantics without disclosure, changes the reference configuration,
measures unequal work, or edits evaluators alongside the optimized
implementation. Burner-authored feature branches may not weaken, delete, skip,
special-case, or rewrite evaluation infrastructure. Benchmark changes are
separate, human-reviewed campaign changes and never earn implementation impact
in the same comparison. Reviewers inspect fast paths and rerun held-out
generated cases whenever a benchmark moves materially.
