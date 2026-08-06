# Benchmark policy

The benchmark program measures useful PyTorch-like behavior, not isolated loops chosen because this implementation happens to win them.

## Gates and aggregation

1. Run formatting, linting, unit, integration, documentation, and differential correctness checks.
2. Validate benchmark provenance and verify that the candidate did not alter campaign-owned evaluators.
3. Measure equivalent `torch_rs` and `torch` Python calls on the same machine, power state, thread count, dtype, shape, and layout. Native execution is timed inside each process so interpreter startup, imports, builds, and dependency installation are excluded.
4. Assign zero performance credit to incorrect and unsupported workload cells.
5. Aggregate capped per-cell parity ratios geometrically within categories, then combine categories at fixed weights. Report uncapped ratios too, but do not let a 10x win in one microkernel offset a 2x loss elsewhere.

Every report includes median, dispersion, samples, warmups, compiler profile, Rust/Python/PyTorch versions, OS, CPU/GPU, and thread settings. No result may silently fall back to a different device or dtype.

## Workload matrix

The durable full suite grows toward all of these categories while keeping prior cells:

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

Shapes include scalars, empty dimensions, primes, powers of two, awkward tails, small latency cases, cache-sized cases, and memory-bandwidth cases. The suite measures one thread and representative multithreaded settings. Quick screening is a strict subset of the same workload definitions; final merges run the full matrix.

## Anti-gaming review

A change is rejected if it recognizes benchmark inputs, caches outputs across independent eager calls, skips synchronization or materialization, uses looser numerical semantics without disclosure, changes the reference configuration, measures unequal work, or edits evaluators alongside the optimized implementation. Reviewers inspect fast paths and rerun held-out generated cases whenever a benchmark moves materially.
