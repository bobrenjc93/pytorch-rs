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

### Reductions

- [Rank-1 `Tensor.sum` release timings](docs/rank1-sum-release-timings.md)
- [Rank-9 `Tensor.sum` release timings](docs/rank9-sum-release-timings.md)
- [Rank-10 `Tensor.sum` release timings](docs/rank10-sum-release-timings.md)
- [Rank-11 `Tensor.sum` release timings](docs/rank11-sum-release-timings.md)
- [Rank-12 `Tensor.sum` release timings](docs/rank12-sum-release-timings.md)
- [`Tensor.mean` and `torch.mean` full-reduction release timings](docs/tensor-mean-release-timings.md)

### Creation

- [`torch.empty`, `torch.zeros`, and `torch.ones` eager CPU factory timings](docs/creation-factory-release-timings.md)

Run the creation-factory driver from a release-built wheel install:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/benchmark_creation_factories.py
```

The script requires PyTorch 2.13 and compares public eager CPU factory calls
for `torch.empty`, `torch.zeros`, and `torch.ones` across scalar, zero-element,
small, and large shapes. It uses identical warmup/sample counts, two reversed
implementation-order passes, one CPU thread by default, and metadata
materialization inside the timed loop. `zeros` and `ones` also include one
final-output sum checksum inside each timed block. `torch.empty` element values
remain unspecified by contract; the current safe `torch_rs` storage
implementation zero-initializes its CPU float32 backing allocation, and the
benchmark records that cost explicitly. Private CUDA driver/runtime probes are
not part of this parity benchmark. Treat the fixed public matrix as
repeatability evidence only until it is paired with a generated-shape validator
run using a held-out seed:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python \
  scripts/validate_creation_factory_benchmark.py --seed <held-out-seed>
```

The generated validator excludes the public fixed shapes and records the seed
and generated workload matrix in its JSON output.

### Elementwise ops

- [`+` and `Tensor.add` release timings](docs/tensor-add-release-timings.md)
- [`torch.sub` and `torch.subtract` release timings](docs/top-level-subtract-release-timings.md)
- [`*`, `Tensor.mul`/`Tensor.multiply`, and `torch.mul`/`torch.multiply` release timings](docs/tensor-mul-release-timings.md)
- [`torch.div` and `torch.divide` release timings](docs/top-level-division-release-timings.md)
- [`Tensor.abs` and `torch.abs` release timings](docs/tensor-abs-release-timings.md)
- [`Tensor.sqrt` and `torch.sqrt` release timings](docs/tensor-sqrt-release-timings.md)
- [`Tensor.reciprocal` and `torch.reciprocal` release timings](docs/tensor-reciprocal-release-timings.md)
- [`torch.nn.functional.softsign` release timings](docs/softsign-release-timings.md)

### Compilation

- [`torch.compile` eager CPU release timings](docs/torch-compile-cpu-release-timings.md)

The CUDA compile measurement boundary is a narrow, fail-closed skeleton. Run it
on the H100 host with a single visible device:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_compile_cuda.py
```

To reserve a different physical GPU, mask exactly one device and pass the same
literal value to `--required-cuda-visible-devices`; benchmark-private CUDA work
still uses logical CUDA device 0 after masking.
Passing an empty `--required-cuda-visible-devices` preserves the local
experimentation escape hatch by skipping the literal environment check; CUDA
work still runs on logical device 0 and records the visible-device count.

The script requires PyTorch 2.13, records GPU, driver, CUDA runtime, `nvcc`,
compile configuration, cold first-call timing, synchronized steady-state
timings, and checksum/correctness evidence for one versioned PyTorch CUDA
reference workload. It also records private benchmark-only `torch_rs` CUDA
driver/runtime evidence, separate from the public `torch.cuda` compatibility
API: device 0 metadata plus a float32 runtime allocation, host-to-device copy,
device-to-host copy, synchronization, checksum roundtrip, and one compiled
torch_rs-owned float32 pointwise kernel launch with synchronized output
checksum verification. The current
`torch_rs` CUDA compile cell is emitted as explicit `zero_credit_unsupported`:
CPU tensors, `backend="eager"`, eager fallback, skipped execution, or
forwarding to installed PyTorch are rejected as eligible CUDA compile evidence.
The public CPU-build `torch.cuda` probe behavior remains unchanged.

### Layout/view ops

- [`Tensor.view`, reshape, flatten, ravel, unbind, and edge-unsqueeze release timings](docs/tensor-view-release-timings.md)

### Linear algebra

- [Rank-2 `@`, `Tensor.matmul`, and `torch.matmul` release timings](docs/rank2-matmul-release-timings.md)

### NN losses

- [`torch.nn.functional.mse_loss` release timings](docs/mse-loss-release-timings.md)
- [`torch.nn.functional.l1_loss(reduction="none")` release timings](docs/l1-loss-release-timings.md)
- [`torch.nn.functional.l1_loss(reduction="sum")` release timings](docs/l1-loss-sum-release-timings.md)

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
