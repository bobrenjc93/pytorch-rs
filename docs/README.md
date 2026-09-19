# Documentation Index

Choose a task below; detailed support boundaries live in the linked contracts.
Burner-managed progress artifacts are generated at merge time, not source documentation.

| Task | Start here |
| --- | --- |
| Install and run a first example | [Repository quickstart](../README.md), [first-success example](../examples/first_success.py) |
| Check an API or unsupported case | [Supported surface](supported-surface.md) |
| Compile a function | [Compiler entry points](#compiler-guides) |
| Reproduce a measurement | [Benchmark policy](../BENCHMARKING.md), [recorded validation](validation-history.md) |

The development host has real H100 GPUs. Follow [optional CUDA setup](troubleshooting.md#optional-native-cuda-runtime)
and use `CUDA_VISIBLE_DEVICES=0` for reserved single-GPU checks. Hardware-only tests
skip when CUDA is unavailable; this does not establish accelerator or training parity.

## Current Contracts

- [Supported surface](supported-surface.md): Exhaustive Python API coverage and unsupported boundary contract.
- [Feature coverage contract](../FEATURES.md): Weighted feature areas and what counts toward coverage.
- [Benchmark policy](../BENCHMARKING.md): Correctness gates, measurement rules, provenance, and anti-gaming policy.
- [Default compiler evaluations](torch-compile-default-evaluator.md): Versioned public-default Inductor coverage and real-CUDA performance gates; legacy scores are non-comparable.
- [Native default CUDA compiler](compile-pointwise-jit.md): Fused pointwise outputs and separate alias-only view/mutation programs through ordinary `torch_rs.compile(fn)`; see its [observable compatibility limits](compile-pointwise-jit.md#observable-differences-from-upstream-default-compilation).
- [Hardware heterogeneity evaluator](hardware-heterogeneity-evaluator.md): Fixed accelerator-family and feature-depth matrix, evidence rules, and real-hardware scoring policy.
- [Generated creation validator](../scripts/validate_creation_factory_benchmark.py): Held-out seeded shape path for creation-factory benchmark review.

## Examples

- [First-success example](../examples/first_success.py): Runnable version of the README first-success assertions.

## Contributor Guides

- [Contributing guide](../CONTRIBUTING.md): Locked setup, environment expectations, test selection, draft workflow, and documentation ownership.
- [Setup troubleshooting](troubleshooting.md): Short fixes for common environment, import, reference dependency, and stale wheel failures.
- [Repository README](../README.md): Install commands, first-success example, scope summary, and validation entry points.
- [Architecture map](../ARCHITECTURE.md): Source map for the Rust core, Python bindings, wrappers, and test layout.

### Compiler guides

Choose the entry point before following an operation guide:

| Entry point | Contract |
| --- | --- |
| `torch_rs.compile(fn)` | [Native default compiler](compile-pointwise-jit.md): supported programs, input admission, views, cache behavior and runtime requirements. |
| `torch_rs.compile(fn, backend="eager")` | [Explicit eager capture](compile-cuda-add.md): bounded graph execution; the operation guides below extend this path. |
| Numerical behavior | [Pointwise numerical contract](compile-pointwise-numerics.md): rounding, realization and FMA rules. |

For measurement tooling and recorded results, see the
[validation history](validation-history.md).

Neither compiler entry point provides full upstream Inductor or training parity.
Historical validation records describe their recorded source revisions; they do
not expand the current contracts.

#### Explicit eager CUDA operations

- [Compiled CUDA squeeze](compile-cuda-squeeze.md): Method and one-argument module/imported singleton removal, shared-storage wrappers and H100 validation.
- [Compiled CUDA view](compile-cuda-view.md): Alias-only constant shapes, strict stride compatibility and H100 validation.
- [Compiled CUDA reshape](compile-cuda-reshape.md): Method and positional native/imported constant tuple/list shapes, rank-0/1/2 alias-or-pack semantics and H100 validation.
- [Compiled CUDA transpose views](compile-cuda-t.md): Method and one-argument module/imported `t`, constant-axis method and three-positional-argument module/imported `transpose` capture, packing composition and H100 validation.
- [Compiled CUDA contiguous](compile-cuda-contiguous.md): Parameterless native view packing, alias semantics and clean-commit H100 validation including review regressions.
- [Compiled CUDA matmul](compile-cuda-matmul.md): Exact capture scope, runnable example, and separate `torch.compile` timing reproduction.
- [Compiled CUDA row sums](compile-cuda-sum-rows.md): Runnable example, constant reduction options, layout/device boundaries, and focused checks.
- [CUDA mul/neg/add graph capture](compile-cuda-add.md): Public bounded eager compilation, device/cache guards, unsupported cases, and reproduction.
  Includes positional module/imported add/neg/ReLU/squeeze/t calls with precise binding guards.
- [Compiled CUDA scalar multiplication validation](compile-cuda-mul-scalar-validation.md): Guarded scalar grammar, independent diagnostics, and H100 checks.
- [Compiled CUDA negation validation](compile-cuda-neg-validation.md): Integrated-commit results, provenance, and current neg/add diagnostic commands.

### Eager tensor operations and diagnostics

- [CUDA scalar `Tensor.add_`](cuda-add-inplace.md): Dense shared-storage mutation, exact alpha/operand limits and synchronous completion; default compilation supports the [alias-only subset](compile-pointwise-jit.md#alias-only-views-and-scalar-mutation), while `backend="eager"` capture rejects mutation.
- [CUDA scalar multiplication validation](cuda-mul-scalar-validation.md): General eager float32 kernel, conversion/layout boundaries, and H100 evidence.
- [Native and compiled CUDA ReLU](cuda-relu.md): Method and trusted top-level capture, IEEE bit semantics, layout bounds and H100 development checks.
- [CUDA negation validation](cuda-neg-validation.md): Contiguous float32 eager scope, H100 checks, and retained evaluation evidence.
- [Text collation diagnostics](text-collation-diagnostics.md): Public-call timings and identity checks for text-led metadata passthrough.
- [CUDA-add diagnostics](cuda-add-diagnostics.md): Latency, sustained throughput, cache saturation, and release provenance.
- [CUDA matrix/vector addition](cuda-add-trailing-vector.md): Bounded eager broadcast support, H100 differentials, and six-case math validation.
- [Native CUDA matrix multiplication](cuda-matmul.md): Contiguous same-device rank-2 float32 matmul, native cuBLAS setup, numerical regressions, and capture commands.
- [CUDA matrix row sums](cuda-sum-rows.md): Contiguous rank-2 float32 dim=1/-1 reduction, numerical behavior, and validation.
- [Rank-2 mean diagnostics](rank2-mean-diagnostics.md): Both reduction axes and layouts at matched CPU worker budgets.

[Compiled CUDA matrix/vector addition](compile-cuda-trailing-vector-validation.md)
records the bounded shape extension and non-scoring diagnostic commands.

## Validation and timing history

[Historical validation and timing catalog](validation-history.md): Recorded revisions,
raw evidence and all release timing groups. These records do not expand current support.
