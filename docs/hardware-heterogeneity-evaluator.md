# Hardware Heterogeneity Evaluator

This Burner evaluation measures how much of PyTorch's accelerator breadth and
feature depth `torch_rs` actually supports. CPU is intentionally excluded: it
is the host baseline, not an accelerator family.

The versioned source of truth is
[`hardware-heterogeneity-matrix-v1.json`](hardware-heterogeneity-matrix-v1.json).
Version 1 gives equal weight to seven stable PyTorch accelerator families:
NVIDIA CUDA, AMD ROCm, Apple MPS, Intel XPU, Google TPU/XLA, Intel Gaudi HPU,
and Meta MTIA. `PrivateUse1`, `meta`, and Vulkan are not counted as accelerator
families: the first is an extension mechanism, the second has no hardware
execution, and the last is not a stable general training backend in the
reference release.

## Score

Each backend is evaluated against stable PyTorch 2.13 programs in eight
capability groups:

| Capability | Weight within each backend |
| --- | ---: |
| Device tensors, placement, and host/device transfers | 15% |
| Dtypes, layouts, views, and indexing | 15% |
| Elementwise math, reductions, and linear algebra | 15% |
| Autograd and training | 15% |
| Neural networks and optimizers | 15% |
| RNG, serialization, and checkpointing | 10% |
| Compilation | 10% |
| Multi-device and distributed execution | 5% |

For backend `b` and capability `c`, let `p(b,c)` be the fraction of fixed
reference-eligible cases that `torch_rs` passes. A case is reference-eligible
only when the declared stable PyTorch stack compiles or runs it correctly on
that accelerator. Each backend score is the weighted sum of its eight
capability fractions. The headline score is the arithmetic mean of all seven
backend scores, multiplied by 100. A backend with no qualifying evidence has a
score of zero; it is never removed from the denominator because the evaluator
machine does not contain that hardware.

The evaluator must also report an unweighted *backend breadth* percentage. A
family counts toward breadth only when real-hardware evidence passes at least
one representative case in all three foundational groups listed by
`breadth_required_capabilities`: device tensors/transfers, core math, and
autograd/training. This supplementary number prevents a single specialized
kernel from being described as general support. The feature-depth matrix,
not breadth alone, is the Burner score.

## Evidence rules

A case earns credit only when it:

- executes on real hardware from the named accelerator family;
- matches PyTorch outputs and required observable semantics;
- uses native `torch_rs` storage and execution rather than installed-PyTorch
  forwarding, CPU fallback, emulation, or a metadata-only shim; and
- is covered by an executable differential test whose workload and result are
  bound to the evaluated commit.

When hardware is unavailable on the evaluator host, a reproducible
hardware-CI or release artifact may provide credit if it records the exact
commit, device, driver/runtime, dependency versions, test matrix, raw results,
and pass/fail accounting. Cross-compilation and mocked availability do not earn
runtime credit. Missing, skipped, unsupported, stale, or unverifiable cases
remain zero; they are not silently excluded.

API probes such as `is_available()`, device parsing, and backend preference
flags are useful compatibility work but do not prove accelerator execution. A
private benchmark-only path earns credit only for the exact cases it executes,
not for the rest of that backend or capability group.

## Local CUDA requirement

The Burner host has eight NVIDIA H100 GPUs. For changes that could affect CUDA
coverage, run the relevant differential cases on one real device with
`CUDA_VISIBLE_DEVICES=0`; reserve more devices only for an explicit
multi-device case. Record the H100 model, compute capability, driver, CUDA
runtime and compiler, PyTorch version, selected device, and memory pressure.
Portable tests must still skip clearly on hosts without matching hardware,
without weakening the hardware-backed acceptance criteria.

Changing the backend set, capability weights, eligibility rules, or zero-credit
rules requires a new matrix schema version and a matching Burner evaluation
definition version. Cases may be added within this contract, but existing
failing cases must not be deleted merely to raise the score.
