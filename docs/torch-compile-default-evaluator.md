# Default compiler parity

These gates compare **`torch_rs.compile(program)` with `torch.compile(program)`**,
without backend, fullgraph, dynamic, mode, or options overrides. The reference
is the `uv.lock`-pinned PyTorch 2.13 default Inductor backend. Matching a backend
name is not evidence of matching its compiler.

The result is a percentage of this finite, representative corpus—not a claim
about the uncountable set of all Python programs, all models, or all hardware.
Program coverage and CUDA speed are separate measurements. Correctly capturing
a native graph without fusion may earn coverage; it earns performance credit
only at its measured speed against default Inductor.

## Run on real hardware

The development host has eight NVIDIA H100 GPUs. CUDA is testable here; do not
replace GPU measurements with CPU results or skip the accelerator denominator.
Check `nvidia-smi` for an available GPU first; this command does not reserve it
or authorize interrupting another user's workload.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric coverage
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric cuda-perf
```

Use `--metric both --output target/default-compile-baseline.json` to capture both
metrics in one run. `both` produces a report, not a single Burner score. The
Burner definitions use the two explicit metric commands above. There is no quick
screen that can replace the full gate. `--diagnostic` permits work in progress
but marks the report unscored and never emits a Burner `score` field.

The wrapper uses a locked, worktree-local Python 3.12 environment and caches,
builds a release native wheel with `maturin build --release --locked`, installs
that exact wheel, and checks extension/import provenance. It retains wheels,
raw compressed observations, compiler logs, and timing reports under
`target/default-compile-eval/`. On hosts using an enterprise CA, configure uv's
system trust store (`UV_SYSTEM_CERTS=true`); do not disable TLS verification.

## Corpus v2

[`torch_compile_default_corpus.py`](../scripts/torch_compile_default_corpus.py)
owns `public-default-compile-v2`: 28 actual programs, two per category. Factories
run unchanged with either framework and only use public APIs and ordinary
tensors. No function-name dispatch, tensor markers, private prepared executors,
or benchmark tensor wrappers are eligible.

| Category | Weight | Programs |
| --- | ---: | --- |
| Tensor arithmetic | 12 | Affine/ReLU; trigonometric composition |
| Broadcasting | 8 | Row bias; rank-3 broadcast |
| Modules, parameters, buffers | 8 | `nn.Linear`; real module with parameter and buffer |
| Inference | 6 | Two-layer MLP; scaled dot-product attention |
| Training/autograd | 8 | Matmul backward; MLP backward, including parameter gradients |
| Python control flow | 8 | Shape-dependent branch; static loop |
| Graph breaks | 8 | Scalar branch; Python scalar materialization between tensor regions |
| Dynamic shapes | 8 | Shape-derived reshape; shape-dependent reduction |
| Mutation, aliasing, views | 8 | Mutating view; returned transpose with external alias mutation |
| Containers/pytrees | 6 | Nested outputs with repeated tensor identity; nested inputs |
| Decompositions | 6 | Layer normalization; GELU |
| Custom functions | 6 | Python helper; custom autograd function and backward |
| Recompilation guards | 4 | Changing scalar argument; changing module attribute |
| Dtype/device transitions | 4 | bfloat16/float32 outputs; explicit CPU/device round trip |

Each program has two input/shape variants, exercised through the **same compiled
wrapper**, plus a changed-value check at each unchanged shape. The second shape
is held out from the initial compilation, not secret from repository readers.
Seeds, shapes, category weights, and tolerances are fixed in the independent
corpus; implementations must not specialize to their names or values. A future
corpus expansion needs a separately reviewed version and new baseline.

Coverage has 112 cells: 28 programs × 2 variants × CPU/CUDA. CUDA performance has
56 cells. Most inputs are float32; matrix batches include 128 and 193 rows, with
separate rank-3, attention, and matmul shapes. This first corpus does not cover
large production models, distributed compilation, every layout/dtype, all
dynamic-shape guard regimes, or every Python graph-break behavior.

Training times compiled forward plus the same public `sum().backward()` call
on both sides; it is not an optimizer-step benchmark. The explicit device-round-
trip program intentionally includes transfers in its timed work. Other CUDA
programs must return CUDA tensors and cannot silently substitute CPU execution.

## Correctness and execution evidence

Reference and candidate run in separate fresh processes. The candidate process
blocks `torch` imports and verifies that all installed Python sources and the
native extension belong to the measured checkout/wheel. Both call the same
public program, with inputs created outside timing using identical values.

The reference must compile with default Inductor, have error suppression off,
produce compiled graphs, and match eager reference outputs and required
gradients. Inductor counters and generated-kernel counts are recorded; a
legitimate external-library matmul or metadata-only view need not generate a
new kernel. Default graph breaks remain enabled and are recorded, not replaced
by `fullgraph=True` or a custom backend returning an FX graph's Python forward.

All tensor values—not just checksums—are compared, along with shapes, strides,
dtypes, devices, gradient requirements, container types, repeated object
identity, mutated inputs, and post-return view aliasing. Tolerances are
`rtol=1e-4`, `atol=1e-4`. The mixed bfloat16/float32 program uses `rtol=8e-3`
symmetrically for reference-versus-eager, cold-versus-warm, and candidate-versus-
compiled-reference comparisons: default Inductor may retain higher precision
between casts, unlike eager bfloat16 intermediate rounding. Metadata remains
exact, including actual bfloat16 output; ordinary float32 programs retain the
tighter tolerance.

An identity compiler fails. After warmup, a profile probe rejects a wrapper
that re-executes the original Python entrypoint, including a wrapper around
eager execution. Changed inputs at the same shape reject cached outputs. This
is an execution check, not proof against arbitrarily malicious source changes;
independent review of lowering and runtime behavior remains required. Native
graph execution need not claim fusion. Legitimate compiled graph-break resume
code is distinct from replaying the original unmodified program.

Any reference, setup, provenance, incomplete-worker, or infrastructure failure
invalidates the whole run and emits **no numeric score**. Ordinary candidate
compile/runtime/semantic failures produce explicit zero-credit cells. Cases
are never dropped because torch_rs does not support them.

## Timing and aggregation

Both implementations use one host compute thread, five warmups, and 17 timing
samples per cell. Each CUDA cell is measured twice, in reversed implementation
orders, with separate initially empty Inductor/Triton cache directories for
each worker. Dataset construction and parameter-gradient resets are outside
timing. A failed cell in either order stays failed; runs are not repeated until
a favorable result appears.

Both sides use the **same CUDA runtime `cudaDeviceSynchronize` binding** before
and after each timed call. This completes real GPU work without making coverage
depend on whether `torch_rs.cuda.synchronize` is exposed. The runtime library,
hash, version, and logical device are recorded. Outputs are materialized and
checked outside timing; cold factory/first-call costs are separate from steady
latency. Raw samples, median, MAD, min/max, compile counters, source/lock/wheel
hashes, toolchain versions, visible devices, memory use, clocks, and GPU/driver
snapshots are retained. Cold runs start with empty disk caches but can reuse
in-process infrastructure across programs; they are not fresh-machine costs.

Coverage averages pass/fail cells within each category, then applies the fixed
weights above. For CUDA performance, take the geometric mean of the two
reference/candidate median-latency ratios for each successful cell, cap it at
one, and use zero for a failed cell. Following [BENCHMARKING.md](../BENCHMARKING.md),
geometrically aggregate those capped cell values within each category, then
sum fixed-weight category contributions. Consequently, any zero cell makes
that category's performance contribution zero. A fast microkernel cannot buy
credit for a missing program. Report the uncapped common-success geometric mean
separately; it is `null` when there are no common successes, not “100% parity.”

## Migration from the old scores

The old coverage gate forced an eager candidate and a custom non-Inductor
reference. The old CUDA gate measured one marker-selected, handwritten kernel
at four shapes through private benchmark tensors. Those measurements describe
their narrow workloads; neither establishes this default-API comparison.

Coverage definition `eval_a61c0e71` is now v4; CUDA performance definition
`eval_6f98c42d` is now v3. Historical 100 scores remain historical, non-comparable
records and must not be used as floors for these definitions. A lower new score
is a **measurement correction**, not an implementation regression. Keep this
campaign change separate from compiler implementation candidates and obtain
human review before adopting it. Burner owns canonical score-history updates;
do not manually rewrite old progress artifacts.

The initial public-default v1 capture (coverage v3 / performance v2) is retained
as pre-adoption evidence. Review found that its bfloat16 tolerance was asymmetric;
v2 applies the same declared allowance to both frameworks and is freshly
rebaselined. Neither capture changes the native implementation or authorizes
reusing scores across definition versions.

The retained `evaluate_torch_compile_coverage.py` and
`benchmark_compile_cuda.py` are legacy diagnostics only. Neither command is
the scoring gate for these new definition versions.
