# Positional scalar binding validation

The shared-cache review fix passes its focused H100 regressions. Logical guards now select
frozen scalar semantics before concrete native executable lookup. The original
18 signed-zero subtest failures pass on H100 without changing their assertions
or resetting the reference within a history.

The [clean post-commit refresh](postcommit-7a74596b/README.md) measures candidate
`7a74596b` and actual main `a281503f` with the unchanged full coverage/CUDA gate.
It also refreshes the actually dispatched CUDA/PTX capture and passes all 12
focused persistent-specialization tests against the clean candidate wheel.
Earlier measurements and their failures remain pinned to their original commits.
Burner's independent review and exact-head qualification remain separate gates.

**The operator-review repairs require a new clean evidence refresh.** Static NaN
guard identity and the two maintained diagnostic consumers changed after
`7a74596b`. Those clean reports and root CUDA/PTX provenance remain unchanged,
but are stale for this repair. The [repair record](operator-review-fix/README.md)
retains the reproduced ninth-NaN rejection, fresh development validation and
separately identified dispatched captures. Burner must regenerate final clean
candidate/main measurements and capture provenance after committing the repairs.

This implementation accepts one or two exact tensors plus
exact built-in float/Boolean arguments in arbitrary positional slots. Positional
integers remain unsupported; literal/captured integer behavior is unchanged.
The [supported boundary](../../compile-pointwise-jit.md) and
[original-IR numerical admission](../../compile-pointwise-numerics.md) apply.
No evaluator, corpus, tolerance, denominator or managed progress artifact changed.

`BindingSource` identifies a public parameter position, global name or closure
cell. Resolution and promotion share the existing successful-cache-entry history
and reset owner. Tensor filtering happens once and preserves unused and repeated
tensors. Parameters never read, or overwritten before their first read, have no
value guards; all public arguments still receive exact-type admission.

## Retained pre-fix failures

The [shape revisit probe](zero-shape-probe.py) retains
[all 20 observations](zero-shape-probe.json.gz), including eight numerical
failures. With one persistent wrapper and `f(scale, x) = x * scale`:

| Call | Scalar | Shape | Native output | Default-Inductor output |
| --- | --- | --- | --- | --- |
| 1 | `+0.0` | `(2,)` | `+0.0` | `+0.0` |
| 2 | `-0.0` | `(3,)` | `-0.0` | `-0.0` |
| 3 | `+0.0` | `(2,)` | `+0.0` | `-0.0` |

The opposite sign order fails symmetrically. Existing captured floats reproduce
the same problem: a separate [baseline-wheel probe](main-capture-shape-probe.py)
reproduced four failures in [ten observations](main-capture-shape-probe.json.gz)
using the wheel built from clean main before implementation. Its SHA-256 is
`5574ecdbf9fdb8d256191dec9d6d5ed299174f84f0cca358e80e48bca5d55f96`;
it was extracted into `target/main-binding-probe-package` and selected with
`PYTHONPATH`, without modifying any checkout or installed environment.
Reference guards inspected from the actual persistent wrapper
are retained in [shape-guards.json.gz](shape-guards.json.gz). Its second graph
generalizes the changed dimension to `2 <= size <= 2147483647` and precedes the
first exact-shape graph. Native's exact metadata lookup instead selects the old
graph. This is an observable sign-bit mismatch within the admitted language,
not an excluded expression or a harmless difference in compilation counts.

The review required one shared representation of logical graph
specializations, source guards and their selection order, with native executable
specialization beneath it. A generalized logical graph must retain its constants
even when another exact broadcast executable is needed. Shape/stride history,
rank changes, zero/singleton dimensions, used/unused sources and tensor
relationships need persistent-default verification. Original-IR admission on
every actual tensor shape must remain unchanged. A zero-specific replacement
rule or a separate positional-argument history would not solve that ownership
problem. The revision implements the shared owner described below.

Five permanent tests in `ScalarShapeSpecializationHardware` exercise parameters,
globals and closures in both sign orders through one-dimensional revisits, rank,
singleton/empty, unused-tensor and broadcast histories. They retain complete
observations before asserting and do not reset the reference within a history.
Their results, including failing assertions, remain in the validation archive.
In the pre-fix capture, three test methods fail with 18 subtest failures; the
rank-transition and broadcast methods pass. These were numerical failures,
including existing captured-scalar behavior, not reference infrastructure errors.
The revised implementation passes all five unchanged methods.

## Shared-cache revision

The existing reset-owned cache contains logical specializations plus a bounded
native executable LRU. Source-identified guards use successful shape/stride and
scalar history, with static zero/singleton dimensions, broadcast relationships
and conditional 32-bit upper bounds. Lookup checks the most recently selected
logical entry first. Generalized entries retain their constants when an older
shape returns. Scalar promotion runs only after every logical guard misses;
nonfinite values specialize on a new trace but remain accepted by a matching
runtime-float guard.

Concrete executors retain the complete filtered tensor tuple, operand indexing
and exact broadcast address formulas. An unused tensor or a changed native ABI
can require a new executable without changing logical scalar semantics. Every
native call still validates actual shapes against the original IR. Successful
execution publishes both cache levels together; failures change neither history
nor selection order. Reset clears both levels. The
[cache contract](../../compile-pointwise-jit.md#cache-behavior) documents bounds.

New regression cases exercise rank/precision selection, nonfinite shape misses,
reversed alias realization, unused scalar/tensor role changes, frozen zero through
new executable compilation, singleton strides and transactional failure. Ten hardware-free
metadata tests cover dimensions above the 32-bit boundary without large GPU
allocations, source history, failed publication, reset and bounded ABI/executable
eviction at a one-specialization limit. These are independent non-corpus tests; evaluator definitions and
numerical assertions remain unchanged.

Current-wheel validation passes 156 pointwise tests in 19 disjoint processes
(four single-GPU reservation skips, all four passed separately on GPUs 0 and 1).
Python 3.14 passes both scalar owners: 58 tests with two reservation skips.
Hardware-free execution runs 19 scalar tests and clearly skips 39 GPU tests.
Both Rust configurations pass 31 pointwise tests; default/Python-bindings builds,
both Clippy configurations, formatting and documentation checks pass.

The [revision validation archive](review-fix-validation.json.gz) retains the full
948-test compiler sweep across 87 disjoint processes and the current-wheel
pointwise rerun. Combining their disjoint owners covers all 965 tests in the
final compiler selection. The earlier sweep wheel differs from the current wheel
only by added contract docstrings and removal of an unused private argument;
both identities and their exact difference are archived. Its neg/add diagnostic
correctly refused those mismatching source bytes, then passed all five tests after
installing the current wheel in the required `.venv`, without changing its harness
or assertions. The initial failure, documentation-anchor and import-path setup
errors, successful reruns, commands and complete selection remain in the archive.
No numerical failure occurred in the revision checks.

The [development CUDA/PTX capture](review-fix-dispatched/provenance.json) uses the
revised wheel and records dirty sources based on `66f4acdd`. Its
[source manifest](review-fix-dispatched/source-manifest.json.gz) identifies the
actual compiler and test bytes. This development record is preserved alongside
the new clean post-commit capture. The original measured evidence below remains
byte-for-byte preserved.

## Reference characterization

The initial [probe](reference-probe.py) keeps each ordinary default-Inductor
wrapper alive for its complete scalar sequence. Its [150 observations](reference-probe.json.gz)
match for parameter, global and closure origins across precision-sensitive values,
Boolean transitions, signed zeros, nonfinite interludes and overflow. Resets occur
only between independent programs/histories, never to remove a mismatch. The
reference owner inspected was PyTorch 2.13's
`torch/_dynamo/variables/builder.py::wrap_symfloat`, including source-identified
`process_automatic_dynamic`, static nonfinite specialization and runtime float
materialization. The native implementation does not import this reference code.

The extended regressions remain in the existing
[admission](../../../tests/test_compile_pointwise_scalar_admission.py) and
[runtime scalar](../../../tests/test_compile_pointwise_runtime_scalars.py) owners.
They cover all positional orders, one/two tensors, mixed captures, repeated
bindings, source role changes, shapes/offsets/identities, 64 combined runtime
scalars, rejection above that budget, integer/object rejection, failed compilation
and launch publication, recompile limits/reset, unchanged inputs/fresh outputs,
current-device restoration, and no body/eager/per-node execution. Actual-shape
admission includes unused tensors and singleton-only linear maps; scalar slots do
not restrict one-tensor expressions. Existing large-integer regressions remain.

## Pre-implementation main baseline

Before implementation, both unchanged public-default-compile-v2 gates ran from
clean main `a281503f3391bbd98a0ab6de2ff8f0c0a55a12d4` with source identity
`6047babf37c578031a0ef8ce76f2d74b2ad5a3e97b144f727c2e81dfd78a6243`.
Both reports have `valid: true` and `diagnostic: false`:

| Gate | Main result | Fixed passing cells |
| --- | ---: | ---: |
| [Coverage](main-coverage.json.gz) | 8% | 6/112 |
| [CUDA performance](main-cuda-perf.json.gz) | 12% | 6/56 |

The performance common-success geometric mean is 2.0181160029262264. This is
baseline evidence, not a candidate gain or general Inductor parity. Reports retain
all cells, unsupported outcomes, both orders, five warmups, 17 samples,
synchronization, cold/steady timings, source/build/wheel identities and GPU
snapshots. Initial setup failures are retained in the validation archive.

## Development validation and reproduction

All Python interpreters, environments, wheels, Cargo outputs, dependencies and
caches used for validation live inside this checkout. GitHub's download proxy
rejected the managed interpreter download; installed standalone Python 3.12.12
and 3.14.5 trees were copied read-only into `target/uv-python` instead. Dependency
resolution remained locked. Rust 1.92.0 and release ABI3 wheels were used.
GPU tests used H100 GPU 0; restoration tests explicitly selected GPUs 0 and 1.
The original runtime checks remain in the development validation archive; the
refreshed clean-commit NVRTC/runtime identities are in [provenance](provenance.json).
The installed `nvcc` is CUDA 12.6; generated code uses NVRTC 13.0 rather than nvcc.

Use the worktree-local interpreter and cache paths from the validation receipt.
The focused commands are:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest \
  tests.test_compile_pointwise_scalar_admission \
  tests.test_compile_pointwise_runtime_scalars
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m unittest \
  tests.test_compile_pointwise_scalar_admission \
  tests.test_compile_pointwise_runtime_scalars
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_compile_pointwise_runtime_scalars.PositionalScalarHardware.test_positional_runtime_parameters_restore_device_on_success_and_failure \
  tests.test_compile_pointwise_runtime_scalars.RuntimeScalarHardware.test_runtime_parameters_restore_current_device \
  tests.test_compile_pointwise_jit.Hardware.test_two_device_restoration_and_module_ownership
cargo test --locked --lib pointwise
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --locked --lib pointwise --features python-bindings
cargo build --locked
cargo build --locked --features python-bindings
cargo clippy --locked --all-targets -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --locked --all-targets --features python-bindings -- -D warnings
cargo fmt --check
```

The pre-fix [validation archive](validation.json.gz) retains commands, logs, exhaustive
disjoint compiler-test selection, supplemental review cases and failed attempts.
The initial sweep collected 941 unique tests in 87 disjoint processes, with
25 hardware/reservation skips. Seven supplemental tests complete the final
948-test selection: two additional binding tests pass, and the five shape-history
tests have the results above. Default and Python-bindings builds, both Clippy
configurations, formatting, 31 focused Rust tests in each configuration, and
12 documentation tests pass. Hardware-free execution of both scalar test owners
passes nine tests and clearly skips 32 hardware cases.
The first collection lacked the checkout root on its import path. An unchanged
matmul isolation test then required installation under `.venv`; its initial
failure and successful rerun in that local environment are both retained. These
were infrastructure failures. Subsequent numerical failures from the shared
shape-history cache are also retained; no failure was removed or hidden by a
reference-only reset. Those results describe the pre-fix implementation; they do not qualify the revision.

## Dispatched code and post-commit evidence

[CUDA source](kernel.cu), [PTX](kernel.ptx.gz),
[source manifest](source-manifest.json.gz) and [provenance](provenance.json)
come from [capture.py](capture.py), an independent non-corpus example. The final
call returns to an earlier scalar value after promotion. The capture observes
the executor selected by successful dispatch and verifies it differs from both the
first static module and the last inserted Boolean specialization. The wheel's
frontend/native bytes are checked against the checkout/loaded extension.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  docs/diagnostics/compile-pointwise-positional/capture.py \
  "$PWD/target/positional-codegen" "$PWD/target/wheels/torch_rs-WHEEL.whl"
```

The current module capture records clean implementation commit
`7a74596b5fa7fc17843eda8d90c885dc8cd72f12`. The [post-commit record](postcommit-7a74596b/README.md)
adds fresh full candidate and actual clean-main coverage/CUDA measurements,
source/build/wheel verification, and 12 passing persistent-specialization tests.
The [earlier clean record](postcommit-4ed105f5/README.md) retains the pre-fix
measurements and all 18 failures; its CUDA/PTX capture is archived byte-for-byte
under that record. Earlier development validation and historical baseline reports
remain unchanged. Burner owns independent review, artifact commits, exact-head
qualification and publication. Fixed-corpus scores do not establish general
Inductor parity.
