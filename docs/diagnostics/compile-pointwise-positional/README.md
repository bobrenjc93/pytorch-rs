# Positional scalar binding validation

**Incomplete milestone: do not qualify this candidate.** The source binding
implementation and broad regression checks are present, but additional
persistent-default tests found numerical failures requiring a shared-cache
prerequisite. The failure evidence is retained below.

This development implementation accepts one or two exact tensors plus
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

## Unresolved prerequisite

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

Completing this increment requires one shared representation of logical graph
specializations, source guards and their selection order, with native executable
specialization beneath it. A generalized logical graph must retain its constants
even when another exact broadcast executable is needed. Shape/stride history,
rank changes, zero/singleton dimensions, used/unused sources and tensor
relationships need persistent-default verification. Original-IR admission on
every actual tensor shape must remain unchanged. A zero-specific replacement
rule or a separate positional-argument history would not solve that ownership
problem. No such approximation was installed.

Five permanent tests in `ScalarShapeSpecializationHardware` exercise parameters,
globals and closures in both sign orders through one-dimensional revisits, rank,
singleton/empty, unused-tensor and broadcast histories. They retain complete
observations before asserting and do not reset the reference within a history.
Their results, including failing assertions, remain in the validation archive.
Three test methods fail with 18 subtest failures; the rank-transition and
broadcast methods pass. These are unresolved numerical failures, including
existing captured-scalar behavior, not reference infrastructure errors.

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

## Fresh main baseline

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
NVRTC and runtime identities are recorded in [provenance](provenance.json).
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

The [validation archive](validation.json.gz) retains commands, logs, exhaustive
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
reference-only reset. Passing earlier suites does not establish this milestone.

## Dispatched code and post-commit handoff

[CUDA source](kernel.cu), [PTX](kernel.ptx.gz),
[source manifest](source-manifest.json.gz) and [provenance](provenance.json)
come from [capture.py](capture.py), an independent non-corpus example. The final
call returns to an earlier scalar value after promotion. The capture resolves
its actual guard key and verifies the dispatched module differs from both the
first static module and the last inserted Boolean specialization. The wheel's
frontend/native bytes are checked against the checkout/loaded extension.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  docs/diagnostics/compile-pointwise-positional/capture.py \
  "$PWD/target/positional-codegen" "$PWD/target/wheels/torch_rs-WHEEL.whl"
```

These are uncommitted development captures. Burner owns independent final review,
commit, clean candidate and actual clean-main coverage/CUDA captures, exact-head
qualification and publication. Refresh the capture with the final installed
wheel and both unchanged gates in that post-commit phase. No worker commit,
publication or candidate scoring claim is made here. Fixed-corpus scores do not
establish general Inductor parity.
