# Positional scalar binding validation

Evidence for the bounded native CUDA pointwise compiler. Start with the
[supported inputs](../../compile-pointwise-jit.md#supported-programs),
[cache contract](../../compile-pointwise-jit.md#cache-behavior) and
[original-IR numerical admission](../../compile-pointwise-numerics.md).
Fixed-corpus scores do not establish general Inductor parity.

The [clean warm-dispatch record](postcommit-ab6a3acd/README.md) includes refreshed
fixed gates and captures. Its balanced H100 comparison is blocked by the committed
diagnostic's missing native synchronization API; all 16 attempts are retained.

## Clean measurements

Each record retains its measured commit, source/build/wheel identities, commands,
failures and raw reports. Historical snapshots supply no current performance
credit; documentation changes do not change their attribution.

| Record | Scope and result |
| --- | --- |
| **Latest: [ab6a3acd](postcommit-ab6a3acd/README.md)** | Clean candidate/actual-main gates, three dispatched-module captures and 36 passing focused tests; retains the failed 16-leg warm-dispatch comparison. |
| [779e512c](postcommit-779e512c/README.md) | Candidate and actual main `a281503f`; full unchanged coverage/CUDA gates, three dispatched-module captures and 16 passing focused tests, including the former signed-zero failures, NaN histories and diagnostic consumers. |
| [7a74596b](postcommit-7a74596b/README.md) | Earlier clean shared-cache revision; 12 persistent tests pass, before the NaN-guard and diagnostic-consumer repairs. |
| [4ed105f5](postcommit-4ed105f5/README.md) | Original clean positional extension; preserves all 18 failing signed-zero subtests and its original CUDA/PTX capture. |

The latest positional [CUDA](postcommit-ab6a3acd/positional-codegen/kernel.cu),
[PTX](postcommit-ab6a3acd/positional-codegen/kernel.ptx.gz),
[source manifest](postcommit-ab6a3acd/positional-codegen/source-manifest.json.gz) and
[provenance](postcommit-ab6a3acd/positional-codegen/provenance.json) come
from the independent non-corpus [capture](capture.py). Its final call revisits an
earlier scalar after promotion and verifies that the dispatched executor differs
from both the first static module and the last inserted Boolean specialization.
The latest record links the other two captures and reproduction commands.

## Retained pre-fix failures

The [shape revisit probe](zero-shape-probe.py) retains
[20 observations, including eight numerical failures](zero-shape-probe.json.gz).
For positive-one tensors and one persistent `f(scale, x) = x * scale` wrapper:

| Call | Scalar | Shape | Native output | Default-Inductor output |
| --- | --- | --- | --- | --- |
| 1 | `+0.0` | `(2,)` | `+0.0` | `+0.0` |
| 2 | `-0.0` | `(3,)` | `-0.0` | `-0.0` |
| 3 | `+0.0` | `(2,)` | `+0.0` | `-0.0` |

The opposite sign order fails symmetrically. The
[baseline-wheel probe](main-capture-shape-probe.py) also reproduces captured-float
failures in [four of ten observations](main-capture-shape-probe.json.gz) on clean
pre-implementation main. That wheel's SHA-256 is
`5574ecdbf9fdb8d256191dec9d6d5ed299174f84f0cca358e80e48bca5d55f96`;
it was extracted into `target/main-binding-probe-package` and selected with
`PYTHONPATH`, without modifying the installed environment.
The [guard inspection](shape-guard-inspect.py) and [record](shape-guards.json.gz)
show the persistent reference's second graph generalizing the changed dimension
to `2 <= size <= 2147483647` and preceding the first exact-shape graph. Native's
old exact-metadata lookup selected the first graph instead: an admitted sign-bit
mismatch, not an excluded expression or compilation-count difference.

Five `ScalarShapeSpecializationHardware` methods in the
[runtime-scalar tests](../../../tests/test_compile_pointwise_runtime_scalars.py)
retain parameter/global/closure histories in both sign orders. Three methods
produced 18 subtest failures across revisits, singleton/empty and unused-tensor
histories; rank-transition and broadcast methods passed. All five later passed
unchanged, keeping both wrappers alive without reference-only resets.

The separate [operator repair record](operator-review-fix/README.md) preserves
the ninth distinct signed/payload NaN's false `recompile_limit=8` rejection and
the stale diagnostic consumers' missing selection/`TypeError` failures. Its
[stdlib-only reproducer](operator-review-fix/review-native-nan-guard-repro.py) and
[validation archive](operator-review-fix/validation.json.gz) retain the original
source/interpreter identities separately from native-wheel validation. NaN
payload identity in outputs was not the failure under test.

## Shared-cache revision

The [cache contract](../../compile-pointwise-jit.md#cache-behavior) describes
logical specialization selection, concrete executors, promotion and reset.
The [revision archive](review-fix-validation.json.gz) records development checks
of that repair; it is separate from the clean measurements above.

| Historical validation | Recorded scope |
| --- | --- |
| Full compiler selection | 948 tests in 87 disjoint processes plus a 156-test pointwise rerun in 19 processes cover 965 distinct final tests. Four single-GPU reservation skips passed separately on GPUs 0 and 1. |
| Other Python runs | Python 3.14 scalar owners: 58 tests, two reservation skips. Hardware-free scalar owners: 19 pass, 39 GPU skips. |
| Rust and documentation | 31 pointwise tests in each of default/Python-bindings configurations; both builds and Clippy configurations, formatting and documentation checks pass. |

The sweep wheel and subsequent pointwise wheel differ only by contract docstrings
and removal of an unused private argument; both identities and the exact diff
are archived. The neg/add diagnostic correctly rejected mismatching source bytes,
then passed all five tests with the corresponding wheel installed in `.venv`.
Initial source-mismatch, documentation-anchor and import-path errors and successful
reruns remain recorded; no numerical failure occurred in these revision checks.
The development [CUDA](review-fix-dispatched/kernel.cu),
[PTX](review-fix-dispatched/kernel.ptx.gz),
[provenance](review-fix-dispatched/provenance.json) and
[source manifest](review-fix-dispatched/source-manifest.json.gz) record dirty
sources based on `66f4acdd`, not a clean candidate measurement.

## Reference characterization

The initial [probe](reference-probe.py) and [150 observations](reference-probe.json.gz)
cover parameter/global/closure precision, Boolean transitions, signed zeros,
nonfinite interludes and overflow. Each ordinary default-Inductor wrapper stays
alive for its scalar sequence; resets separate independent histories only.
The inspected PyTorch 2.13 owner was
`torch/_dynamo/variables/builder.py::wrap_symfloat`, including
`process_automatic_dynamic`, static nonfinite specialization and runtime float
materialization. Native code does not import it. Permanent regression owners are
[admission](../../../tests/test_compile_pointwise_scalar_admission.py) and
[runtime scalars](../../../tests/test_compile_pointwise_runtime_scalars.py),
including positional-integer rejection and existing captured/literal large integers.

## Pre-implementation main baseline

The original [coverage](main-coverage.json.gz) and
[CUDA performance](main-cuda-perf.json.gz) reports measure clean main
`a281503f3391bbd98a0ab6de2ff8f0c0a55a12d4`, source identity
`6047babf37c578031a0ef8ce76f2d74b2ad5a3e97b144f727c2e81dfd78a6243`.
Both are valid, non-diagnostic reports: coverage 8% (6/112), CUDA performance 12%
(6/56), common-success geometric mean 2.0181160029262264. They retain every cell,
unsupported outcomes, both orders, five warmups, 17 synchronized samples,
cold/steady timings, build identities and GPU snapshots. These are baseline
measurements, not candidate gains.

## Development validation and reproduction

The pre-fix [validation archive](validation.json.gz) preserves commands, logs and
setup failures: 941 tests in 87 disjoint processes with 25 hardware/reservation
skips, then seven supplemental tests completing the 948-test selection. Two
binding tests passed; the five shape-history methods retain the failures above.
Both builds and Clippy configurations, formatting, 31 Rust tests per configuration
and 12 documentation tests passed. Hardware-free scalar tests passed nine and
skipped 32. Missing checkout-root imports and the matmul isolation test's required
`.venv` installation caused initial setup failures; successful reruns are retained
separately from the numerical failures. This pre-fix run does not qualify the repair.

That development setup used locked dependencies, Rust 1.92 release ABI3 wheels,
and worktree-local Python 3.12.12/3.14.5 trees copied from installed standalone
interpreters after the managed download proxy failed. All environments, builds
and caches stayed inside the checkout. H100 tests used GPU 0, with restoration
checks on GPUs 0 and 1. Installed nvcc was 12.6; generated kernels used NVRTC 13.0.

For later NaN-guard and diagnostic-consumer development checks, commands, build
identities and captures, see the [operator repair record](operator-review-fix/README.md).
For clean reproduction, use the [latest measured record](postcommit-ab6a3acd/README.md#reproduction-and-provenance).
Burner's independent review and unchanged full evaluation gates remain separate
from these focused diagnostics.
