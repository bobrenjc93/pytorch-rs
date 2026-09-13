# Operator-review repairs

The NaN guard and diagnostic-consumer repairs address the two operator findings
against `b9d5fd1a6f1d493e7e26aa3f2db32fcfb3b99444`. This directory records development
validation from that commit plus the recorded working-tree changes. It does not
claim a clean candidate measurement. The subsequent [clean refresh](../postcommit-779e512c/README.md)
measures committed repairs at `779e512c` and actual main, including dispatched-module
provenance. This development archive and previous measurements remain unchanged;
no evaluator, corpus, tolerance, denominator or managed artifact changed.

## NaN guard

The preserved [operator reproducer](review-native-nan-guard-repro.py) has SHA-256
`fcdc50711e4f55501e8122a99172d01f48ae8bae039ed1dec209296963248397`.
It executes the production frontend with only its native boundary redirected,
using the actual cache class. The [validation archive](validation.json.gz) retains
the original operator report and a local pre-fix reproduction: eight distinct
signed/payload NaNs consume the default eight specializations, and the ninth
raises `NotImplementedError`. This is a cache-domain failure; no numerical kernel
or native build participates in that stdlib-only reproduction. Its source and
interpreter identities are recorded separately from the new native wheel.

Installed PyTorch 2.13 `torch/_dynamo/guards.py:2707–2716` uses an exact-float plus
`isnan` guard. The repair canonicalizes only the logical key, retaining the
original scalar in frozen values, typed IR and runtime operands. The same packed
float-key representation still feeds the existing finite-promotion owner.
The unchanged reproducer now records one specialization for all nine values.

Permanent CPU cache-control and H100 regressions exercise parameter, global and
closure origins with twelve signed/payload NaNs in both orders, every value
followed by warm reuse, then finite promotion and further NaNs. Both ordinary
default compiled wrappers stay alive throughout each history. Numerical checks
remain exact outside NaNs, including signed zero; NaN output payload identity is
not required. The tests also check frozen/raw binding bits, runtime ABI bits,
cache counts, reset, unchanged inputs, fresh outputs and no original-body replay.

## Diagnostic consumers

The maintained bounded-broadcast and pointwise JIT scripts select the executor
from the successful-call LRU. The bounded example returns to its first shape
after dispatching another module. Tests exercise the actual cache class and an
actual native CUDA shape revisit, so the selected executor need not be the first
cached or most recently created module.

The separate development captures retain actual source manifests and provenance:

- [Bounded broadcast](bounded-capture/provenance.json), including a shape revisit.
- [Pointwise JIT](jit-capture/provenance.json), with a hash for each observed dispatch.
- [Positional bindings](positional-capture/provenance.json), using the new wheel.

Existing measured CUDA/PTX, baseline reports and failure archives remain
byte-for-byte unchanged. The older broad-broadcast script remains explicitly
historical; its program is outside the current numerical admission boundary.

## Reproduce

Use a worktree-local locked release wheel and local caches. The archive preserves
build/install logs, wheel/native/source hashes, reference guard source identity,
all test commands and outcomes, and the source diff. The managed-Python download
failed through the proxy; a standalone Python 3.12.12 installation was copied
read-only into this checkout before creating its local environment. All
subsequent dependencies, builds, interpreters and caches stayed inside the worktree.
Burner held the `gpu` and `cpu-heavy` resources; CUDA tests used H100 GPU 0.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_scalar_admission.SharedCacheGuards.test_nan_payloads_share_static_guard_without_rewriting_frozen_values \
  tests.test_compile_pointwise_runtime_scalars.PersistentSpecializationHardware.test_signed_nan_payloads_reuse_static_guard_then_promote_for_all_sources \
  tests.test_compile_pointwise_jit.Admission.test_diagnostic_capture_selectors_follow_successful_executor_lru \
  tests.test_compile_pointwise_jit.Hardware.test_diagnostic_capture_selectors_follow_actual_shape_revisit
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  docs/diagnostics/compile-pointwise-broadcast/capture_bounded.py target/new-bounded-capture
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  docs/diagnostics/compile-pointwise-jit/capture.py target/new-jit-capture
```

The four focused tests pass. The surrounding selection runs 123 tests across ten
separate owner processes, with four explicit device-reservation skips. Hardware-free
validation runs 70 tests, passing 30 and skipping 40 GPU cases. Formatting and
documentation checks pass. Two broadcast owners initially failed import because
they require `tests/` on `PYTHONPATH`; both pass with that path set. Initial errors,
successful reruns and the final frozen-IR assertion check remain in the archive.
