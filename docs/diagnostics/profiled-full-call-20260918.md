# Profiled default-compiler guard matching

The retained change replaces temporary generator expressions in `TensorGuard`
and `ShapeGuards` matching with ordered loops. It checks the same predicates,
sizes, strides, properties, cross-input equalities and index bounds, with the
same short-circuit behavior. It introduces no cache or native API; numerical
planning, module reuse, result reconstruction and completion are unchanged.

The investigation began on `c48b9188`, whose `src`/`python` content equals incoming
`9a436482`. A worktree-local `cProfile` diagnostic covered four independent
programs, two shapes and fresh/reused inputs, 200 calls per case. For affine/257,
it attributed about 12.9 µs/call inclusively to specialization selection and
10.4 µs to the prepared native call. These are **instrumented attributions, not
latency claims**. Native admission, launch and completion remain combined in that
measurement; separate post-call synchronization medians were 2.29–2.49 µs,
including Python/ctypes overhead. Nsight was an unavailable toolkit stub, so no
device-only trace was obtained. This does not explain a historical scored result.

A native method-guard batching prototype was tried first. Full public-call
measurements regressed by roughly 2–5 µs in many cases. It was removed; its patch,
five passing tests and all `candidate-*` observations remain in the bundle.
The retained loop-only variant is labeled `shape-*`.

Uninstrumented measurements used ordinary default `torch_rs.compile(fn)` and
default PyTorch 2.13.0+cu130 `torch.compile(fn)`: 76 independent developer cells,
fresh and reused inputs, five warmups, 17 single-call samples, separate processes
and compiler caches, and both framework orders. Input creation/readback stayed
outside timing; identical runtime synchronization bracketed calls. First-shape
and wrapper-creation costs are separate. Every supported cell passed complete
output/metadata/alias and signed-zero checks. Native processes blocked PyTorch
imports and checked warm original-body replay separately.

Against incoming, retained cell medians were lower in **61/76** and **62/76**
cells in the two orders; median cell reductions were **1.21/1.30 µs**. Against
accepted `60202557`, they were lower in **53/58** and **58/58** common supported
cells (median reductions **3.54/5.07 µs**). Accepted main's 18 view rejections per
order remain recorded. These are descriptive developer observations, not a
weighted corpus score, universal speedup or evidence of canonical qualification.

For example, fresh affine/257 native medians were 47.47/44.83 µs incoming and
43.65/43.86 µs retained; the paired default PyTorch medians were 68.93/69.53 µs.
Slow observations also remain: reused affine/257 was
43.90/30.41 µs retained versus 32.20/31.65 µs incoming. No outliers, failed attempts
or unsupported cells were removed, and no canonical evaluator was run.

All captures used GPU0 H100 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`,
driver 580.82.07, runtime/NVRTC 13.0, NVCC inventory 12.8/V12.8.93, Rust 1.92.0
and Python 3.12.12. Release wheels, imports and writable caches were worktree-local.
The retained native extension is byte-identical to incoming; the Python frontend
changed. Captures identify actual source/wheel/library hashes and explicitly
record the **uncommitted author sources**. They are not clean-commit evidence.

The [bundle](profiled-full-call-20260918.tar.xz) preserves profiles, commands,
build/import receipts, all samples, outputs, failures and both patches. Its 16
NPZ files are byte-identical to two members of the
[existing raw archive](default-compile-full-call-20260918-raw.tar.xz).
`bundle-manifest.json` records a complete byte-preserving mapping; this deduplicates
array bytes only, never timing/provenance. To replay offline:

```bash
mkdir -p target/profiled-call-replay
tar -xJf docs/diagnostics/profiled-full-call-20260918.tar.xz -C target/profiled-call-replay
.venv/bin/python -B -I target/profiled-call-replay/profiled-call/bundle.py restore \
  target/profiled-call-replay/profiled-call \
  docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz
.venv/bin/python -B -I scripts/diagnose_compile_full_call.py --compare \
  target/profiled-call-replay/profiled-call/shape-reference-second.json \
  target/profiled-call-replay/profiled-call/shape-native-first.json
```

[Offline verification](profiled-full-call-20260918-replay.json) covers all eight
pairs, including the reverse order and rejected prototype. Final checks: portable
compiler **406 tests, 195 skipped**; focused GPU/compiler **204 tests, 4 skipped**;
diagnostic/documentation **22 passed**. These Linux checks do not reproduce
macOS or Python 3.14 CI.

The required clean-commit, release-build three-way H100 capture remains
outstanding. Burner must first commit this implementation and Main separately
admit continuation; independent review and ordinary evaluation remain later gates.
