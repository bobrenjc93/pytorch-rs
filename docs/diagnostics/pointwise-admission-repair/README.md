# Native admission and source-distribution repair

The default pointwise frontend now obtains its ordered metadata records and
whole-input validation through one private native call. The legacy metadata
and validation APIs remain intact, and every prepared run still validates its
current inputs independently. This changes admission overhead, not supported
operations, numerical behavior, Program identity, or executable ownership.

New source packages exclude large binary evidence archives while preserving
the three mandatory history archives, offline verification fixtures, build
inputs, provider bytes and all declared licenses. Historical Git evidence is
unchanged. Contributor guidance now identifies real NVRTC and C-compiler
prerequisites; README licensing matches the existing package metadata.

## Current author-only direct-scratch revision

The [phase record](warm-repair-scope.md) binds `gelu-b355-direct-scratch-20260917`
to parent `b35595e4`. Selected direct execution now omits unused invocation
scratch; VM and identity-absent execution retain it. The shared output allocation,
launch and completion owner remains in place. No speedup is claimed.

Development checks passed 17 selected Rust tests (the canonical allocation/failure
witness also passed with Python bindings) and five selected Python tests: one
frontend mock and four real GPU controls on the fresh release wheel. The witness
covered 21 direct/selected-VM/legacy-VM cases, including empty calls and distinct
launch/completion errors. All selections had no skips or hardware early returns.
Default test compilation, Clippy with and without Python bindings, formatting
and diff checks passed. This is a subset, excluding the timing-producing
`test_gpu_dispatch_real_native_and_default_histories` test unchanged.

[Development receipts and binary provenance](../gelu-program-integration/direct-scratch-development-manifest.json)
retain the wheel, command streams, three rejected test-fixture attempts, initial
Clippy errors and the test launcher's failed import attempt. These are dirty-source
correctness checks, not postcommit or performance evidence. Completion errors
were injected after real synchronization; a device-not-completed fault and
restoration from a different active device ordinal were not exercised on GPU0.
Burner must commit and stop: review, performance measurement and merge remain
outstanding pending explicit continuation.

## Completed archive repair; native batch stopped

The [current phase record](warm-repair-scope.md) binds rejection of `f83a858`
and permits no new diagnostic profile or timing campaign. The proposed stateless
native method-check batch was stopped before implementation: an ordinary class
can retain a string-subclass namespace key, and its equality callback executes
during the proposed mapping-proxy lookup. Exact tuple/name/metaclass checks
alone cannot meet the callback-free private-boundary requirement. Broader
namespace validation or a trusted-owner registry is outside this bounded repair.
Production method guards remain unchanged; no CUDA performance repair is claimed.
The hermetic reproducer, design outcome and untimed checks are retained in the
[repair archive](../gelu-program-integration/method-guard-repair-manifest.json).

All 28 scoped historical archive copies were removed after the supplied Main
remote observations and local original-blob verification; post-removal retrieval
verified every byte again. Use the [historical retrieval guide](../gelu-program-integration/archive-preservation.md).
This reduces checkout size by 1,612,770,062 bytes, not full-clone history.
The focused untimed selection passed 65 tests with one explicit two-device skip;
12 documentation checks, three new provenance checks and 13 existing archive
checks passed. The timing-producing dispatch-history test was excluded. Tested
Python 3.12/native-extension hashes and GPU0/driver/NVRTC/runtime identities are
retained with the receipts; the existing wheel was verified, not rebuilt or
relabeled. Independent archive review passed.

## Completed warm-call repair

The [repair scope/status](warm-repair-scope.md) binds the rejection of
`34dcc474` and the later public guidance. Warm calls now construct invocation-local
binding projections during input snapshotting, reusing immutable Program roots.
Cold calls and reentrant Program replacement retain the original resolver.
Input guards, native admission and independent prepared-run validation remain.

The one [before/after call-profile pair](warm-profile-summary.json) completed
280 calls per side (256 profiled), with identical cases, inputs/outputs,
interpreter, GPU0 and NVRTC/runtime 13.0. The before production source matches
`34dcc474`; the after source is the recorded **uncommitted** repair. Both use
source-checked release wheels. The second recursive `parameter()` traversal
disappeared. Profiled cumulative execute time was mixed:

| Case | Before / after, ms across 64 profiled calls |
| --- | --- |
| Unary flat | 6.308 / 5.917 |
| Unary nested | 9.225 / 10.279 |
| Binary flat | 9.726 / 7.054 |
| Binary nested | 12.001 / 12.392 |

These are profiler observations, not ordinary latency, reference parity or score
evidence; both nested totals increased. No timing reroll is authorized.
[Raw profiles, wheels, outputs, checks and failures](../gelu-program-integration/warm-repair-manifest.json)
remain auditable. Six new tests and 55 GELU/identity/failure controls passed.
The broader suite ran 414 tests: 403 passed, nine skipped, and two errors in
the original new fixtures; the corrected module passed separately. CUDA-hidden
controls passed 25 tests with 17 skips. All 25 documentation/offline archive
checks passed. Local design and code review passed.

That broad suite also invoked an existing dispatch-history test which emitted
**unplanned timings** (408 native and 408 reference calls). This scope deviation
is retained separately and excluded from performance claims; it is not
retroactively admitted as another phase. The three required offline archives
remain intact. Historical archive removal, blocked during that phase, is now
complete under the later supplied proof and [retrieval contract](../gelu-program-integration/archive-preservation.md).
No official gate has been rerun. Earlier clean measurements below do not measure
this repair; the no-extra-phase boundary prevents a replacement timing campaign.
Any later source change that invalidates this profile must be reported.

## Earlier clean-commit evidence


The [clean capture](postcommit-728d147.json) measures implementation commit
`728d147bfdb2b05e261cb1a4ea1df9e26900c348` against rejected `8006d7e0` under
the unchanged [six-leg protocol](protocol.md). Both native wheels were rebuilt;
C was built and installed from its fresh source package. Every leg records a
clean source checkout inside this worktree. All six legs and offline verification
passed on GPU0 with NVRTC/runtime 13.0 and unchanged PyTorch 2.13.0+cu130.

The geometric mean B/C steady latency ratio was **1.061×**, and reference/C
was **0.866×**. Four steady comparisons regressed: small binary in forward
order, small unary in reverse order, and the 1153-element held-out case in both
orders. The largest first-call regression was 6.7%; nested churn regressed
18.5% and 7.5%. All cells, samples and dispersion remain in the
[raw archive manifest](../gelu-program-integration/postcommit-728d147-manifest.json).
These observations do not establish an official CUDA score or GELU link-cost parity.

The installed wheel passed 11 targeted admission/GELU tests, eight diagnostic
integrity controls passed, and 12 untimed selected-executable captures verified
provider/license inclusion in wheel and sdist. The extracted sdist passed 12
offline archive tests with one explicit Git-history skip. No production, test,
dependency or measurement-tool changes accompanied this capture. Canonical
qualification remains separate; the completed eight-leg integration was not replayed.

## Pre-commit development evidence

The [frozen protocol](protocol.md) compares rejected commit
`8006d7e05636b3322da86ae89c41040900ceee00` with the **pre-commit repair snapshot** and
unchanged default PyTorch 2.13.0+cu130. All six ordered B/R/C/C/R/B processes
passed on reserved H100 GPU0, with NVRTC 13.0 and CUDA runtime 13.0. The separate
toolkit `nvcc` was 12.6. The [derived summary](development-summary.json) binds
actual source snapshots, dirty status, local builds/imports and runtime files.
These are development measurements, not clean-commit evidence or qualification.

Across the 16 paired cells, the geometric mean B/C steady latency ratio was
**1.023×**, and reference/C was **0.817×**. Four native comparisons were slower:
extent-509 unary in both orders, extent-509 nested in reverse order, and the
49157-element held-out case in reverse order. The largest first-call regression
was 9.7%; reverse-order unary and held-out churn calls regressed 7.3% and 7.4%.
All samples, dispersion, outputs and orders remain in the capture. This modest,
mixed diagnostic result does not establish an official CUDA score improvement
or measure GELU link-cost performance. No unchanged timing was rerolled.

The [raw archive manifest](../gelu-program-integration/admission-repair-development-manifest.json)
indexes command streams/receipts, source snapshots, both comparator wheels,
the source-package-built candidate wheel, both slim source-package attempts,
input/output captures, package inventories and local review. Retrieve the original
parts from [ancestor Git blobs](../gelu-program-integration/archive-preservation.md),
then concatenate them in manifest order to recover the verified gzip archive. It excludes build caches,
whole environments and older archives. The earlier
[GELU integration evidence](../gelu-program-integration/README.md) remains pinned
to its original sources; its completed eight-leg phase was not replayed.

## Checks and limits

The required-check follow-up fixes the shared documentation smoke-test failure:
the contributor guide exceeded its existing 120-line bound. It now links to
the unchanged NVRTC instructions in troubleshooting. With CUDA hidden, Python
3.14's full suite passed (6,350 run, 603 skipped); all 12 Python 3.12 documentation checks
passed after the fix. The preceding 3.12 full run had only that one failure.
[Command receipts, failures and the fresh wheel](../gelu-program-integration/ci-gate-repair-manifest.json)
are retained separately from the measurements above; GitHub log access was blocked.

- Default Rust tests: 486 passed; Python-enabled Rust unit tests: 303 passed.
  Formatting and Clippy passed with both feature configurations checked.
- Pointwise Python suite: 182 passed, four explicit multi-GPU skips. Its initial
  discovery included an imported test class twice; that import was corrected.
- GELU/identity/compiler-failure/consumer suite: 55 passed. The installed
  source-package wheel then passed 11 admission/GELU tests and 12 untimed
  selected-executable/package captures.
- Each actual extracted source package passed 12 offline archive tests, with
  one explicit unavailable-Git-history skip. Provider and license inclusion
  passed for the installed wheel and source package. Maturin's sole build-input
  transformation, adding `package.readme`, is explicitly bound in C's build record.
- With CUDA hidden, 56 portable admission/cache/consumer controls passed.
  Eight diagnostic corruption controls and local independent design/compatibility
  review passed. Release-unobservable counters remain unknown.

Preserved failed development attempts include the initial Rust fixture's private
field access, its unsupported CUDA clone, and a baseline-environment dependency
attempt before correcting the local interpreter. No external environment was
modified. Canonical review, full qualification and exact-head CI remain separate;
no official evaluator, denominator, tolerance or managed progress artifact changed.
