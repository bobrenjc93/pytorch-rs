# Warm-call and historical-archive repair

Starting source: `34dcc47454b3faaa27ef46a5189b34e81d3802f2`.
The confirmed rejection is `leaf-evaluation_0f7369bf`, completed
2026-09-17T00:32:59.255Z: CUDA 33.4352 versus 34; polish 74 versus 80.
This record follows the later public repair guidance supplied with that rejection;
it is not evaluation evidence or authority to change the gates.

The original integration development/eight-leg phase and the native-admission
six-leg development/clean-postcommit phases are complete. Their bytes, failures
and source identities remain intact. This repair prospectively permits **one**
before/after Python call-profile pair, with no additional timing campaign at
author or post-commit handoff. Correctness tests and normal Burner qualification
remain required. Stale claims or conflicts with an outstanding capture must be
reported without launching another phase or relabeling evidence.

Before either profile, the cases and limits are fixed here: unary `-x * scale`
and binary `-x + y * scale`, each with matched flat and fresh nested-container
inputs, contiguous float32 CUDA tensors of shape `(1021,)`, and scale `0.5`.
Each case makes one cold call, five unprofiled warmup calls, then 64 individually
profiled ordinary public compile invocations: **280 compiled invocations per
side**, 256 profiled. Fresh containers and current tensors are constructed
outside profiling; every result is checked outside profiling and retained until
that case ends. GPU0 only; the interpreter, runtime, input values, cases and
policy match across sides. Profiler durations are diagnostic, never parity scores.

Outputs: `target/warm-repair/profile-before/` and
`target/warm-repair/profile-after/`. Each side records actual source/build,
installed files, interpreter, driver/compiler/runtime and GPU identities at entry.
A failed or incomplete side is preserved, not rerolled. The implementation must
follow the before profile, preserve all admission and independent run guards,
and receive focused correctness checks before the after side.

The six named historical archive groups may leave this checkout only after
complete inventory, blob/size/hash, ancestry and pushed-remote reachability
proofs succeed. Otherwise copies remain and the missing proof is reported.
The three mandatory offline archives and PR1924 inventories are unchanged.

Status: the planned pair is complete (280 compiled calls per side, 256 profiled),
with matching runtime/build checks and outputs. The invocation-local projection
change and focused correctness/review checks are complete; nested profiled
execute totals increased, so no parity or score improvement is claimed. The
frozen pre-profile version of this record is retained as
`target/warm-repair/profile-plan.md`, matching both captures' scope hashes.
See the [current evidence index](README.md#current-warm-call-repair).

The broad correctness suite inadvertently invoked an existing dispatch-history
test that records timings: 408 native and 408 reference calls, retained under
`target/dispatch-smoke-hmruts7w`. This is an unplanned scope deviation, not a
retroactively admitted phase, and supplies no performance claim. Its commands,
outputs and failure context remain in the new archive.

Archive checkout reduction is blocked: all 28 original files, hashes and
ancestor blobs were verified, but GitHub API access was denied and git remote
access returned HTTP 403. All copies remain; a fresh verifiable remote
reachability observation is still required before removal.

Remaining: Burner independent review and qualification; external remote proof
for archive removal. No extra author/post-commit timing phase is authorized.
Existing measurements remain bound to their original sources. A stale claim
or fresh-capture conflict must be reported without another phase. Both handoffs
must open with scope/status and this record's path.
