# Namespace and broadcast guard repair

The repair moves exact-string namespace scanning into a private stateless native
predicate and removes the first identity combine in broadcast element counting.
Python keeps namespace selection, error ordering and every-call validation.
PyO3's public critical section surrounds both iterator construction and scanning;
this does not establish free-threaded Python qualification. Other guards,
numerical planning, owners, synchronization and cache publication stay unchanged.

The once-only instrumented profile at
`target/structured-warm-profile.iQmC81/profile.json` has SHA256
`8f03f5f519a755fdb20fb0649484ebba1d661871df07ee8982184c9a48140d8f`.
It measured 1,024 warm calls for each of four public functions at `17ccacb98`.
Its attribution totals are not uninstrumented latency or a speedup estimate.
The failed setup at `target/structured-warm-profile.eGOEoY` remains retained.
The negative full qualification at `17ccacb98` remains authoritative.

## Fixed paired capture plan

This plan is declared before any repair timing. Run it once after Burner commits
the implementation, using two new worktree-local environments with verified
wheel members. A is the sealed `17ccacb98` release wheel:
`target/default-compile-eval/wheels.QdELd0/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl`,
SHA256 `5c722b483df0031bddd06ad5c588425e262390be45eb615e92594329b159fd7a`.
B must be a release wheel built from the clean repaired commit. Preserve the old
profile environment. Use the same CPython 3.12 interpreter and locked dependencies
for A and B; record both source identities, wheel/import hashes, commands,
timestamps, cache directories and GPU/runtime/compiler identities.

Use the four functions from retained `profile-script.py`, copied verbatim without
its profiling instrumentation: `literal(x): x*1.25+0.75`, `pair(x,y): x*y+x`,
`broadcast(x,y): x+y`, and `structured(x)` returning
`(x*1.25, {'neg': -x, 'repeat': x*1.25}, x.shape[1])` with the **same local product
object** in both product slots. Inputs are contiguous native CUDA float32:
`x` is shape `(37,23)` filled with `1.0`, `y` is that shape filled with `0.5`,
and the broadcast right operand is shape `(23,)` filled with `0.5`.
Use ordinary `torch_rs.compile(fn)` defaults throughout.

Run four fresh processes in fixed **A/B/B/A** order, each visiting the cases in
literal/pair/broadcast/structured order with fresh wrappers and process caches.
Use physical GPU 0 (`CUDA_VISIBLE_DEVICES=0`), one host thread, checked
`cudaSetDevice(0)`/`cudaDeviceSynchronize`, and record GPU UUID/utilization/memory
before and after. Ensure shared-resource ownership and no competing benchmark
process before starting; do not interrupt other jobs.

Record the first call separately, then five warmups and 17 samples of 256 calls
per case. Synchronize before each sample; time the 256-call loop with
`perf_counter_ns`, retaining each result and checking device synchronization after
each call. Divide elapsed nanoseconds by 256. Materialize the final sample result
outside timing. No profiler, tracing, validation or cache inspection runs inside
the timed loop. Retain every raw sample; report per-case medians for both blocks
and, for each case, median(A1 ∪ A2) / median(B1 ∪ B2), then the
geometric mean of those four case ratios. Do not select a
faster block, discard slow samples, change the schedule or reroll unchanged code.

Report correctness separately: outside timing, compare the four functions to
ordinary stock PyTorch default compilation in a separate reference process,
using the timed inputs and a second same-shape set cycling signed zeros, finite
signed values, infinities and NaNs. Keep existing `rtol=1e-5`, `atol=1e-6`, NaN
and zero-sign assertions. Check shapes/dtypes/devices, nested topology, repeated
product identity, distinct negation storage, current shape metadata, unchanged
inputs and retained-output freshness. Audit no body replay and stable warm cache
identities outside timing. A semantic/setup failure invalidates the diagnostic;
preserve its raw files and do not report a passing speedup.

Keep new raw reports, samples, source/PTX/plans, setup failures and checksummed
archives under a new `target/default-compile-eval/run-*` root. Check in only a
compact summary with exact artifact paths/hashes. This A/B diagnostic produces no
qualification score; Burner's reviewed delivery owns frozen full qualification.
The clean capture below completes this plan; it was run once without retries.

## Clean committed capture

The [source-bound report](postcommit-d0f965a2.json) measures
`d0f965a29b2d8ad02219f746c1f5ed6a0edbe89a` with a clean worktree and a new release
wheel. It passed 66 structured/prepared/guard checks, nine separate two-device
checks, 73 native tests and 171 portable tests on each CPython 3.10–3.14, plus
Rust 1.92 formatting and both Clippy configurations. Hardware skips are recorded
separately. The native compiler used NVRTC 13; runtime and library hashes are in
the report and raw provenance.

The declared A/B/B/A run completed on physical H100 GPU 0. A reused the sealed
`17ccacb98` wheel; B used the clean `d0f965a2` wheel. Separate correctness processes
matched ordinary stock PyTorch 2.13 default compilation on the declared original
and changed inputs, including signed zeros and nonfinite values. Metadata,
aliases, freshness, unchanged inputs and warm execution/cache audits passed.
Timing followed the fixed plan above, including synchronization after every call.

Each timing entry below is microseconds per call. Pooled columns use all 34
samples for that implementation; all 272 samples and separate first-call times
remain retained. Ratios above one mean lower measured latency after the repair.

| Case | A1 | B1 | B2 | A2 | Pooled A | Pooled B | A/B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| literal | 31.176 | 29.829 | 29.800 | 31.499 | 31.418 | 29.809 | 1.0540 |
| pair | 38.166 | 35.226 | 35.750 | 38.973 | 38.918 | 35.652 | 1.0916 |
| broadcast | 37.677 | 35.359 | 35.772 | 38.614 | 38.073 | 35.535 | 1.0714 |
| structured | 33.562 | 32.263 | 32.677 | 34.259 | 33.937 | 32.450 | 1.0458 |

The four-case geometric-mean ratio is **1.06555**. This diagnostic does not measure
default-Inductor performance parity or change the saved negative qualification.
The report links the exact commands, wheel/source/import/runtime identities,
raw numerical output, generated native source/PTX/plans and readback-verified
archive under `target/default-compile-eval/run-postcommit-d0f965a2/`. Preserve that
canonical report root before cleanup; it is not an off-host backup.

## Development validation

The uncommitted repair passed Rust 1.92 formatting and both CI Clippy
configurations, 330 one-H100 checks plus the remaining nine two-device checks,
and 73 native tests. CPython 3.10–3.14 each passed 171 portable checks and skipped
168 hardware cases. The seven new guard tests are included in those counts;
portable skips do not establish GPU behavior. Focused source review found no
actionable issue. These development results remain separate from the clean
capture above.

Receipts, wheel/import/source hashes, all logs and 3,789 numerical CUDA/PTX/plan
captures are retained at `target/default-compile-eval/run-guard-repair-17ccacb/`.
Its `validation-summary.json` has SHA256
`245d3316a3485a5f82de7a5087e3c52c4a156efb8555e031a0866d3aa7fc1a55`.
The readback-verified `raw-evidence.tar.gz` (261,700,010 bytes) has SHA256
`1c5c6c10d853c527fd7fe223df59ae4201dbb220f0cec0552db08b5aa7a6c37d`. It also retains byte-exact
copies of both historical profile attempts and the sealed before wheel. Raw
artifacts remain ignored, worktree-local evidence; preserve the canonical report
root before cleanup. No comparative timing or qualification score was produced.
