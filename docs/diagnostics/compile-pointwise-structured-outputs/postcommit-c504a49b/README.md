# Clean capture at c504a49b: history-dependent failure retained

Measured implementation: `c504a49b3186c26cae49c1422af650b16258fce4`, against `main` at
`30a3b504ef4d43bf2958998cc39545996cc09970`. This supplies the clean-commit capture deferred in the
[nonlinear repair record](../review-nonlinear-regions.md). The previous
three-size reproducer passes, but the **structured-output numerical milestone
remains incomplete** because the compilation-history-dependent failure persists.

## Results

The unchanged committed checks ran against a freshly built release wheel:

| Check | Passed | Explicit skips |
| --- | ---: | ---: |
| Python pointwise suite on H100 | 287 | 8 device tests |
| Dedicated two-physical-device checks | 8 | 0 |
| Release native pointwise and ownership | 47 | 0 |
| Release Python-conversion failure ownership | 1 | 0 |
| Portable pointwise suite, each CPython version | 143 | 152 hardware tests |

Portable versions are 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware
skips are not GPU passes. Native-extension verification passed. The suite includes
1,120 nonlinear-region comparisons, 600 sibling-product comparisons, and the
existing identity, freshness, current metadata, cache reuse, helper/loop/branch
and failure-atomicity checks. The committed nonlinear test resets the reference
compiler before each shape, then uses ordinary `torch.compile(fn)` without
backend/mode/fullgraph/dynamic overrides. Native wrappers persist across shapes.
This setup tests cold-reference partitions; it does not resolve the history issue.

The suite retained 1904 materialized native/reference comparisons with
CUDA/PTX, each containing one native kernel entry. The isolated no-PyTorch-import
and no-original/helper-replay check passed.

Two archived diagnostic programs were copied byte-for-byte and rerun unchanged:

- Previous reproducer: `p=x*y; q=-p; r=p.sin(); return (q,r)`, with positive
  float32 inputs `x=y=1e-38`. All six first/repeated comparisons pass at sizes
  1, 13 and 257. Both implementations return `q=-0` at size 1 and `q=+0` at
  sizes 13 and 257.
- History diagnostic: seven bodies, four return forms, sizes 1/2/3/13/257,
  four value histories and two calls, with persistent reference wrappers.
  **24 of 1,120 comparisons fail**, all in `p=x*x; q=-p; r=p.sin()` with
  `(q,r)` or `(r,q)` at sizes 3/13/257 after size 2. Native returns `q=+0`
  while the reference retains `q=-0`. This is the documented automatic-dynamic
  optimization-hint difference, not an infrastructure failure or a pass.

The passing suite and failing diagnostic use the existing strict tolerances and
zero-sign assertions. No workload, assertion, implementation or external producer
was changed to suppress the failure. Raw outputs and reference compiler caches
for both outcomes are preserved.

## Provenance and retention

[Measurements](measurements.json.gz) record clean before/after status, every
tracked file hash, timestamped commands, cache state, wheel/import identities,
loaded libraries, GPU snapshots and outcomes. All tracked files remained
byte-identical throughout capture. Installed package and extension members
matched the wheel on all five interpreters.

Wheel SHA256: `ceb2d1c12679a81eba88c1d3888bab7cc23cb3e05ec01a67f56231be68915b35`.

The locked offline release build used a fresh Cargo target and initially absent
reference caches. Ordinary H100 runs used `CUDA_VISIBLE_DEVICES=0`; device checks
reserved only `0,1`. Captures record Rust 1.92.0, Maturin 1.15.0, driver 580.82.07,
PyTorch 2.13.0+cu130, NVRTC 13.0
and CUDA runtime 13000. PATH `nvcc` reports 12.6.85; NVRTC compiled native kernels.
Runtime probes reused the report-local caches populated by the suite. No timing
or performance credit is claimed.

The [raw archive](raw-captures.tar.gz) contains successful and failed logs,
raw outputs, CUDA/PTX and orchestration scripts. The [retention manifest (XZ)](raw-retention-manifest.json.xz)
identifies byte-verified wheel, committed-source, reference-cache and nested
`target/dispatch-smoke-*` archives under
`target/default-compile-eval/structured-outputs-postcommit-c504a49b/`.
Earlier checked-in evidence was hash-verified unchanged. Older nested diagnostics
retain their original identities and receive no current credit. No cleanup or
external write was performed; Burner's canonical report-root observer owns
external archival.

Only new evidence and the focused guide link change in this step. Implementation,
tests, dependencies, benchmark harnesses, evaluation definitions and managed
progress artifacts remain unchanged. No official score or unrelated repository
suite was rerun. This capture neither approves the branch nor resolves the
remaining numerical blocker.

The current [retention manifest](raw-retention-manifest.json.xz) and
[verification log](verification.log.xz) use lossless XZ transport; the raw archive
and historical container hashes remain unchanged. See the dated
[transport mapping and decode commands](../README.md#artifact-transport) for exact
old/new hashes, and the [retention limits](../README.md#provenance-and-retention-limits)
for corrected observer coverage and later operator custody.
