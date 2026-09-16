# Warm ownership bookkeeping repair

This is non-scoring development evidence for the narrow repair of `4993a379f0b57ca97f9e15ef5f63db98991d4b5f`.
Burner's confirmed CUDA result for that implementation remains **33.0539/100**,
below the accepted baseline's 34. No new score or ordinary-call speedup is claimed.

The frontend now certifies native ownership once after binding and the shape
snapshot guard, before accounting, execution or publication. Selected hits still
check the recorded executor against the current map object. The existing bounded
scan still prunes stale owners after success. Frozen native owners preserve the
admitted relationship; this does not protect manually fabricated inconsistent
private preparation tuples. No Rust, numerical, codegen or scoring change is made.

## Bounded attribution

The same [profiler](../profile-ownership.py) ran on worktree-local release wheels
before and after the source edit. Both captures retain their actual dirty status,
HEAD, source and wheel hashes, installed-file checks, loaded libraries and GPU
inventory. The before capture preceded the production change; the after capture
measures an uncommitted patch over the recorded HEAD, not a clean implementation
commit. Each profile checks all retained outputs outside profiling.

| Phase | Preparations before | Calls | Queries before | Queries after |
| --- | ---: | ---: | ---: | ---: |
| Cold | 0 | 1 | 0 | 1 |
| Newest warm hit | 1 | 16 | 32 | 0 |
| Warm miss | 1 | 1 | 1 | 1 |
| Newest/older warm hits | 3 | 32 | 128 | 0 |
| Newest/older warm hits | 8 | 32 | 288 | 0 |

The old hit makes P+1 native ownership queries for P current preparations. The
repair removes those queries while retaining Python iteration and map lookups;
it adds explicit admission on the cold path. Profiler seconds in the raw records
are instrumentation measurements, not ordinary-call latency. These observations
do not establish that ownership queries caused the entire canonical loss.

GPU work used only visible GPU0, H100 UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07. The captures pin
worktree-local CUDA runtime and NVRTC 13.0 and verify their actual providers.
No two-GPU claim is made. Build and test logs, the environment, profile records
and source bindings are retained in the [archive](development-evidence.tar.gz)
and [manifest](manifest.json). All 32 member hashes and both source bindings were
verified. The two wheels' native extension bytes are identical.

## Validation

The H100 pointwise suite ran 386 tests successfully with 9 explicit multi-GPU
skips. All 9 real native ownership/compiler-failure tests and all 290 Rust tests
passed. The portable preparation suite ran 29 tests successfully with 5 hardware
skips; formatting, Clippy with warnings denied, and all 33 frozen consumer
controls passed. The archive includes the executed commands and full logs.

The four new portable tests cover cold/warm-miss owner mismatch and query
exceptions, once-per-bind query counts across multiple retained preparations,
selected/non-selected owner replacement, failure snapshots and exact byte
accounting. Their intentional run against the pre-repair wheel failed (one
failure and four subtest errors); that log is preserved rather than discarded.

## Clean post-commit evidence

Burner committed the repair as `fc6c345bd71205626329627cd4c28a085af6072e`.
The [new clean comparison](../postcommit-fc6c345/README.md) completed all eight
ordinary-default B/C/reference legs and passed the unchanged verifier, retaining
every sample and cold/churn/small-case regression. The previous complete and
failed archives remain unchanged. Independent review, unchanged full qualification
and exact-head CI remain required; the development profiles above do not replace
those gates or measure ordinary-call latency.
