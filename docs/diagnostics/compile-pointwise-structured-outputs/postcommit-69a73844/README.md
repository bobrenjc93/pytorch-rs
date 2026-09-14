# Clean structured-output capture at 69a73844

Measured implementation: `69a738444ee7c71044730d2bccecb735cad8b161`, against `main` at `30a3b504ef4d43bf2958998cc39545996cc09970`. This supplies the clean-commit evidence deferred in the [history repair record](../review-history/README.md). **The numerical milestone remains incomplete.** The history regression is repaired, but the finite-result and output-order failures remain.

## Results

The unchanged committed compiler checks ran against a fresh source-bound release wheel. The pointwise suite retains the strict tolerances, nonfinite checks, signed-zero assertions, persistent wrappers and no-original/helper-replay tests. The failing finite regression is not an expected-failure marker or an infrastructure skip.

| Check | Measured result |
| --- | --- |
| Python pointwise suite, H100 | 300 total: 291 passed, 8 skipped, **1 failed** |
| Required two-device checks | 8 passed, no skips |
| Release native pointwise/ownership checks | 48 passed |
| Conversion-failure ownership check | 1 passed |
| Portable checks, each CPython 3.10–3.14 | 145 passed, 155 hardware skips |

Portable versions are 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware skips are not GPU passes. The native-extension source check and all installed wheel member comparisons passed. The suite retained 1,959 materialized native/reference result captures and their generated CUDA/PTX, including 1,120 nonlinear history comparisons and 52 fresh/persistent one-input comparisons.

The size-1 regression remains: `p=x*y; q=p-x; r=p.sin().sin().sin().sin().sin().sin().sin(); return (q,r)` produces native `q=1024.0` versus reference `1192.0928955078125` for `x=1e10`, `y=1.0000001192092896`. Sizes 13 and 257 pass this test.

The unchanged retained order diagnostic was also rerun from this clean commit. Its admitted 2,117-node, 8,211-instruction bodies differ only in return order, with identical native Graphs. At shape `(2,)` with the same finite input values:

| Return order | Native q | Default PyTorch q |
| --- | ---: | ---: |
| `(q,r)` | 1024.0 | 1024.0 |
| `(r,q)` | 1024.0 | 1192.0928955078125 |

Native execution reuses the same executor object and one executable cache entry. Both diagnostic processes exit normally; that does **not** make the second numerical result a pass. Default reference output traversal changes its locality ordering and subsequent numerical partition. The missing order-aware execution plan remains an implementation blocker; this evidence step changes no compiler behavior.

## Provenance and retention

[Measurements](measurements.json.gz) record clean before/after status, tracked source hashes, timestamped commands, cache state, wheel/import identities, GPU snapshots and runtime versions. Every tracked file remained byte-identical during all measured runs. The isolated runtime probe executed native compilation before importing PyTorch and verified aliases, structured results and the ordinary default reference.

Wheel SHA256: `55d7aceaaf9aadfa791e6e2450b3d9c00f33a41be660bd45ada7b42463550524`.

The locked offline release build used a fresh Cargo target and initially absent reference caches inside this worktree. Ordinary runs used `CUDA_VISIBLE_DEVICES=0`; required device tests used only `0,1`. Captures record Rust 1.92.0, Maturin 1.15.0, PyTorch 2.13.0+cu130, H100/driver 580.82.07, NVRTC 13.0 and CUDA runtime 13000. PATH nvcc reports 12.6.85; NVRTC compiled native kernels. The runtime diagnostic reused the report-local suite caches; the order probe used separate, initially absent compiler-cache directories. These are numerical checks, not performance measurements.

The [raw capture](raw-captures.tar.gz), [retention manifest](raw-retention-manifest.json.gz) and [verification](verification.log) preserve successful and failed outputs, CUDA/PTX, generated reference code, commands and provenance. Byte-verified wheel, committed-source, compiler-cache and nested-diagnostic archives remain under `target/default-compile-eval/structured-outputs-postcommit-69a73844/`, covered by Burner's canonical report-root observer. No cleanup or external write was performed; no external archival path is claimed. Previous reports and every prior measured artifact remain unchanged.

Only new evidence and accompanying compiler-guide links change. Implementation, tests, dependencies, benchmark harnesses, evaluation definitions and managed progress artifacts are unchanged. No official scoring corpus or unrelated repository suite was rerun. This capture does not approve the branch or resolve the numerical blockers.
