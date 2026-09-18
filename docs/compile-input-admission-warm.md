# Default compile warm-call author experiment (2026-09-18)

This historical author experiment measured the production source later committed
as Q (`3779fd0e6ac6fea2e0216d5c8c944423873c9023`). Its
[frontend change](../python/torch_rs/_compile_pointwise.py) grouped guard owners:
all 38 expected/missing method identities retain their order, with one current
namespace read per class owner on every invocation; prepared-plan cleanup runs
only upon executor eviction in the existing locked, successful publication.
Admission coalescing already existed in C and is not a new change here.

C is `9b640c20dc7bee602ef997180eea6cce72d86e77`, based on
`60202557b4f110d07777f585e804ab5f55e1ff7b`. R names Q's precommit author
source, identified by snapshots and hashes below. It does not measure the
subsequent publication-recovery correction described at the end of this note. These are development
measurements, not clean-commit qualification or a replacement for C's recorded
negative evaluation. No frozen evaluator was invoked or changed.

An exact-C whole-call diagnosis preceded the edit. Its seven ordinary warm-call
medians were 34–47 µs. Separate cProfile captures observed 38 mappingproxy lookups
and one preparation-publication call per invocation; their instrumented times
are not ordinary latency samples. The predeclared comparison then used all seven
unchanged `warm-dispatch-gpu-v2` histories: literal arithmetic, tensor arithmetic,
broadcast, repeated alias, shape revisit, promoted capture and eight boolean entries.

Each of four rounds used separate processes for C, R and ordinary default
PyTorch `torch.compile`, alternating C/R order. Every process used fresh caches,
two setup traversals, five warmups and 17 samples, one host thread, identical
CUDA barriers around whole public-call traversals, and materialized output checks
outside timing. All 12 legs passed; no round or slow result was excluded.
Ratios below divide C latency by R latency; values above one favor R.

| History | Round 1 | Round 2 | Round 3 | Round 4 |
| --- | ---: | ---: | ---: | ---: |
| Literal unary | 1.059 | 1.013 | 1.031 | 0.977 |
| Tensor arithmetic | 1.117 | 1.031 | 1.084 | 1.221 |
| Broadcast add | 1.050 | 1.040 | 1.046 | 1.030 |
| Repeated alias | 1.041 | 1.042 | 1.051 | 1.035 |
| Shape revisit | 1.051 | 1.057 | 1.080 | 1.368 |
| Promoted capture | 1.052 | 1.038 | 1.071 | 1.314 |
| Eight boolean entries | 1.088 | 1.056 | 1.062 | 1.068 |
| Geometric mean | 1.065 | 1.040 | 1.061 | 1.136 |

The combined diagnostic ratio is 1.075 (about 7% lower elapsed time); thread-CPU
round ratios are 1.062, 1.043, 1.063 and 1.134. This supports a narrow full-call
improvement in this capture. Literal unary regressed in round 4, and that round
also shows larger variability elsewhere. There is no confidence interval,
general CUDA speedup claim, new score or continuation approval. Raw reference
latencies and all samples remain in `comparison.json` and the per-leg reports;
all C/R outputs matched exactly and matched default Inductor under unchanged
metadata, IEEE-zero and numerical checks.

GPU0 was H100 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07,
with `CUDA_VISIBLE_DEVICES=0` and existing canonical gpu/cpu-heavy leases.
The local environment used CPython 3.12.14, PyTorch 2.13.0+cu130, libcudart 13.0,
NVRTC 13.0, Rust 1.92.0 and release/locked/offline wheel builds. Installed nvcc
was 12.6.85; these pointwise kernels compile through NVRTC. A separately labeled,
untimed post-capture loader check identifies the native NVRTC provider. C and R
wheels contain the identical native extension; their packaged Python differs.

The historical targeted checks passed 62 tests; two second-GPU tests were skipped.
Coverage includes all owner identities and pre-admission order, no-scan warm
calls, eviction/survivor order and charges, oversized plans, no publication on admission/preparation/run/reconstruction failure,
reset, and real CUDA fresh/offset/unused/duplicate/scalar/empty/broadcast inputs.
New CUDA coverage crosses the default executor limit through unused inputs.
The first new-test fixture hit the logical limit prematurely; it was corrected.
A broadcast import-path failure was corrected by rerunning that module with the
repository test directory on PYTHONPATH. Both failures and a corrected untimed
provider-helper failure are retained. Production was unchanged through testing.

The prior archive is retained as an **operator-host artifact**, not a tracked or
publicly downloadable repository file, at:

```text
/tmp/burner-approved-default-compile.UA2M6I/compile-input-admission-warm-author-cli.9qETRD/evidence/author-evidence.tar.gz
```

Its manifest and receipt are adjacent. This location is not a promise of permanent
storage. Inside the archive, `commands.md`, `plan.json`, `paired-receipt.json`,
`comparison.json`, `audit.json` and `retention-manifest.json` index all rounds,
failures, snapshots and wheel identities; `source-tests` holds the historical
final tests. No historical artifact or measurement was regenerated for this repair.

SHA-256 identities (full manifests are in the archive):

```text
C frontend: 079c48945c7e0d460c57807732da1eeb7d9db46c8943c184d79df53b344de20f
R frontend: 64e8dee17d344b5ed5c1c25f17951276208f17bafc3ba1146b7b71d9f4bf117a
Final test: 14d18b93b68cd328a0e6b299cc296175e35e601873723c03cb2247cbc731fe32
Archive: f667dfa3cf48e377f899a8af3678f3f026115a4abfc70392d8d5d71730b1cb2b
Manifest: 8b7272431b957b008594216feaadd5013d18cd565392efd328b59bd4f562aa74
```

## Historical publication-recovery correction F: author validation

F (`8256303d7526a2d3176a520ad9a36751fa3cb764`) added failure-only clearing
of prepared retention and charges after executor reinsertion failure. Its staging
order still allowed repeated failing misses to accumulate executors; the capacity
correction below supersedes that behavior. The following validation and hashes
remain historical evidence of F's source, not of the capacity correction.

Precommit validation of F passed 49 focused checks plus 15
additional CUDA checks; two second-GPU checks were skipped. The
[regressions](../tests/test_compile_pointwise_warm_overhead.py) inject both faults,
recover through a different survivor or newest key, check ownership/order/charges
and reset, alternate retained executors without scans, and exercise empty-plan
eviction. Real default public calls vary inputs and recheck retained old outputs
after eviction/reset. The first new recency assertion incorrectly required the
bookkeeping tuple itself to retain identity; the corrected test checks the
preparation identity and charges. Both fixture snapshots and the failed run remain.

Historical F operator-host artifacts are in worktree-local
`target/compile-input-admission-publication-repair/`, including
`repair-evidence.tar.gz`. These ignored files are not public repository downloads;
retention beyond this host is not promised. Each command has an execution-time
argv/cwd/environment/start/end/return-code receipt and separate stdout/stderr.
`tested-source-final.json`, `binding.json` and `audit.json` bind the final source,
wheel and imports. The isolated interpreter used `-B`, `sys.flags.optimize=0`,
CPython 3.12.14 and PyTorch 2.13.0+cu130; GPU0 UUID matches the historical device,
with libcudart/NVRTC 13.0 provider hashes recorded. Build and dependency commands
used locked inputs. No new timing or frozen evaluation was run. Historical archive
hashes above were verified read-only. Archive contents passed a round-trip audit.

```text
Repair frontend: b20c9bfbe7ea22bc3902a50f99c9955ca1574c551928506aca7e461742d4cbe7
Final test: 7ee6eceb6ed93c25639ed9244346fcd0d4e8f183cfa6e209d0284897c6a60949
Tested source manifest: ce669272cebc8e6d4a16f7da56feecb6a03208b1725b39f80fe3a4d625704fcc
Repair wheel: 975ebde6a3a7cb6ca6fa96938c709acd4b5c4d29f48e7f8df92b0e5760548790
Repair archive: 62b99bf8a9b21b13d807970f2f3be5924e606af18d3c71036ebe72797e3ff38c
Repair archive manifest: b1cf69df26dce057ecc6ae476f03057ccb1a721381b8cecc873ad892998a7843
```

## Capacity-safe staging correction: author validation

The successor to F stages eviction bookkeeping after successful reconstruction
and before any cache publication, only on a capacity-increasing executor miss.
Starting with a bounded cache, repeated failures at that staging boundary leave
all cache contents, owner identities, recency orders and charges unchanged.
Retained hits avoid staging/scans. F's failure-only preparation clearing on failed
executor reinsertion remains. These specific boundaries do not imply universal
allocation-failure atomicity; the guide's earlier phase-limited guarantee remains.

Fresh validation passed all 49 focused checks and 15 additional real CUDA checks;
two checks requiring a second GPU were skipped. One staging fault stays enabled
across five distinct unused-input-shape misses in each recovery order. Every
failure checks the limit, exact charge, owners/order and original exception identity
before retained-hit recovery, successful later eviction and reset. The existing
public CUDA history changes used-input values and rechecks retained old outputs.
All commands passed on their first attempt. No timing or scoring was run; neither
Q's timings nor F's validation above measure this successor.

New operator-host evidence is in
`target/compile-input-admission-capacity-repair/capacity-evidence.tar.gz`, with
adjacent `retention-manifest.json`. These ignored host files are not tracked or
publicly downloadable repository files and do not promise permanent storage.
The archive contains the final source, execution-time command/return/timestamp
receipts, stdout/stderr, wheel/import bindings and GPU0/provider metadata.
`tested-source.json`, `binding.json` and `audit.json` identify the tested uncommitted
source against F. Validation used the isolated CPython 3.12.14 interpreter with
`-B` and explicit `sys.flags.optimize=0`, PyTorch 2.13.0+cu130 and CUDA/NVRTC 13.0.
Historical scratch and evidence remain unchanged.

SHA-256 identities for this correction:

```text
Frontend: fff728e9222a948dd1fb752a527380f789cd74577ad27ad3eb742d31fd564624
Final test: 2d895e0eb606cbd596bffdb1318df8b17e0f2d541f9308c936faa84d046e960b
Tested source manifest: c4d5dca63aeb092f537198f6ebb35473d5956d1a1c69c92502ddbb42bd58f3f6
Wheel: bd5f7bf16d41a28f6fd788f7bf61438b577de5a543b36d26776cdddd7c1ae18e
Archive: 68c0fee38f7054facb3aab208e9ef1d84060859278f32be7b546b76953dd8dd3
Archive manifest: 283ab4d776bb022185df816b95adc543e1102fbb76c5bdebd4b47fab44facd7c
```
