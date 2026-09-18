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

## Recorded correctness validation

The tested production and test files match G
(`8ee0c29458dcf7b057581830d50344f4c66fd996`). Its recorded focused log has
49 successful checks. The additional selected group has 13 successful GPU-path
methods, two successful metadata-only methods, and two explicit two-device skips.
The metadata methods inspect generated source and rejected shapes; they do not
execute GPU work. This breakdown is supported by source and logs, not independent
per-test device traces.

The [cache guide](compile-pointwise-jit.md#recompilation-and-reset) is the canonical
home for current publication guarantees; the
[regressions](../tests/test_compile_pointwise_warm_overhead.py) cover their scoped
failure boundaries. No later repair was timed. These recorded tests are not a
fresh benchmark, clean-commit qualification or general compile-parity claim.

Original G evidence remains under `target/compile-input-admission-capacity-repair/`.
The operator retained an exact copy at:

```text
/tmp/burner-approved-default-compile.UA2M6I/compile-input-admission-warm-capacity-safe-staging-cli.DwzjJq/evidence/
```

These are operator-host locations, not tracked or publicly downloadable repository
files; permanent storage is not promised. `capacity-evidence.tar.gz` and the
adjacent `retention-manifest.json` identify the retained artifacts. The archive
contains the tested production/test snapshot against parent F (`8256303d7526a2d3176a520ad9a36751fa3cb764`),
command receipts, stdout/stderr and source/wheel/import bindings. Its documentation
snapshot predates G's finalized note. `tested-source.json`, `binding.json` and
`audit.json` record those identities; earlier failures and intermediate identities
remain in retained evidence. This prose correction does not regenerate evidence.

```text
Tested-source manifest: c4d5dca63aeb092f537198f6ebb35473d5956d1a1c69c92502ddbb42bd58f3f6
Archive: 68c0fee38f7054facb3aab208e9ef1d84060859278f32be7b546b76953dd8dd3
Archive manifest: 283ab4d776bb022185df816b95adc543e1102fbb76c5bdebd4b47fab44facd7c
```
