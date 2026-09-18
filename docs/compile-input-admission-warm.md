# Default compile warm-call author experiment (2026-09-18)

This author-only revision changes `python/torch_rs/_compile_pointwise.py`:
all 38 expected/missing method identities retain their order, with one current
namespace read per class owner on every invocation; prepared-plan cleanup runs
only upon executor eviction in the existing locked, successful publication.
Admission coalescing already existed in C and is not a new change here.

C is `9b640c20dc7bee602ef997180eea6cce72d86e77`, based on
`60202557b4f110d07777f585e804ab5f55e1ff7b`. R is the uncommitted author
implementation, identified by snapshots and hashes below. These are development
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

Final targeted checks passed 62 tests; two second-GPU tests were skipped.
Coverage includes all owner identities and pre-admission order, no-scan warm
calls, eviction/survivor order and charges, oversized plans, failure atomicity,
reset, and real CUDA fresh/offset/unused/duplicate/scalar/empty/broadcast inputs.
New CUDA coverage crosses the default executor limit through unused inputs.
The first new-test fixture hit the logical limit prematurely; it was corrected.
A broadcast import-path failure was corrected by rerunning that module with the
repository test directory on PYTHONPATH. Both failures and a corrected untimed
provider-helper failure are retained. Production was unchanged through testing.

Retained evidence is in [target/compile-input-admission-warm](../target/compile-input-admission-warm/):
`commands.md` indexes actual invocations and every failure; `plan.json`,
`paired-receipt.json`, `comparison.json`, `audit.json`, and `retention-manifest.json`
index the complete raw logs, reports, snapshots and wheel identities. Final tests
are in `source-tests`; measured C/R snapshots remain immutable. The verified
`author-evidence.tar.gz` is ready for retention before scratch cleanup.

SHA-256 identities (full manifests are in the archive):

```text
C frontend: 079c48945c7e0d460c57807732da1eeb7d9db46c8943c184d79df53b344de20f
R frontend: 64e8dee17d344b5ed5c1c25f17951276208f17bafc3ba1146b7b71d9f4bf117a
Final test: 14d18b93b68cd328a0e6b299cc296175e35e601873723c03cb2247cbc731fe32
Archive: f667dfa3cf48e377f899a8af3678f3f026115a4abfc70392d8d5d71730b1cb2b
Manifest: 8b7272431b957b008594216feaadd5013d18cd565392efd328b59bd4f562aa74
```
