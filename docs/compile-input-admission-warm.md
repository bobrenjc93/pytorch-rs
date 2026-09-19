# Default compile admission evidence

The [compiler guide](compile-pointwise-jit.md#bounded-positional-input-trees) owns
current semantics; the [input](../tests/test_compile_pointwise_structured_inputs.py)
and [cache](../tests/test_compile_pointwise_warm_overhead.py) tests cover their
contracts. Diagnostics below are unscored author evidence, not qualification or
general CUDA parity. Historical captures do not measure subsequent changes.

## Warm source resolution (2026-09-19)

Baseline N is `c46e6a0ea53fd3b924ba64161e6c7724d895eb01`; candidate R is clean
commit `fd38f8241158198da1e07ccc2f95ec6af4867cd0`. R projects retained source
identities from each call's admitted snapshots. One binding conversion serves both
projection and full ordered materialization for lowering/private `resolve()`.
Captures remain eager. This
avoids descendant construction on retained hits without retaining topology plans;
direct path reads can revisit prefixes and scan dictionary-key tuples.
The cited historical H profile guided this choice; its overlapping instrumented
costs are not ordinary latencies or causal proof.

The clean-commit recapture required by review reused the unchanged fixed
comparison scripts, order, workloads and checks, including all seven
independent histories below, with four balanced rounds and ordinary default
`torch.compile`/Inductor. Each fresh process/cache ran two setup traversals, five
warmups and 17 samples with one host thread. A traversal made four public calls
across two shapes/scalar values and separately allocated input states reused
across traversals. Matched CUDA barriers bracketed the whole traversal; output,
metadata, IEEE-zero, current-owner and freshness checks ran outside timing.
All 12 legs passed; N/successor outputs matched exactly and passed the existing
reference checks. All raw setup/warm/sample wall and thread-CPU times are retained.
Ratios divide N latency by successor latency; above one favors the successor.

| History | Round 1 | Round 2 | Round 3 | Round 4 |
| --- | ---: | ---: | ---: | ---: |
| flat_one | 0.997 | 1.070 | 0.983 | 1.090 |
| flat_two | 0.996 | 1.012 | 0.966 | 1.063 |
| tuple_two | 1.106 | 1.094 | 1.061 | 1.162 |
| list_two | 1.065 | 1.080 | 1.059 | 1.163 |
| dict_two | 1.156 | 1.133 | 1.022 | 1.247 |
| mixed_two | 1.119 | 1.123 | 1.107 | 1.217 |
| deep_one | 1.176 | 1.104 | 1.080 | 1.214 |
| Geometric mean | 1.086 | 1.087 | 1.039 | 1.163 |

Combined ratio: 1.093, about 8.5% lower elapsed time. Both flat controls regressed
in rounds 1 and 3. Every nested history improved against N in every round, but
remained slower than default Inductor. The per-history reference comparisons and
all raw samples remain in the reports. Thread-CPU round ratios were 1.081, 1.089,
1.048 and 1.162. This bounded diagnostic has no confidence interval and establishes
neither general CUDA parity nor qualification. No scoring or favorable retry
followed.

The earlier source-bound portable/GPU suite passed 181 tests with three explicit
two-device skips. Two subsequently added tests also passed: older-specialization projection
after a newer shape miss, and real GPU0 reordered/offset/fresh-input history with
retained outputs across reset. Earlier test snapshots remain in the evidence.
Admission, guards, promotion, aliases, native current-input checks, bounded cache
retention and scoped failure recovery were exercised; arbitrary allocation-failure
atomicity is not claimed. Production stayed unchanged throughout validation.
The initial prepared tests used the existing hard-coded worktree cache at
`target/prepared-reference-caches`, outside this request's scratch. Their rerun
from `source-R` passed four tests with one two-device skip and placed caches
inside the requested scratch; both runs remain recorded.

The dedicated scratch virtualenv used CPython 3.12.14 (`optimize=0`, `isolated=0`),
PyTorch 2.13.0+cu130, H100 GPU0 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`,
driver 580.82.07 and `CUDA_VISIBLE_DEVICES=0` under gpu/cpu-heavy leases. Release
wheels were built locked/offline with Rust 1.92.0; their native extension bytes
match. Barriers used libcudart 13.0; kernels reported NVRTC 13.0. A separate untimed
post-capture public call recorded the loaded NVRTC 13.0.88 provider; it is not a
per-leg library trace. Installed nvcc was 12.6.85.

Clean capture location: `target/compile-warm-source-resolution/committed-fd38/`.
`declaration.json` fixes the recapture before timing; `run.sh` invokes the unchanged
`paired.py` and `compare.py`. `build-N.json` and `build-R.json` record committed
source snapshots, clean checkout identity, wheel hashes and actual local paths;
per-leg reports verify installed wheel bytes before execution. Commands, return
codes, timestamps, stdout/stderr, interpreter flags, provider/device observations,
all samples and the final binding audit are retained. The candidate wheel was
built from the clean checkout before this documentation-only update. No target
implementation, tests, workload or numerical tolerance changed for recapture.

`committed-evidence.tar.gz` and `retention-manifest.json` package the new evidence
for Burner retention, with an explicit member/hash manifest. They cover tested
source/build/test snapshots, wheels, scripts, receipts and reports, not dependency
environments, a complete checkout or this subsequently updated note. Ignored
scratch is neither a public download nor a promise of permanent storage.

The original precommit comparison is preserved unchanged in the parent scratch's
`author-evidence-final.tar.gz` (SHA-256
`f39f4d494b37c1292c345f001028c18eb47f64c2ef011c1e06bf5ebdb14b1787`).
Its round ratios 1.064, 1.054, 1.052 and 1.100 (combined 1.068, approximately
6.3% lower elapsed time), every per-history result and all original failures remain
recoverable there. Those preliminary figures do not satisfy clean-commit
provenance; the table above reports the new capture rather than pooling runs.

```text
Measured commit: fd38f8241158198da1e07ccc2f95ec6af4867cd0
Candidate wheel SHA-256: 457108527c1b4865d8df27a07196bbe10eeaf0107597fb1f5667ff6a841f68cc
Comparison SHA-256: 79cd3204978c792eda9c58fe19599456ccdfa01d1d5278305651f8c74de041a7
Archive SHA-256: aa6c693936e66304545bc8455bd542eb32be0c71f0ffc128a06ef9a8f9f82ad3
```

## Historical evidence and custody

Historical captures remain unchanged, including every round, failure, slowdown,
source identity and original note. Their timings do not measure this successor.

- **H/N snapshot simplification:** N's prior precommit capture against H
  `0d4ed07976368a8f79b9c8d5766f59f053e46fcd` had round geomeans 1.011, 1.019,
  1.009 and 0.966, combined 1.001: flat, not a demonstrated general benefit.
  Deep nesting regressed to 0.786 in round 4; every nested history was slower
  than Inductor. All 12 legs passed; tests had 110 successes/two two-device skips.
  The complete tables, synchronized traversal scope, initial pre-wheel failures
  and raw samples are in the capture recorded at
  `target/compile-nested-input-warm-binding/author-evidence.tar.gz`, SHA-256
  `4deeabb656f5651c1df04bafc14642b8ddbdbcbbbfaa13faf076b1dcaa1d5407`.
- **Q/C guard and publication experiment:** historical precommit Q
  (`3779fd0e6ac6fea2e0216d5c8c944423873c9023`) versus C
  (`9b640c20dc7bee602ef997180eea6cce72d86e77`) had round geomeans 1.065, 1.040,
  1.061 and 1.136, combined 1.075. Literal unary regressed in round 4 (0.977),
  which had greater variability. These synchronized whole-traversal measurements
  were not clean-commit qualification or general parity; later publication
  repairs were not timed. Every per-history ratio and the full measurement scope
  remain in the original note in the [unchanged historical archive/index](diagnostics/compile-input-admission/README.md),
  alongside all 55 original admission files, both full JSON reports and failures.
- **G correctness capture:** 49 focused successes, plus 13 GPU-path successes,
  two metadata-only successes and two explicit two-device skips. These are
  source/log-supported counts, not per-test device traces. Its archive holds
  the tested production/test snapshot against F, with documentation finalized
  afterward.

Burner removed the earlier Q/G worktrees and original scratch directories.
Recorded operator copies below and the ignored N capture above are not tracked
or publicly downloadable repository files; permanent storage is not promised.
Adjacent manifests/receipts retain detailed identities. The repository archive
has its own member/hash manifest and extraction instructions with working links.

```text
Q: /tmp/burner-approved-default-compile.UA2M6I/compile-input-admission-warm-author-cli.9qETRD/evidence/author-evidence.tar.gz
SHA-256: f667dfa3cf48e377f899a8af3678f3f026115a4abfc70392d8d5d71730b1cb2b
G: /tmp/burner-approved-default-compile.UA2M6I/compile-input-admission-warm-capacity-safe-staging-cli.DwzjJq/evidence/capacity-evidence.tar.gz
SHA-256: 68c0fee38f7054facb3aab208e9ef1d84060859278f32be7b546b76953dd8dd3
```
