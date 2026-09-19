# Default compile admission evidence

The [compiler guide](compile-pointwise-jit.md#bounded-positional-input-trees) owns
current semantics; the [input](../tests/test_compile_pointwise_structured_inputs.py)
and [cache](../tests/test_compile_pointwise_warm_overhead.py) tests cover their
contracts. Diagnostics below are unscored author evidence, not qualification or
general CUDA parity. Historical captures do not measure subsequent changes.

## Warm source resolution (2026-09-19)

Baseline N is clean `c46e6a0ea53fd3b924ba64161e6c7724d895eb01`. The tested
precommit successor projects retained source identities from each call's admitted
snapshots. One binding conversion serves both projection and full ordered
materialization for lowering/private `resolve()`. Captures remain eager. This
avoids descendant construction on retained hits without retaining topology plans;
direct path reads can revisit prefixes and scan dictionary-key tuples.
The cited historical H profile guided this choice; its overlapping instrumented
costs are not ordinary latencies or causal proof.

The fixed comparison reused the previous diagnostic machinery and all seven
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
| flat_one | 1.015 | 0.994 | 0.978 | 1.013 |
| flat_two | 0.990 | 0.964 | 0.954 | 1.077 |
| tuple_two | 1.123 | 1.065 | 1.095 | 1.137 |
| list_two | 1.130 | 1.104 | 1.044 | 1.117 |
| dict_two | 1.084 | 1.085 | 1.097 | 1.111 |
| mixed_two | 1.025 | 1.127 | 1.110 | 1.106 |
| deep_one | 1.086 | 1.053 | 1.101 | 1.146 |
| Geometric mean | 1.064 | 1.054 | 1.052 | 1.100 |

Combined ratio: 1.068, about 6.3% lower elapsed time. Every nested history improved
in every round, but flat controls regressed in some rounds and most nested cases
remained slower than default Inductor. Thread-CPU round ratios were 1.067, 1.063,
1.055 and 1.092. This supports a narrow whole-call improvement, without a
confidence interval or broad parity claim. No timing reroll or scoring followed.

The focused portable/GPU suite passed 181 tests with three explicit two-device
skips. Two subsequently added tests also passed: older-specialization projection
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

Capture location at author time: `target/compile-warm-source-resolution/`.
`plan.json`, `paired-outcomes.json`, `comparison.json`, all per-leg reports,
command receipts/stdout/stderr, `build-N.json`, `build-R.json`, tested source
snapshots and `audit.json` bind reproduction to actual source/wheel/import paths.
`author-evidence-final.tar.gz` and `retention-final-manifest.json` package that evidence for
Burner retention. This ignored scratch is not a public download or permanent
storage. Snapshots cover tested code/tests/build inputs and final documentation,
not dependency environments or a complete checkout. No clean-commit measurement
is claimed for the successor.

```text
Successor frontend SHA-256: bc621e50219577170095acea129e4c42b5c7e4ad01eaa9ccf4ba94f1374cecfc
Comparison SHA-256: 47185e288245068e20ebc11aaa797e5a71803f72660dd795e699d74d25d05652
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
