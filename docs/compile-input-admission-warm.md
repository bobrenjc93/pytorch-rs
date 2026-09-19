# Default compile admission evidence

The [compiler guide](compile-pointwise-jit.md#recompilation-and-reset) owns the
current behavior and scoped publication guarantees. The [input tests](../tests/test_compile_pointwise_structured_inputs.py)
and [cache tests](../tests/test_compile_pointwise_warm_overhead.py) cover those
contracts. This index separates author diagnostics from historical measurements;
none establishes general CUDA parity or replaces qualification.

## Nested snapshot author diagnostic (2026-09-19)

H is clean commit `0d4ed07976368a8f79b9c8d5766f59f053e46fcd`. N is the
subsequent tested precommit [frontend](../python/torch_rs/_compile_pointwise.py),
which reuses exact immutable tuples, consumes dictionary pair snapshots directly,
and admits leaves within the existing walker. Binding resolution and cache
publication are unchanged. An H whole-call profile preceded this edit; its
instrumented costs are separate from ordinary samples.

The predeclared four balanced rounds used seven independent histories, two setup
traversals, five warmups and 17 samples in fresh processes and compiler caches.
Each traversal made four ordinary public calls across two shapes and scalar
values, using four separately allocated input states reused across traversals.
Results and unchanged inputs were checked outside timing, including current input
aliases and fresh computed outputs. Identical CUDA barriers bracketed each whole
traversal; per-call latency divides that elapsed time by four. All 12 legs and
all samples are retained. H/N results matched exactly and matched ordinary default
`torch.compile`/Inductor under the existing numerical/metadata/IEEE-zero checks.
Ratios divide H elapsed time by N elapsed time; values above one favor N.

| History | Round 1 | Round 2 | Round 3 | Round 4 |
| --- | ---: | ---: | ---: | ---: |
| flat_one | 0.988 | 0.980 | 0.984 | 0.994 |
| flat_two | 1.048 | 1.069 | 1.025 | 0.993 |
| tuple_two | 1.187 | 1.062 | 1.018 | 1.032 |
| list_two | 0.938 | 0.991 | 1.002 | 1.025 |
| dict_two | 0.902 | 1.043 | 1.018 | 0.966 |
| mixed_two | 1.001 | 1.019 | 1.018 | 0.989 |
| deep_one | 1.038 | 0.976 | 1.001 | 0.786 |
| Geometric mean | 1.011 | 1.019 | 1.009 | 0.966 |

The combined ratio is 1.001 (about 0.1%); this does not establish a general
warm-call benefit. Round 4 regressed overall, including a 0.786 deep-nesting
ratio. All nested histories remained slower than default Inductor. Every slowdown
is retained; no tuning or additional timing run followed this result.

The focused suite ran 112 tests: 110 succeeded and two explicit two-device
checks skipped.
The suite includes real GPU0 public calls, current/fresh/offset/unused/repeated
inputs, nested scalar and broadcast histories, old output retention, eviction,
reset and scoped failure recovery. It does not establish arbitrary Python
allocation-failure atomicity. The three new portable tests check snapshot
consistency, rejection order and tuple/dictionary depth bounds.

The dedicated scratch virtualenv used CPython 3.12.14, PyTorch 2.13.0+cu130,
H100 GPU0 (`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`), driver 580.82.07,
`CUDA_VISIBLE_DEVICES=0`, one host thread and the existing gpu/cpu-heavy leases.
Release wheels were built locked/offline with Rust 1.92.0. The timed
barrier used libcudart 13.0; kernels reported NVRTC 13.0. A separate untimed
post-capture public call observed the native NVRTC provider at CUDA 13.0.88;
this is not a per-leg library trace. Installed nvcc was 12.6.85.
The native extension bytes are identical between H and N. Receipts record
`optimize=0`, `isolated=0`, actual commands/timestamps/exit codes and import paths.
An early snapshot/install/diagnosis sequence ran before the H wheel was ready;
those failures produced no timings and remain recorded alongside the successful
retry. No production or timing-plan changes followed the comparison.

Capture location at author time: `target/compile-nested-input-warm-binding/`.
`plan.json`, `paired-outcomes.json`, `comparison.json`, per-leg reports,
`build-H.json`, `build-N.json`, source snapshots and command receipts provide
reproduction and identity; `author-evidence.tar.gz` and `retention-manifest.json`
package them for Burner retention. This ignored scratch is not a tracked/public
download or permanent storage. The archive contains tested production/test/build
snapshots and final documentation, not the dependency environments or a complete
checkout. Documentation was finalized after testing; no clean-commit capture is
claimed for N.

```text
N frontend SHA-256: cb81b0b1e253c32f8541376a805ce18587e94b9cff52a717587c677f642e829f
Comparison SHA-256: 958c076e24a34f252cd865af6c72cf10ca4fd71a2c840697babc5fe6a7d3ca60
```

## Historical evidence and custody

The [historical archive and member manifest](diagnostics/compile-input-admission/README.md)
losslessly preserve all 55 original admission diagnostic files, including both
full JSON reports and every failure, plus H's original warm-call note. Its
extraction instructions restore the original evidence links. Historical paths,
source identities and measurements remain unchanged; none measures N.

Q (`3779fd0e6ac6fea2e0216d5c8c944423873c9023`) was measured as precommit R
against C (`9b640c20dc7bee602ef997180eea6cce72d86e77`). Four rounds measured
synchronized whole-public-call traversals with two setup traversals, five warmups,
17 samples, one thread, separate processes/caches, alternating C/R order and
ordinary default Inductor comparisons. All 12 legs passed. C/R ratios were:

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

The combined ratio was 1.075; thread-CPU round ratios were 1.062, 1.043, 1.063
and 1.134. Literal unary regressed in round 4, which also had greater variability.
These are narrow historical author measurements without a confidence interval,
not clean-commit qualification, general compile parity or measurements of the
later publication repairs. Full scope, reference latencies, failures and source
identities remain in the archived note and operator-host Q capture.

G (`8ee0c29458dcf7b057581830d50344f4c66fd996`) recorded 49 focused successes;
its additional group had 13 GPU-path successes, two metadata-only successes and
two explicit two-device skips. These are source/log-supported counts, not
per-test device traces. Its archive holds the tested production/test snapshot
against F; documentation was finalized afterward. No later repair was timed.

Burner removed the historical worktree and original scratch directories.
The Q and G copies below were verified at this author step. They are
**operator-host artifacts**, not tracked/public downloads or promises of permanent
storage; adjacent manifests/receipts carry detailed identities.

```text
Q: /tmp/burner-approved-default-compile.UA2M6I/compile-input-admission-warm-author-cli.9qETRD/evidence/author-evidence.tar.gz
SHA-256: f667dfa3cf48e377f899a8af3678f3f026115a4abfc70392d8d5d71730b1cb2b
G: /tmp/burner-approved-default-compile.UA2M6I/compile-input-admission-warm-capacity-safe-staging-cli.DwzjJq/evidence/capacity-evidence.tar.gz
SHA-256: 68c0fee38f7054facb3aab208e9ef1d84060859278f32be7b546b76953dd8dd3
```
