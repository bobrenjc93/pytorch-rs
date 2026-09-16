# Grouped live method guards (non-scoring)

This repair reuses only the two guard hunks from held commit `9009c8ea`.
Each call fetches a fresh live namespace from each of the two original native
classes, preserving all 38 ordered identity checks, expected objects, missing
entries and errors before metadata or execution. The public compile docstring
now describes direct/VM selection and exact executable identity reuse. No Rust,
numerical, cache, supported-surface or evaluator changes are included.

The authoritative rejection of `6f0123a2cc521af9fab3dc19f1c256b28684f0db`
reported CUDA **32.90710217090202 versus 34**, with unchanged 20/56 support.
The previous ownership repair did not clear that gate. Native slowdowns and
reference movement do not establish a guard-specific cause. The held GELU
branch's earlier grouping had mixed results and also failed qualification;
its implementation is reused, not its performance credit.

## Bounded diagnosis

Before editing production, [method-guard-block.py](../method-guard-block.py)
extracted the actual old and reviewed grouped blocks from their source commits.
Both arms ran on the same actual native class objects, interpreter and installed
extension, whose bytes were checked before and after. Separate instrumented
controls counted **38 → 2 namespace fetches**, with **38 comparisons** in both
arms and identical flattened owner/name/expected identities. Substitute-owner
mutation controls in the tests establish structure only.

Uninstrumented block timing used before/grouped/grouped/before order, five
1,000-iteration warmup batches per leg, then 17 samples of 20,000 calls per leg.
All 68 elapsed samples are retained. Median nanoseconds per block were:

| Leg | Before | Grouped |
| --- | ---: | ---: |
| First pair | 2524.116 | 1735.043 |
| Reversed pair | 2535.609 | 1770.600 |

These are block timings including the common wrapper and outer loop, not
ordinary compile-call latency or a prediction of the canonical score. They
show removed work without assigning the historical loss to these guards.
The capture records HEAD `6f0123a`, untracked diagnostic/test files, exact
extracted source, source hashes, package/interpreter bindings and iterations.
It is development evidence, not a clean capture of the new implementation.

The [archive](development-evidence.tar.gz) and [manifest](manifest.json) retain
raw diagnosis, commands, source and wheel bindings, checks and failure logs.
The rebuilt test wheel has different native extension bytes from the earlier
wheel used by **both** block arms; both hashes are retained. A supplementary
check's failed cross-build equality assumption is preserved. No cross-wheel
latency comparison is made, and Rust sources remain unchanged.

## Validation and limits

The final H100 pointwise suite ran 396 tests with 9 explicit multi-device skips;
all remaining tests passed. All 9 native identity/compiler-failure tests, 290
Rust tests, formatting, Clippy with warnings denied and 33 frozen consumer
controls passed. The portable guard suite ran 10 tests with one hardware skip.
Wheel/source, extracted AST, historical fixture and docstring/link checks passed.

The first H100 attempt had five subtest setup errors because the new test used
unsupported CUDA `ones` creation. The final test uses supported CPU creation
followed by `.to('cuda:0')`; the failed log and original test are retained.
Review also found a shallow-checkout dependency on historical Git objects in a
new test. A minimal provenance-bound local fixture removed that dependency;
its AST was checked against the original, and final tests read only local files.

Tests retain cold/warm admission, all 38 binding mutations and failure prefixes,
hostile objects, live proxies, multiple retained preparations, receipts,
byte accounting and recovery. GPU work used only GPU0, H100 UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, with worktree-local CUDA runtime
and NVRTC pins. No multi-GPU support is claimed.

## Clean post-commit evidence

Burner committed this repair as `4e291f5272b4b872ddc1704398b088b1917affc8`.
The [new clean comparison](../postcommit-4e291f5/README.md) completed all eight
ordinary-default B/C/reference legs and passed the unchanged verifier. Every raw
sample, selected-invocation control and cold/churn/small-case regression is
retained. Previous complete and failed archives remain unchanged. Independent
review, full qualification and exact-head CI still decide merge eligibility;
neither development checks nor block timing clear that gate.
