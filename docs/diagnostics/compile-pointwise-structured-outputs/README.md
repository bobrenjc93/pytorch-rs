# Structured-output compiler evidence

The [clean `6ec4f2dd` capture](postcommit-6ec4f2dd.json) records 59 structured/prepared
checks, nine separate two-H100 checks, 73 native checks, and 164 portable passes
per CPython 3.10–3.14 interpreter. The two hardware skips in the focused run passed
in the separate device run; each portable run skipped 168 hardware cases.
The report identifies the release wheel, imports, runtime and 3,789 numerical
captures with native CUDA/PTX and plans. Raw artifacts remain in the canonical
worktree-local report root recorded there, with a verified archive hash; they are
not another checked-in payload or an off-host backup. This capture produces no
performance score and does not replace independent review or full qualification.
Formatting and both CI Clippy configurations also passed. The initial incomplete
portable setup and its successful retry are retained in the report.
The [prior `9f09bc2a` capture](postcommit-9f09bc2a.json) remains unchanged and
bound to that earlier source revision.

## Historical captures

These are historical, source-bound correctness records for the bounded CUDA
pointwise compiler. The latest measured source here is
[`c238ca63`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-c238ca63/README.md);
the evidence was committed at `1315dce10e04481f3571850c35845d642b191225`.
They do not qualify later code, establish general Inductor equivalence, or
demonstrate CUDA performance parity.

For current behavior, start with the [compiler guide](../../compile-pointwise-jit.md#bounded-nested-results)
and [numerical contract](../../compile-pointwise-numerics.md).
The [validation summary](validation-summary.json) records the measured source,
test counts and exact bundle identities. The checked-in tests and frozen
`.burner` eval definitions are unchanged by this archive cleanup.

## Numerical history

Earlier failures remain part of the record. Development captures measured
uncommitted changes; “clean” identifies a source snapshot, not a passing result.
The [full chronology](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/README.md#numerical-history) links all 16 records.

| Clean source | Outcome and later findings |
| --- | --- |
| [`36954660`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-36954660/README.md) | Initial capture; review found returned-product FMA errors. |
| [`90de52a9`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-90de52a9/README.md), [`4554bbe8`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-4554bbe8/README.md) | Returned-product repair exposed sibling/nonlinear errors; four of six nonlinear calls still failed. |
| [`c504a49b`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-c504a49b/README.md) | Original reproducers passed, but 24 of 1,120 expanded history calls failed. |
| [`69a73844`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-69a73844/README.md) | One permanent test failed; finite-result and output-order errors remained. |
| [`523d51d7`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-523d51d7/README.md) | Scoped repair passed; later review found duplicate-root FMA errors. |
| [`232cc734`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-232cc734/README.md), [`0f68f8bd`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-0f68f8bd/README.md) | Source-bound duplicate-root and producer-identity checks passed. |
| [`c238ca63`](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-c238ca63/README.md) | Signed-product use repair: 304 H100 pointwise tests, eight separate two-device checks and 64 native checks passed. |

The latest capture also records one conversion-failure check, 1,050 signed-product
comparisons, and 149 portable passes plus 163 hardware skips per CPython 3.10–3.14
interpreter. Skips are not GPU passes. See the linked report for exact runtime,
source-drift observations and limitations.

## Retrieve exact records

The [archive manifest](archive-manifest.json) inventories all 78 historical files
(301,344,873 bytes): original paths, sizes, Git blob IDs, SHA-256 hashes and
immutable-commit download URLs. Every file was independently downloaded and
byte-verified on 2026-09-15 before this cleanup. A separate on-host recovery copy
was retained outside disposable worktrees; it is not an off-host backup.

For example, download the latest measurements without cloning the repository:

```bash
curl --fail --location --output measurements.json.gz \
  https://raw.githubusercontent.com/bobrenjc93/pytorch-rs/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/postcommit-c238ca63/measurements.json.gz
sha256sum measurements.json.gz
# 68851aa43f18d05be88194205ab408cb17d91a98c63798c8c666ac717676c95d
```

Alternatively, retrieve a path with
`git show 1315dce10e04481f3571850c35845d642b191225:<original-path>`.
A clone missing that revision can first fetch the retained PR history with
`git fetch origin refs/pull/1997/head`. Keep the containing history reachable;
Burner's merge-commit path preserves this ancestor when the PR is merged.
This removes generated payloads from the current checkout, **not Git history
or full-clone bandwidth**. Historical document links stay within their original
tree; the [transport notes](https://github.com/bobrenjc93/pytorch-rs/blob/1315dce10e04481f3571850c35845d642b191225/docs/diagnostics/compile-pointwise-structured-outputs/README.md#artifact-transport) retain older container identities.

## Retention limits

The separate review7 `independent-review-c45b41f` directory was removed before
its planned copy. Only its inventory/hashes and selected audit data survived,
not all original raw bytes. Its 80 held-out comparisons were fresh-wrapper,
single-call size-13 cases, not a compilation-history suite. Later bundles do
not fill this gap.

The frozen Burner collector covered fixed-evaluator `run-*` reports, **not**
these custom diagnostic directories; older claims implying that coverage were
incorrect. Separate operator copies preserved selected historical artifacts,
not every lost file. Those supplemental copies are outside this manifest.
No historical failure, numerical tolerance, evaluation denominator or
qualification requirement is changed by this reorganization.
