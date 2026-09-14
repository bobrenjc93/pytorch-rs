# Structured-output compiler evidence

The latest measured production revision in this record set is
[`0f68f8bde413605572ebe1aa878b474dbb90a5d9`](postcommit-0f68f8bd/README.md),
against main `30a3b504ef4d43bf2958998cc39545996cc09970`.
Its source-bound capture records 303 passing H100 pointwise tests with eight
explicit two-device skips, eight separately passing two-device checks, and
63 passing native pointwise/ownership checks. The report also records the portable
interpreter checks, 720 duplicate-product comparisons and 16 independent/reused
producer comparisons. Skips are not GPU passes.

The [signed-product use repair](review-signed-uses/README.md) records subsequent
uncommitted development: 304 passing H100 pointwise tests, eight separate
two-device checks and 64 native checks. Its 1,050-comparison regression and
original failures are retained. Clean-commit qualification awaits Burner’s commit.

These are bounded correctness captures against ordinary stock `torch.compile`,
not general Inductor equivalence, a performance result, or qualification of
later commits. The [usage guide](../../compile-pointwise-jit.md#bounded-nested-results)
and [numerical contract](../../compile-pointwise-numerics.md) own supported behavior.

## Numerical history

Records are chronological. “Clean” identifies a recorded source/build capture,
not an assertion that every numerical check passed. Development records measured
uncommitted changes; later successful checks do not erase their earlier failures.

| Record | Source and outcome |
| --- | --- |
| [Initial clean capture](postcommit-36954660/README.md) | `36954660`: initial structured-output checks; later review found returned-product FMA errors. |
| [Returned-product repair](review-returned-products.md) | Development after `597d1f3e`; original cancellation, infinity-sign and zero-sign failures retained. |
| [Clean returned-product repair](postcommit-90de52a9/README.md) | `90de52a9`: scoped repair checks passed; later review found sibling-product failures. |
| [Sibling-product repair](review-sibling-products.md) | Development repair and passing controls; nonlinear partition-dependent zero-sign failures remained. |
| [Clean nonlinear blocker](postcommit-4554bbe8/README.md) | `4554bbe8`: four of six nonlinear reproducer calls still failed. |
| [Nonlinear-region repair](review-nonlinear-regions.md) | Development current-size planning; expanded probes retained compilation-history failures. |
| [Clean history blocker](postcommit-c504a49b/README.md) | `c504a49b`: six original reproducer calls passed, but 24 of 1,120 expanded history calls failed. |
| [History repair](review-history/README.md) | Development after `c13d660d`; sampled history repaired, finite-result and observable-order blockers remained. |
| [Clean finite/order blocker](postcommit-69a73844/README.md) | `69a73844`: one permanent pointwise test failed; finite-result and output-order discrepancies remained. |
| [Realization/order repair](review-realization/README.md) | Development after `0f7e6296`; order-aware regions and rounded imports, not clean-commit qualification. |
| [Clean realization repair](postcommit-523d51d7/README.md) | `523d51d7`: scoped checks passed; later review found duplicate-observable-root FMA errors. |
| [Duplicate-root repair](review-duplicate-roots/README.md) | Development after `245e91f0`; pre-fix failure and 720 passing repair comparisons retained separately. |
| [Clean duplicate-root capture](postcommit-232cc734/README.md) | `232cc734`: source-bound duplicate-root repair checks; evidence added at `c45b41fd`. |
| [Clean producer-identity capture](postcommit-0f68f8bd/README.md) | `0f68f8bd`: independent and reused producers preserve their distinct realization semantics; source-bound numerical and portable checks passed. |
| [Signed-product use repair](review-signed-uses/README.md) | Development after `cba212e3`: sign rewrites retain external uses for FMA selection; original finite/overflow failures and passing repairs retained. |

Each report links its measurements, raw capture, manifests and available
verification records. Measured production revisions are distinct from the
commits adding evidence: for example, clean `523d51d7` was recorded in
`245e91f0`, and clean `232cc734` in `c45b41fd`.

## Provenance and retention limits

The `0f68f8bd` capture includes its full raw report tree in the candidate's
checked-in archive, including wheel, compiler/test sources and compiler caches;
only rebuildable Cargo output is excluded. Its initial portable setup failures
are retained alongside successful retries. No external observer coverage is
assumed, and this new capture does not repair the historical custody gap below.

Source checks are observations, not continuous monitoring. For clean `523d51d7`
and `232cc734`, full tracked-file fingerprints matched the baseline after each
command; the source-subset fingerprints matched before and after. Preserve the
individual records' build, runtime and source-drift limitations.

The linked historical payloads remain checked in. Paths under
`target/default-compile-eval/structured-outputs-*` describe disposable capture
workspaces, not durable download locations. The frozen Burner collector covered
its fixed-evaluator `run-*` reports, not these custom diagnostic roots; older
claims implying automatic coverage of these roots were incorrect. Separate
operator retention later preserved selected manifest-bound historical artifacts
outside the worktree. That on-host custody is neither an off-host backup nor
numerical qualification, and is distinct from the original capture's actions.

The separate review7 `independent-review-c45b41f` root was removed during
canonical cleanup before its planned external copy. The surviving audit retains
a file inventory and hashes plus selected driver/log/output data, not all
original raw bytes. Its observed 80 held-out comparisons were fresh-wrapper, single-call
size-13 cases, not a compilation-history suite. The clean `232cc734` bundle does
not fill that custody gap, and partial audit data is not a replacement archive.

The fixed evaluation definitions, denominators, numerical tolerances and
unsupported-category credit remain unchanged. These records grant no review,
evaluation or full-qualification waiver.

## Artifact transport

On 2026-09-14, a separate CPU-only trial produced these lossless `.xz`
transports from the original files at
`c45b41fd64b19e03e91585897d78db55e8f53931`. Full decoded TAR/JSON/log streams
matched by `cmp`, byte count and SHA256; TAR members and metadata were not
extracted or rewritten. The compressed-container bytes are different.

| Current download | Original bytes | XZ bytes | Saved payload bytes |
| --- | ---: | ---: | ---: |
| [review-history/raw-captures.tar.xz](review-history/raw-captures.tar.xz) | 26,307,060 | 15,373,228 | 10,933,832 |
| [postcommit-c504a49b/raw-retention-manifest.json.xz](postcommit-c504a49b/raw-retention-manifest.json.xz) | 1,666,350 | 116,252 | 1,550,098 |
| [postcommit-c504a49b/verification.log.xz](postcommit-c504a49b/verification.log.xz) | 187,684 | 2,144 | 185,540 |

The three payloads total 15,491,624 bytes instead of 28,161,094, saving
12,669,470 bytes. This changes storage only: no scientific capture was rerun,
no historical failure was removed, and the review7 custody gap above remains.

Historical names and container hashes in measurements, manifests and logs
still identify the original files, not the new transports. Those records are
not rewritten. Original containers remain recoverable from the Git blobs below
and separately retained operator copies; current downloads retain their complete
decoded content rather than requiring history-only retrieval.

```text
review-history/raw-captures.tar.gz -> review-history/raw-captures.tar.xz
  original Git blob: 13eb894b0b3c4a826495180a0dfd5237d8ec2b8f
  original SHA256:  d6cee3eab29472fd7abe5835409afbbac64d31a86c0e54bc275ee6260479bb79
  XZ SHA256:        56bf7ffaed46c76c7920469aa6f41b97a2d26609f519cc162e7f5229f601d6dd
  decoded bytes:    184944640
  decoded SHA256:   c6bad0afdadaffc46725dfa9f9a56707d12de1b6a9ebfc042497b6a3d656ce37

postcommit-c504a49b/raw-retention-manifest.json -> postcommit-c504a49b/raw-retention-manifest.json.xz
  original Git blob: 3a46ca18c5157bc4ab929589e0fc101132f99b7d
  original SHA256:  404f892b6f5c691bb76f0002293144247e7de034b8537d19a864c5ff61233552
  XZ SHA256:        27f8aa81ae143de9d20368615bd5d999b321d78f6181c96b2ba4938e6b4d320d
  decoded bytes:    1666350
  decoded SHA256:   404f892b6f5c691bb76f0002293144247e7de034b8537d19a864c5ff61233552

postcommit-c504a49b/verification.log -> postcommit-c504a49b/verification.log.xz
  original Git blob: deaa9dda00716015d9e5f14a25530d9a10ab2f76
  original SHA256:  b525097b1caad3a6b549d8cab681b13801243973d4e3e709666f1f9a49cb27f5
  XZ SHA256:        ceb40e66f5cc0ad69693e06f4a59e2f18deac78b82b2db9efdc92eb3b7ad81ef
  decoded bytes:    187684
  decoded SHA256:   b525097b1caad3a6b549d8cab681b13801243973d4e3e709666f1f9a49cb27f5
```

Verify a downloaded container against its XZ SHA256 above. From this directory,
the following Bash commands stream-decode without overwriting files or extracting
TAR members. Require each pipeline to exit zero, then compare its output hash
with the decoded SHA256 above; a partial stream's hash is not verification:

```bash
set -o pipefail
xz --decompress --stdout -- review-history/raw-captures.tar.xz | sha256sum
xz --decompress --stdout -- postcommit-c504a49b/raw-retention-manifest.json.xz | sha256sum
xz --decompress --stdout -- postcommit-c504a49b/verification.log.xz | sha256sum
```

The retained trial result SHA256 is
`84cb4140ce29001996160bb433e4cf88a6c61ce87d3af5279f63bcdc38e95416`.
It records 23 normal exit-zero commands and matching full streams; the operator
separately observed actual outer exit zero. This is transport-integrity evidence,
not compiler qualification or an evaluation-score claim.
