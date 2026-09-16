# Failed private Erf integration (historical)

The CUDA13 experiment failed 4 of 335 output comparisons (24 elements) under
the unchanged numerical policy. Production changes were removed; GELU remains
unsupported. The failure precedes the final contraction in `g*g + x*y`, where
`g=gelu(x)` and `y=-x`. At x=5, the native/reference sums differ by
3.814697265625e-6 against an allowance of 1.0001335144042969e-6.

The complete original evidence is immutable at commit
`934719a35559bc987120088fd126e8237a16727b`. This directory keeps the
[experimental patch](implementation.patch), [original probe](probe.py),
[counterexamples](failure-elements.json), [build receipt](build-and-checks.json),
[logs](checks), and [independent audit](independent-audit.md).
These measure the unapplied development patch, not current production code.

The 19 MB cache/raw archive and its large manifest have been removed from the
latest tree to reduce diagnostic bulk. [Raw references](raw-references.json)
pin their commit, paths, sizes and SHA256 digests. Restore copies inside this
worktree when needed:

```bash
mkdir -p target/gelu-history
git show 934719a35559bc987120088fd126e8237a16727b:docs/diagnostics/compile-gelu-stage1/raw-evidence.tar.gz > target/gelu-history/raw-evidence.tar.gz
git show 934719a35559bc987120088fd126e8237a16727b:docs/diagnostics/compile-gelu-stage1/raw-manifest.json > target/gelu-history/raw-manifest.json
```

The [historical checksum list](historical-SHA256SUMS.txt) describes that original
snapshot, including its old README. Original measurements and provenance are
unchanged; no paths or scores have been relabeled. The snapshot does not prove
uninterrupted reference history: `run_and_get_code` resets Dynamo per call, and
retained-output filenames overwrite earlier snapshots. Those limitations do
not invalidate the static counterexamples.

The [vendor-provider retry](../compile-gelu-vendor/README.md) stopped earlier,
at compiler input selection. Neither attempt supplies a public feature,
performance improvement, or qualification certificate.
