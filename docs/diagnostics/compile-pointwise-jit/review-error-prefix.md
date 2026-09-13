# Compile rejection prefix compatibility

Native pointwise rejection again starts with `torch.compile():`, retaining
`native CUDA pointwise:` and the detailed reason and documentation link.
The unchanged top-level compile compatibility test reproduced the regression:
49 tests ran, with one failure. After rebuilding and installing the local
release wheel, all 49 top-level compile tests and nine pointwise admission
tests passed. No tests, evaluator definitions or numerical behavior changed.

The [validation bundle](review-error-prefix.json.gz) preserves the original
failure, build and passing logs, commands and after-fix source/wheel/native
hashes. These are unscored development checks based on `b1e0dd4` plus the
one-line prefix repair. Temporary build paths may disappear after cleanup.

The existing clean [campaign capture](postcommit-74602c/postcommit.json) remains
pinned to `74602c07` and predates this repair. Fresh campaign capture requires
Burner's next clean implementation commit; no previous report is relabeled.
