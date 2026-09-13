# Compile rejection prefix compatibility

Native pointwise rejection again starts with `torch.compile():`, retaining
`native CUDA pointwise:` and the detailed reason and documentation link.
The unchanged top-level compile compatibility test reproduced the regression:
49 tests ran, with one failure. After rebuilding and installing the local
release wheel, all 49 top-level compile tests and nine pointwise admission
tests passed. That implementation repair changed no tests, evaluator definitions
or numerical behavior.

The [validation bundle](review-error-prefix.json.gz) preserves the original
failure, build and passing logs, commands and after-fix source/wheel/native
hashes. These are unscored development checks based on `b1e0dd4` plus the
one-line prefix repair. Temporary build paths may disappear after cleanup.

The preceding clean [campaign capture](postcommit-74602c/postcommit.json) remains
pinned to `74602c07` and predates this repair. The operator subsequently added
the requested prefix assertion in `08e37fe5`, retaining all no-callback checks
and the unchanged top-level compatibility test. Substituting the old prefix in
memory now fails the new assertion; the committed helper passes.

The [fresh capture](postcommit-08e37f/postcommit.json) measures clean `08e37fe5`
with both unchanged campaign commands and generated-code evidence. It also
records 84 pointwise tests (two explicit multi-device skips), 90 top-level/backend
checks and 16 documentation/archive checks. No previous report is relabeled.
The original failing [CI run](https://github.com/bobrenjc93/pytorch-rs/actions/runs/34726354726)
remains linked as historical evidence; these local checks do not replace fresh
independent review, CI or merge qualification.
