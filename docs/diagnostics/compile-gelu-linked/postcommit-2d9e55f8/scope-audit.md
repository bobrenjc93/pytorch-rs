# Post-commit evidence scope audit

Clean implementation commit inspected:
2d9e55f820f49efd817f3d52b7cbc9c69894d3ad.
Read-only git status was clean. The complete diff against main contains the
bounded native/frontend GELU change, tests, license metadata and staged evidence
previously reviewed; no evaluator scripts, corpus, weights, tolerances or
benchmark helpers changed. This report is not a new implementation approval.

## Required current-code captures

1. Run the existing unchanged public_probe.py from the clean commit for all
   five legs: native eager, reference eager, native default compiled, reference
   default compiled and native eager with deliberately absent NVRTC. Retain
   the complete declared matrices, changed inputs, post-call storage records,
   output retention/context checks, timings/controls, observed exits and actual
   source/import/build identities. Compare modes separately using its unchanged
   policy. This directly fulfills the explicitly deferred final current-code
   capture; prior dirty public-attempt-002 does not fulfill it.
2. Complete the unchanged fixed public-default-compile-v2 coverage and CUDA
   performance gate with its full denominator and both CUDA orders. The
   repository supports --metric both as one report covering both metrics;
   diagnostic mode is insufficient. Preserve unsupported/incorrect cells and
   all slow measurements. Retain actual report, raw worker records, logs,
   wheel/setup/build/source/import identities and process outcome.

No additional required measurement was identified beyond these captures.
Canonical independent review/full qualification/exact-head CI remain separate
Burner delivery gates, not a reason to rerun unrelated suites in this limited
post-commit evidence step. The fixed gate's fresh wheel/setup already provides
new build provenance; notice and source inclusion can be checked against that
wheel without introducing another build or changing a harness.

## Preserve historical evidence

Do not regenerate or relabel the earlier private prerequisite, offline provider
selection/generation pairs, failed stock-Erf/NVCC attempts, prior capture-harness
failures, staged development tests, or unrelated baselines. Their purpose is to
record actual intermediate states and the conditional implementation sequence.
They were explicitly staged records before this post-commit step, and should
remain bound to their original source/build identities. Likewise retain the
old dirty public/fixed diagnostic captures with their honest limitations; add
fresh current-code evidence rather than silently replacing historical results.
No unchanged historical workload needs a new run merely because its original
working path later disappears.

## Compact fixed-worker retention

A suitable lossless transport for large .json.gz workers is .json.xz containing
exactly the original decompressed byte stream. Verify the original gzip hash
against the evaluator report first, then record original path/size/SHA256,
uncompressed byte count/SHA256 and converted xz byte count/SHA256 in an
accompanying transport manifest. Independently check byte equality after both
decompressions. Do not parse/reserialize JSON, remove tensors, edit provenance,
change report hashes, or claim the xz bytes are the original gzip artifact.
The manifest bridges the unchanged evaluator identity to the alternate transport.

All current captures must remain rooted inside this worktree. Do not write
tracked evidence documentation until measurements have finished; doing so
would turn later legs into dirty captures. After successful captures, changes
must be evidence/documentation only. The measured code commit can then remain
2d9e55f8 underneath Burner's later artifact-only commit, as the task permits.

This audit ran no compiler, build, test, framework or GPU workload. At the time
of inspection the author's fixed gate was still building/running; no result or
successful qualification is inferred.

Assembled 2026-09-16T07:51:54.164623+00:00.
