# Bounded Python dispatch diagnosis

The actual 885264b candidate remains unqualified: dedicated CUDA score
11.360180035407296 (stored 11.4) versus main 12. Its same-source coverage run
scored 12 for CUDA. The negative result is retained; neither a favorable rerun
nor CPU evidence may substitute for qualification.

Main and an independent source reviewer agree that a small stdlib-only
phase/call-count comparison is the next useful diagnostic before any production
change. Main's old exact-metadata lookup cannot replace the corrected logical
specialization owner without reopening signed-zero/shape-history defects.

Compare immutable main a281503f with candidate 885264b, using the actual frontend
AST and actual cache class, following the existing NaN-reproducer technique.
Redirect only relative imports to isolated synthetic package, Tensor, function
mode and native boundaries. No real torch/torch_rs import, CUDA work, build,
checkout, server, state mutation, scoring, or merge. The native boundary returns
a sentinel and counts metadata/validation/compilation/launch calls: it provides
no native admission, numerical, GPU-latency or full public-entrypoint evidence.
Direct cache construction deliberately omits the WeakSet registry and global
reset lifecycle. CPU timings include harness iteration, namespace updates,
sentinel assertions and mocked native bookkeeping. These are explicit limits,
not lifecycle or native-boundary verification.

Use seven predeclared common frontend histories: literal unary arithmetic,
two-tensor arithmetic, one-stage broadcast, repeated tensor alias, shape revisit,
promoted captured float, and eight static captured-Boolean specializations.
Construct and warm each wrapper before profiling. Retain all cases and require
zero warm analysis/lowering/compilation/original-body calls, equal expected
metadata/validation/launch counts, unchanged cache cardinality and default limit.

Primary result: 128 profiled warm calls per case and source, reporting production
helper and dataclass-method call counts without wrapping or altering them.
Assert that all 128 compiled-wrapper calls are counted and candidate Graph hash
calls are visible. Native metadata uses the exact real tuple order and dtype
spelling. The intended Python 3.12.12 dataclass generator supplies frontend
module globals to generated init/equality/hash methods; the filter includes them.
Secondary diagnostic only: four unprofiled 1024-call batches in predeclared
main/candidate, candidate/main, candidate/main, main/candidate order. Report every
batch and median, not a best result; do not equate mocked CPU latency with CUDA.
Use isolated Python 3.12.12 with no site imports. Hash source objects before and
after and preserve the script/output separately from qualification receipts.

If counts and CPU timing identify redundant work, profile/validate it later on
real H100s under canonical resource ownership with a predeclared balanced plan.
Any eventual optimization should reuse immutable projection/key material inside
the existing Program/Specialization/lowering owners, preserving all guards,
newest-successful selection, success-only publication, reset and bounded recency.
No parallel same-shape fast cache, corpus/shape dispatch, weakened validation,
reference setting change or removal of the negative result is licensed here.
