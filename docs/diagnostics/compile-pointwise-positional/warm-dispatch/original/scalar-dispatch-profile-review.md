# Pre-execution review: bounded scalar dispatch diagnostic

Verdict: PASS for the described unscored, stdlib-only diagnostic. This is a
source/design review, not a diagnostic result or production qualification.

## Reviewed identity

- Design: `scalar-dispatch-profile-design.md`, SHA-256
  `abd3a0a8242a8ec0d2c673958663935d6fd4016fd2fcdc5f6e61161eb401c795`.
- Script: `profile-scalar-dispatch.py`, SHA-256
  `bb708311bdd4e2c59430e794658cc58782253b8e8a34ea77e447d713810fda39`.
- Main: `a281503f3391bbd98a0ab6de2ff8f0c0a55a12d4`.
- Candidate: `885264b5319d76165e6be8d6df405450b3830c47`.
- Runtime sources are read from pinned Git objects; no checkout is required.

The final recheck read both revised files completely. Changes resolve the two
earlier findings and add the suggested profiling sanity checks; no additional
actionable finding remains in this bounded review.

## Evidence and limits checked

- Metadata now matches `885264b5:src/python.rs:9669`: shape, strides,
  requires-grad flag, `torch.float32`, device and storage offset.
- Production frontend function bodies are not rewritten. Relative imports are
  removed and an explicitly synthetic package/Tensor/function-mode/native shell
  is supplied. The actual cache class is extracted, but direct construction
  bypasses the WeakSet registry and global reset lifecycle. That omission and
  the absence of full public-entrypoint evidence are now explicit.
- The intended Python 3.12.12 `dataclasses.py` was inspected at
  `/home/bobren/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/dataclasses.py`,
  SHA-256 `d242aea5fcf6408b1c1f622442f88f68b9526ce1f8bd2890d74a144677c427d9`.
  Its class processing supplies frontend module globals to generated
  initialization, comparison and hash functions. The profiler's globals filter
  therefore includes those warm generated methods. No code-object-set rewrite
  is required. All compiled calls and positive candidate Graph hash counts must
  now be observed, preventing an empty/misfiltered profile from passing silently.
- The seven predeclared common histories remain unchanged. Setup warms each
  wrapper before collection. Checks require no warm analysis, lowering, native
  compilation or original-body calls; exact expected mock-boundary call counts;
  stable cache cardinalities; and the source-derived default limit of eight.
- All 128 profiled calls and all four unprofiled 1024-call batches per source
  remain predeclared. The complete main/candidate, candidate/main,
  candidate/main, main/candidate order and every batch are reported, with no
  best-result selection. Profiling is disabled during timing.
- Source hashes are checked before loading and source bytes reread afterward.
  Framework imports are forbidden. Timings explicitly include harness updates,
  assertions and mock bookkeeping; they cannot establish native admission,
  numerical correctness, lifecycle behavior or GPU latency.

The retained CUDA result, 11.360180035407296 (stored 11.4) versus main 12, remains
blocking. This diagnostic neither replaces that result nor authorizes a
production change, scoring rerun or merge.

No diagnostic execution, framework import, build, GPU work or production/state
change was performed by this reviewer. Only this new scratch review was written.
