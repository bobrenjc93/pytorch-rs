# Pointwise compiler evidence archive

For the supported compiler contract, current measurements, and reproduction
commands, start at the [current evidence index](README.md). This archive records
earlier implementations and the repair history. Reports, logs, and original
failures remain byte-for-byte unchanged at their existing paths.

## Initial implementation

The clean `5b93c983` capture predates review fixes and does not measure the current
candidate:

- [Coverage](candidate-coverage.json.gz), [CUDA performance](candidate-cuda-perf.json.gz),
  [receipt](postcommit.json), and [logs](postcommit-logs.json.gz).
- [Generated CUDA](kernel.cu), [PTX](kernel.ptx.gz),
  [provenance](provenance.json), and [source manifest](source-manifest.json.gz).
- The [original development diagnostic](candidate-diagnostic.json.gz) is
  explicitly unscored dirty-source evidence, including original observations.
- The author [inventory](validation.json), [logs](logs.json.gz), and
  [IEEE probes](initial-ieee-probes.json.gz) retain initial validation, failures,
  and repairs at their original source identities.

## First review repair

Clean `60abd863` predates the second numerical review fixes. Its weighted
coverage/performance scores were 6/12; its timings do not measure later code.

- [Coverage](postcommit-60abd/candidate-coverage.json.gz),
  [CUDA performance](postcommit-60abd/candidate-cuda-perf.json.gz),
  [receipt](postcommit-60abd/postcommit.json), [logs](postcommit-60abd/postcommit-logs.json.gz),
  and [codegen provenance](postcommit-60abd/provenance.json).
- [Development validation](review-fixes.md) and [bundle](review-fixes.json.gz)
  preserve regressions and original failures. The exhaustive compiler selection
  covered 69 files and 805 cases.

## Second review repair

Clean `53c10058` predates the third review fixes. Its 6/12 measurements retain
their original code and build identities.

- [Coverage](postcommit-53c100/candidate-coverage.json.gz),
  [CUDA performance](postcommit-53c100/candidate-cuda-perf.json.gz),
  [receipt](postcommit-53c100/postcommit.json), [logs](postcommit-53c100/postcommit-logs.json.gz),
  and [codegen provenance](postcommit-53c100/provenance.json).
- [Development validation](review-round2.md) and [bundle](review-round2.json.gz)
  preserve the original 24 failing subcases, before/after numerical probes, and
  checks at their original source identity.

## Third review repair

[Development validation](review-round3.md) and its [bundle](review-round3.json.gz)
preserve 30 original failing subcases, intermediate failures, numerical probes,
and final checks. The 124 implementation-source and six regression-module hashes
match implementation `dbd1a0f6`. The record includes 406 default / 433 bindings
Rust tests, 96 backend/entrypoint checks, Clippy, documentation, and separate
two-device checks. These dirty-source development runs are unscored. The clean
`dbd1a0f6` capture predates the composed constant-tensor fix:

- [Coverage](postcommit-dbd1a0/candidate-coverage.json.gz),
  [CUDA performance](postcommit-dbd1a0/candidate-cuda-perf.json.gz),
  [receipt](postcommit-dbd1a0/postcommit.json), [logs](postcommit-dbd1a0/postcommit-logs.json.gz),
  and [codegen provenance](postcommit-dbd1a0/provenance.json).

## Composed constant-tensor repair

[Development validation](review-constant-tensors.md) and its
[bundle](review-constant-tensors.json.gz) preserve the 48 originally failing
subcases, numerical probes, generated code and final checks. These dirty-source
runs remain unscored. Clean `0c836a49` predates the precision and zero-origin
repair and retains its original measured identity:

- [Coverage](postcommit-0c836a/candidate-coverage.json.gz),
  [CUDA performance](postcommit-0c836a/candidate-cuda-perf.json.gz),
  [receipt](postcommit-0c836a/postcommit.json), [logs](postcommit-0c836a/postcommit-logs.json.gz),
  and [codegen provenance](postcommit-0c836a/provenance.json).

## Constant-folding precision and zero-origin repair

[Development validation](review-folding.md) and its [bundle](review-folding.json.gz)
preserve original and intermediate failures, output-bit probes, generated code,
and build/test commands. Its 124 source hashes and seven regression-module
hashes match `40a57b36`; these development checks remain unscored. Clean
`40a57b36` predates the shared-product, runtime-sine and warm-scalar repair and
retains its original measurements:

- [Coverage](postcommit-40a57b/candidate-coverage.json.gz),
  [CUDA performance](postcommit-40a57b/candidate-cuda-perf.json.gz),
  [receipt](postcommit-40a57b/postcommit.json), [logs](postcommit-40a57b/postcommit-logs.json.gz),
  and [codegen provenance](postcommit-40a57b/provenance.json).

## Shared products, runtime sine and warm captured scalars

[Development validation](review-boundaries.md) and its [bundle](review-boundaries.json.gz)
preserve the original numerical failures, intermediate failures, generated
CUDA/PTX and command logs. The 124 implementation and ten regression-module
hashes match `f745c45c`; the development measurements remain explicitly unscored.
The current evidence index links the subsequent clean capture.

Temporary worktree paths in these captures may no longer contain the original
build or raw observations after cleanup. Checked-in reports, manifests, generated
code, and compressed logs retain what was captured. No failed measurement was
overwritten or promoted as a score.

## Globals keys, static zero guards and constant unary precision

Clean `f745c45c` predates this repair. Its original captures remain unchanged:

- [Coverage](postcommit-f745c4/candidate-coverage.json.gz),
  [CUDA performance](postcommit-f745c4/candidate-cuda-perf.json.gz),
  [receipt](postcommit-f745c4/postcommit.json), [logs](postcommit-f745c4/postcommit-logs.json.gz),
  and [codegen provenance](postcommit-f745c4/provenance.json).

[Development validation](review-static-guards.md) and its
[bundle](review-static-guards.json.gz) retain the original admission, signed-zero
and constant-sine failures, intermediate checks and generated-code evidence.
All 124 implementation and thirteen regression-module hashes match `5fc75c41`;
these development runs remain unscored. The current evidence index links the
subsequent clean capture.
