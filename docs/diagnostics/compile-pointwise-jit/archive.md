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
runs remain unscored; clean `0c836a49` measurements are linked from the current
evidence index.

Temporary worktree paths in these captures may no longer contain the original
build or raw observations after cleanup. Checked-in reports, manifests, generated
code, and compressed logs retain what was captured. No failed measurement was
overwritten or promoted as a score.
