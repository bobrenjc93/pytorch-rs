# Pointwise compiler evidence archive

Start at the [current evidence index](README.md) for the supported contract,
latest measurements and reproduction commands. This page is the historical
record, not another compiler guide or a current performance claim.

Seven superseded clean captures are consolidated in the checked-in
[history.tar.gz](history.tar.gz). The [manifest](history-manifest.json) lists
each original relative path, byte count and SHA-256, plus the archive hash.
Every member is byte-for-byte identical to its file in source commit
`ed6ad9eaaea17ee500f5464183c00984115e7fb3`, verified before consolidation.
The original gzip payloads, failures, measured commits and provenance are
unchanged. No evidence was moved to an external service or dropped.

## Read an archived capture

From the repository root, verify the archive without extracting or executing
any member (Python standard library only):

```bash
python3 docs/diagnostics/compile-pointwise-jit/verify_archive.py
```

Use a prefix from the table below to read a report directly, for example:

```bash
tar -xOf docs/diagnostics/compile-pointwise-jit/history.tar.gz \
  postcommit-5fc75c/candidate-coverage.json.gz | gzip -dc
```

Each prefix contains the same eight original files: `candidate-coverage.json.gz`,
`candidate-cuda-perf.json.gz`, `postcommit.json`, `postcommit-logs.json.gz`,
`kernel.cu`, `kernel.ptx.gz`, `provenance.json` and `source-manifest.json.gz`.
The archive preserves these logical paths; they are no longer separate checkout
directories. Git history also retains their original locations.

| Measured clean source | Archive prefix | Development record for that repair |
| --- | --- | --- |
| `60abd863` | `postcommit-60abd/` | [First review](review-fixes.md), [bundle](review-fixes.json.gz) |
| `53c10058` | `postcommit-53c100/` | [Second review](review-round2.md), [bundle](review-round2.json.gz) |
| `dbd1a0f6` | `postcommit-dbd1a0/` | [Third review](review-round3.md), [bundle](review-round3.json.gz) |
| `0c836a49` | `postcommit-0c836a/` | [Composed constants](review-constant-tensors.md), [bundle](review-constant-tensors.json.gz) |
| `40a57b36` | `postcommit-40a57b/` | [Precision and zero origins](review-folding.md), [bundle](review-folding.json.gz) |
| `f745c45c` | `postcommit-f745c4/` | [Shared products, sine and scalar histories](review-boundaries.md), [bundle](review-boundaries.json.gz) |
| `5fc75c41` | `postcommit-5fc75c/` | [Globals, zero guards and constant unary precision](review-static-guards.md), [bundle](review-static-guards.json.gz) |

These historical captures scored 6/12 on the frozen weighted coverage/CUDA
performance gates. Their timings belong only to their recorded source and build
identities. The development records preserve original and intermediate failing
cases, numerical probes and repair checks; dirty-source runs remain unscored.

The later scalar-zero contraction, nonfinite binding and offset-cache repair is
documented in [review-zero-boundaries.md](review-zero-boundaries.md) and its
[bundle](review-zero-boundaries.json.gz). Its clean `42959e14`
[coverage](postcommit-42959e/candidate-coverage.json.gz),
[performance](postcommit-42959e/candidate-cuda-perf.json.gz),
[receipt](postcommit-42959e/postcommit.json) and
[logs](postcommit-42959e/postcommit-logs.json.gz) remain directly accessible.
These reports predate the rejection-message clarification in `74602c07`;
the current evidence index links the fresh capture of that revision.

## Initial implementation

The initial clean `5b93c983` capture predates every review repair and remains
directly accessible for comparison:

- [Coverage](candidate-coverage.json.gz), [CUDA performance](candidate-cuda-perf.json.gz),
  [receipt](postcommit.json), and [logs](postcommit-logs.json.gz).
- [Generated CUDA](kernel.cu), [PTX](kernel.ptx.gz),
  [provenance](provenance.json), and [source manifest](source-manifest.json.gz).
- The [original development diagnostic](candidate-diagnostic.json.gz),
  [author inventory](validation.json), [logs](logs.json.gz), and
  [IEEE probes](initial-ieee-probes.json.gz) retain initial observations,
  failures and repairs at their original source identities. These development
  diagnostics are not scores.

## Retention limits

Temporary worktree paths recorded inside captures may no longer contain the
original build or raw observations after cleanup. Checked-in reports, manifests,
generated code and compressed logs are the durable record. Consolidation does
not recreate missing temporary artifacts, relabel a failed measurement, or
replace independent review and merge qualification.
