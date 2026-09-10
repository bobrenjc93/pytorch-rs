# PR #1930 historical boolean-buffer evidence

These 26 original `target/` files support the
[boolean conversion report](../../../boolean-buffer-semantics.md) from
PR #1930, source commit `ed078e1a2dec6ad7c01f6c74c36cce52165d6fc3`, against
base `03075466b077c69ed1f7a4f047ebb33004d3306b`. They are historical evidence;
their original source/build/import paths deliberately identify the source job,
even after that environment is removed. They do not measure or award performance
credit to the combined checkout. [Combined validation](../../../boolean-stack-integration.md)
is reported separately.

Retention verified SHA256
`88c5f642589e15b66467bc74a94d244eda9bccdbf2e0e9580a63026a41b0737a` of the
read-only local snapshot `boolean-buffer-evidence.tar.gz`, checked all 30
members were unique regular files with safe relative paths, and copied the 26
`target/` members without changing their bytes. The four omitted source members
(`src/python.rs`, both tensor-buffer test files, and the report) were verified
byte-for-byte against Git at `ed078e1a`; Git already preserves them.
No environment, wheel, native binary, or cache is included.
The baseline traceback log contains two original trailing spaces, and the Rust
test log ends with a blank line. Local `.gitattributes` exceptions cover only
those respective whitespace rules and files, preserving their checksummed bytes.

- [Driver](target/measure_bool.py): historical data; do not execute archived
  scripts against their old absolute paths.
- Raw measurements: `target/baseline-*-timing.json` retains the initial runs;
  `target/before-*-conversion.json` and `target/after-*-conversion.json` retain
  the before/after matrix used by the report. `.venv` means host GCC 3.12.13;
  `managed312` means Clang 3.12.12; `managed314` means Clang 3.14.5.
- `target/baseline-*-test.log`, `target/fixed-*-test.log`, dependency and build
  logs, and Rust/fmt/clippy logs retain the original validation/setup history.
- [Post-commit verification](target/post-commit-evidence/verification.json)
  and [build log](target/post-commit-evidence/build.log) bind the repaired bytes
  to `ed078e1a`; the original wheel is intentionally excluded.
- [SHA256SUMS](SHA256SUMS) covers all 26 payload files. Run `sha256sum -c
  SHA256SUMS` from this directory, or the portable audit from the repository:
  `python -m unittest tests.test_boolean_buffer_evidence`.

| Source interpreter | Initial baseline | Report before | Report after |
| --- | --- | --- | --- |
| GCC 3.12.13 | [JSON](target/baseline-.venv-timing.json) | [JSON](target/before-.venv-conversion.json) | [JSON](target/after-.venv-conversion.json) |
| Clang 3.12.12 | [JSON](target/baseline-managed312-timing.json) | [JSON](target/before-managed312-conversion.json) | [JSON](target/after-managed312-conversion.json) |
| Clang 3.14.5 | [JSON](target/baseline-managed314-timing.json) | [JSON](target/before-managed314-conversion.json) | [JSON](target/after-managed314-conversion.json) |

Every raw cell remains, including the 17 incorrect GCC baseline cells per
baseline/before run and all slow repaired cells. Incorrect cells earn no
performance credit. All 81 repaired cells matched the reference. Dense
noncanonical Clang 3.12 contiguous conversion slowed from 2.49 ms to 33.83 ms
(13.61x); reversed/strided cases slowed 4.11x/3.95x. Canonical-buffer geometric
mean improvements were 1.19x (GCC 3.12), 1.16x (Clang 3.12), and 3.41x
(Clang 3.14). These bounded, fixed-seed CPU observations do not establish broad
parity. The original report retains sample ranges, layouts, setup costs,
and the Clang 3.14 dense-layout regressions.
