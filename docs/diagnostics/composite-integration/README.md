# Composite integration checks

Author validation used the integrated production code at
`e3a3ba3bd11f90378000b780b29389e5b1195c59`. The logs and receipts below remain
pinned to that completed run. The implementation, tests, and measurement
harnesses are unchanged between that revision and Burner's committed
implementation `519f375c1f11cf66e3e83f3158c632cd171ec7a3`.

The subsequent [CUDA](../../cuda-neg-validation.md) and
[text](../../text-collation-diagnostics.md) evidence refresh freshly built and
measured clean `519f375c`; its [build receipt](../composite-cuda-neg/build-record.json)
and [focused evidence checks](../composite-cuda-neg/evidence-checks.log) describe
that capture. It did not rerun the author full suites or baseline measurements
below. Their provenance and results have not been reattributed to the later run.

The [check receipt](checks-record.json) retains actual commands, directories,
environment, timestamps, exit statuses, source fingerprints, and log hashes.
The [installed-wheel receipt](installed-provenance.json) verifies all 59 Python
sources against this checkout and records the separate stripped extension hash.

These reports replace the prior dirty-merge diagnostics. The retained full-suite
run is CPython 3.12.14; CPython 3.14.7 has focused reference/GPU checks and
baseline comparisons below. No full 3.14 pass is claimed.

| Check | Result | Raw evidence |
| --- | --- | --- |
| `rust-tests` | 365 passed | [rust-tests](rust-tests.log) |
| `rust-default-tests` | 354 passed | [rust-default-tests](rust-default-tests.log) |
| `clippy` | passed | [clippy](clippy.log) |
| `fmt` | passed | [fmt](fmt.log) |
| `focused` | 75 tests; OK | [focused](focused.log) |
| `gpu-focused` | 69 tests; OK (skipped=5) | [gpu-focused](gpu-focused.log) |
| `two-device` | 5 tests; OK | [two-device](two-device.log) |
| `python-full` | 5304 tests; FAILED (failures=2, skipped=10) | [python-full](python-full.log) |
| `compile-evaluation` | passed | [compile-evaluation](compile-evaluation.log); [raw report](compile-evaluation.json) |
| `python314-focused` | 25 tests; OK (skipped=1) | [python314-focused](python314-focused.log) |
| `historical-setup` | passed | [historical-setup](historical-setup.log) |
| `historical-tests` | 6 tests; OK | [historical-tests](historical-tests.log) |
| `quickstart` | passed | [quickstart](quickstart.log) |

The unchanged full compile evaluator reports all 38 reference-eligible cases
passing. The unchanged CUDA math evaluator retains six fixed cases and all
three original seeds: native negation and same-shape addition pass, with all
four unsupported cases retained. Neither result establishes universal compiler
or CUDA parity or a repository-wide performance improvement.

The [provenance audit](provenance-audit.log) checks source/evaluator/matrix and
receipt/report/log hashes, local worker interpreter/package/extension/runtime
paths, immutable successful CUDA inputs, all failure slots, and the 100 text
cells' balanced orders, samples, medians, MAD, and geometric means.
[Documentation links](links.log), the README assertion-only quickstart,
historical setup disclosure, and `git diff --check` pass.

All four historical workload JSON files and result sections remain unchanged.
The historical source tree, archive and lockfile hashes match
`e2f40ff16f8aba5216bc51699b56571e7e82a3e4`; its later same-code setup capture
remains explicitly distinct from the original compute measurements. No historical
compute workload was rerun. Evaluator definitions, weights, denominators,
calibration, and Burner-managed progress/history remain unchanged. The complete
source PR ancestry is preserved; no branch, commit, or delivery action was made.

## Independently reproduced baseline failures

A fresh release build of the verified `git archive` of base
`31310e8fd1a1ff4f8dadcca93566dba70d115241` runs inside this worktree's
`target/integration/baseline`, with a separate empty build target. The
[baseline receipt](baseline-provenance.json) and [build log](baseline-build.log)
record its archive/extension hashes, command, compiler and timestamps. All 864
archived tracked files remain unchanged. The buffer and factory reference tests
and factory implementation are identical between base and candidate.

- CPython 3.12 reproduces the two noncanonical boolean-buffer assertions on both
  source revisions, at hash seeds 0 and 1: [base, seed 0](baseline-reference312-seed0.log),
  [candidate, seed 0](candidate-reference312-seed0.log),
  [base, seed 1](baseline-reference312-seed1.log),
  [candidate, seed 1](candidate-reference312-seed1.log).
- CPython 3.14 passes those same buffer tests on both revisions and both seeds:
  [base, seed 0](baseline-reference314-seed0.log),
  [candidate, seed 0](candidate-reference314-seed0.log),
  [base, seed 1](baseline-reference314-seed1.log),
  [candidate, seed 1](candidate-reference314-seed1.log).
- Factory-keyword ordering reproduces three assertions on both revisions at the
  previously recorded seed 6 on Python 3.12
  ([base](baseline-factory312-seed6.log), [candidate](candidate-factory312-seed6.log))
  and seed 10 on Python 3.14
  ([base](baseline-factory314-seed10.log), [candidate](candidate-factory314-seed10.log)).
  Python 3.14 seed 6 passes on both
  ([base](baseline-factory314-seed6.log), [candidate](candidate-factory314-seed6.log)).
  Set iteration and keyword insertion order depend on the interpreter/hash seed.

The full suite uses its ambient hash seed; none of its assertions were suppressed
or weakened. Separate deterministic comparisons use the previously recorded
failure seeds to distinguish baseline behavior from integration regressions.
Reproduce from either source root with its matching `PYTHONPATH=python`:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  tests.test_tensor_buffer_reference -v
PYTHONHASHSEED=6 PYTHONPATH=python .venv/bin/python -m unittest \
  tests.test_nn_factory_kwargs_reference -v
```

For the archived baseline, run in `target/integration/baseline` using the absolute
path to the composite interpreter. For Python 3.14 use its local interpreter and
hash seed 10 for the factory reproduction. Exact commands for every captured run
are retained in the check receipt.
