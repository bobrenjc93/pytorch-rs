# Composite integration checks

These checks ran on the uncommitted integrated source identified by the
[CUDA build receipt](../composite-cuda-neg/build-record.json). They are local
integration diagnostics, not clean-final-commit attestation or overall Burner
scores. The source fingerprint and extension binding are retained with the
[CUDA capture](../../cuda-neg-validation.md) and
[text benchmark](../../text-collation-diagnostics.md). The pending merge, Git
index, and implementation commit remain Burner's responsibility under the task's
no-commit/no-outside-writes constraints. The [check receipt](checks-record.json)
records the source fingerprint, installed wheel/extension hashes, and raw log hashes.

| Check | Result | Raw evidence |
| --- | --- | --- |
| `cargo test --locked --all-targets --features python-bindings` | 365 passed | [log](rust-tests.log) |
| `cargo test --locked --all-targets` | 354 passed | [log](rust-default-tests.log) |
| `cargo clippy --locked --all-targets --features python-bindings -- -D warnings` | passed | [log](clippy.log) |
| `cargo fmt --check` | passed | [log](fmt.log) |
| Python 3.12 wheel build, native provenance, full discovery (`scripts/test-python.sh`) | 5,304 tests; five baseline failure assertions; ten skips | [full log](python-full.log) |
| Python 3.14 wheel, full discovery | 5,304 tests; three baseline failure assertions; one environment-path failure; ten skips | [full log](python314-full.log) |
| Python 3.14 stack validator at required `.venv` path plus buffer references | nine passed | [rerun log](python314-canonical-path.log) |
| Focused collation/convert/split/reference/README checks | 75 passed | [log](focused.log) |
| Final README and mixed-text reference checks | 21 passed | [log](docs-final.log) |
| Fresh source-copy GPU/negation/compile-boundary/evaluator checks, device 0 | 69 tests; five expected skips | [log](gpu-focused.log) |
| Negation/addition/transfer ownership, devices 0 and 1 | five passed | [log](two-device.log) |
| Unchanged full compile-coverage evaluator | 38/38 eligible cases passed | [report](compile-evaluation.json), [stderr](compile-evaluation.log) |

The assertion-only `examples/first_success.py`, 193 documentation links/anchors,
historical setup validator and six disclosure tests, and `git diff --check`
passed. Source/evaluator/matrix/report/build/log hashes and local worker paths
were checked against the retained CUDA/text reports. Historical workload JSON
and result sections remain byte-for-byte unchanged; the historical source tree,
archive and lockfile hashes match `e2f40ff16f8aba5216bc51699b56571e7e82a3e4`.
No historical compute workload was rerun. Evaluator definitions, weights,
denominators, calibration, and Burner-managed progress/history remain unchanged.

## Independently reproduced baseline failures

A `git archive` of base `31310e8fd1a1ff4f8dadcca93566dba70d115241` was extracted
under this worktree's `target/integration/baseline`, built with
`cargo build --locked --release --features extension-module` into the separate
`target/integration/baseline-build`, and loaded through that export's Python
package. [Baseline provenance](baseline-provenance.json) records its archive and
extension hashes; all 864 archived tracked files were verified unchanged after
building/testing. The buffer and factory reference tests and factory helper are
byte-for-byte identical between base and candidate.

Using the same local CPython 3.12.14/PyTorch 2.13.0 environment:

- The two noncanonical boolean-buffer reference assertions fail identically in
  base and candidate. Both hash seed 0 ([base](baseline-reference-seed0.log),
  [candidate](candidate-reference-seed0.log)) and seed 1
  ([base](baseline-reference-seed1.log), [candidate](candidate-reference-seed1.log))
  reproduce the failures. The native-only buffer tests pass; it is the PyTorch
  differential that differs. On CPython 3.14.7 the same base reference tests pass
  ([log](baseline-reference314-seed0.log)).
- Three factory-keyword ordering assertions fail identically at `PYTHONHASHSEED=6`
  ([base](baseline-factory-seed6.log), [candidate](candidate-factory-seed6.log)).
  The [seed probes](factory-seed-probes.json) retain every attempted seed from 2
  through 6. Seeds 0 and 1 pass the factory checks; seed 6 was selected to reproduce
  the existing failure, not to turn the full suite green. Set iteration and keyword
  insertion order vary with the hash seed. Python 3.14 reproduces the same three
  assertions at seed 10 ([base](baseline-factory314-seed10.log),
  [candidate](candidate-factory314-seed10.log)); its [probe record](factory314-seed-probes.json)
  retains seeds 0 through 10. Seed 6 passes on 3.14
  ([base](baseline-factory314-seed6.log), [candidate](candidate-factory314-seed6.log)),
  illustrating the interpreter-dependent ordering.

Reproduce the reference failures from the respective checkout using the local
interpreter, the matching `PYTHONPATH=python`, and `PYTHONHASHSEED=0` or `6`:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  tests.test_tensor_buffer_reference tests.test_nn_factory_kwargs_reference -v
PYTHONHASHSEED=6 PYTHONPATH=python .venv/bin/python -m unittest \
  tests.test_nn_factory_kwargs_reference -v
```

For the archived base, run in `target/integration/baseline` and use the absolute
path to the composite's `.venv/bin/python`. Full-suite failures are retained;
no tests or assertions were suppressed or weakened for these baseline issues.

## Python 3.14 environment-path correction

The initial full 3.14 run used `target/integration/venv314`; the unchanged stack
validator explicitly requires an executable under the checkout's `.venv`.
Its sole non-factory failure was this provenance requirement, before accepting
its generated smoke report. Temporarily putting the 3.14 environment at `.venv`
and rerunning that exact test plus the buffer references passed all nine tests.
The original 3.12 and 3.14 environments were then restored, and the retained
CUDA/text provenance was rechecked. No validator or test was modified, and the
original failing full log remains retained. This smoke check generated new test
fixtures only; it did not replace any historical workload measurements.
