# Dynamic squeeze consumer review revision

Independent review reproduced a dynamic cache-hit rejection in negation, ReLU
and addition after squeeze, with and without explicit contiguous packing. The
original capture remains pinned to `e835f7f173af4ffc7783560d9b03709660fa3141` and
does not validate this repair. These are non-scoring correctness diagnostics.

The repair checks unary/binary declarations against their original inputs while
planning runtime ranks independently. New seeded public execution regressions
cycle between scalar, vector, matrix and empty results, starting with both scalar
and matrix captures. They check repeated calls, one cached graph, reference output
metadata/data, fresh results, and absence of Python body or per-node replay.
Malformed late declarations and outputs still fail before the native bridge;
unpacked strided arithmetic remains unsupported. The frontend metadata check
also runs without CUDA.

Measurements use a fresh exact-source release wheel built from the uncommitted
review revision over `d5d9869591832a9809477a12d932fa8463286f9a`. They are development
evidence, not a clean-commit capture. Burner must commit this repair before the
required fresh clean-commit capture can be generated. Previous build records,
baseline measurements and clean-capture artifacts remain unchanged.

The original six-expression probe passed in reference PyTorch but rejected each
native vector-to-matrix transition. The first added GPU regression reproduced
16 rank-guard failures. A subsequent test run exposed an incorrect test expectation:
malformed output metadata uses the existing exact `ValueError`, while malformed
operation declarations use exact `CompileTraceUnsupportedError`. The final tests
assert those contracts; both failed runs are retained.

Setup reused the worktree-local uv-managed CPython 3.12.14 `.venv`, locked
dependencies and download caches. The repository build capture tool ran offline
with `--allow-dirty` and an empty release target; native code uses driver JIT of
embedded PTX. The host compiler reports nvcc 12.6.85, while native and reference
actually load the same worktree-local CUDA 13.0 runtime with PyTorch 2.13.0+cu130
and driver 580.82.07. Ordinary GPU checks use physical H100 0; only restoration
uses 0,1. Command receipts include UUID and memory/utilization snapshots. The
`gpu` resource was declared; snapshots are not reservations and no jobs were
interrupted. Rust checks reuse the unchanged local build cache.

| Final check | Result |
| --- | --- |
| Six original reproducer expressions, three calls each | 18 native/reference matches |
| Focused new regressions and frontend metadata checks | 5 passed |
| Complete compiler sweep, all 57 `test_compile*.py` modules | 603 run, 14 expected device skips |
| `test_top_level_compile.py` | 49 passed |
| CUDA-hidden squeeze and CPU/reference checks | 23 run, 13 hardware skips |
| Two-device restoration and unused-device guards | 1 passed |
| Native graph/planner tests, H100 and CUDA-hidden | 15 passed each; hardware cases return early when hidden |
| Formatting, python-bindings Clippy, four guide examples, docs smoke | Passed |

[Validation receipts](validation.json) bind commands, source, native binary, tests,
logs and GPU snapshots. [Build provenance](build-record.json) and
[final verification](verification-final.json) check the exact installed wheel and
all Python sources. The unchanged [input manifest](../inputs.json) and
[environment recipe](../postcommit-e835f7f/environment.sh.txt) are reused; only the
new squeeze test hash differs and is recorded separately. Recipe files alongside
the receipts reproduce this development capture; they do not replace the
repository build tool, test suite, or pending clean-commit capture.
