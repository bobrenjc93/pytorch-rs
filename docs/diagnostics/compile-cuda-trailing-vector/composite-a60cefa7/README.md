# Integrated CUDA trailing-vector and Python compatibility evidence

These are final correctness captures of the clean Burner merge commit
`a60cefa7c2ea7792bace10381218365a14da5fd6`, integrating #1941, #1937 and #1938.
The implementation needed no additional source repair after independent review.
This addition closes the compiler leaf's missing first-capture evidence gap.
It contains reports/evidence only; Burner owns the artifact commit and delivery.
Any later implementation or diagnostic-harness change requires fresh captures.

The frozen 38-case compiler corpus, private four-workload CUDA benchmark,
feature weights and Burner-managed progress files were not changed. This report
claims bounded correctness, not performance or universal compiler coverage.
Command elapsed times and stack-preflight samples are not performance credit.
Existing historical evidence was left unchanged.

## Final captures

| Capture | Cases | Native passes | Native unsupported | Reference eligible | Unexpected failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Python 3.12, GPU0](python312/trailing-single-312.json) | 228 | 200 | 28 | 226 | 0 |
| [Python 3.14, GPU0](python314/trailing-single-314.json) | 228 | 200 | 28 | 226 | 0 |
| [Python 3.12, GPUs0,1](python312/trailing-two-device-312.json) | 12 | 6 | 6 | 6 | 0 |
| [Python 3.14, GPUs0,1](python314/trailing-two-device-314.json) | 12 | 6 | 6 | 6 | 0 |

All unsupported and reference-ineligible cases retain zero local credit and
raw error strings. The two ineligible GPU0 cases use mixed CPU/CUDA inputs;
the six ineligible two-device cases use mixed CUDA ordinals. Supported cases
compare materialized values and complete metadata against eager and compiled
PyTorch using both fixed seeds and both fullgraph policies.

The focused [3.12 log](python312/focused-312.log) and
[3.14 log](python314/focused-314.log) each ran 49 tests: 48 passed and the
GPU0-only invocation skipped the separate two-device class. Both explicitly
pass `test_large_empty_signed_byte_strides_all_policies` and
`test_large_empty_dynamic_cache_replans_wrapped_strides`, including both operand
orders and offset views. The separate two-device regressions ([3.12](python312/focused-two-device-312.log),
[3.14](python314/focused-two-device-314.log)) each passed their one
device-ordinal/restoration test. Tanhshrink and variadic
atleast_1d public/reference tests are included in each focused run.

Full-suite totals are separate from the focused runs above:

| Full suite | Tests run | Passed | Skipped | Failures |
| --- | ---: | ---: | ---: | ---: |
| [Python 3.12 exact-HEAD fresh-wheel suite](python312/exact-head-312.log) | 5443 | 5426 | 17 | 0 |
| [Python 3.14 clean-commit release-wheel suite](python314/full-suite-314.log) | 5443 | 5426 | 17 | 0 |

The exact-HEAD script verified all committed files and rebuilt a fresh wheel
before testing; it cleaned its temporary checkout afterward. Both suites used
GPU0. Their hardware/mode-specific skips are included in these totals.

Formatting and Clippy passed, including Clippy with Python bindings. Native
`cargo test --locked --all-targets` passed 359 tests. The Python-bindings Rust
suite passed 370 tests with the Python 3.14 runtime selected. An additional
[explicit-runtime CUDA run](python312/rust-cuda-runtime.log) passed eight tests
using the canonical Python 3.12 environment's CUDA 13 runtime on GPU0.
The [unchanged full compiler evaluator](python312/compile-coverage-312.log)
reported 38/38 reference-eligible cases and 100/100 on its frozen definition;
this is not an expanded-coverage claim.

## Identity, setup and limitations

Each Python version has its own real canonical `.venv`, installed release wheel,
source export and build target inside the current composite worktree. Python
3.12 ran from the composite root. Python 3.14 ran from
`target/python314`, an identical committed source export with its own `.venv`.
The environment preflight, native-extension check, documentation smoke tests
and contained stack preflight passed before each version's focused workload.

The raw reports embed the complete build records, source manifests and harness
hashes. Standalone [3.12](python312/build-record.json) and
[3.14](python314/build-record.json) build records retain the original bytes.
[Verification](python312/verify-final-captures.log) checked all 1,820 committed files,
source-manifest hashes, reported source/harness hashes, installed native and
wheel hashes, clean status and contained source/build/import paths. Independent
review repeated those checks and found no retention-blocking issue.

Actual interpreters were Python 3.12.14+meta built with Clang 21.1.0 and Python
3.14.5 built with Clang 22.1.3; the full compiler/build strings are recorded in
the raw environments. Both used PyTorch 2.13.0+cu130, CUDA runtime 13000 and
NVIDIA H100 GPU0, with driver 580.82.07. Only the bounded two-device checks used
GPUs0,1. Rust/Cargo were 1.92.0. Builds used locked dependencies, release,
thin LTO, one codegen unit and `extension-module`. Available nvcc was 12.6.85;
the implementation uses driver-JIT embedded PTX and does not invoke nvcc.

The [capture helpers](capture/) preserve the local recorder, environment,
preflight and verification scripts at their execution-time contents.
[Independent review](independent-review.md) is recorded separately.
Raw command receipts record commands, working directory, clean source commit,
start/end timestamps, environment, status and log hashes. Build dependency and
compiler logs, initial environment setup logs, failed attempts and successful
retries are retained. The original Python 3.12 environment setup preceded the
receipt wrapper; its raw log and UTC preflight are retained, without inventing
an exact setup duration. Dependency caches initially started empty inside each
source root. uv caches were warmed by the environment setup; Cargo downloaded
crates during the first builds. The existing build tool's fixed `build_cache`
string is inaccurate for the Cargo download-cache state in these captures;
the [3.12 build log](python312/build.log) and [3.14 build log](python314/build.log)
give the actual first-build cold-download history. Each release build target was new.

The first Python 3.14 build attempt returned 1 because Git interpreted the nested
export as a subdirectory and returned an empty archive. It was retained as
[build-314.log](python314/build-314.log). The successful retry explicitly selected
the outer repository's read-only Git metadata with `GIT_DIR` and set
`GIT_WORK_TREE` to the nested source root; neither source nor Git metadata was
modified. The invocation was:

```bash
export GIT_DIR="$(git rev-parse --absolute-git-dir)"
cd target/python314
export GIT_WORK_TREE="$PWD"
. target/integration-env.sh
.venv/bin/python target/capture-command.py build-314-retry \
  .venv/bin/python scripts/build_cuda_add_diagnostic.py \
  --name composite-final-314-retry --revision HEAD
```

The Python 3.12 Rust bindings test first returned 101: this host's Python build
advertises `libpython3.12.a`, which is absent. The first Python 3.14 bindings run
then returned 127 because its shared library was not on the runtime search path.
Both failed logs are retained. Selecting that interpreter with `PYO3_PYTHON`
and prepending its actual `sysconfig.get_config_var("LIBDIR")` to
`LD_LIBRARY_PATH` produced the passing 370-test run. These are setup failures,
not omitted test failures. No external interpreter or toolchain was modified.

Fresh [neg_add_v1](python312/legacy-neg-add-312.json) and
[mul_neg_add_v1](python312/legacy-mul-neg-add-312.json) replays intentionally return
1 with four and two expectation failures respectively: their unchanged old
definitions still expect matrix/vector addition to be rejected. These are
current-commit replays of historical definitions, not relabeled historical
measurements. Their raw passing native outputs and obsolete expectations remain
visible. The current `trailing_vector_v1` definition has zero expectation failures.

## Delivery gates

Independent implementation and evidence review found no blocking defect.
Local checks are documented above; they do not stand in for Burner-managed
current-definition no-regression evaluations, published exact-head CI or managed
source-PR disposition. Those remain mandatory before final approval. No branch,
commit, push, PR or disposition action was performed by the integrator.
