# Compiled CUDA scalar multiplication validation

The [compiler guide](compile-cuda-add.md) owns the supported grammar and cache
contract. PR #1925's native `Tensor::mul_scalar` kernel is the prerequisite;
this change adds scalar graph nodes and bytecode lowering, without kernel
changes, fusion, Python-body execution or installed-PyTorch forwarding.

`tests/test_compile_cuda_mul_scalar.py` generates ordinary programs independently
of compiler dispatch with grammar seed `481903`. It checks both operator orders,
positional methods and top-level calls, local constants, guarded scalar globals
and helper captures, all existing fullgraph/dynamic policies, changed data on
cache hits, recompile limits, signed-zero/NaN guards, code and callable rebinding,
nested containers and repeated input/output aliases. Real CUDA tests cover
scalar, empty, irregular tail, offset and singleton-stride layouts, IEEE edge
values and scalar conversion, plus every element of a 17,000,003-element result
(seed `713924`). A Python profiling hook rejects original-body execution;
a subprocess blocks PyTorch imports. CPU tests validate scalar-node construction
and unsupported metadata without requiring a GPU.

Unsupported CUDA layouts, dtypes, autograd, tensor-tensor multiplication, scalar
call arguments and closures reject before execution or cache insertion. Tests
also construct malformed late nodes and intercept operation execution to prove
preflight rejection, including invalid declared output metadata on dynamic
scalar nodes. CUDA dtypes other than float32 and grad-requiring CUDA
tensors cannot be constructed publicly, so those compiler boundaries additionally
use explicit metadata tests. Two-device tests check driver pointer ownership,
cache separation, current-device restoration and rejection on GPUs 0,1.

The historical `mul_neg_add_v1` diagnostic retains two obsolete matrix/vector
rejection expectations and now exits 1 on GPU 0. Preserve its raw results; use
[trailing-vector validation](compile-cuda-trailing-vector-validation.md) for the
new capability. Historical measurements below remain attributed to their sources.

## Reproduction

Build a release wheel from the current source, install it in a fresh worktree-local
`.venv` using `uv sync --locked --no-install-project --group dev --group reference`,
and keep Cargo, CUDA, Python and temporary caches inside the worktree.
Verify installed Python sources and the extension hash before running:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python .github/scripts/verify_native_extension.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest tests.test_compile_cuda_mul_scalar
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_compile_cuda_mul_scalar.CompileCudaMulScalarDeviceTests
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.diagnose_compile_cuda_mul_scalar \
  --case-set mul_neg_add_v1 --output target/compile-cuda-mul-scalar.json
```

The separate `mul_neg_add_v1` diagnostic uses grammar seed `791406` and exact
input seeds `19092675` and `19092676`. It retains generated Python source and
hashes, source/native hashes, toolchain/runtime provenance, full materialized
output hashes and metadata, both fullgraph outcomes and unsupported results.
Its stock PyTorch 2.13 reference uses `backend="eager"` and must match stock
eager execution on both data sets. This establishes correctness coverage only;
it does not measure Inductor CUDA performance parity. Setup errors and unexpected
outcomes produce a nonzero exit and remain in the report.

The fixed 38-case compile corpus, frozen addition-only diagnostic, existing
neg/add diagnostic, private four-workload CUDA benchmark and all scoring weights,
denominators and historical evidence are unchanged. No speed improvement is
claimed for this unfused execution path.

## PR #1933 post-commit history

The preserved PR #1933 evidence measures its clean implementation commit
`e8a4488aa0751348a8e91f744b2d76e934576ba9`. The
[source post-commit record](diagnostics/compile-cuda-mul-scalar/post-commit-e8a4488/results.json)
indexes fresh reports, exact commands, timestamps, exit statuses, and the
pre-commit reports they supersede. All measurements completed with empty git
status before these reports and this documentation were added. This step
changes no implementation, dependencies, tests, harnesses, or scoring inputs.

Release wheels were rebuilt from that commit in separate fresh worktree-local
Cargo targets and Python 3.12.13/3.14.7 environments, using the locked dev and
reference groups. Existing worktree-local dependency downloads and the managed
interpreter were reused; build targets and CUDA/compiler caches were new.
[Python 3.12 build provenance](diagnostics/compile-cuda-mul-scalar/post-commit-e8a4488/build-provenance-312.json)
and [Python 3.14 build provenance](diagnostics/compile-cuda-mul-scalar/post-commit-e8a4488/build-provenance-314.json)
record the original interpreter builds, packages, toolchains, wheel/native hashes,
runtime libraries, source hashes, and source-worktree import paths. Installed
Python files match the committed source and wheel contents; each installed
extension matches its wheel. The native hash remains
`8836dea002699a0d738a347428d8fdeac794a797cfb601b70574db35b0261628`.

| Refreshed diagnostic | Cases | Reference eligible | Native pass | Native unsupported | Expectation failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mul_neg_add_v1`, GPU 0, Python 3.12 and 3.14 (each) | 104 | 102 | 84 | 20 | 0 |
| `mul_neg_add_v1`, GPUs 0,1, Python 3.12 | 8 | 6 | 6 | 2 | 0 |
| `neg_add_v1`, GPU 0, Python 3.12 | 168 | 164 | 128 | 40 | 0 |

Every case, reference eligibility decision, full output hash, generated source,
and unsupported outcome matches the earlier report. Six focused scalar-guard
and diagnostic-accounting checks passed on each interpreter; all five original
two-device checks passed together. The unchanged evaluator again passed its
38 reference-eligible cases. Reference tracing remains stock PyTorch 2.13's
eager backend; these results do not establish universal Python coverage or
Inductor CUDA performance parity. No timing workload was rerun or relabeled.

The author test suites and baseline failures below retain their original
measurement identities and raw files. They were not rerun merely to repeat
author validation. That source refresh supplied committed-code evidence for PR #1933. Its paths
and measurements remain pinned to that revision; the composite refresh below
supersedes its current-candidate claims and does not replace independent review
or other merge gates.

## Composite integration evidence

The composite combines PR #1933 with contiguous rank-3 `functional.linear`
bias support from PR #1934. Its implementation commit is
`90dc97fd476342eceb5c45261e60404b36819d61`. A detached, clean checkout of that
commit under this composite worktree supplies the source for fresh release
wheels, tests and repository diagnostics. The integration edits only descriptive
coverage and evidence; implementation, tests, frozen corpora, weights and
scoring definitions are unchanged. In particular, [FEATURES.md](../FEATURES.md)
now includes matching or singleton rank-1 bias for contiguous rank-3 input and
links the [detailed linear contract](supported-surface.md#nn-and-data-helpers).

The source post-commit, interrupted and pre-commit bundles above and below are
preserved byte-for-byte with their original identities. Fresh composite evidence
supersedes the PR #1933 post-commit bundle for current-candidate validation;
source measurements do not grant current-candidate performance credit.

The [composite record](diagnostics/compile-cuda-mul-scalar/composite-90dc97fd/results.json)
indexes actual commands, UTC timestamps, exit statuses and superseded reports.
[Python 3.12 provenance](diagnostics/compile-cuda-mul-scalar/composite-90dc97fd/provenance-312.json)
and [Python 3.14 provenance](diagnostics/compile-cuda-mul-scalar/composite-90dc97fd/provenance-314.json)
verify committed source, wheel contents and installed native hashes. Separate
fresh Cargo targets built release abi3 wheels with Rust 1.92.0, thin LTO and one
codegen unit. Both native extensions hash to
`54edb430965423ec8cc7a203dea4054559ae74346989a66f35905256b25fbfcf`.
Python 3.12.13 was installed inside the worktree; Python 3.14.5 reused the host
base interpreter read-only. All virtual environments, packages, build outputs,
compiler caches and reports are inside this composite worktree.

Both use PyTorch 2.13.0+cu130 and worktree-local CUDA runtime 13.0 (13000) on
H100s with driver 580.82.07. The public scalar path uses driver JIT of embedded
PTX; available nvcc 12.6.85 is unused by that path. GPU masks are explicitly `0`
for single-device checks and `0,1` for ownership/restoration checks.

| Composite diagnostic | Cases | Reference eligible | Native pass | Native unsupported | Expectation failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mul_neg_add_v1`, GPU 0, Python 3.12 and 3.14 (each) | 104 | 102 | 84 | 20 | 0 |
| `mul_neg_add_v1`, GPUs 0,1, Python 3.12 | 8 | 6 | 6 | 2 | 0 |
| `neg_add_v1`, GPU 0, Python 3.12 | 168 | 164 | 128 | 40 | 0 |

Every case outcome, generated program and complete materialized output hash
matches the source post-commit record. The frozen compile evaluator passes all
38 reference-eligible cases. These scalar diagnostics use the declared stock
PyTorch eager reference backend and establish bounded correctness only; they
do not establish universal compile coverage or a CUDA performance improvement.

The integration audit also confirms that the checkpoint's dynamic scalar-node
preflight fix, singleton-stride specialization test and corrected two-device
reference reset are already present. No implementation repair was needed for
those source findings. The rank-3 linear change retains bias-free layouts,
contiguous matching/singleton bias, empty and offset inputs, strided weights and
biases, fresh storage, error precedence and the autograd boundary.

| Composite check | Result |
| --- | --- |
| Full Python 3.12 and 3.14 suites, separate canonical environments, CUDA hidden | 5,398 checks per interpreter, OK; 150 skips each |
| Rank-3 linear/public-reference and documentation checks | 52 passed |
| Python 3.14 scalar CUDA/compiler/accounting checks, GPU 0 | 25 checks, OK; three two-device skips |
| Original two-device Python set, GPUs 0,1 | Five passed |
| Rust debug all targets, without / with Python bindings | 357 / 368 passed |
| Rust CUDA release integration, GPU 0 | Eight passed |
| Rust CUDA storage/cleanup unit checks, GPUs 0,1 | Eight passed |
| Rust formatting and clippy, with and without Python bindings | Passed |
| Rust release all-target gate | 160 library tests passed; two unchanged NaN-payload failures |

The full initial H100 suite exercised 5,398 checks with 15 skips and two setup
failures: the documented CLI selected an uninstalled environment, and the stack
validator rejected a noncanonical environment. Its five-test CLI followup passed.
Canonical full suites then passed on both interpreters. The stack validator
requires real `.venv` directories; replacing temporary environment links fixed
setup without changing its assertions. Initial host-Python linking, environment
layout, audit and nonexistent-target attempts remain in the record with their
actual outcomes and explicit supersession. The initial unused host wheel is not
the measured managed-Python release build.

The release failures are
`absolute_difference_sum_same_shape_contiguous_fast_path_matches_composition`
and `squared_difference_sum_contiguous_fast_path_matches_composition`. Both test
names and all four assertion values exactly match the preserved
[baseline log](diagnostics/compile-cuda-mul-scalar/recovery/baseline-rust-release.log).
Core tensor code and build inputs are byte-identical to that baseline. These
assertions remain unchanged, and the release gate is **not passing**. Burner’s
independent review, calibrated evaluations and delivery gates remain required.

The unchanged six-case CUDA math evaluator again passes three cases on each of
its original three seeds; unsupported broadcasting, reduction and matmul cells
retain zero credit. Its initial missing-extension setup produced a retained
zero-credit report; the fresh corrected report follows installation of the
verified wheel extension into the evaluator-required source package.

The separate fixed four-workload CUDA performance corpus passes all four cells
in both measured cache conditions. The reported reference/native steady-state
geometric means are **1.3810×** with fresh driver/Inductor/Triton caches and
**1.2889×** with reused caches; the unchanged coverage-adjusted formula reports
100% for each run. Both runs use the full matrix and default five warmups,
17 samples and three repeats. Private native CUDA libraries from correctness
checks are reused in both runs; their source and binary hashes, cache inventories
and empty GPU process inventories are retained in the
[cache receipt](diagnostics/compile-cuda-mul-scalar/composite-90dc97fd/cache-before.json).
Those libraries use nvcc 12.6.85 targeting `sm_90`. No timing selection or retry
was performed. These bounded private-corpus measurements do not attribute a
speed gain to the new scalar graph grammar or change any historical timing.
The [fresh](diagnostics/compile-cuda-mul-scalar/composite-90dc97fd/cuda-performance-fresh.json)
and [reused](diagnostics/compile-cuda-mul-scalar/composite-90dc97fd/cuda-performance-reused.json)
reports retain every workload, sample, correctness result and timing dispersion.

## Pre-commit recovery record

Recovery started at main `f88b7e8bb622d8aad58819b7985fb39d279a51c1`, which
contains PR #1925, and applied checkpoint
`9a1512de97392beadde70875f58f35f76f4cbe7e` without committing. The original
worktree was not modified. The author's pre-commit evidence measured the
**uncommitted recovery candidate**, identified by the source hashes in
[Python 3.12 provenance](diagnostics/compile-cuda-mul-scalar/recovery/final-provenance-312.json)
and [Python 3.14 provenance](diagnostics/compile-cuda-mul-scalar/recovery/final-provenance-314.json).
Its recorded base commit alone does not identify the tested implementation.
The refreshed diagnostics above now provide the clean-commit evidence.

The recovery fixes dynamic scalar-node preflight: invalid declared dtype,
device, gradient, layout, offset, or missing metadata now rejects before any
graph operation executes. The retained reproducer failed all seven cases on
the checkpoint. The singleton-stride test now uses equal-shaped contiguous
views with distinct strides, checks both cached specializations, and rejects
a mismatched graph input before execution. Reference Dynamo state is reset
between two-device policy scenarios without increasing limits or relaxing
reference eligibility. The public compile docstring and owning guides agree
on the bounded mul/neg/add grammar.

Both environments were freshly synchronized with `uv sync --locked
--no-install-project --group dev --group reference`. Python 3.12.13 uses the
host GCC 11.5.0 build; managed Python 3.14.7 uses Clang 22.1.3. Both install
NumPy 2.5.1, Maturin 1.14.1, PyTorch 2.13.0+cu130, CUDA bindings 13.3.1, and
CUDA runtime package 13.0.96. Separate fresh Cargo targets built release abi3
wheels using Rust 1.92.0, thin LTO and one codegen unit. Installed Python files
match current source byte-for-byte; both native extensions hash to
`8836dea002699a0d738a347428d8fdeac794a797cfb601b70574db35b0261628`.
H100 tests use driver 580.82.07 and worktree-local `libcudart.so.13` reporting
runtime 13000. Kernels use driver JIT of embedded PTX 6.0/sm_50; available
nvcc 12.6.85 is not used by these builds.

| Author check | Result |
| --- | --- |
| Combined Python 3.12 compiler, CUDA, multiplication, buffer and documentation set | 249 checks, OK; 11 hardware-specific skips under GPU mask `0` |
| Python 3.14 compiler/CUDA and documentation set | 167 checks, OK; five two-device skips under mask `0` |
| Complete original two-device set on GPUs `0,1` | Five checks, all passed together |
| Rust debug, all targets with Python bindings, CPU mask | 368 checks passed; hardware-only bodies detect unavailable devices |
| Rust CUDA release integration on GPU `0` | Eight checks passed |
| Rust formatting and clippy with Python bindings | Passed |

The [recovery record](diagnostics/compile-cuda-mul-scalar/recovery/results.json)
indexes commands, exit statuses and logs. The separate diagnostics report:

| Diagnostic | Cases | Reference eligible | Native pass | Native unsupported | Expectation failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mul_neg_add_v1`, GPU 0, Python 3.12 and 3.14 (each) | 104 | 102 | 84 | 20 | 0 |
| `mul_neg_add_v1`, GPUs 0,1, Python 3.12 | 8 | 6 | 6 | 2 | 0 |
| Existing `neg_add_v1`, GPU 0, Python 3.12 | 168 | 164 | 128 | 40 | 0 |

[Evidence checks](diagnostics/compile-cuda-mul-scalar/recovery/evidence-integrity.json)
verify every scalar report's source/native hashes, generated sources, complete
output hashes, and unsupported outcomes. The fixed compile evaluator still
passes its 38 reference-eligible cases; that result measures only its frozen
corpus. The private four-workload CUDA benchmark and all historical performance
evidence are unchanged. No timing campaign or performance gain is claimed.

These Python runs are targeted sets, not full Python-suite results. The
boolean-buffer checks pass on pinned Python 3.12, as do the five factory-kwargs
reference checks. No unrelated buffer, factory, or reduction fixes were made.
`cargo test --release --locked --all-targets` still reports 160 passed and two
NaN-bit failures in the existing `absolute_difference_sum` and
`squared_difference_sum` composition tests. An unchanged main export, with
every exported file verified against git, reproduces the same failing assertion
values with the same release flags; see
[baseline provenance](diagnostics/compile-cuda-mul-scalar/recovery/baseline-source-provenance.json)
and [baseline log](diagnostics/compile-cuda-mul-scalar/recovery/baseline-rust-release.log).
The NaN assertions remain unchanged.

All original validation files are retained byte-for-byte under
[interrupted/](diagnostics/compile-cuda-mul-scalar/interrupted/archive-manifest.json),
with archived scripts renamed to inert `.txt` data. Archive SHA256:
`fa6052b1170f1afcdf5826d073e4246fc541f5d055b6e6c262c68fc549548fd9`.
This preserves the exploratory NumPy 2.5.3/Maturin 1.15.0 environment, the
246-check run with two documentation failures and its 12-check docs followup,
the 123-check run with the stride-fixture failure and isolated correction,
the five-test two-device run with a reference recompile-limit failure and isolated
followup, and both original diagnostic reports. They are historical outcomes,
not pinned final validation. New recovery logs also retain an initial baseline
export setup failure caused by requesting an absent `benches` directory;
the corrected export omitted that path.
The whole-diff whitespace check reports whitespace already present in retained
raw logs; the source-only diff check passes. Raw logs are not rewritten to
silence that check.
