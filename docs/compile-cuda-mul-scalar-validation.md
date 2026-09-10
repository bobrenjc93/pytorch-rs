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

## Checkpoint recovery and pinned validation

Recovery starts at main `f88b7e8bb622d8aad58819b7985fb39d279a51c1`, which
contains PR #1925, and applies checkpoint
`9a1512de97392beadde70875f58f35f76f4cbe7e` without committing. The original
worktree was not modified. Final evidence measures this **uncommitted recovery
candidate**, identified by the source hashes in
[Python 3.12 provenance](diagnostics/compile-cuda-mul-scalar/recovery/final-provenance-312.json)
and [Python 3.14 provenance](diagnostics/compile-cuda-mul-scalar/recovery/final-provenance-314.json).
The recorded base commit alone does not identify the tested implementation;
Burner performs subsequent review and exact-head evaluation.

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

| Final check | Result |
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
