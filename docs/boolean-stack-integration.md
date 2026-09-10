# Boolean buffers and stack provenance integration

This integration combines PR #1930
(`ed078e1a2dec6ad7c01f6c74c36cce52165d6fc3`) and PR #1931
(`85c8c1e5842ac394f182232dadb7c45d0a6c6b89`), both based on
`03075466b077c69ed1f7a4f047ebb33004d3306b`. Both source commits remain in the
managed merge ancestry. Boolean conversion and its tests match #1930; stack
validation and its tests match #1931. No frozen corpus, weight, reference case,
benchmark workload, dependency, or unrelated backend behavior changes here.

The [boolean report](boolean-buffer-semantics.md) and
[stack provenance report](top-level-stack-provenance.md) retain their source
validation histories. In particular, #1931's 5,372-test run had the two baseline
boolean failures plus a documentation-placement failure. That source PR moved
the link out of the exact historical-report index and passed all 12 documentation
checks. The boolean failures are addressed by #1930's actual conversion fix;
the original differential cases remain unchanged.

The [historical evidence bundle](diagnostics/boolean-buffer/source-pr-1930/README.md)
retains all 26 supplied measurement/driver/log/verification files with checksums
and a portable audit. Original source paths and timings remain historical; no
source-PR timings are presented as combined-checkout performance measurements.

## Combined validation

The implementation under test is merged commit
`d501bfada1a0c7ffb23f29fc8f950c77e695a671`. The integration adds documentation,
the historical payload, and its two audit tests on top of that commit; production
and benchmark code remain identical. The independent combined review found no
correctness defects and separately verified the archive and retained bytes.

On 2026-09-09, both interpreters used this worktree's canonical `.venv`, locked
NumPy 2.5.1 and PyTorch 2.13.0+cu130, and fresh release wheels built with
Rust/Cargo 1.92.0 and Maturin 1.14.1 (`--release --locked`, `extension-module`,
ABI3 Python 3.10, thin LTO, one codegen unit). CPython 3.12.13 is the host
GCC 11.5.0 build; CPython 3.14.5 is the managed Clang 22.1.3 build. The host
environment imports through `lib64 -> lib`, exercising the original alias bug.
Both report nonzero conversion of noncanonical boolean bytes.

Both builds produced native SHA256
`2bbb513fc30c386c5fe55195af8e6a9b2e8cfe66e6b47d883436f4f5e7c84044`;
all 60 installed package files matched each freshly built wheel, and native
extension provenance passed. Caches, managed Python, temporary fixtures, build
outputs and logs are under this worktree's `target/`. The full suites expose
only GPU 0 (NVIDIA H100, driver 580.82.07). PyTorch loads the environment's
`nvidia/cu13/lib/libcudart.so.13` and `libnvrtc.so.13` (CUDA 13.0); system
`nvcc` is 12.6.85. These are correctness checks, not new performance timings.

| Check | Combined result |
| --- | --- |
| Canonical GCC 3.12 focused buffer/creation/API, stack validator, docs, and evidence audit | 183 passed, 17.253 s |
| Canonical Clang 3.14 same focused checks | 183 passed, 20.146 s |
| Clean exact-HEAD GCC 3.12 full suite | 5,375 tests, 14 skips, no failures; 436.355 s |
| Canonical Clang 3.14 full suite including the new evidence audits | 5,377 tests, 14 skips, no failures; 569.139 s |
| Clang 3.14 portable path and evidence checks with site packages disabled | 11 passed |
| Rust formatting and clippy, with and without Python bindings | Passed |
| Rust all-target tests, without / with Python bindings | 357 / 368 passed |
| Full frozen v13 compile-coverage evaluator | 38/38 eligible cases passed; 100/100 within that corpus |
| Restored canonical GCC 3.12 provenance, original boolean differential case, and documentation checks | Provenance passed; 13 tests passed |

The focused command on each canonical environment was:

```sh
.venv/bin/python -m unittest \
  tests.test_tensor_buffer tests.test_tensor_buffer_reference \
  tests.test_tensor_data tests.test_tensor_data_reference \
  tests.test_as_tensor tests.test_as_tensor_reference \
  tests.test_scalar_tensor tests.test_scalar_tensor_reference tests.test_python_api \
  tests.test_top_level_stack_benchmark_artifact tests.test_readme_quickstart \
  tests.test_boolean_buffer_evidence -v
```

The portable command adds `-S` and selects
`tests.test_top_level_stack_benchmark_artifact.StackRuntimePathTests` plus
`tests.test_boolean_buffer_evidence`. Rust checks use `--locked --all-targets`,
with `--features python-bindings` for the bindings variant; clippy adds
`-- -D warnings`. The full compile gate is
`bash scripts/evaluate_torch_compile_coverage.sh` with its default full subset.
The clean gate is `bash scripts/test-python-exact-head.sh`: it exports and
verifies the merged HEAD under this worktree's `target/`, builds a fresh wheel,
and runs the full suite in that export's canonical `.venv`. It tests the merged
implementation and original source tests; the two new evidence audits and
integration documentation are additionally exercised by the working-checkout
focused runs and the canonical 3.14 full suite (`scripts/test-python.sh`).
Local validation logs are in `target/integration-logs/`; they are separate from
the checked-in historical bundle. Canonical `.venv` is restored to host GCC
CPython 3.12.13 after the 3.14 run. Both full suites passed without suppressing
or changing the original boolean differential cases. The 3.14 diagnostics from
hostile keyword equality and non-leaf gradient probes did not fail tests.
