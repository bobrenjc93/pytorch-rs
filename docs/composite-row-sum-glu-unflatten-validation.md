# CUDA row sums, vector GLU gradients, and unflatten integration

This repair replaces the row source's float64 accumulator with a native float32
reduction. Four accumulators, alignment heads/tails, descending shuffle-down,
geometry-dependent shared-memory reduction, and device-configured CTA partials
preserve the required cancellation/overflow behavior and wide decimal accuracy.
Partial scratch survives both launches and synchronous completion, including
errors. Public rank, layout, dtype, dimension, autograd, and compilation limits
are unchanged.

All three requested source heads are ancestors of the original composite merge
`24ca75e` and the committed repair below. GLU's dedicated vector backward ordering and unflatten's native view,
shape inference, aliasing, error, and first-order gradient behavior were already
integrated; no replacement of those implementations was needed. Existing Rust
and Python regressions remain, with additional row-sum differentials for device
geometry, all four offset alignments, random dynamic range, and nonfinite values.

README scope, FEATURES (including the rank-2 zeros wording), supported-surface
contracts and overview tables, documentation navigation, and architecture now
include bounded CUDA row sums. Unflatten appears in view navigation. Strict
documentation assertions were expanded. Public sum docstrings retain their
PyTorch-compatible text; GLU's existing docstring and detailed backward contract
remain accurate.

The original row-sum artifacts are unchanged historical source evidence. Their
source fingerprint was recomputed from git objects at
`f1040cdf723cf9b172f27d50d3580bbb21f3c430` and matched the preserved build receipt.
They do not establish combined-candidate correctness or performance.

## Qualification boundary

Burner committed the implementation as
`02535d5fb191285b0e2ca62677a1ec5d29edbb01`. The clean capture below now fulfills
the deferred post-commit measurement step. The earlier `--allow-dirty` capture
remains explicitly labeled `precommit-diagnostic`, with `clean_checkout=false`;
its raw files and original provenance have not been rewritten.

Combined independent review, all ten exact-definition evaluation gates against
the current baseline, and final exact-head CI remain required before managed
merge. These correctness captures claim no performance score.

No evaluator, corpus, weight, benchmark validator, dependency lock, or
Burner-managed progress artifact was changed. No branch, commit, push, or PR was
created.

## Clean post-commit capture

The [clean evidence bundle](diagnostics/composite-row-sum-glu-unflatten/postcommit-02535d5/README.md)
measures `02535d5fb191285b0e2ca62677a1ec5d29edbb01` from this composite's real
`.venv`, with empty git status before and after every capture. The existing
repository build tool ran without `--allow-dirty`, built a fresh release wheel
in an empty Cargo target, and verified unchanged source. Evidence publication
followed measurement; this step changes only reports and documentation.

The unchanged six-case CUDA math evaluator passed five cases at both selected
seeds `7763153567161607008` and `2618969910755569448`, including
`cuda_f32_sum_axis`. Unsupported CUDA matmul remained zero in the denominator.
The focused Python run completed 49 tests (48 passed, one two-device
mask skip); that restoration test then passed separately with devices 0,1.
The suite includes all committed row-sum, GLU, unflatten and strict README checks.
Full Rust and Python 3.12/3.14 suites were not repeated in this evidence step;
their earlier complete records remain below.

A separate-process reproduction captured 30 exact cancellation/overflow cases
(`3e38` and float32 maximum, 1/2/17 rows, both keepdim forms), 12 wide-decimal
cases at widths 65,539/1,000,000/1,000,003, and the finite GLU overflow regression.
All passed on the new extension. Reference and candidate processes were
separate, and candidate processes imported no PyTorch. The broader committed
suite covers random dynamic range, vector/geometry boundaries, offsets,
subnormals, nonfinite values, empty axes, storage and gradient behavior.

The [audit](diagnostics/composite-row-sum-glu-unflatten/postcommit-02535d5/evidence-audit.json)
verifies the source fingerprint, all installed/wheel Python sources, native
binary and wheel, interpreter, evaluator, actual runtime paths and hashes, and
local worker package/executable identities. The new extension SHA-256 is
`e805ae81c1cfe51b141c8694ee5bd75e7dffdcb9e83c8796f88f79711213cf69`.
It matches the previous binary bytes, but this capture independently binds a
fresh build and measurements to the clean committed implementation. Raw logs,
commands, timestamps, cache state and hashes are retained. No capture failed.

## Precommit environment and validation

Ordinary GPU work used `CUDA_VISIBLE_DEVICES=0`; two-device runs used only `0,1`.
Hardware: NVIDIA H100, compute capability 9.0, driver 580.82.07. Reference PyTorch
was `2.13.0+cu130`; both math-evaluator worker roles actually loaded this
composite's `.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`,
version 13000. Available nvcc was 12.6.85 and was not used for the native build:
the driver JITs embedded PTX. Rust/Cargo 1.92.0 built release abi3 wheels with
thin LTO, one codegen unit, and `extension-module`.

Python 3.12 uses the composite's real `.venv`. Python 3.14.7 uses a matching
source snapshot and real `.venv` under `target/python314-checkout`, entirely
within this worktree. The snapshot's implementation, tests, harness, and Python
package files match the combined source. The generated benchmark validators'
checkout/`.venv` identity checks were preserved.

| Check | Result |
| --- | --- |
| `cargo fmt --check` | Passed |
| Clippy, all targets, with and without `python-bindings`, warnings denied | Passed |
| Full Rust all-target checks | 372 reported passes; one two-device body skipped |
| Full Rust all-target checks with `python-bindings` | 383 reported passes; one two-device body skipped |
| Separate native two-GPU guard test | 1 passed; no skip |
| Full Python 3.12 suite | 5,491 tests, 19 skips, no failures |
| Separate Python 3.12 two-GPU suite | 16 passed; no skips |
| Corrected full Python 3.14 suite | 5,491 tests, 19 skips, no failures |
| Separate Python 3.14 two-GPU suite | 16 passed; no skips |
| Portable row-sum suite with no visible GPUs | All 11 hardware cases explicitly skipped |
| Original six-case CUDA math evaluator | Five supported cases passed at both seeds; unsupported matmul remained zero |

Each successful full Python suite's 19 skips were 16 two-device tests (all passed separately), two
pre-3.12 recursion tests, and one macOS libc++ test.

The fixed math seeds were `7763153567161607008` and `2618969910755569448`.
All six cases and both seeds remain in the report. Source, wheel, native binary,
interpreter, evaluator, runtime, and worker package identities are retained.
The report confirms source stability during execution, independent reference
and candidate processes, native device-pointer observations, preserved inputs,
and no candidate PyTorch imports. These are correctness diagnostics, not
performance measurements or final qualification evidence.

The same captured extension reproduced the requested `[1e8,1,1,-1e8]`, `3e38`,
and float32-maximum cancellation/overflow patterns for 1, 2, and 17 rows and
both keepdim forms. Widths 65,539, 1,000,000, and 1,000,003 passed for positive
and negative decimal values without changing tolerances. GLU's `x=[1e20,-20]`,
upstream `1e20` case produced identical native/reference gradients:
`[206115373056.0, 2.0611537240190114e31]`.

## Precommit corrections retained in the logs

An initial test invocation ran before the first wheel build completed and
failed to find the wheel/package; it was rerun after installation. Initial
Clippy checks caught a documentation-markup issue and a float comparison in the
new test. The test now compares bits, retaining its exactness, and both final
Clippy configurations pass. No numerical assertion or tolerance was weakened.
The Python 3.14 download reported that an existing external executable shim
could not be replaced; the interpreter was installed inside this worktree and
used by explicit path. Subsequent tool/cache paths were explicitly local.

The initial complete Python 3.14 run reported 5,491 tests, one setup error, and
20 skips. The snapshot had omitted the unchanged `.burner/evaluations.json`
contract, and one evaluator test skipped because its source-package native
extension was absent. The snapshot now references the original contract for
read-only inspection and contains an identical copy of its installed extension.
Neither the contract nor test assertions were edited. Both affected tests were
rerun, followed by a complete corrected Python 3.14 suite; the original log and
correction receipts are retained.


## Retained precommit records

Both interpreters used the same native extension bytes, SHA-256
`e805ae81c1cfe51b141c8694ee5bd75e7dffdcb9e83c8796f88f79711213cf69`.
The final audits confirm the measured production fingerprint still matches the
combined tree. All 172 local links checked in the six canonical/navigation
contracts resolved.

- [Raw logs, command receipts, and checksums](diagnostics/composite-row-sum-glu-unflatten/precommit/README.md).
- [Python 3.12 full run](diagnostics/composite-row-sum-glu-unflatten/precommit/python312-full.log), [Python 3.14 corrected full run](diagnostics/composite-row-sum-glu-unflatten/precommit/python314-final.log), and [initial Python 3.14 failure](diagnostics/composite-row-sum-glu-unflatten/precommit/python314-full.log).
- [Fresh precommit build receipt](diagnostics/composite-row-sum-glu-unflatten/precommit/capture-build-record.json) and [unchanged six-case evaluator output](diagnostics/composite-row-sum-glu-unflatten/precommit/cuda-math.json).
- [Numerical reproduction](diagnostics/composite-row-sum-glu-unflatten/precommit/numerical-reproduction.log), [source audit](diagnostics/composite-row-sum-glu-unflatten/precommit/source-audit.json), and [runtime/process audit](diagnostics/composite-row-sum-glu-unflatten/precommit/evaluation-audit.json).
