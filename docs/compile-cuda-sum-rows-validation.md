# Compiled CUDA row-sum validation history

For supported argument forms and a runnable example, use the
[compiled row-sum guide](compile-cuda-sum-rows.md). This page records source
implementation and development captures under their original commit identities;
it does not certify a later revision or replace the current delivery gates.

## Implementation and measurement boundaries

The graph contains `call_reduction`, target `sum`, with separate normalized
`(dim, keepdim)` reduction options. Output metadata is `(M,)` or `(M, 1)`,
canonical contiguous strides, CUDA float32 on the input ordinal, offset zero,
and no gradients. It is a shape-changing reduction, not a unary elementwise
operation. A single node uses `_compile_trace_reduction`; composed graphs
use the native `SumRows` operation in `_compile_trace_cuda_graph`. Both call
`Tensor::sum_rank_two_dimension(1, keepdim)` and reuse the existing CUDA kernel,
storage bounds, synchronization, completion and device-restoration guards.
The complete Python graph and native composed plan are validated before
execution. No original Python body or reference PyTorch is executed by the
candidate.

## Current scoring boundary

The implementation PR now preserves `scripts/evaluate_cuda_compilation.py`
byte-for-byte from its main-branch base `563d5596`. That observer still watches
`_compile_trace_unary` for the row-sum case; the native implementation correctly
uses the shape-changing `_compile_trace_reduction` entry point instead.
Successful native outputs alone do not satisfy the observer's recorded-call
contract, so the row-sum slot must remain zero under this scoring campaign.
The existing five slots and six-case denominator remain unchanged.

The observer adaptation is isolated in
[campaign PR #1970](https://github.com/bobrenjc93/pytorch-rs/pull/1970), which
requires separate human review before adoption. This implementation does not
adopt that campaign or earn its observation-related hardware-score increase.
Held-out GPU tests and changed-input comparisons still verify the real native
capability independently of that score.

Earlier captures below used the adapted reduction observer and retain their
original implementation and harness identities. Their 6/6 results are not
current canonical-observer scores. The fresh clean-commit captures below
use the restored observer and record 5/6 with zero row-sum credit.
Programs, seed policy, input distribution, tolerances,
compile options, denominator, weights, the 38-case compiler corpus and the
four-workload private performance suite are unchanged. Negative controls still
reject Python forwarding, counterfeit hooks, body execution, re-lowering,
malformed evidence and incorrect results.

During local separation checks, the first hardware-test run failed on row-sum
seeds `1927` and `83719` because a new assertion expected `passed`. The unchanged
worker correctly returned `failed` after rejecting the unobserved native
return, while preserving both executions and their outputs. The assertion was
corrected to require that rejection; the rerun passed all six cases at both
seeds without production or evaluator changes. These were dirty local checks,
not the deferred clean capture. The initial failure output was truncated and
its complete raw log was not archived; it is not represented as a passing run.

## Clean canonical-observer capture (380af38)

Fresh measurements use clean commit `380af38d9d8b8936569c23b2e77cc0ff970b98d0`.
The scoring evaluator matches `main` byte-for-byte. The native implementation
is unchanged from the earlier captures, but those captures used a different
observer and are preserved only under their original identities.

The release ABI3 extension was rebuilt with the committed, locked tooling in
an initially absent local Cargo target. Worktree-local Python 3.10–3.14
interpreters were installed; existing local Python dependencies and the Cargo
registry cache were reused. All 23 build/measurement receipts record an empty
Git status and unchanged source hashes before and after execution. Installed
Python sources and the installed/source native extensions matched the build
used by evaluator workers. The [build receipt](diagnostics/compile-cuda-sum-rows/postcommit-380af38/build-record.json),
[setup record](diagnostics/compile-cuda-sum-rows/postcommit-380af38/setup.json),
[provenance audit](diagnostics/compile-cuda-sum-rows/postcommit-380af38/audit.json),
[source hashes](diagnostics/compile-cuda-sum-rows/postcommit-380af38/source-files.json)
and [artifact hashes](diagnostics/compile-cuda-sum-rows/postcommit-380af38/artifact-sha256.json)
bind the capture to this worktree and commit.

| Check | Result | Raw evidence |
| --- | --- | --- |
| Held-out reduction/boundary tests | 14 tests; one expected two-device skip | [log](diagnostics/compile-cuda-sum-rows/postcommit-380af38/held-out.stderr.log) |
| GPU0/GPU1 restoration | One test passed | [log](diagnostics/compile-cuda-sum-rows/postcommit-380af38/two-device.stderr.log) |
| Python 3.10–3.14 lowering/native execution | Six tests passed per version; 3.12 included above | [3.10](diagnostics/compile-cuda-sum-rows/postcommit-380af38/python-3.10.stderr.log), [3.11](diagnostics/compile-cuda-sum-rows/postcommit-380af38/python-3.11.stderr.log), [3.13](diagnostics/compile-cuda-sum-rows/postcommit-380af38/python-3.13.stderr.log), [3.14](diagnostics/compile-cuda-sum-rows/postcommit-380af38/python-3.14.stderr.log) |
| Observer/accounting negative controls | 17 tests passed | [log](diagnostics/compile-cuda-sum-rows/postcommit-380af38/observer-controls.stderr.log) |
| Frozen-observer hardware regression | Passed all six cases at both test seeds, requiring row-sum rejection and correct outputs | [log](diagnostics/compile-cuda-sum-rows/postcommit-380af38/frozen-observer-hardware.stderr.log) |
| Canonical hardware compilation, capture 1 | 5/6; row-sum slot zero on both evaluator-selected seeds | [report](diagnostics/compile-cuda-sum-rows/postcommit-380af38/hardware-1.json) |
| Canonical hardware compilation, capture 2 | 5/6; row-sum slot zero on both evaluator-selected seeds | [report](diagnostics/compile-cuda-sum-rows/postcommit-380af38/hardware-2.json) |
| Frozen compiler corpus | 38/38 | [report](diagnostics/compile-cuda-sum-rows/postcommit-380af38/frozen38.stdout.log) |
| Private CUDA performance, fresh caches | 4/4; 1.2833x common-success ratio, 100.00% capped result | [report](diagnostics/compile-cuda-sum-rows/postcommit-380af38/performance-fresh.json) |
| Private CUDA performance, reused caches | 4/4; 1.2225x common-success ratio, 100.00% capped result | [report](diagnostics/compile-cuda-sum-rows/postcommit-380af38/performance-reused.json) |

Capture 1 selected seeds `6029585457145424764`, `3746966417343885870`;
capture 2 selected `2377763515611782021`, `385647285228949986`. The four
row-sum candidate trials retain status `failed` and error `invalid execution
or compilation evidence`. Their initial and changed-input outputs match the
reference, but the frozen observer records zero native returns from its old
unary hook. These outcomes remain zero in the unchanged six-case denominator.
The existing five cases pass. No observer adaptation or score increase is
adopted by this capture.

Both private kernel caches and the first run's CUDA/Inductor/Triton caches
were initially absent; the second run reused them. The four-workload suite
retained five warmups, 17 samples, three repetitions and equal weights.
All reported slow results, warnings and rejected outcomes remain. Timing
artifacts contain summary statistics, cold/factory accounting and checksums,
not individual latency samples; their medians and distributions cannot be
reconstructed from those summaries. These timings do not measure the new
generic row-sum graph or establish universal compilation/performance parity.

The host was NVIDIA H100 with driver 580.82.07. Ordinary runs used GPU0;
only restoration used GPU0/GPU1. Rust 1.92.0 built the extension; native row
sums use driver-JIT PTX, while the private kernels used nvcc 12.6.85. Main
workers used local PyTorch 2.13.0+cu130 and CUDA runtime 13.0; standalone
Python checks recorded the read-only system CUDA 13.0.96 runtime and no
PyTorch imports. Interpreter/import/build/cache paths resolve inside this
worktree; system compiler/driver/runtime paths are identified separately.

All build and capture commands exited successfully; this does not turn the
four rejected row-sum trials into passes. Historical evidence and documented
development failures remain unchanged. The deferred canonical-observer
capture is complete; separate observer-campaign review, independent candidate
review and Burner's ordinary ten no-regression gates remain delivery steps.

## Clean implementation-commit capture

This historical capture at clean commit `472ed6c26404176686a0cde6cb32b0722979be84`
validated the earlier candidate with its adapted observer, before the scoring
separation described above. The changes since `e9adfdca` contained only
evidence and documentation, and all 873 recorded source, test and evaluator
hashes matched that capture. However, the earlier local build/interpreter directories
had been removed and the installed extension differed from that capture's
hash. That run therefore rebuilt and recaptured its provenance; it does
not rewrite or reuse the earlier measurements as new results.

The release ABI3 wheel was built with `maturin build --offline --release
--locked` in an initially absent local Cargo target. Worktree-local Python
3.10.21, 3.11.16, 3.12.14, 3.13.15 and 3.14.7 distributions were installed.
The existing local Python dependencies and Cargo registry cache were reused;
no lockfile, implementation, test or evaluator changed. The installed Python
sources and native extension matched the build used by evaluator workers.
All 22 build/measurement command receipts record an empty Git status and
unchanged source hashes before and after execution.

See the [build receipt](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/build-record.json),
[setup record](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/setup.json),
[provenance audit](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/audit.json),
[source hashes](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/source-files.json),
and [artifact hashes](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/artifact-sha256.json).
The archived [capture commands](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/capture.py.txt)
invoke the committed repository tools without changing their cases or scoring.

| Check | Result | Raw evidence |
| --- | --- | --- |
| Held-out reduction and boundary tests | 14 tests; one expected two-device skip | [log](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/held-out.stderr.log) |
| GPU0/GPU1 restoration | One test passed | [log](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/two-device.stderr.log) |
| Python 3.10–3.14 lowering/native execution | Six tests passed per version; 3.12 included above | [3.10](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/python-3.10.stderr.log), [3.11](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/python-3.11.stderr.log), [3.13](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/python-3.13.stderr.log), [3.14](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/python-3.14.stderr.log) |
| Observer/accounting negative controls | 17 tests passed | [log](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/observer-controls.stderr.log) |
| Fixed hardware compilation, capture 1 | 6/6, both evaluator-selected seeds | [report](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/hardware-1.json) |
| Fixed hardware compilation, capture 2 | 6/6, both evaluator-selected seeds | [report](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/hardware-2.json) |
| Frozen compiler corpus | 38/38 | [report](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/frozen38.stdout.log) |
| Private CUDA performance, fresh caches | 4/4; 1.2735x common-success ratio, 100.00% capped result | [report](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/performance-fresh.json) |
| Private CUDA performance, reused caches | 4/4; 1.2080x common-success ratio, 100.00% capped result | [report](diagnostics/compile-cuda-sum-rows/postcommit-472ed6c/performance-reused.json) |

The hardware captures selected seeds `2634653737215571408`,
`3320354213672585458` and `6071828783403128758`, `1384939189734677963`,
respectively. Both retain the existing five passes and the row-sum pass,
including changed-input executions through the same wrapper. The six cases,
input distribution, tolerances, compile options, denominator and weights are
unchanged. Unsupported behavior outside these cases receives no new credit.

The private performance runs retained all four workloads, five warmups,
17 samples, three repetitions and equal weights. Both private kernel caches
were initially absent, as were the first run's CUDA/Inductor/Triton caches;
the second run reused them. All reported slow results and warnings remain.
The timing artifacts retain summary statistics, cold/factory accounting and
checksums, not individual latency samples. Their medians and distributions
cannot be independently reconstructed from the saved summaries. This suite
does not measure the new generic row-sum graph or establish universal parity.

The host was NVIDIA H100 with driver 580.82.07. Ordinary runs used GPU0;
only the restoration test used GPU0/GPU1. Rust 1.92.0 built the release
extension. The native row-sum kernel uses driver-JIT PTX; the unchanged
private performance kernels used nvcc 12.6.85. Main candidate/reference workers
used local CUDA runtime 13.0 and PyTorch 2.13.0+cu130; standalone Python checks
recorded the read-only system CUDA 13.0.96 runtime and no PyTorch imports.
Interpreter, import, cache and build paths resolve inside this worktree;
system compiler/driver/runtime paths are identified separately.

No build or capture command failed. Earlier evidence bundles and development
failures remain byte-for-byte unchanged. Independent review and Burner's
ordinary ten no-regression gates remain separate delivery steps.

## Earlier clean implementation-commit capture (e9adfdca)

The post-commit measurements below use clean implementation commit
`e9adfdca71626190a17e36544af97b1066499bbe`. They complete the previously deferred
clean-build capture; the development records below remain unchanged and keep
their original source identities and failures.

The release ABI3 wheel was rebuilt with the committed repository tooling in a
new, initially absent Cargo target directory. The local Python environments
and locked dependency cache were reused. Installed Python sources matched the
checkout, and the installed extension matched the extension used by evaluator
workers. Every build and measurement checked the commit, an empty Git status,
and source hashes before and after execution. All outputs were staged under
ignored `target/row-sum-postcommit-e9adfdca/`; they were copied byte-for-byte into
this evidence bundle only after the captures finished.

See the [build receipt](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/build-record.json),
[provenance audit](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/audit.json),
[source hashes](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/source-files.json),
and [artifact hashes](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/artifact-sha256.json).
Per-command receipts retain exact commands, timestamps, source checks,
stdout/stderr hashes, and GPU inventory/utilization/memory snapshots.

| Check | Clean-commit result | Raw evidence |
| --- | --- | --- |
| Held-out reduction and boundary tests | 14 tests; one expected two-device skip | [log](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/held-out.stderr.log) |
| GPU0/GPU1 restoration | One test passed | [log](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/two-device.stderr.log) |
| Python 3.10–3.14 lowering/native execution | Six tests passed per version (3.12 included above) | [3.10](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/python-3.10.stderr.log), [3.11](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/python-3.11.stderr.log), [3.13](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/python-3.13.stderr.log), [3.14](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/python-3.14.stderr.log) |
| Evaluator observer/accounting negative controls | 17 tests passed | [log](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/observer-controls.stderr.log) |
| Fixed hardware compilation, capture 1 | 6/6, both independently selected seeds | [report](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/hardware-1.json) |
| Fixed hardware compilation, capture 2 | 6/6, both independently selected seeds | [report](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/hardware-2.json) |
| Frozen compiler corpus | 38/38 | [report](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/frozen38.stdout.log) |
| Private CUDA performance, fresh caches | 4/4; 1.1636x common-success ratio, 100.00% capped result | [report](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/performance-fresh.json) |
| Private CUDA performance, reused caches | 4/4; 1.2903x common-success ratio, 100.00% capped result | [report](diagnostics/compile-cuda-sum-rows/postcommit-e9adfdca/performance-reused.json) |

Capture 1 selected seeds `548378513193054655` and `2591242177926228750`;
capture 2 selected `2419986187499636499` and `4536423993897679006`. Each case also
executed changed inputs through the same wrapper. The existing five passes and
the newly implemented row-sum slot passed in both captures. The performance
runs retained five warmups, 17 samples, three repetitions, all four workloads,
and equal weights. Existing private kernel caches were preserved separately;
fresh kernel and CUDA/Inductor/Triton caches were used for the first run and
reused for the second. The timing reports retain median, MAD, variance, sample
count, minimum and maximum, plus cold/factory accounting and output checksums.
They do not retain individual latency samples: ratios can be recomputed from
the reported medians, but those medians and distributions cannot be
independently recomputed from the saved summaries. The hardware correctness
reports separately retain raw input and output values. No capture or
validation command failed in this step.

These results cover bounded inference correctness and the unchanged private
performance workload. They make no universal compilation, accelerator,
training or performance parity claim. Unsupported behavior outside the fixed
cases has not gained credit. The six hardware cases, two evaluator-selected
seeds per capture, changed-input executions, distribution, tolerances, compile
options, weights, 38-case compiler corpus, and four-workload performance suite
were preserved. The audit verifies that the evaluator's sole change from
`main` is observing the real native reduction hook.

The machine used NVIDIA H100 GPUs and driver 580.82.07. Ordinary runs used
`CUDA_VISIBLE_DEVICES=0`; the restoration check alone used `0,1`. Rust 1.92.0
built the release extension with thin LTO and one codegen unit. The native row
sum uses existing driver-JIT PTX and does not invoke nvcc. Main reference and
candidate workers used the local PyTorch 2.13.0+cu130 environment and CUDA
runtime 13.0; standalone Python-version checks recorded the read-only system
CUDA 13.0.96 runtime and no reference PyTorch imports. The private performance
kernels used nvcc 12.6.85. All interpreter, import and build paths resolve within
this worktree; system compiler/driver/runtime paths are recorded separately.

This evidence step does not approve the branch or replace independent review
or Burner's ordinary ten no-regression gates. Those remain delivery steps.

## Preserved development validation

Evidence under `diagnostics/compile-cuda-sum-rows/development/` records the
author's pre-commit worktree, including failed development attempts. Build receipts
bind the parent commit, production source and diff hashes, and native extension
hash; these are **development receipts, not clean implementation-commit
receipts**. The first independent hardware capture preserved the existing five
passes and passed the previously zero row-sum slot on both evaluator-selected
seeds. This is six bounded inference correctness cases, not universal compiler,
accelerator, training or performance parity.

The independent regressions use held-out shapes and normally distributed
inputs, changed values through the same wrapper, nonzero offsets, singleton
strides, zero rows/columns, keepdim metadata, composed reductions, live method
mutation, invalid options/metadata, late graph rejection, stream completion,
and GPU 0/GPU 1 device restoration. Python 3.10–3.14 each exercise their own
real keyword bytecode and native CUDA execution. Hardware-only checks skip
explicitly when CUDA is unavailable.

The copied Python distributions, virtual environment, dependencies, Cargo
registry, build outputs and all caches were confined to this worktree. The
reference is PyTorch 2.13.0+cu130. GPU inventory and runtime paths are recorded
in the evidence. Rust 1.92.0 builds the release ABI3 extension; the reduction
uses existing driver-JIT PTX, so nvcc is unused for this native operation.
The host nvcc version is separately recorded.

The final development wheel passed the following checks:

| Check | Result |
| --- | --- |
| Independent reduction suite | 14 tests, one expected two-device skip; separate two-device test passed |
| Python 3.10, 3.11, 3.12, 3.13, 3.14 | Six lowering/metadata/native-CUDA tests passed on each interpreter |
| Existing compiler and evaluator regressions | 139 tests passed |
| Existing CUDA compiler regressions | 62 tests, six expected two-device skips |
| Rust default / Python bindings | 389 / 401 tests passed |
| Rust formatting and both Clippy configurations | Passed |
| Installed wheel provenance | Passed; source and installed extension hashes agree |
| Fixed hardware compilation | 6/6 in both development captures, two evaluator-selected seeds per capture |
| Frozen compiler evaluator | 38/38 |
| Unchanged private performance suite | 4/4 in each of two runs |

The hardware seeds were `1136685960689225517`, `4173613457994413490`
(first capture) and `1713849872941315970`, `1407868679530581933`
(final wheel). Every case also reused its wrapper on the evaluator's changed
input dataset. All unsupported cells outside these fixed cases retain their
existing status; no larger capability score is inferred.

The performance runs retained five warmups, 17 samples, three repetitions,
synchronization, output materialization, all four shapes and equal weights.
The first used fresh CUDA/Inductor/Triton caches and built the private kernels;
the second reused those caches. Their common-success geometric-mean ratios
were 1.3339x and 1.2911x, respectively (both reports capped at 100% for this
private suite). The saved artifacts retain timing summary statistics,
cold/factory accounting, checksums and GPU snapshots, not individual latency
samples. Ratios can be recomputed from the reported medians; the underlying
medians and latency distributions cannot be independently reconstructed.
These timings do not measure the new generic row-sum graph and are
not a clean-commit performance non-regression claim. The private kernels used
nvcc 12.6.85; native reduction and reference workers used CUDA runtime 13.0.
Python-version-only native checks selected the read-only system CUDA 13.0.96
runtime; their exact paths and empty reference-import sets are recorded.

Failed attempts remain in the evidence bundle. Early test setup assumed
unsupported CUDA factories, strided slicing, gradient enabling and dtype
exports; reference wrappers also hit their recompilation limit, and a dynamic
cache test initially ignored the existing concrete-stride guards. These were
corrected without changing the implementation or reference tolerances. Two
old tests expected compiled sum to be unsupported and now assert the bounded
success while retaining their rejection checks. An initially copied editable
install pointer caused the isolated wheel regression to import the parent
checkout and fail; the pointer was removed and a fresh local wheel installed
before final validation. Missing `typing_extensions` in standalone interpreter
attempts was corrected locally (the final Python 3.12 test uses the local
virtual environment). No failed attempt was discarded or represented as a
passing gate.

The development stage deferred the clean-commit captures now reported above.
Neither capture alone approves a later revision. Independent review and the
ordinary ten no-regression gates still precede Burner's managed merge; the
raw historical artifacts are not rewritten to reflect later delivery status.

Reproduce after preparing a genuinely local environment and local cache paths:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v \
  tests.test_compile_sum_lowering tests.test_compile_cuda_sum_rows
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v \
  tests.test_compile_cuda_sum_rows.CompileCudaSumDeviceTests
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_cuda_compilation.py \
  --build-record target/row-sum/build-record.json \
  --output target/row-sum/hardware-compilation.json
.venv/bin/python scripts/evaluate_torch_compile_coverage.py
```

Use the existing [build receipt procedure](hardware-heterogeneity-evaluator.md#reproduce-and-bind-a-native-build)
for the actual revision and extension. Omitting `--seed` lets the hardware
evaluator select two independent seeds; preserve every case and outcome.
