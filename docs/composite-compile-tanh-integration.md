# Immutable compiler analysis and owned CPU tanh integration

The composite preserves the complete reviewed changes from compiler source
`9e467a251ecf43ab38b01ff27253d3aa5b6d2ebf` and tanh source
`03ef9ee74a81ae8a498b74762d8884cb0446c6d3`, based on main
`b7936239ceb7d7713a15ecd8a2c82c4bddd6d214`. No additional production change was
needed. The tanh persistent-view-marker repair and all its regressions remain
intact, including empty views after clearing the base's gradient flag.

[The integration tests](../tests/test_compile_tanh_composition.py) exercise
eager tanh consuming owned CPU outputs of supported compiled negation and
addition. They compare values, metadata and gradients with pinned PyTorch 2.13,
using the existing per-operator tanh/tanhshrink tolerances and signed-zero
checks. Coverage includes:

- Scalar through rank-three outputs and empty dimensions; method, top-level,
  functional tanh and the existing tanhshrink composition.
- Changed values in input/capture allocations, shared operands and input/capture aliases,
  weighted repeated output use, accumulated gradients, and graph release.
- Input/capture gradient-flag transitions, capture rebinding, helper code and
  callable replacement, root code replacement, inactive-branch globals, and
  recovery after an invalid active capture.
- Rejection of tanh inside the compiled grammar, and of rank-four nonleaf or
  nonfinite compiled outputs by eager tanh, preserving usable parent graphs.

Calls to compiled programs are instrumented to reject execution of their
original Python bodies. Existing compiler tests retain the independent checks
for native execution, callable/method guards, unsupported CPU scalar globals,
cache limits, failure recovery, whole-CUDA-graph validation and device restoration.
Sigmoid support is unchanged. Native CUDA compilation remains bounded
`backend="eager"`; CPU tanh gradients do not implement accelerator autograd.

## Source evidence preservation

Every source-authored file changed by either included revision was compared
byte-for-byte with that revision after integration: 257 compiler/publication
files and nine tanh files. The development publication's 143 indexed artifact
hashes and the postcommit publication's 110 indexed hashes were verified;
the indexes themselves are also unchanged. The complete 2,923-file source
manifest was checked against its measured implementation `4cb0432`.

The [source development series](diagnostics/compile-cuda-graph/static-analysis-b7936239-development/README.md)
and [source clean series](diagnostics/compile-cuda-graph/postcommit-4cb0432/README.md)
retain their original identities, measurements, raw profiles, failures,
corrected BLAKE2b-128 audit receipt and contention observations. They measure
the source revisions identified in those publications, not this composite.
Their original worktree paths are retained as historical provenance. The four
source clean all-cell native ratios remain approximately 0.3346, 0.6599, 0.0863
and 0.9167. Neither isolated performance nonregression nor repeatable speedup
has been established. Reference drift and source scheduling credit provide no
composite performance credit.

## Validation and delivery boundary

Local checks are development correctness checks, with logs, command receipts,
before/after GPU inventories, environment setup and build products under
`target/integration/`. They are not the final clean performance publication.
The first test invocation preceded completion of the build; a later invocation
exposed a replacement-helper fixture using an unsupported globals namespace.
That fixture was corrected to use the root namespace. The first broad H100 run
also correctly rejected an editable installation in the installed-wheel test.
The release wheel was then installed and verified before rerunning the suite.
The original failed logs remain in that local directory.

A stronger capture-value fixture also exercised tanhshrink near zero. Applying
tanh's relative tolerance to that subtraction failed by one float32 rounding
step; the final test uses the established tanhshrink tolerance from its existing
reference suite. Tanh retains its stricter existing tolerance. No pre-existing
test or assertion was relaxed, and both exploratory failure logs are retained.

Completed development checks on 2026-09-10:

| Check | Result |
| --- | --- |
| Rust formatting | Passed |
| Rust all targets, default features | 389 passed |
| Rust all targets, Python bindings | 401 passed |
| Clippy all targets, Python bindings, warnings denied | Passed |
| Installed release wheel provenance | Passed |
| Focused CPU/H100 compiler, tanh and sigmoid suites | 292 run, six hardware/device-mask skips, no failures |
| Portable no-GPU focused suite, Python 3.12 | 226 run, 43 hardware skips, no failures |
| Portable no-GPU focused suite, Python 3.14 | 226 run, 43 hardware skips, no failures |
| Final integration test file after fixture refinements, each Python version | Four passed on each |
| Separate two-GPU graph/matmul restoration and mixed-device checks | Two passed |
| Unchanged frozen compiler evaluator | 38/38 eligible cases passed |

The local interpreters were CPython 3.12.14 and 3.14.7 with pinned PyTorch
2.13.0+cu130. The native release wheel used Rust 1.92.0, thin LTO, one codegen
unit and `extension-module`/abi3-py310. H100 checks used GPU 0, with only GPUs
0 and 1 exposed for the separate two-device check, driver 580.82.07 and the
local CUDA 13 runtime/cuBLAS. The available nvcc was 12.6.85; these native graph
operations use driver-JIT PTX/cuBLAS rather than nvcc compilation. Before/after
inventories for the successful H100 and two-GPU runs were empty. This is not an
isolation or timing claim.

GitHub `test` and `Python 3.14 compatibility` jobs subsequently succeeded at
both exact source heads listed above. These are source CI results, not
exact-head combined CI for the final composite.

Burner must commit the final implementation/tests before the required fresh
empty-target builds and paired capture. Use unchanged actual main `b7936239`
as baseline and the final committed composite as candidate, each rooted inside
this worktree and using identical local CPython, PyTorch, libraries, toolchain
and unchanged diagnostic harness. Declare the new series COMPLETE only after
collecting four baseline and four candidate reports: all 12 cases, primary
798431, held-outs 481723/926051, and primary repeat 798431, both timing orders,
all raw/first calls, cache states and failures. Keep that series separate from
the source publications.

Admissions must remain paused and other Burner source/reviewer/evaluator work
must finish before that capture. Keep physical `CUDA_VISIBLE_DEVICES=0` and
the existing diagnostic/math guards unchanged; record before/after process
inventories for every run, retain all cells, and disclose any foreign activity.
Empty snapshots alone do not prove isolation. Withhold isolated performance
claims if contention persists.

Use the existing repository build/capture tools and publish complete tracked
source/test/harness manifests, commands, cache records, build/wheel/native
hashes, interpreter/reference/runtime/compiler identities and raw reports.
Separate measured implementation and evidence-only publication commits; later
implementation/test/harness edits invalidate affected clean captures. All UV
caches and Python/tool executable links must remain worktree-local.

Final delivery still requires independent exact-head review, all ten unchanged
current-definition nonregression gates and successful exact-head combined CI.
The frozen 38-case compiler corpus, four-shape CUDA corpus, scoring definitions
and Burner-managed progress artifacts are unchanged. No score increase,
hardware credit or performance milestone is claimed by this integration.
