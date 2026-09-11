# Clean-commit CUDA squeeze capture after review

Measured **`f8f8244c2cfa81e255c1814d1bd101c5d846b03d`** from a clean checkout with
a fresh exact-source release wheel on H100. All 19 setup/build/verification/check
commands began and ended at that commit with empty git status. This completes
the capture deferred by the [dynamic-consumer review revision](../review-dynamic-consumers/README.md).
The [earlier clean capture](../postcommit-e835f7f/README.md), development records,
baseline measurements and failed attempts remain unchanged.

| Check | Result |
| --- | --- |
| Public H100 squeeze/reference tests, including dynamic arithmetic consumers | 15 run, 1 two-device skip |
| Complete compiler sweep, all 57 `test_compile*.py` modules | 603 run, 14 expected device skips |
| `test_top_level_compile.py` | 49 passed |
| Squeeze restoration and unused-device guards on GPUs 0 and 1 | 1 passed |
| Native graph/planner/storage tests on H100 | 15 passed |
| CUDA-hidden frontend and CPU squeeze/reference tests | 23 run, 13 hardware skips |
| CUDA-hidden native graph/planner tests | 15 passed; hardware sections return early |
| Rust eager squeeze and view/autograd regressions | 6 passed |
| Formatting, Clippy default/python-bindings, four guide examples | Passed |

These unchanged tests exercise the four compiler policies, fresh shared-storage
wrappers, exact layouts/bits, nested identity, whole-graph prevalidation and
unsupported forms. Dynamic regressions include scalar/vector/matrix/empty
transitions through negation, ReLU and addition, cold/repeated calls without
Python body or per-node replay, and rejection of unpacked strided arithmetic.

[Build provenance](build-record.json), [verification](verification.json) and
[command receipts](capture.json) bind the source, installed Python files, native
binary, wheel, interpreter, inputs, logs and GPU snapshots. Verification matches
before and after all measurements. The original input manifest and previously
recorded review test hash are verified and reused; no tests, workload matrix,
reference, denominator or unsupported outcomes changed.

Setup reused the canonical worktree-local uv-managed CPython 3.12.14 `.venv`,
synced locked dev/reference dependencies, and used pinned Rust 1.92.0. The release
build ran offline with an empty target; local download, CUDA/Inductor and unchanged
Rust test caches were reused. Native and reference loaded the same worktree-local
CUDA 13.0 runtime, with PyTorch 2.13.0+cu130 and driver 580.82.07. Installed nvcc
was 12.6.85; native code uses driver JIT of embedded PTX and does not invoke nvcc.
GPU resource use was declared: ordinary checks used GPU 0, and only restoration
used 0,1. UUID/utilization/memory snapshots are observations, not reservations;
no jobs were interrupted.

The [guide](../../../compile-cuda-squeeze.md) now points here. Its publication
and README/docs smoke check are recorded separately in `publication-check.json`,
after the evidence-only working diff exists. The measured commit may precede a
later evidence-only commit. These are non-scoring correctness diagnostics and
make no performance or general compiler, training or hardware-parity claim.
Independent review and Burner delivery gates remain required.
