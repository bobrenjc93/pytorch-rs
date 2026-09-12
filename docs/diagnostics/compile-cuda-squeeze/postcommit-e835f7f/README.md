# Clean-commit CUDA squeeze capture

Measured implementation **`e835f7f173af4ffc7783560d9b03709660fa3141`** on an
NVIDIA H100 with a recreated canonical `.venv` and a fresh release wheel.
All 19 setup/build/verification/test commands began and ended at that commit
with empty git status. This completes the capture deferred by the
[development record](../README.md), which remains unchanged, including its
baseline and failed first attempt. These are non-scoring correctness diagnostics.

| Check | Result |
| --- | --- |
| Public H100 squeeze/reference differentials | 11 run, 1 two-device skip |
| Complete `test_compile*.py` sweep, all 57 modules | 599 run, 14 two-device skips |
| `test_top_level_compile.py` | 49 passed |
| Squeeze restoration and unused-device guards on GPUs 0 and 1 | 1 passed |
| Native graph/planner/storage checks on H100 | 15 passed |
| CUDA-hidden frontend and CPU squeeze/reference regressions | 19 run, 10 hardware skips |
| CUDA-hidden native graph/planner checks | 15 passed; hardware cases return early |
| Rust eager squeeze and view/autograd regressions | 6 passed |
| Formatting, Clippy default/python-bindings, four guide examples | Passed |

The hardware tests exercised the unchanged assertions for all four compiler
policies, cold/repeated calls, dynamic singleton removal, exact bits, layouts,
shared storage and fresh wrappers, nested output identity, prevalidation and
rejected forms. No test, workload, unsupported outcome or denominator changed.
Only the compiler sweep explicitly required by the task and focused checks were
rerun; unrelated full Rust/Python suites were not repeated.

[Build provenance](build-record.json) records the actual source, release wheel,
native binary, interpreter, dependencies and toolchain. [Verification](verification.json)
checks every installed Python source against the checkout and verifies the native
binary against that build before and after measurements. The existing
[input manifest](../inputs.json) was verified and reused without duplication.
[Command receipts](capture.json) include timestamps, exit statuses, clean status,
log hashes, actual environment and GPU UUID/utilization/memory snapshots.
All command logs and the exact evidence recipe bytes are retained alongside them.

Setup used uv-managed CPython 3.12.14, pinned Rust 1.92.0 and locked dependencies.
It reused the local managed interpreter and download caches, recreated `.venv`,
and started with empty release-build and Rust-test targets. CUDA/Inductor caches
were reused. `uv python install --no-bin` and a local `UV_PYTHON_BIN_DIR` kept this
setup inside the worktree; this capture did not alter the historical setup incident.
Both native and reference loaded the same worktree-local CUDA 13.0 runtime,
with PyTorch `2.13.0+cu130` and driver `580.82.07`. Installed `nvcc` was 12.6.85;
the native build uses embedded PTX and does not invoke it. GPU resource use was
declared, ordinary checks used `CUDA_VISIBLE_DEVICES=0`, and only restoration used
`0,1`. Snapshots are observations, not reservations; no other jobs were interrupted.

The [guide](../../../compile-cuda-squeeze.md) links this capture. Its updated
navigation is checked separately in [publication-check.json](publication-check.json)
after the evidence-only working diff exists; that documentation check is not
represented as a clean-code measurement. The measured commit may precede the
later evidence-only commit. Independent review, evaluation and Burner delivery
remain required. No latency, general compiler, training or hardware-parity claim
is made; PR1970/PR1971 remain separate unadopted campaigns.
