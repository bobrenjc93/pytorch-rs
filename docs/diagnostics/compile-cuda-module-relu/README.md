# Trusted native ReLU call validation

Non-scoring development evidence from exact main
`b74c2051b823518cc09d5a29f8237a3de9863516`. The unchanged frozen probe reproduced
32 module/imported ReLU gaps before implementation edits, with all 12 existing
operation controls and three unrelated-ReLU-rebinding controls passing.
The original audited baseline and probe bytes are preserved alongside this run.

The frontend retains the genuine PyO3 module function during package startup
and lowers positional module/direct-import calls through the existing ReLU node.
CPU module-call rejection, CUDA dtype/layout/gradient boundaries, the four
compiler policies and per-used-field dependency guards are unchanged. See the
[public compiler contract](../../compile-cuda-add.md).

The rejection placeholder `m.relu(x)` in `test_compile_cuda_relu.py` is now
`m.relu(x, out=None)`: it still checks unsupported binding rejection before
native work. The module-arithmetic hostile-binding fixture uses `native.abs`
instead of newly supported `native.relu`. No #1982 owner-startup regression was
removed. New positive tests exercise genuine ReLU calls explicitly.

[Command receipts](commands.json), [shared input manifests](input-manifests.json),
[release builds](builds.json), [production fingerprint](production-source.json),
[environment](logs/environment.log) and [integrity audit](verification.json)
bind these checks to source, inputs, installed Python/native files and release
wheels. The fresh baseline and development native libraries have identical bytes;
only frontend Python changes are required. Test-only fixture revisions after
the wheel build are recorded as input-manifest deltas; production wheel inputs
remain identical to the final checked sources. [The baseline outcome](baseline.json)
and [original audited evidence](audited-baseline.json) retain their original bytes.

The new seeded spelling test checks the original 32 cells under all four policies
(128 policy/cell combinations), with cold/repeated Inductor and native calls.
Other tests cover generated compositions, same-input aliasing, raw IEEE bits,
scalar/empty/offset/singleton layouts, dynamic squeeze ranks, nested outputs,
lifetimes, hostile/deleted bindings, exact exceptions and failure recovery.

## Results

| Check | Result |
| --- | --- |
| Complete final-source compiler sweep, 60 modules | 678 run, 662 passed, 16 device-mask skips |
| Focused CUDA/reference and #1982 regressions | 41 run, 37 passed, 4 two-device skips |
| Final CUDA-hidden/frontend/CPU reference regressions | 117 run, 88 passed, 29 hardware skips |
| Two-physical-GPU restoration checks | 4 passed |
| Rust default all-targets | 395 passed, 0 ignored |
| Rust Python-binding CUDA graph/planner | 15 passed |
| Rust Python-binding graph/planner with CUDA hidden | 15 passed; hardware sections return early |
| Formatting and default/Python-binding Clippy | Passed |
| Final README/docs smoke | 12 passed |
| Updated CUDA ReLU documentation example | Passed |

Counts describe test functions; generated subcases are additional. The full
compiler selection includes every `test_compile*.py` and `test_top_level_compile.py`,
without exclusions. Recompile-limit, restoration and failure-recovery assertions
remain enabled. No timings or scores are inferred from test durations.

## Retained development failures

The first frontend run had one failed subcase: a module-wide `__getattr__` trap
intercepted Python's private submodule import before ReLU lookup. The corrected
startup fixture still tests joint fake/hostile callable replacements and deletion
before frontend import. Separate deleted-field tests trap attribute callbacks.
The failed log and a reversible delta to that fixture are retained.
The first docs smoke run found an altered navigation label and the not-yet-created
evidence index. The original tested label was retained and the index added;
subsequent docs smoke passed. No production fix or assertion weakening was needed.

## Reproduction and delivery

Use the worktree-local environment recipe with Python 3.12.14, locked dev and
reference dependencies and Rust 1.92.0. Only the verified complete relocatable
interpreter was copied; the virtual environment and release wheels were built
fresh. Native and reference tests select the local CUDA 13.0 runtime; installed
nvcc is 12.6.85, while native kernels use driver JIT of embedded PTX.
Resources are `gpu` and `cpu-heavy`; GPU 0 is used normally and GPUs 0–1 only for
restoration. UUID/utilization/memory snapshots are observations, not reservations.

Burner owns commits, review/revision, draft PR delivery and fresh merge gates.
This worktree contains development evidence, not a fabricated implementation
commit or postcommit capture. Once Burner creates the implementation commit,
repeat the source-bound release build and commands at that clean commit and
publish a separate capture without replacing these development bytes. Existing
review metadata and blocked-push safeguards belong to that delivery workflow.
PRs #1970/#1971 remain separate unadopted human-review campaigns.

Copy the recorded [environment recipe](recipes/env.sh.txt) into
`target/relu-validation/env.sh` and source it. Python setup used the verified
[interpreter inventory](bootstrap.json) and [bootstrap recipe](recipes/bootstrap.py.txt)
because the uv catalog could not download CPython 3.12.14; no project environment
or native wheel was copied. Run commands from this worktree's root:

```sh
. target/relu-validation/env.sh
uv sync --locked --no-install-project --group dev --group reference --python "$UV_PYTHON_INSTALL_DIR/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12"
.venv/bin/python scripts/build_cuda_add_diagnostic.py --name module-relu-reproduction
.venv/bin/python .github/scripts/verify_native_extension.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_relu tests.test_compile_cuda_relu tests.test_compile_cuda_module_arithmetic tests.test_cuda_relu
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_relu.ModuleReluDeviceTests tests.test_compile_cuda_relu.CompileCudaReluDeviceTests tests.test_compile_cuda_module_arithmetic.ModuleArithmeticDeviceTests tests.test_cuda_relu.CudaReluDeviceTests
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest -v tests.test_compile_cuda_module_relu tests.test_compile_cuda_module_arithmetic tests.test_compile_cuda_relu tests.test_top_level_compile tests.test_relu_reference tests.test_nn_functional_relu tests.test_nn_functional_relu_reference tests.test_compile_static_analysis tests.test_compile_sum_lowering tests.test_compile_tanh_composition
CUDA_VISIBLE_DEVICES=0 cargo test --locked --all-targets
CUDA_VISIBLE_DEVICES=0 cargo test --locked --features python-bindings --lib cuda_graph
CUDA_VISIBLE_DEVICES= cargo test --locked --features python-bindings --lib cuda_graph
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --all-targets --features python-bindings -- -D warnings
cp docs/diagnostics/compile-cuda-module-relu/recipes/compiler-sweep.py.txt target/relu-validation/compiler-sweep.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python target/relu-validation/compiler-sweep.py
.venv/bin/python -m unittest -v tests.test_readme_quickstart
```

For a clean-commit capture, give the build a new name and `--revision HEAD`, then
repeat the same checks against that installed wheel. The frozen rejection probe
is only for the unchanged starting main, not the supported implementation.
The Rust CUDA-hidden result counts test functions whose hardware sections return
early; it is not evidence of GPU execution. No performance was measured and no
coverage/performance corpora, evaluator definitions, weights or managed progress
artifacts were changed. All preceding repository evidence remains untouched.
