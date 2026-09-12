# Owner initialization review revision

The independent review found that the lazy frontend captured the writable
native `_VariableFunctionsClass` export on its first import. Replacing that
export beforehand could authorize a counterfeit `neg` or execute a hostile
attribute callback. The pre-revision H100 probe reproduced eager `[1, -3]`
versus compiled `[-1, 3]`, and a hostile owner invoked its `mul` attribute hook.

Package initialization now eagerly loads the existing compiler state module,
which retains the genuine immutable owner. The lazy frontend imports that
identity instead of consulting the writable export. No operation dispatch,
kernels, accepted dtypes/layouts, gradients or CPU support changed.

Two new subprocess tests start before either frontend module is imported. They
cover namespace substitution, hostile attribute access and owner deletion;
module and direct-import counterfeit rejection; canonical aliases and unused
fields; all four policies; cold/warm calls; and cache recovery with
`recompile_limit=1`. Callback counters and exact exception classes are checked.
The original test for rebinding after frontend import is preserved.

## Validation and provenance

This directory records development validation based on
`7ff7cf3bb3496b077479db7938b3ebb457d6415c`, with an immutable export of the
modified source and a freshly built release wheel. It is not a clean-commit
capture. The [previous clean capture](../postcommit-e2378805/README.md) measured
`e2378805`, before this fix; its bytes and all earlier evidence remain unchanged.
**Burner must commit this revision and run a fresh clean-commit capture.**

| Check | Result |
| --- | --- |
| Complete compiler sweep, 59 modules | 666 run, 651 passed, 15 skipped |
| Focused CUDA/reference arithmetic | 34 run, 29 passed, 5 skipped |
| CUDA-hidden frontend, CPU compiler/eager regressions | 118 run, 109 passed, 9 skipped |
| Two-GPU restoration | 2 run, 2 passed, 0 skipped |
| Rust default all-targets | 395 passed, 0 ignored |
| Python-binding graph checks on GPU 0 | 15 passed, 0 ignored |
| CUDA-hidden native graph checks | 15 passed, 0 ignored |
| Docs/README smoke after adding report page | 12 run, 12 passed, 0 skipped |
| Formatting; default and Python-binding Clippy | Passed |
| Fresh-process owner regression pair | 2 passed, including all six subprocess cases |

The original counterfeit probe now raises `CompileTraceUnsupportedError`. With
only the owner export replaced by a hostile object, genuine negation returns
`[-1, 3]` on both cold and warm calls without invoking the callback.

The command receipts retain the pre-fix six failed subcases and the initial docs
smoke failure caused by publishing its navigation link before this report page
existed. The unchanged docs smoke passed after the page was added. No failing
assertion was removed or weakened. All 32 seeded Inductor input cells are
unchanged; the test-source hash changes because of the additional regressions.

The run reused the verified worktree-local CPython 3.12.14 interpreter and
canonical `.venv`, synced locked dev/reference dependencies, and built a fresh
release wheel with Rust 1.92.0 in an empty Cargo target. Cargo/uv download and
CUDA/Inductor/Triton caches were warm. Full interpreter inventory and installed
Python/native-to-wheel/source identity were checked before and after execution.
Native and reference used the local CUDA runtime 13.0 (version 13000), with
PyTorch `2.13.0+cu130` and driver 580.82.07. Installed nvcc was 12.6.85; the native
build uses embedded PTX and driver JIT, without invoking nvcc.

Resource use was `gpu` and `cpu-heavy`. Ordinary execution used H100 GPU 0,
UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, with
`CUDA_VISIBLE_DEVICES=0`; restoration additionally used GPU 1,
UUID `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. CUDA-hidden checks used an empty
mask. Per-command snapshots are observations, not reservations. All files and
caches stayed inside this worktree. This is functional validation, without
performance, fusion, training or hardware-parity claims.

## Reproduction

Use the locked contributor setup, [local environment settings](../env.sh.txt),
and the repository build tool with a fresh name:

```sh
. docs/diagnostics/compile-cuda-module-arithmetic/env.sh.txt
python3 scripts/build_cuda_add_diagnostic.py --name owner-startup-reproduction
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_arithmetic
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest -v tests.test_compile_cuda_module_arithmetic.ModuleArithmeticFrontendTests
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_arithmetic.ModuleArithmeticDeviceTests tests.test_compile_cuda_boundary.CompileCudaDeviceTests
```

[Command receipts](commands.json) contain every actual command, exit status,
timestamp, duration, input identity and GPU snapshot.
[Build provenance](build-record.json.gz) preserves the complete build-tool
output losslessly; [capture metadata](capture.json) links it to the installed
wheel and the retained failures. [Input deltas](input-delta.json) reuse the
unchanged prior manifest rather than duplicating it. The recorded `.py.txt` and
`checks.sh.txt` files are capture recipes; the complete sweep uses the unchanged
[parent sweep selector](../compiler_sweep.py.txt), covering all 59 modules.
Raw logs and their hashes are retained. Test-method counts include overlapping
suites and should not be summed as unique cases. CUDA-hidden Rust hardware
sections return early and are not reported as ignored tests.
