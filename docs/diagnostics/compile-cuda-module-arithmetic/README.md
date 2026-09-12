# Native CUDA module arithmetic development validation

This is non-scoring development evidence based on
`64dd296ee65bff37d5d2868506de2e61aa5effb5`. A fresh release build reproduced
32 rejected module/imported add/neg/negative cases, with all 12 existing
operator/module-mul controls and three unrelated-rebinding controls passing.
The frozen external probe was executed unchanged before source edits.

The change adds positional native call spellings and load-specific binding
snapshots, using existing arithmetic nodes and kernels. It also prevalidates
single-operation CUDA output specifications before the native hook. CPU global
call capture remains unsupported. The old trailing-vector rejection fixture's
`m.add(x, y)` placeholder is now `m.add(x, y, alpha=1)`, preserving its purpose.
See the [public contract](../../compile-cuda-add.md).

## Results

| Check | Result |
| --- | --- |
| Complete final-source compiler sweep, 59 modules including the public entrypoint | 664 run, 649 passed, 15 expected device-mask skips |
| Focused module arithmetic plus eager CUDA add/neg reference checks | 32 run, 27 passed, 5 two-device skips |
| CUDA-hidden frontend, CPU compiler and eager add/neg reference checks | 116 run, 108 passed, 8 hardware skips |
| New and existing device-restoration checks on GPUs 0 and 1 | 2 passed |
| Rust default all-targets, explicitly selecting the local CUDA 13.0 runtime | 395 passed, 0 reported ignored |
| Python-binding native graph/planner checks with GPU 0 | 15 passed |
| CUDA-hidden native graph/planner checks | 15 passed; hardware sections return early |
| Formatting; default and Python-binding Clippy | Passed |
| README/docs smoke and the compiled CUDA guide example | 12 tests passed; example passed |

Counts are unittest/Rust test functions; parameterized subcases include the 32
Inductor spelling cells and all four policies. Runtime, kernel and helper body
checks are functional evidence, not performance measurements. The final
restoration-fixture correction was followed by both the two-device run and the
complete final-source sweep; executable Python/Rust wheel inputs were unchanged.

[Command receipts](commands.json), [build provenance](builds.json),
[input manifests](input-manifests.json), [frozen baseline outcomes](baseline.json),
and [verification](verification.json) bind the checked source, native binary,
wheel, interpreter and inputs. Input manifests share one base plus hash-verified
deltas. Large historical logs are losslessly gzip-compressed; receipts retain
original log hashes and published paths. The final sweep log remains plain text.
All earlier repository evidence bytes are unchanged.

## Reproduction

All paths and caches belong to the current worktree. Resource use is `gpu` and
`cpu-heavy`; GPU 0 is used normally, and only restoration needs GPUs 0 and 1.
Snapshots are observations, not reservations. No other jobs were interrupted.

Use the locked contributor setup with Python 3.12.14, `UV_PYTHON_INSTALL_DIR`
and `UV_PYTHON_BIN_DIR` inside `target`, and pinned Rust 1.92.0. This run reused
only the complete verified relocatable interpreter distribution identified in
`interpreter.json`; it created a fresh `.venv`, synced locked dev/reference
dependencies and built fresh release wheels. No environment, wheel or installed
native package was copied. `env.sh.txt` records the worktree-local settings.

```sh
. target/env.sh
python3 scripts/build_cuda_add_diagnostic.py --name module-arithmetic-reproduction
.venv/bin/python .github/scripts/verify_native_extension.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_arithmetic tests.test_cuda_add tests.test_cuda_neg
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_compile_cuda_module_arithmetic.ModuleArithmeticDeviceTests
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest -v tests.test_compile_cuda_module_arithmetic tests.test_top_level_compile tests.test_top_level_add tests.test_top_level_add_reference tests.test_top_level_neg tests.test_top_level_neg_reference tests.test_top_level_negative tests.test_top_level_negative_reference
CUDA_VISIBLE_DEVICES=0 cargo test --locked --all-targets
CUDA_VISIBLE_DEVICES=0 cargo test --locked --features python-bindings --lib cuda_graph
CUDA_VISIBLE_DEVICES= cargo test --locked --features python-bindings --lib cuda_graph
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --all-targets --features python-bindings -- -D warnings
.venv/bin/python target/compiler_sweep.py
.venv/bin/python -m unittest -v tests.test_readme_quickstart
```

`compiler_sweep.py.txt` selects every `test_compile*.py` plus
`test_top_level_compile.py`, without exclusions. Copy the recorded `.txt` recipes
into `target` when reproducing; they are development capture helpers, not
scoring infrastructure. Native/reference execution loads the same local CUDA
13.0 runtime. Installed nvcc is 12.6.85; native kernels use driver JIT of embedded
PTX and the build does not invoke nvcc.

## Retained failures and revisions

Initial new fixtures omitted metadata fields, tried to patch a missing module
`__getattr__` without creating it, and fed an unpacked transpose to matmul.
Those fixture errors were corrected without expanding supported layouts.
The exact native Tensor type bypasses a subsequently attached Python
`__torch_function__` hook and is sealed against subclassing; tests now assert
those actual boundaries and separately reject active override modes.

A CPU regression caught changed helper error precedence: scanning all helper
opcodes before resolving its globals changed the established rejection message.
The frontend now collects helper global loads without prematurely validating
other opcodes, retaining the old order. Existing assertions remain unchanged.

An additional Inductor comparison found one signed-zero difference: native and
eager PyTorch returned `-0.0` for `neg(+0.0)`, while Inductor returned `+0.0`.
The retained failed log records the exact bits. The final test requires native
bit parity with eager PyTorch and zero-tolerance numerical parity with Inductor,
as in the frozen baseline. It retains zero inputs and full bit checks against
the eager reference. No kernel or frozen case was changed to hide this difference.

The owner audit also demonstrated that rereading the mutable exported owner
slot could trigger a hostile object's attribute hook. The frontend now pins the
immutable native owner at import; a hardware-free regression checks that an
exported-slot replacement cannot provide new identities or execute callbacks.
The earlier in-progress sweep was stopped for this fix and its interrupted
status is retained. The first sweep launcher attempt also retained its import
errors; the final launcher explicitly adds the worktree root to `sys.path`.
The first docs smoke ran during wheel replacement and saw the temporary package
uninstall, and also caught an exact navigation-text assertion. The final run
waited for installation and preserved the original navigation sentence while
adding the new scope on its continuation line; the existing test was unchanged.

The first restoration attempt also named a nonexistent existing test class and
patched `native.neg` while the program used a direct imported `n` alias. The
alias correctly stayed valid. The fixture now patches its used `native.add`
field; both it and the existing two-device boundary test pass on GPUs 0 and 1.
The complete sweep was repeated after this final fixture-only correction.

## Delivery boundary

These are uncommitted source-bound captures, not clean-commit or final-score
claims. This implementation task explicitly prohibits commits and PR actions.
Burner must commit the implementation, then run a separate clean-commit capture
and its normal independent review/revision and full delivery gates. Use the
same recipes with `--revision HEAD`, verify empty git status before/after,
and record the resulting commit separately; preserve this development evidence.
PR #1970 and #1971 remain separate unadopted campaigns. No scoring corpus,
evaluator, weights, hardware matrix, observer, managed progress artifact or
Burner contract was changed.
