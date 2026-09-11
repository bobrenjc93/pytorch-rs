# Clean-commit CUDA reshape capture

Measured **`3d44687d186bcc0f4d3a003676b5719353c80ab4`**, the committed implementation, using a fresh canonical
worktree-local `.venv`, uv-managed CPython 3.12.14 installed inside this worktree,
locked dev/reference dependencies, and an empty release build target. All build
and check receipts record clean status before and after measurement. Artifacts
and this documentation were published only after the measurements completed.

| Check | Result | Log |
| --- | --- | --- |
| Reshape H100 differentials and guards | 12 passed; 1 skipped | [reshape-final](reshape-final.log) |
| Existing compiler / frozen corpus regressions | 222 passed; 10 skipped | [compiler-regressions](compiler-regressions.log) |
| Context restoration / unused mixed-device captures | 4 passed | [two-device](two-device.log) |
| CUDA packing regressions | 6 passed; 1 skipped | [cuda-packing](cuda-packing.log) |
| CUDA-hidden portability | 2 passed; 11 skipped | [hidden](hidden.log) |
| Rust graph planning / prevalidation | 9 passed | [rust-final](rust-final.log) |
| Rust packing | 1 passed | [rust-packing](rust-packing.log) |
| CPU layout / reference / docs | 132 passed | [cpu-layout](cpu-layout.log) |
| Clippy, formatting, installed imports and guide example | passed | [Clippy](clippy.log), [format](format.log), [imports](imports.log), [Python files](python-imports.log), [example](docs.log) |

The [build record](release/build-record.json), [setup receipt](setup-record.json),
[preflight](preflight.log) and [audit](audit.json) bind code, interpreter, wheel,
native binary, installed Python files, commands and measured inputs. One
[input manifest](measured-inputs.json) serves all check receipts. The
[runner](validate.py.txt) preserves the committed task's requested checks;
[capture](capture.py.txt) requires this clean commit and hashes command files.
[Publication](publish.py.txt) verifies receipts before copying artifacts.

H100 tests use physical GPU 0 (logical `cuda:0`); only restoration checks expose
0,1. Every receipt includes UUID/index, utilization and memory snapshots before
and after its command; those snapshots do not reserve hardware. The selected
native/reference runtime is CUDA 13.0, driver 580.82.07, PyTorch 2.13.0+cu130,
and H100 capability 9.0. Rust/Cargo 1.92.0 builds release with thin LTO, one
codegen unit and `extension-module`. nvcc 12.6.85 is installed but unused:
native kernels use driver-JIT embedded PTX. Rust checks pin the same local
CUDA 13 runtime and print uncaptured hardware-test output. The uv cache was
populated inside this worktree; Cargo used a copied local registry and an empty
build target. All generated files, caches and environments stayed in this worktree.

The seeded reshape checks cover view/pack layouts, raw bits, identity,
mutation/lifetimes, cold/warm and static/dynamic behavior, composition, public
binding rejection, cached metadata, method replacement and blocked body/reference
replay. The review regressions for Tensor-valued arguments, known-invalid dimensions
and mixed literal/local tuples also pass across every supported policy, before
early/late execution and on repeated rejected calls. Nonconstant shapes, mixed
outputs, tuple helper/arithmetic operands and unused mixed tuples stay unsupported;
the established output-leaf diagnostics are preserved.
CUDA-hidden tests retain their hardware skips. These are non-scoring
correctness diagnostics, not performance measurements or a coverage-score change.

This completes the clean-commit capture deferred by the unchanged
[partial-argument review development bundle](../validation-review/README.md). The original
[development bundle](../README.md), [first clean capture](../postcommit-1ad5fd62/README.md)
and [preceding clean capture](../postcommit-7533cbbc/README.md)
remain pinned to their original code; their baseline, failures and build identities
are intact. The [initial candidate inspection](candidate-inspection.json)
verifies that prior inventory and the committed final development input hashes.
Implementation, dependencies, tests, workloads, evaluator definitions, hardware
matrix, managed progress and Burner contracts are unchanged in this step.
Independent review, evaluations and merge gates remain required; PR1970/PR1971
remain separate unadopted campaigns.
