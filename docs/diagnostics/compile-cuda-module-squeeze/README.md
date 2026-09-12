# Trusted native squeeze call validation

Non-scoring implementation evidence for one-positional-argument native
`torch.squeeze(x)` and genuine direct-import aliases, using the existing CUDA
rank-0–2 squeeze view node. No kernel, method, backend, dtype, gradient or rank
boundary changes. See the [public contract](../../compile-cuda-squeeze.md).

The unchanged frozen probe (SHA-256
`6963e58a3afead071932db39af353559732d9bf8ef210555e1587108127dac2c`)
reproduced all 32 gaps on clean main
`66fd4d43e80e7c545f0cdb9f17f421cdc0012657`, before implementation edits,
using a fresh locked source-export release build in this worktree.
All 12 existing-operation and three unrelated-rebinding controls passed.
The original probe, audited external results and failed first attempt are
preserved as lossless gzip files, with original byte hashes in
[external provenance](external-provenance.json). The first external attempt
failed in NumPy comparison because `tolist()` lost the empty `(0,1)` rank.
The frozen correction reshapes logical values only after independent metadata
validation. That harness error is not a native execution failure.

The new test matrix covers those 32 cells under all four existing policies,
cold/repeated native and Inductor execution, shared-storage bits and mutation,
scalar/empty/odd/offset/noncontiguous views, nested identities and lifetimes,
dynamic singleton changes and consumers, trusted startup, helper loads,
per-used-field guards, canonical substitutions, exact exceptions, failed-cache
recovery, whole-graph prevalidation and two-physical-GPU restoration.
The existing negative `m.squeeze(x)` placeholder becomes
`m.squeeze(x, dim=0)`; its unsupported-binding purpose is unchanged.
The #1982 owner and #1983 ReLU startup regressions remain unchanged.

The first local focused run retained four setup errors: three used unsupported
direct CUDA `ones` construction and one used direct scalar CUDA `zeros`.
Use CPU construction followed by `.to('cuda:0')`, as in the existing method
suite. A subsequent added rank-sensitive consumer test expected the wrong
exception for invalid transpose axes; it now checks the existing `IndexError`.
Original test source snapshots and logs are retained. The first run had already
loaded its tests when fixture edits began; its final traceback may display a
newer source line. Its command input manifest and archived first source identify
the actual executed code. The first two-device attempt also used the raw-byte reader without selecting
the output device, causing a CUDA invalid-argument error and contaminating the
following ReLU control. The device test now uses the existing transfer-aware
comparison helper and checks context restoration immediately after execution.
A fresh-process rerun passes all four device tests. Raw IEEE assertions remain
in the single-device view tests. No production fix or boundary expansion was needed.

The first local full-sweep launcher selected all 61 modules but omitted the
repository root from `sys.path`, producing 61 loader errors before any tests ran.
Its log and original launcher are retained; the corrected launcher explicitly
adds the worktree root and keeps the same complete module selection.

## Results

| Check | Result |
| --- | --- |
| Exact-main frozen baseline | 32 expected gaps; 12 operation controls and 3 rebinding controls pass |
| Complete final-source compiler sweep, 61 modules | 693 run, 676 passed, 17 device-mask skips |
| Focused CUDA/reference | 38 run, 36 passed, 2 two-device skips |
| Final CUDA-hidden/frontend/CPU/reference | 119 run, 75 passed, 44 hardware skips |
| Two-physical-GPU restoration | 4 passed |
| Rust default all-targets | 395 passed, 0 ignored (CUDA hidden) |
| Rust Python-binding graph/planner, H100 | 15 passed |
| Rust Python-binding graph/planner, CUDA hidden | 15 passed; hardware sections return early |
| Formatting and default/Python-binding Clippy | Passed |
| Native extension/source verification | Passed |
| README/navigation smoke and CUDA documentation example | 12 tests and example passed |

Counts are test functions, not generated subcases. The 32 original spelling cells
are also tested under all four policies (128 policy/cell combinations), with
cold and repeated native/Inductor execution. Command durations are not performance
measurements. Earlier failed attempts remain in the receipts and logs.

## Evidence records

[Command receipts](commands.json) bind actual exit codes, exact test totals,
GPU snapshots and environment to shared [production](production-manifests.json)
and [test input](input-manifests.json) manifests. The full base manifests are
stored once as lossless gzip files; revisions contain only changed/removed
entries. SHA-256 keys use `json.dumps(files, sort_keys=True).encode()`.
[Release build records](builds.json) bind fresh wheel/native hashes to immutable
source exports; the baseline full source manifest is shared by the development
delta. The native binaries are byte-identical: this is a frontend-only change.
[Verification](verification.json) checks the installed native and final production
sources against the development build. Test-only fixture revisions after the
build are identified by each command's input manifest; they do not change
production wheel inputs.

The [test source](../../../tests/test_compile_cuda_module_squeeze.py) fixes seeds
198401–198404, shape/layout recipes, raw IEEE inputs, operation expressions and
all four policies. The original baseline additionally records every input hash.
Logs and failed-source snapshots retain their original uncompressed bytes.
`focused-final` loaded the third fixture snapshot; the subsequent change only
corrected the two-device comparison test, which is skipped in that run.
`restoration-final`, `hidden-final` and the corrected complete compiler sweep
use the final checked-in test source.

## Reproduction and delivery

Use the [recorded environment](recipes/env.sh.txt) and worktree-local Python
3.12.14 distribution, locked dev/reference
dependencies, Rust 1.92.0, a fresh source-export release wheel and the recorded
environment recipe. Only the verified complete relocatable interpreter was
copied; no project virtual environment, wheel or installed package was copied.
The native and reference runtime is CUDA 13.0; nvcc is 12.6.85, while native
kernels use driver JIT of embedded PTX. Resources are `gpu` and `cpu-heavy`;
ordinary checks use GPU 0, restoration checks use 0–1. Recorded inventory,
UUID, utilization and memory snapshots are observations, not reservations.

These are development measurements of uncommitted sources, not a clean candidate
commit capture. Burner owns the implementation commit, separate fresh clean-commit
capture, independent review/revision, same-branch draft PR and fresh merge gates.
For the clean capture, rebuild with the existing source-export builder using
`--revision HEAD` after the implementation commit, then rerun the recorded focused,
hidden, restoration, Rust, formatting/Clippy, documentation and
[complete compiler selection](recipes/sweep.py.txt) commands with new receipt names.
Those steps cannot be performed by this implementation agent under the explicit
no-commit/no-PR/no-`.burner`-writes instruction. They remain delivery requirements,
not assumed successful results. Preserve review metadata and blocked-push
safeguards. PR1970/PR1971 remain separate unadopted human-review campaigns.

The frozen corpora, evaluator definitions/weights, benchmark evidence, hardware
matrix, observer and managed progress artifacts are unchanged. No performance,
general compiler, Inductor, training or hardware-parity claim follows from these
tests.
