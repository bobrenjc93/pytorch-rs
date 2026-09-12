# Native positional transpose capture

Non-scoring development evidence for exactly three positional arguments to native
`torch.transpose(x, dim0, dim1)` and genuine direct-import aliases. Both axes are
exact integer constants. The frontend retains the genuine callable at package
startup and reuses the existing rank-0–2 CUDA float32 no-grad transpose view
recorder, planner and executor. No kernel or backend boundaries change. See the
[compiler contract](../../compile-cuda-t.md).

Before implementation edits, the unchanged frozen probe (SHA-256
`bc98874d08f12e7deb9e6f5284b0703104d292e6bed4fbbb49fe58c20030590f`)
reproduced 96 expected compilation gaps on clean main
`9795835e3265c18546029015ff04c06ae6c08f0a`, with 48 operation controls and four
unrelated-rebinding controls passing. Native eager and reference Inductor cold
and repeated execution passed. The original probe, supplied audit, output,
preflight and recipes are preserved losslessly under `external/`, with original
byte hashes in `external/provenance.json`. The local baseline log retains every
case's input hash, axes, shape, expression and original exception.

The new tests cover those 96 cells under all four policies (384 combinations),
cold/repeated native and reference execution, offset and positive-stride views,
raw IEEE bits, shared mutations, parent deletion, output lifetime, nested wrapper
identity, dynamic rectangles and explicit packing before arithmetic. Startup tests
jointly replace/delete public and native exports before the first frontend import.
Per-used-field guards preserve old t/squeeze/ReLU/add/neg/mul/matmul programs and
retained aliases, including cold/warm/lowered reuse with `recompile_limit=1`.
Malformed late nodes/outputs and mixed/unused inputs reject before native work.
The existing method-transpose rejection fixture changes only
`m.transpose(x, 0, 1)` to unsupported `m.transpose(x, 0, dim1=1)`; explicit positive
coverage lives in the new module suite. Earlier startup and guard tests remain.

The first prevalidation check incorrectly expected every malformed declaration
to raise `CompileTraceUnsupportedError`. Existing transpose axis validation
instead raises exact `TypeError` for booleans and `IndexError` for out-of-range
axes. The corrected assertions pass without production changes. Original logs
and test snapshots are retained. The first focused process had already loaded
its test module before these fixture edits; its input manifest and archived source
identify the executed version, even if traceback source lines display newer text.
The two-device expected tensor was also corrected before its first execution to
retain transpose's singleton stride, and inverse canonical-substitution coverage
was added. Both are test-only revisions after the release build.

## Results

| Check | Observed result |
| --- | --- |
| Exact-main frozen baseline | 96 expected gaps; 48 operation and 4 rebinding controls pass |
| Original 96 spelling cells, all four policies | 384 combinations pass, native/Inductor cold and repeated |
| Complete final-source compiler sweep, 63 modules | 721 run: 702 passed, 19 hardware-mask skips |
| New transpose suite within the complete sweep | 13 run: 12 passed, 1 two-device skip |
| Final focused boundary/startup/bit checks | 7 passed |
| Final CUDA-hidden/frontend/CPU/reference | 148 run: 86 passed, 62 hardware skips |
| Two-physical-GPU restoration | 5 passed |
| Rust default all-targets, CUDA hidden | 395 passed, 0 ignored |
| Rust Python-binding graph/planner, H100 | 15 passed |
| Rust Python-binding graph/planner, CUDA hidden | 15 passed; hardware sections return early |
| Formatting and default/Python-binding Clippy | Passed |
| Native extension/source verification | Passed |
| Documentation/navigation and CUDA example | 12 tests and example passed |

Counts are test functions unless explicitly labeled combinations. The first full
focused run had 13 tests: 11 passed, one hardware skip and the fixture expectation
error described above. The first boundary rerun had six passes and that same
error; the corrected boundary run passed all seven. Logs and snapshots retain
both failures. Durations are command wall time, not performance measurements.

## Provenance and reproduction

[Command receipts](commands.json) record actual commands, exit statuses, totals/skips, effective
CUDA masks (including leading `env` overrides), before/after GPU inventory and
source/test manifests. Logs preserve their original uncompressed bytes as gzip.
[Release builds](builds.json) bind fresh locked release wheels and native hashes to immutable
source exports. Full manifests are shared once; later versions store deltas.
The baseline was built before source edits. The development export binds the
production sources; command manifests separately bind subsequent test edits.
The documentation smoke recipe records the documentation hashes.
Pure Rust commands do not claim a Python-extension identity. The initial command
recorder and its later revision (which also hashes local launch recipes) are both
preserved; publication recipes contain no measurements of their own.

Only the verified relocatable Python 3.12.14 interpreter was copied, with complete
file/mode/internal-link inventory verification and isolated path checks. No project
environment, wheel or package was copied. All environments, caches, builds and
artifacts stay in this worktree. Rust is pinned to 1.92.0; dev/reference packages
use `uv.lock`. Native and reference runtime are CUDA 13.0, driver 580.82.07; nvcc is
12.6.85 and native kernels use driver JIT of embedded PTX. Command receipts record
H100 UUIDs and masks: GPU 0 normally, GPUs 0–1 for restoration. Resources are `gpu`
and `cpu-heavy`; utilization snapshots are observations, not reservations.

Source `recipes/env.sh.txt` from the worktree root, prepare the local interpreter
using the adapted bootstrap recipe, then create `.venv` with
`uv venv --python "$UV_PYTHON_INSTALL_DIR/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12" .venv`.
Run the recorded builder and test commands. `recipes/sweep.py.txt` selects every `test_compile*.py` plus
`test_top_level_compile.py`. These are development measurements, not performance
measurements or changes to the scoring denominator.

## Delivery boundary

The implementation agent may not commit, create PRs or write `.burner`. A separate
fresh clean-commit capture therefore requires Burner's managed implementation
commit. After that commit, rebuild with the existing source-export builder using
`--revision HEAD`, then rerun the recorded focused, complete compiler, hidden/CPU,
two-device, Rust, formatting/Clippy and documentation commands with new receipts.
That capture, same-branch draft PR, independent review/revision and fresh merge
gates remain delivery requirements; no result is assumed. Preserve review metadata
and blocked-push safeguards. PR1970/PR1971 remain separate unadopted campaigns.
Evaluator definitions, scoring corpora, historical benchmark evidence, hardware
matrix, observer and managed progress artifacts remain unchanged.
