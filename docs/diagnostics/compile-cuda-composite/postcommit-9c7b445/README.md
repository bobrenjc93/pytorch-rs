# Clean combined CUDA diagnostic and accounting evidence

Measured implementation: `9c7b445c46373909392a2bce2e97c1b956240462`.
Every build, preflight and capture receipt records clean Git status before and
after execution. The implementation includes the original shared-interpreter
provenance correction and the procfs portability regression fix. No
`--allow-dirty` capture is used here. The evidence-only publication commit is
pending Burner delivery and is distinct from the measured implementation.

## Captures

The [hardware report](hardware.json) ran all six fixed inference compilation
cases on H100 GPU0 at predeclared seeds `6132505239447970515` and
`8782175947437908558`. All 12 reference trials passed with both initial
and changed-input executions. The candidate earned **5/6** slots, with 20
successful executions across the five supported cases and two seeds. Row sum
failed initial bytecode lowering with `KW_NAMES` at both seeds, before a
changed-input call could run, and retains its zero-credit slot. Raw failures,
materialized values, input-preservation checks, CUDA pointer observations,
original-body/import blockers and native-return evidence are retained.

| Routing report | Physical device | Seed | Complete cells |
| --- | --- | --- | --- |
| [Ordinary default](default.json) | GPU0, `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1` | 798431 | 12/12 passed |
| [Explicit UUID](explicit.json) | GPU2, `GPU-3aa416f0-b0e5-dbfc-0613-04b3426a6c0e` | 798431 | 12/12 passed |

The [declaration](plan.json) precedes measurement. Both frameworks used logical
`cuda:0` and the same physical device within each report. Before/after driver
UUID/PCI, native runtime PCI-to-inventory UUID and reference runtime UUID agree.
Inventories and command receipts retain index, UUID, utilization and memory;
idle observations are not reservations. Reports ran sequentially with no other
validation work during timed runs. No other user's jobs were controlled.

[Mismatch](reject-mismatch.json), [multiple-device](reject-multiple.json) and
[absent-mask](reject-absent.json) attempts each exited 1 before package imports
or timed work, retaining all 12 pending cells and zero aggregate. Every full
report, failure, first-call observation and raw sample remains intact. The
sample audit verified 14,880 checked timed calls across both reports, identical
inputs, policy and wheel, and unchanged fixed-denominator aggregation.

## Build and environment

The [plan](plan.json), [fresh build receipt](build.receipt.json),
[build record](build-record.json) and [environment preflight](environment-preflight.json)
bind source, wheel, installed/native worker extensions, packages and runtimes.
A locked release wheel was rebuilt in a previously absent local Cargo target,
using Rust 1.92.0, thin LTO and one codegen unit. The existing verified local
Cargo registry, CPython 3.12.12 and locked dependency environment were reused;
no dependencies were installed or changed. The fresh project wheel was installed
with `--no-deps`. Its installed extension was copied to the evaluator's local
source package and equal hashes verified.

The resolved executable, `sys.base_prefix` and interpreter SHA-256 are recorded,
not merely the `.venv/bin/python` invocation. Candidate and reference packages,
CUDA runtime/cuBLAS libraries, reference PTX assembler, build/cache/temp paths
all resolve inside this composite worktree. Installed Rust and the system
NVIDIA driver are recorded separately. PyTorch is 2.13.0+cu130 with CUDA runtime
13.0; both GPUs are H100s with compute capability 9.0 and driver 580.82.07.
Installed nvcc is recorded but unused by native execution, which uses cuBLAS
and driver-JIT embedded PTX; reference Triton/PTX assembler identities are explicit.

Measurement compiler caches began empty and were retained across orders and
reports under the existing first-call policy. Focused checks used separate
caches. The accounting runner creates its own temporary worker caches under
this worktree as defined by the committed runner. Command receipts record
selected environment controls without provider metadata or unrelated process
arguments. Exact receipt/preflight/capture/audit helper sources are preserved
as evidence text in [the supporting-source manifest](supporting-sources.json).

## Verification and limits

[Focused preflights](focused-preflights.log) ran 51 tests: 50 passed and the
explicit two-GPU boundary test skipped under the single-GPU mask. These cover
documentation, diagnostic structural goldens, matrix/accounting/isolation,
procfs portability, and real-GPU compilation boundaries. The
[wheel verifier](wheel-verifier.log) and [final artifact audit](final-artifact-audit.log)
passed. The [summary](validation-summary.json) retains exact accounting and
provenance checks. The initial [publication audit attempt](artifact-audit.log)
compared generator tuples directly with JSON arrays and failed; the corrected
[audit helper](audit-v2-source.py.txt) normalizes the expected JSON representation.
No measurement, implementation, test, harness or definition changed for that repair.

The [scope audit](scope-audit.json) verifies unchanged production code and
frozen math/transfer, four-workload performance and 38-case compiler corpora
against merged main `46db0021e8db4b563327ac4b8290eb7eab4318f4`. Existing matrix
fields and weights are unchanged apart from the committed six-case compilation
set. All 102 included historical source-publication files remain byte-for-byte.
The two original six-report GPU-selection publications and original hardware
capture/receipt keep their source/build identities and original paths; none is
repurposed as combined evidence. The hardware source's shared interpreter is
still documented as shared. Earlier dirty development reports remain separate.

These captures establish bounded existing capability and hardware routing.
They claim no native compiler gain, broader Python compilation, new accelerator
family, library speedup, or automatic overall-score increase. GPU0/GPU2 timings
are not a performance comparison. Independent review of the complete diff,
all current-definition no-regression gates, exact-head CI and managed source-PR
disposition remain required after Burner publishes the evidence. Any prompt-score
movement requires the normal three-sample median confirmation; new admissions
remain paused until this cohort finishes. This evidence step does not approve
the branch. Burner-managed progress artifacts were not changed.
