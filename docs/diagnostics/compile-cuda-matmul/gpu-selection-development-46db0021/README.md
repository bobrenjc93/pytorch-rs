# Explicit GPU selection: development validation

Six complete public diagnostic reports passed all 72 cells. This establishes
routing/provenance and unchanged work, with **no library speedup claim**.
The same installed native release wheel and reference runtime were used for
both selection paths. Timings across different physical GPUs are not compared.

| Seed | Ordinary physical GPU0 | Explicit physical GPU2 |
| --- | --- | --- |
| 798431 (primary) | [12/12](default-798431.json) | [12/12](explicit-798431.json) |
| 481723 (held out) | [12/12](default-481723.json) | [12/12](explicit-481723.json) |
| 926051 (held out) | [12/12](default-926051.json) | [12/12](explicit-926051.json) |

The [premeasurement declaration](plan.json) selected GPU0
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1` with ordinary mask `0`, and GPU2
`GPU-3aa416f0-b0e5-dbfc-0613-04b3426a6c0e` with its full UUID as both mask
and `--gpu-uuid`. Both are H100s. Inventories include every device's UUID,
physical index, PCI bus, utilization and memory before/after each attempt.
Idle snapshots were not reservations; no other user's jobs were controlled.

Every successful report independently binds driver UUID/PCI, native-runtime
PCI-to-inventory UUID, and PyTorch UUID to the same physical device at logical
`cuda:0`, before and after work. The [artifact audit](validation-summary.json)
verified all 44,640 timed calls and output checks, complete raw samples, equal
input hashes/policy/wheel across paths, and unchanged aggregation. All failures
remain in the fixed denominator: [mismatched identity](reject-mismatch.json),
[multiple devices](reject-multiple.json), [absent mask](reject-absent.json), and
[dirty checkout without opt-in](reject-clean-dirty.json) each exited 1 with
12 pending zero-valued cells, zero aggregate and full inventories.

The base is merged main `46db0021e8db4b563327ac4b8290eb7eab4318f4`.
The measured diagnostic SHA-256 is
`2bd01a0aa73f924c2ee7f10882bba8c180c42cfa2a4d97504b0eff1fc8fef1d6`.
All captures are explicitly **development-uncommitted**, not clean-commit
qualification. No implementation or evidence-only publication commit was
created under the no-commit delivery instruction. Burner must commit, run clean
validation, managed review, all current gates and exact-head CI before delivery.
Test coverage and contributor prose were refined during capture; measured
diagnostic and production code were unchanged. Historical evidence is untouched.

The [build receipt](build-record.json) binds production source, release wheel
and installed extension hashes. Every report verifies installed Python sources
and retains local runtime/library/compiler identities. Python was 3.12.14,
PyTorch 2.13.0+cu130, driver 580.82.07, and selected libcudart 13.0. Available
nvcc was 12.6.85 (unused by native matmul); native uses cuBLAS/driver-JIT PTX.
Reference Triton used its local ptxas 12.8.93. Rust was 1.92.0 with a locked
release build, thin LTO and one codegen unit. Compiler disk caches were retained
and their initial contents recorded under the unchanged first-call policy.

Checks: [42 focused tests, two multi-device skips](focused-tests-final.log),
[99 evaluator/accounting tests, 14 skips](evaluator-accounting-tests.log),
[native extension provenance](native-provenance.log), and formatting passed.
Golden tests bind cell generation, programs, full measurement loop, timing
policy and aggregation to the merged baseline. [Independent review](review.md)
found no blocking routing/work/accounting issues after its provenance finding
was fixed. [Scope verification](scope-check.json) confirms production and frozen
evaluators/corpora are unchanged. [Setup/check failures](setup-notes.json) and
their original available logs are retained alongside every command receipt.
