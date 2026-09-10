# Clean GPU-selection evidence

Measured implementation: `4e75e9ae177b37d022799394732db49ad95358a6`.
All six captures ran from this clean commit without `--allow-dirty`, using a
fresh release wheel. The checkout remained clean through every build, check
and diagnostic command. These reports complete the previously deferred clean
capture; the [development records](../gpu-selection-development-46db0021/README.md)
and all historical publications remain byte-for-byte unchanged.

| Predeclared seed | Ordinary GPU0 | Explicit UUID-selected GPU2 |
| --- | --- | --- |
| 798431 (primary) | [12/12 passed](default-798431.json) | [12/12 passed](explicit-798431.json) |
| 481723 (held out) | [12/12 passed](default-481723.json) | [12/12 passed](explicit-481723.json) |
| 926051 (held out) | [12/12 passed](default-926051.json) | [12/12 passed](explicit-926051.json) |

The [declaration and initial inventory](plan.json) precede measurements. The
default mask was `0`, physical UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`. The explicit mask and `--gpu-uuid`
were both `GPU-3aa416f0-b0e5-dbfc-0613-04b3426a6c0e`, physical index 2. Both
devices are H100s. GPU2 initially reported 0% utilization and 4 MiB used.
An idle snapshot is not exclusive access; no other user's jobs were controlled.

Each run verifies that driver UUID/PCI, native-runtime PCI-to-inventory UUID,
and PyTorch UUID independently bind logical `cuda:0` to the selected physical
device before and after work. Full inventories retain utilization and memory.
The same installed wheel, reference, seeds, inputs and policy were used on both
paths. This establishes routing and provenance, **not a library speedup**;
timings across the two GPUs are not compared.

[Mismatched identity](reject-mismatch.json), [multiple devices](reject-multiple.json)
and [absent mask](reject-absent.json) were rejected with exit status 1, all 12
planned cells retained and zero aggregate. Their command receipts bind the
clean commit even when selection fails before the diagnostic's source checks.
Every attempt, raw timing sample, failure and command receipt is retained.

[Setup metadata](setup.json) and the [build record](build-record.json) identify
the fresh locked Maturin wheel build, install, source/extension/wheel hashes,
commands, durations and local paths. Existing local packages and Cargo registry
were reused; build target and CUDA/Triton/Inductor caches began empty. Compiler
disk caches were then retained across orders and reports under the unchanged
first-call policy. Python was 3.12.14, PyTorch 2.13.0+cu130, driver 580.82.07,
and selected libcudart 13.0. Available nvcc 12.6.85 was unused by native matmul;
reference Triton used its local ptxas 12.8.93. The native release build used
Rust 1.92.0, thin LTO and one codegen unit.

Validation passed: [10 diagnostic/golden tests](diagnostic-goldens.log),
[installed-extension provenance](native-provenance.log), and the
[artifact audit](validation-summary.json), including 44,640 checked timed calls,
full sample counts, identical inputs/policy/wheel across routing paths, physical
identities, current local artifact hashes and unchanged aggregation.
[Diff and structural checks](scope-and-workload-check.json) confirm that the
measured definitions and frozen evaluators match dispatch baseline `46db0021`.

The evidence-only publication commit is pending Burner delivery; it is distinct
from measured implementation `4e75e9ae`. This step changes only new evidence and
its documentation. It does not approve the branch or replace independent
review, evaluations, merge gates or exact-head CI. No commit was created here.
