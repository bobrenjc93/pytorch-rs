# Complete clean public comparison — 2026-09-16

The required eight-leg B/C/reference protocol completed and passed the unchanged
offline verifier. All 96 cells, 1,632 samples, 480 warmups, 96 first calls and 96
additional churn calls are retained. Exact B/C comparison and finite reference
tolerance/metadata checks passed; the verifier recorded zero raw bit differences
for these fixtures. This supplementary diagnostic produces no score or merge
approval.

Candidate C was clean `8f2002d4fe1d579b8a0eb7d9d6016d9c6be01519`;
its changes after implementation commit
`47b97f203b35f936dc9fc0d38ef1255a68764f93` contain only prior evidence and
documentation. Baseline B was clean
`60202557b4f110d07777f585e804ab5f55e1ff7b`. The consumer SHA256 remained
`1861fca73674d8c89f6285b07883ab9a83e6719c1b6a0c12cd2be1ae7c4c8b42`,
and the [frozen protocol](../protocol.md), workloads, checks and tolerances were
unchanged. Both native packages were release-built and installed from wheels
bound to their own clean source inventories inside this worktree.

The [complete result tables](results.md) report every family/size in both orders,
all 72 median ratios, all first-call times and every ordinary churn call. Shared
arithmetic improved from 48.353 to 35.364 microseconds at size 257 in forward
order, and from 46.480 to 36.886 in reverse order (1.367× and 1.260×). At size
65537 it improved from 55.574 to 36.666 and from 55.674 to 37.577 microseconds
(1.516× and 1.482×). These repeated-call improvements have costs: small broadcast
regressed in forward order, small structured output regressed in reverse order,
and small over-cap execution regressed in both orders.

Cold costs also regressed for substantial arithmetic: the first small shared
call took 28.428/29.050 ms for C versus 12.294/11.472 ms for B in forward/reverse
order. The over-cap first small call took 137.816/137.018 ms versus
53.913/53.970 ms. The final negation preparation revisit took 86.490/73.220
microseconds for C versus 77.917/68.974 for B. These are ordinary completed-call
costs, not isolated compiler timings. All slow samples and dispersion remain in
the raw records; no order was pooled or discarded.

Reference comparisons are separate from those B/C gains. C remained slower than
default PyTorch in most cells: reference/C median ratios ranged from 0.247 to
1.137 across both orders. No native before/after ratio is treated as a default
compiler parity score, and no unchanged full qualification gate was rerun here.

## Runtime collision repair

[Attempt 1](../postcommit-47b97f2/README.md) remains intact, including its failed
reference record and complete baseline leg. The review requested a new attempt
after fixing the dependency collision. A new baseline environment was copied
inside this worktree. Its 25 CUDA library files were compared with the pinned
directory in the candidate environment; every byte hash matched. The new
baseline's `nvidia/cu13/lib` directory was preserved under
`lib-before-shared-identity`, then replaced by a relative symlink to the pinned
worktree-local CUDA directory. Both environments therefore resolve the same
library files while retaining separate framework packages and native wheels.
The old failed-attempt environment was not modified.

The archive retains the executed `dependency-setup.py` and its before/after
inventory, symlink target and timestamps. Separate untimed reference setup
processes used the frozen consumer's `Synchronizer`, GPU identity and
`validate_libraries` functions. Both passed with exactly one actual mapped
runtime. No PyTorch code, library bytes, environment-wide settings, consumer
checks or implementation were patched. All eight subsequent legs independently
passed the same runtime checks before and after execution.

## Checks and provenance

| Check | Result |
| --- | --- |
| Hardware-free consumer controls before freeze | 33 passed |
| B/C release builds and wheel/source/install checks | Passed |
| Reference dependency setup in both environments | Passed |
| Candidate fixed-fixture preflight | Passed |
| Serial order B, R, C, R, R, C, R, B | All eight completed |
| Frozen offline verifier | Passed, no failures |
| Post-capture source/wheel/library integrity | Passed |
| Prior failed-attempt archive integrity | Unchanged |

Both candidate control phases retained one exact negation executable owner across
`3,5,5,7,9,11,13,15,17,19,21,5`. The retained second 5 performed zero host-plan
construction or accounting; the evicted final 5 rebuilt one preparation while
using the same executable. Distinct broadcast addresses, executable eviction,
preparation pruning, reset, retained outputs and exact selected-owner receipts
passed. Eligible families selected direct code; the unchanged over-cap fixture
selected VM with 325 instructions at both sizes. Release-unobservable NVRTC,
module-load, native Program-build and upload counters remain labeled
unobservable; the existing focused tests cover those counters.

GPU0 was H100 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver
580.82.07, with exactly one visible device. Worktree-local CUDA runtime 13.0 and
NVRTC 13.0 providers were pinned by actual path and hash. The build recorded
Rust 1.92.0, Python 3.12.12 and installed nvcc 12.6; the native JIT used NVRTC.
Reference PyTorch was `2.13.0+cu130`. Every leg had a fresh process and fresh
cache directories, one host thread, five warmups and 17 retained-output samples,
with the same runtime synchronization boundary. Native NVRTC library preload
remained outside first-call timing as declared before capture. No additional GPU
was used and no foreign job was interrupted.

The [manifest](manifest.json) inventories 39 hash-verified records in
[attempt-2.tar.gz](attempt-2.tar.gz), including setup and installation commands,
freeze/build records, runtime checks, all eight raw legs, candidate receipts,
verification, all sample statistics, and final integrity/GPU snapshots. Original
record paths are inside this worktree. Both source inventories were revalidated
while clean before these report-only changes. The archive also includes the
exact serial launcher; use fresh attempt directories for any separately
authorized reproduction.

This completes the review's missing comparison. The complete record, including
cold and small-case regressions, is ready for independent review. Unchanged
qualification and Burner's merge gates remain separate requirements.
