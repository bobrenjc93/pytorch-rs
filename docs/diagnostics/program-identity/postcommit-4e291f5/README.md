# Clean public comparison after grouped method guards

Clean candidate `4e291f5272b4b872ddc1704398b088b1917affc8` completed all eight
frozen ordinary-default B/C/reference legs against accepted main
`60202557b4f110d07777f585e804ab5f55e1ff7b`. The unchanged verifier passed all
96 cells, 1,632 samples, 480 warmups, 96 first calls and the 96 additional churn
calls. Exact B/C and finite reference/metadata comparisons passed, with zero raw
bit differences across the compared outputs. This diagnostic produces no score.

The [complete tables](results.md) retain every cell in both orders, all 72 ratios,
factory/first-call times and every churn call. The archive retains every raw
sample and Q1/Q3/MAD/min/max; no sample, workload or order was removed or rerun.
This compares the complete candidate with accepted main. It does not isolate the
guard repair or establish that guards caused the earlier canonical loss.

## Results and limits

Shared arithmetic improved in both native orders. At size 257, B/C steady
medians were 47.862/35.865 microseconds forward and 46.130/33.952 reverse
(1.335× and 1.359×). At size 65537 they were 58.658/39.149 and 58.378/37.828
microseconds (1.498× and 1.543×).

Regressions remain visible. Small structured output was slower forward
(55.514 to 56.536 microseconds), as was small over-cap execution
(123.978 to 124.318 microseconds). Both improved in reverse order. Order movement
also matters: baseline large structured medians were 67.632 forward and 55.614
reverse. These two orders are not a statistical guarantee.

Cold small shared calls cost C 29.886/29.135 ms versus B 12.320/12.331 ms.
Cold small over-cap calls cost 97.757/95.851 ms versus 15.148/14.998 ms.
The final negation preparation revisit cost C 73.751/70.817 microseconds versus
B 71.057/69.475. These are completed ordinary calls, not isolated compiler times.

C was slower than default PyTorch in 21 of 24 paired steady cells;
reference/C median ratios ranged from 0.244 to 1.256. Native gains do not establish
public reference parity or a qualification improvement. Earlier canonical
rejections remain authoritative for their measured revisions; independent review,
unchanged full qualification and exact-head CI still control this candidate.

## Checks and provenance

All 33 hardware-free controls passed before freezing the clean commit. Separate
B/C release builds, installed Python/native wheel bytes and clean source
inventories passed validation. B used a detached checkout inside this worktree.
Both inventories were revalidated after capture, before these report-only edits.
Builds recorded Rust 1.92.0, Python 3.12.12 and nvcc 12.6; actual native JIT
compilation used pinned NVRTC 13.0. Reference PyTorch remained `2.13.0+cu130`.

The proved CUDA layout remedy was reused without changing library bytes: all 25
B library files matched before preserving that directory and linking it to the
common worktree-local directory. Both untimed reference runtime checks passed.
Every leg verified actual runtime/compiler providers and GPU0 H100 UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07. Only GPU0 was used.
Workers ran serially with one host thread, fresh local caches, the identical
CUDA 13.0 completion barrier, five warmups and 17 retained outputs per cell.
No profiling or additional GPU work ran during ordinary timing.

Candidate preflight and both post-timing control phases passed. Equivalent
negation retained one executable through preparation eviction, with zero
host-plan construction/accounting on the retained hit. Distinct broadcast
addresses, eviction/pruning, reset and retained-owner checks passed. Over-cap
fixtures selected VM with 325 instructions at both sizes; the other five
families selected direct code. Release-unobservable compiler/load/Program-build/
upload counters remain explicitly unobservable, not zero.

The [manifest](manifest.json) inventories 63 hash-verified files in
[attempt-4.tar.gz](attempt-4.tar.gz), including setup/build commands, library and
source checks, every ordered raw leg, selected-invocation controls, verification
and complete statistics. The protocol, consumer, fixed evaluations and prior
archives are unchanged. The [development diagnosis](../method-guard-repair/README.md),
[previous complete capture](../postcommit-fc6c345/README.md) and earlier failed
attempts retain their original source identities and results.
