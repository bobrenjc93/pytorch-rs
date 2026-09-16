# Clean public comparison after the warm ownership repair

The frozen eight-leg protocol completed for clean candidate
`fc6c345bd71205626329627cd4c28a085af6072e` against accepted main
`60202557b4f110d07777f585e804ab5f55e1ff7b`. The unchanged verifier passed all
96 cells, 1,632 samples, 480 warmups, 96 first calls and 96 additional churn
calls. Exact B/C and finite reference/metadata checks passed, with zero raw bit
differences for these fixtures. This diagnostic produces no qualification score.

This capture follows the committed once-per-bind ownership repair. It compares
the complete candidate with accepted main, so its ratios do not isolate the
repair's contribution or explain the earlier canonical score loss. The
[development profiles](../warm-ownership-repair/README.md),
[previous complete comparison](../postcommit-8f2002d/README.md) and
[failed first attempt](../postcommit-47b97f2/README.md) retain their original raw
records and source identities.

## Results and limits

The [complete tables](results.md) show every cell in both orders, all 72 ratios,
factory/first-call times and every churn call. All 17 raw samples and dispersion
statistics remain in the archive; no slow sample or order was removed.

Shared arithmetic improved in both orders: at size 257, B/C medians were
47.792/36.285 microseconds forward and 46.531/36.054 reverse (1.317× and 1.291×).
At size 65537 they were 58.658/40.210 and 57.457/39.219 (1.459× and 1.465×).
Small structured output regressed in forward order (54.242 to 55.173
microseconds). Small over-cap execution regressed in both orders (122.335 to
125.340 and 123.396 to 124.388 microseconds).

Cold and churn costs also remain visible. The first small shared call cost
30.402/29.025 ms for C versus 12.762/11.390 ms for B in forward/reverse order.
The first small over-cap call cost 97.417/97.751 ms versus 14.964/15.119 ms.
The final negation preparation revisit cost 72.500/72.980 microseconds for C
versus 68.523/65.509 for B. These are completed ordinary calls, not isolated
compiler timings.

C was slower than default PyTorch in 22 of the 24 paired steady cells;
reference/C median ratios ranged from 0.236 to 1.119. Native B/C gains do not
establish public reference parity or a score improvement. Independent review,
unchanged full qualification and exact-head CI remain separate requirements.

## Checks and provenance

All 33 hardware-free consumer controls passed before the clean freeze. B and C
were separately release-built with the committed consumer, installed into
separate worktree-local environments, and checked against their own clean
source inventories and wheel bytes. B used a detached checkout inside this
worktree; no canonical base or external installation was modified.

The previously proved CUDA dependency-layout remedy was reused: all 25 library
files were hash-equal before B's library directory was preserved and replaced
with a link to the common pinned directory. Both untimed reference setup checks
passed. Every subsequent leg independently checked actual providers. The setup
helpers are byte-identical to the earlier archived helpers; the serial launcher
only changes the attempt directory.

Candidate preflight and both post-timing control phases passed. Equivalent
negation code retained one exact executor through preparation eviction; the
retained hit performed zero host-plan construction or accounting. Distinct
broadcast addresses, eviction, pruning, reset and retained owners passed. The
over-cap fixture selected VM with 325 instructions at both sizes; all other
matrix families selected direct code within the frozen caps. Release-unobservable
NVRTC/module-load/Program-build/upload counters remain labeled unobservable.

GPU0 was H100 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07.
Worktree-local CUDA runtime 13.0 and NVRTC 13.0 providers were pinned by actual
path and hash. Builds recorded Rust 1.92.0, Python 3.12.12 and nvcc 12.6; the
native JIT used NVRTC. Reference PyTorch remained `2.13.0+cu130`. All eight
workers ran serially with one host thread, fresh local caches, identical runtime
synchronization, five warmups and 17 retained outputs. Profiling and dependency
checks were outside these timings. Only GPU0 was used; the final snapshot showed
0% utilization and 4 MiB on each GPU.

The [manifest](manifest.json) inventories 52 hash-verified files in
[attempt-3.tar.gz](attempt-3.tar.gz): clean freeze/build records, setup commands
and library checks, all eight raw legs, selected-invocation controls, verification,
complete statistics, and post-capture integrity checks. Both source and installed
wheel inventories were revalidated while clean before these report-only edits.
The protocol and consumer hashes remain unchanged. No prior archive, scoring
definition, corpus, tolerance or Burner-managed progress artifact was changed.
