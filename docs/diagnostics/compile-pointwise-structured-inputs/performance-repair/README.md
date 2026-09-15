# Structured-input frontend performance repair

The confirmed canonical rejection remains in force: CUDA **26.3441 versus
baseline 28**, coverage **18**, at rejected commit
`5ef07240386593dc4cfaff075fac3ce88d47ab2c`. No canonical command was rerun.
This record supports a narrow source repair for normal changed-head review and
qualification; it does **not** establish recovery of the score or general
performance parity.

The repair caches `BindingSource`'s immutable structural hash and restores direct
flat argument admission. On encountering a container, admission discards the
flat prefix and restarts the complete existing bounded walker. Structural
identity, scalar canonicalization/history, native ABI/numerics and transactional
cache publication are unchanged. Root-leaf resolution was not optimized.

## What was measured

[DECLARATION.md](DECLARATION.md) fixes the six non-corpus cases, shapes, random
seed, successful scalar/shape history, output checks, process order and timing
boundary. Each process keeps one wrapper per case alive across shapes. Each
supported case has 12 completed-GPU blocks of 300 calls per process, with two
processes per variant. Outputs are retained during timing and all materialized
and checked outside it. Cold calls are recorded separately. Replay guards cover
later seeds and warmups, **not the first cold call**; see the retained correction.

This is a **common-native/frontend comparison** using isolated processes and
ordinary public `torch_rs.compile(fn)`. The base and rejected frontend sources
were exported from Git into this worktree, and the two prototypes were measured
before being selected. Public wrapper closures verified actual frontend
selection. It is not an independently built end-to-end baseline or a PyTorch
performance comparison. The base's structured cases remain explicitly
unsupported; no flattened surrogate substitutes for them.

| Phase | Control → experiment | Flat median latency changes | Wide structured median |
| --- | --- | --- | --- |
| `before2` | Base → rejected | 30.68→33.21, 35.98→39.24, 31.00→34.92, 34.55→39.54 µs | Base unsupported |
| `after2` | Rejected → hash-only | 33.21→32.55, 39.40→37.68, 34.90→33.26, 39.94→37.07 µs | 12.808→14.297 ms |
| `flat` | Hash-only → hash + direct flat admission | 32.56→32.16, 37.82→36.83, 33.53→32.42, 36.87→36.20 µs | 12.890→15.048 ms |

Flat columns are one Tensor, two Tensors, runtime scalars and shape history,
in declaration order. Use comparisons **within each phase**: prototype names
`repaired` and `hash-flat` mean hash-only and combined respectively.
[summary.json](summary.json) contains every case's mean, median, quartiles,
minimum, maximum and individual process medians; the archive retains every block.

The result is mixed. Flat medians improve in both controlled comparisons, but
remain above the earlier base medians. Wide structured latency worsens, and
long-tail blocks matter: the `flat` phase's two-Tensor mean rises from
38.08 to 62.68 µs, with retained blocks near 323 and 365 µs/call. Hash-only also
has a 321 µs/call block in `after2`. No outliers were removed. The independent
plan/implementation check accepted the small repair for ordinary qualification
with these material caveats, not as recovered performance. Direct-flat-only
is an unmeasured simpler alternative; root inlining was not tried.

The initial `before`/`after` phases overlapped due to author orchestration error.
Both affected children were interrupted and **all timings from those phases are
invalidated**, including their completed base leg. Original receipts, partial
blocks and errors remain preserved. Replacement phases ran sequentially under
an exclusive process lock. CPU profiles are diagnosis only. None of this changes
or dismisses the original negative canonical measurement.

## Provenance and validation

All builds, imports, source exports, caches and output are rooted in
`full-leaf-agent_ea088824`. The existing setup-only release command built the
common native extension, then rebuilt/reinstalled after source selection.
The selected production source exactly matches the measured `hash-flat.py`
prototype (SHA256 `43b1a283edc9d5bddd2ba3b9972463ac74947c0846d685454879835b1c830e33`).
The common extension SHA256 is
`29c47a9b6058b7036e8e15f3d7ae68665e7789000dc1522b0445318c145d1c04`.

Runs used CPython 3.12.14, one host thread, `CUDA_VISIBLE_DEVICES=0`, H100
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07 and loaded CUDA
runtime 13.0. Separate post-selection compiled-executor metadata confirms NVRTC
13.0; host nvcc is 12.6.85. That later metadata check is labeled separately
from timed workload records. Actual paths, hashes, commands, GPU snapshots,
statuses and process exits are retained.

The new tests cover equal source/hash/dictionary semantics including collisions,
hash memoization, a structured restart after flat Tensor/scalar prefixes,
repeated Tensor occurrences, edge bounds, ignored-root transitions and failed-call
recovery. Existing numerical, helper, loop, alias, retention and transactional
assertions remain intact. The rebuilt worktree package passed:

- CPython 3.12.14 GPU-backed pointwise suite: **359 passed, 9 skipped, 0 failed**
  (368 executions; skips require an explicit two-device reservation).
- Focused portable contracts and loop admission: **34 passed, 0 skipped, 0 failed
  on each** of CPython 3.10.21, 3.11.16, 3.13.15 and 3.14.7.
- Focused review trace: 10 passed, covering every new hash/admission executable
  line. These repeat selected tests; they are not additional distinct tests.

`records/validation-summary.json` in the archive links exact commands and exits
and verifies each interpreter's installed source/native hashes. The pre-change
hash test's intentional failure is retained separately from these passing checks.

## Evidence and remaining gates

[diagnostic-records.json.gz](diagnostic-records.json.gz) preserves raw process records/logs, original and
corrected declarations, all source variants, setup/provenance, validation and
review reports. Text members carry byte lengths and SHA256 hashes. The included
Main canonical-report excerpts are explicitly historical **projections**, not
the unavailable full raw reports; their original hashes and limits are retained.
`capture.py` and the archived orchestration commands reproduce the declared
workloads; regenerate in a worktree-local release environment with fresh isolated
processes and the same ordering/counts. These diagnostics are not scoring tools.

This is dirty-source development evidence after a recreated worktree. The
[85fe118 correctness capture](../postcommit-85fe118/README.md) predates this source
repair and records its original worktree; it is not current-candidate validation.
The [c648878 clean-commit capture](../postcommit-c648878/README.md) now records
focused correctness and build/import checks in this checkout. It adds no timing
claim and does not replace these raw diagnostic records. Independent review and
the unchanged canonical gates remain with Burner. Preserve all earlier captures,
failed diagnostics and the negative qualification. No merge is authorized here.
