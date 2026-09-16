# Pointwise admission repair: fixed public-path diagnostic

Frozen before measurements. This is a non-scoring comparison of rejected
`8006d7e05636b3322da86ae89c41040900ceee00` (B), the actual native admission
repair (C), and unchanged default PyTorch 2.13.0+cu130 (R). It does not replay
the earlier eight-leg integration experiment or replace canonical evaluation.

Run exactly six serial fresh processes in order **B, R, C, C, R, B**, using
GPU0 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`. Both R processes use the
same environment as C. Pin identical existing NVRTC and CUDA runtime files,
record their loaded identities, and use one host thread and fresh process-local
CUDA/Inductor/Triton caches. No instrumentation belongs in timed calls.

Each process executes these families in order, using ordinary
`framework.compile(fn)` with no options:

1. `unaryflat`: `return (x.sin() + x * 0.125).relu()`; extents 509, 32749.
2. `binaryflat`: `return (x + y) * 0.75 - x * 0.125`; extents 509, 32749.
3. `nested`: `x,y=data['pair']; p=x*y; return {'out':(p+x,-p),'alias':x}`;
   extents 509, 32749.
4. `heldout`: `p=(x-y).cos(); return (p*x+y).sin()`; extents 1153, 49157.

Create one compiled callable per family and visit its extents in the listed
order. Inputs are contiguous CUDA float32 vectors, using the unchanged
`program-identity/consumer.py` `values` formula: first-call phase 0, five warmup
phases 1 through 5, then 17 sampled phases 8 through 24. Inputs are created
outside timing. Complete both sides of each timed call with the identical
`cudaDeviceSynchronize` barrier. Release first/warmup output owners before
sampling. Retain all 17 sampled output owners until sampling completes, then
materialize every output and verify ownership and input preservation. Retain
first-call outputs as encodings too. After both extents, revisit the first
extent with phase 25 on the same callable and record that churn call separately.
The factory, first-call, steady and churn costs remain separate.

Preserve all input/output encodings, raw nanosecond samples, median/quartiles/MAD,
source and build identities, commands/timestamps, actual installed wheel/native
bytes and hashes, runtime/compiler/GPU snapshots, and failed or partial attempts.
Native B/C outputs must match exact encoded finite bits, including signed zero;
reference comparison keeps the existing `1e-4 + 1e-4 * abs(reference)` finite
bound. Structure, dtype/device/shape/stride/grad and alias metadata match exactly.
Non-finite results fail this finite-input diagnostic. Every cell and both process
orders remain visible, including slow results. Ratios are observations, not a
score or a promise of improvement. Do not reroll unchanged timings.

Build and source changes must be visible: development repairs remain labeled
with actual dirty status and source snapshots; clean committed evidence must
bind its actual commit. B remains at the rejected commit. Historical artifacts
are never relabeled as repaired source. The verifier is offline and must reject
incomplete, reordered, differently bound, incorrect or failed legs.
