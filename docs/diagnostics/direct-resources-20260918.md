# Direct executable resource experiment

Author-only developer observations; no canonical evaluation or qualification.
The retained change integrates the exact-Program executable owner from donor
`cef34b14d865ee6c207120b1ee3701a8e5e26bd4`, then removes unused direct-only
instruction uploads and invocation scratch. VM resources, numerical planning,
launch geometry, completion and success-only cache publication remain in their
existing owners. See the [live compiler contract](../compile-pointwise-jit.md).

## Sources and capture

Four separately built release variants were measured:

- **B / accepted:** git-exported `60202557b4f110d07777f585e804ab5f55e1ff7b`.
- **H / incoming:** git-exported `a9045e8b14dd5ee63ff19537860514227c379692`.
- **integration:** H plus the donor integration, before resource removal.
- **elision-final:** integration plus direct resource removal and its tests.

The last two are **uncommitted author sources**, bound to archived source bytes,
patches, release wheels, verified wheel RECORDs and loaded native/frontend hashes.
They are not clean-commit evidence. B and H were freshly captured from verified
source exports in this worktree; no historical timing was relabeled.

The unchanged `scripts/diagnose_compile_full_call.py` (SHA256
`4c32f0329f7cd602517ffd7b34fcc324f5b6ce5c89c5dde3b82ba479224cb70e`)
used ordinary default `compile(fn)` on both frameworks: eight independent
programs, fresh/reused inputs, both framework orders, separate processes/caches,
five warmups and 17 synchronized single-call samples. Input construction and full
readback were outside timing. Wrapper creation and first-shape calls are recorded
separately. Native captures block PyTorch imports; warm body-replay probes run
outside timing.

All eight strict paired comparisons passed (572 compared cells). Across the
16 framework runs there were 1,180 successful captured cells, 20,060 timing
samples and 36 retained accepted-base rejections for views/mixed views. Full arrays, metadata, aliases, rejections and raw
samples are preserved. Exceptional-value coverage comes from the separate tests,
not from an assumption about these diagnostic inputs.

GPU0 was H100 `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07,
with `CUDA_VISIBLE_DEVICES=0` under the owner's lease. Loaded CUDA runtime and
NVRTC were 13.0; nvcc 12.8/V12.8.93 was inventory, not the pointwise compiler.
Reference was PyTorch 2.13.0+cu130, Python 3.12.14+meta and Rust 1.92.0.
Pre/post GPU inventory, actual library paths/hashes and exact commands/exits are
in the bundle. No shared Python environment was installed into or repaired.

## Observations and limits

These are unweighted geometric means of **baseline/candidate per-cell median
latency ratios**, over common successful cells in both orders; greater than one
means lower candidate latency. All unsupported and slower rows remain in the raw
reports. This is not the scoring corpus, a capped score or a general parity claim.

| Comparison | Fresh | Reused | Slower candidate cells, fresh / reused |
| --- | ---: | ---: | ---: |
| H / integration | 0.996× | 0.999× | 31/76 / 32/76 |
| integration / elision-final | 1.040× | 1.034× | 11/76 / 18/76 |
| H / elision-final | 1.037× | 1.032× | 16/76 / 22/76 |
| B / elision-final | 1.165× | 1.220× | 1/58 / 0/58 |

Integration alone did not establish a speedup over H. The combined change is
retained for the verified removal of unused resources and its bounded aggregate
observations, with material limitations: the 262,147-element affine reused cell
in native-first order was 0.603× versus H. Median first-shape-call costs also rose
from H's 274/278 µs (fresh/reused) to 340/435 µs. Those calls include compilation
or preparation where needed; they are not pure compile-time measurements. The
final reference/native descriptive ratios were 1.545×/1.425×, not universal wins.
No cause of a historical scored regression is inferred.

Separate `*-ownership.json` receipts record 166 owner/churn events per variant,
with full outputs and individual public-call costs. Actual live executable
objects are counted separately from distinct PTX hashes, outside timing. Exact
Program changes can increase module count (the nested history used three owners
versus H's one); equivalent Programs reuse their owner. Pure views retained zero
pointwise owners/preparations and selected receipt `None`. Direct scalar-affine
native preparation retention fell from 360 to 216 bytes, with unchanged counts
of six instructions/four registers. This does not bound total module memory or
prove device execution from a receipt.

## Validation and replay

Integration: 298 native tests; pointwise GPU 420 passed/9 skipped; CUDA-hidden
232 passed/197 skipped; identity/add_ 24 passed. Final elision: 300 native tests;
CUDA-hidden 232 passed/197 skipped; focused GPU 184 passed/3 skipped plus one
loader error, followed by all seven tests of the missing module passing.
Replay/quickstart: 25 passed. Formatting, Linux Clippy with warnings denied and
diff checks passed; this does not reproduce macOS CI.

The failed validation attempts are retained: the initial elision native-test build exited
101 because a new context test accessed a private field from the wrong module;
moving that test to the existing JIT test module fixed it. The targeted Python
run exited 1 because an existing sibling import needed the tests directory;
the missing module was rerun with that import path, without changing source or
wheel. No numerical mismatch or tolerance adjustment was involved. The initial
elision release wheel/source remains archived, but supplied no timing capture.
An initial packaging wrapper also rejected an unsupported `TarFile` compression
argument; the corrected XZ stream wrapper changes no captured data or manifest format.

[Raw bundle](direct-resources-20260918.tar.xz): **fa22bd33ced112cbfe6851cc1c69e35f5b8fc931fbeda88c20388b63cca3e68d**.
[Offline verification receipt](direct-resources-20260918-checks.json) records the
relocated replay and documentation checks. The bundle includes source snapshots,
all five attempted release wheels, source/wheel
mappings, tests, commands/exits, arrays, provider receipts, summaries and failures.
The existing byte-preserving bundle helper maps matching arrays to the immutable
[base raw archive](default-compile-full-call-20260918-raw.tar.xz); all other arrays
are embedded. The XZ envelope uses a 128 MiB dictionary; the existing helper
generates unchanged payload/manifest formats. Original absolute paths are
provenance, not replay dependencies.

From a checkout, using its NumPy-capable Python and a worktree-local directory:

```bash
mkdir -p target/direct-replay
tar -xJf docs/diagnostics/direct-resources-20260918.tar.xz -C target/direct-replay
.venv/bin/python -I -B target/direct-replay/profiled-call/bundle.py restore target/direct-replay/profiled-call docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz
.venv/bin/python -I -B scripts/diagnose_compile_full_call.py --compare target/direct-replay/profiled-call/elision-final-reference-second.json target/direct-replay/profiled-call/elision-final-native-first.json
```

Use the other seven recorded pairs identically; `comparison-summary-*.json`
contains exact commands and every descriptive row. The emitter and identity
sources retain the requested donor hashes. Protected files and prior historical
payloads are unchanged. A separately admitted clean post-commit release-build
capture, independent review and ordinary evaluation remain outstanding.
