# Guard allocation investigation (author sources)

Neither proposed optimization is retained. The metadata-copy removal was slower
in both framework orders; singleton-product and combined trials were inconsistent.
Production `python/` and `src/` match incoming
`64d6d79b33b2233962530bd6df51488fb321c9b9`. This output adds guard contract tests
and documentation navigation, not a performance improvement or authority for an
unchanged-code CUDA evaluation rerun.

## Measurements and decision

The unchanged [full-call diagnostic](../../scripts/diagnose_compile_full_call.py)
ran its eight independent programs and all 76 cells per build/order, including
fresh and reused inputs. Both frameworks used public `compile(fn)` defaults in
separate processes and compiler caches. Each cell has five warmups and 17
single-call samples, synchronization around the public call, complete output and
alias checks outside timing, and separately recorded cold cost. Native processes
blocked PyTorch imports. These are descriptive developer observations, not scores.

Accepted `60202557b4f110d07777f585e804ab5f55e1ff7b`, incoming `64d6d79b`, and each
prototype used fresh release native builds and separate extracted wheels.
Prototype patches, source hashes, wheel bytes, native hashes, exact commands and
all samples are retained in the [evidence bundle](guard-allocation-20260918.tar.xz).
The author checkout was dirty; archived baseline source identities are verified
separately. No capture is presented as clean post-commit evidence.
The [final validation receipt](guard-allocation-20260918-validation.json) records
the archive hash, byte-preserving recompression, relocation replay and final
quickstart checks.

Median of matched cell latency reductions versus incoming, in microseconds
(positive means lower latency; these are not confidence estimates):

| Trial | Native first | Native second | Cells lower than incoming | Decision |
| --- | ---: | ---: | --- | --- |
| Reuse six-field metadata tuple | -0.516 | -0.511 | 24/76; 26/76 | Discard |
| Both proposals | -0.371 | +1.032 | 24/76; 58/76 | Discard |
| Singleton index-bound product only | +0.196 | -0.356 | 43/76; 36/76 | Discard |

All ten corrected reference/native pairs passed the existing comparator:
accepted main had 58 supported and 18 unsupported cells per order; incoming and
each prototype had 76 supported cells. All slower cells and accepted-main
comparisons remain in `summary.json` and `singleton-summary.json`. Differences
from accepted main include prior retained changes and do not establish a gain
from these proposals. No profiler result or historical score is used to attribute
an allocation hotspot or explain a scored regression.

GPU0 was H100 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07,
under the owner's GPU/CPU-heavy lease. Corrected captures loaded CUDA runtime and
NVRTC 13.0 with compute_90; nvcc inventory was 12.8/V12.8.93, not the pointwise
compiler. Python was 3.12.14+meta and upstream PyTorch 2.13.0+cu130. Per-process
receipts retain toolchain versions, runtime paths, NVRTC version/options and
before/after device snapshots. The final validation receipt separately timestamps
library file hashes collected after the captures. Dependencies and imports were
copied/selected into this worktree; the shared pytorch-env was untouched.

## Failures and validation

The first complete four-build attempt selected fallback NVRTC 11.4, which rejects
compute_90. Its numerical cells failed; it is not performance evidence. An untimed
explicit-NVRTC-13 probe then exposed a missing builtins search path. Selecting
the local NVRTC and its local library directory fixed setup; the complete matrix
was repeated once in fresh processes/caches with identical source and wheels.
`failed-attempt.tar` preserves every original receipt, array, log and recovery
probe; `attempts.json` distinguishes this recovery from the measured trials.

Each prototype passed 15 portable guard tests, including the proposed branches.
With production restored, compiler portable tests passed (408 run, 195 skipped),
and replay/documentation tests passed (25 run). The GPU suite ran 206 tests with
four skips: five isolated subprocess tests initially could not import the
extracted wheel. Installing that exact wheel into the verified local virtualenv
resolved four; one archived diagnostic additionally required a byte-identical
local interpreter copy instead of a standard virtualenv symlink. All five then
passed. A failed retry with duplicated unittest identifiers is also preserved.
These Linux checks do not reproduce macOS or Python 3.14 CI.

Nine native CUDA graph bridge tests passed. The first link attempt failed because
this Python's sysconfig names an absent static archive; an explicit test-only
PyO3 configuration linking a worktree copy of its existing shared library passed.
Both commands, library identity and complete outputs are retained.

The tests retain exact index-bound products/cardinalities, zero-before-broadcast
incompatibility, native offset metadata and five-field persisted observations.
The existing admission, cache, alias, mutation and numerical suites are unchanged.
Saved diagnostic arrays are finite and nonzero; their signed-zero comparison is
vacuous and provides no exceptional-value coverage.

## Offline replay

Use a worktree-local Python with NumPy. The unchanged archive-local helper checks
payload hashes and the committed historical base; this bundle embeds its NPZs.
The existing comparator verifies whole-NPZ and individual array content hashes.
Original absolute paths remain provenance, not replay dependencies.

```bash
mkdir -p target/guard-allocation-replay
tar -xJf docs/diagnostics/guard-allocation-20260918.tar.xz -C target/guard-allocation-replay
.venv/bin/python -I target/guard-allocation-replay/profiled-call/bundle.py restore target/guard-allocation-replay/profiled-call docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz
for build in accepted incoming metadata combined singleton; do
  .venv/bin/python -I scripts/diagnose_compile_full_call.py --compare target/guard-allocation-replay/profiled-call/$build-reference-second.json target/guard-allocation-replay/profiled-call/$build-native-first.json
  .venv/bin/python -I scripts/diagnose_compile_full_call.py --compare target/guard-allocation-replay/profiled-call/$build-reference-first.json target/guard-allocation-replay/profiled-call/$build-native-second.json
done
```

The nested failed-attempt tar uses `tar -xf` and the same restore procedure;
its comparator errors are expected. Historical payloads were not rewritten.
No canonical evaluator or clean-commit performance capture was run. Any future
retained executable improvement still requires separately admitted post-commit
evidence and independent review.
