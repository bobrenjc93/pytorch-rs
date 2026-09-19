# Clean-commit three-way public-call capture

This non-scoring H100 capture measures retained commit
`ed6146650aa1bf8d5ba13a17337b6331372c400c`, accepted main
`60202557b4f110d07777f585e804ab5f55e1ff7b`, and incoming implementation
`9a436482d3e49dd40df7ce1d75198890b16b8f95`. It fulfills the deferred clean-commit
capture after the [author investigation](profiled-full-call-20260918.md).
Historical timings, unsuccessful experiments and author receipts remain unchanged.

All twelve processes recorded a clean worktree at `ed614665`. Accepted/incoming
sources were fresh git-archive exports inside this worktree, verified byte-for-byte
after capture. Each revision received one fresh release build with `--locked`,
an explicit local Python interpreter, and separate wheel/import/build directories.
No environment installation was needed. The bundle retains exact wheel bytes,
source/import/library hashes, commands, build logs and test output.

The unchanged [developer diagnostic](../../scripts/diagnose_compile_full_call.py)
uses ordinary `framework.compile(fn)` defaults for both frameworks: eight
independent programs, 76 cells, fresh/reused inputs, five warmups and 17 separately
synchronized single-call samples. Both framework orders use separate processes
and fresh CUDA/Inductor/Triton caches. Input construction and complete readback
are excluded consistently; public wrapper work and synchronization are timed.
Wrapper construction and first-shape calls are recorded separately. Native
processes block PyTorch imports and check warm original-body replay outside timing.

Both orders passed output, metadata and alias comparisons for all supported cells:
accepted main **58 checked / 18 unsupported**, incoming and retained **76 checked**
each. All reference cells passed. There were **14,892 saved timing samples**, no
capture errors, and no discarded slow cells or retries. Saved arrays contain no
zero, NaN or Inf values, so their signed-zero checks are vacuous and do not
establish exceptional-value numerical coverage or runtime line coverage.

Descriptive retained-versus-prior observations (first/second framework order):

| Prior revision | Common supported cells | Retained lower latency | Median per-cell reduction |
| --- | ---: | ---: | ---: |
| Accepted main | 58 | 57 / 55 | 4.628 / 4.176 µs |
| Incoming | 76 | 53 / 66 | 0.772 / 1.473 µs |

The remaining cells were slower. Every cell's samples and medians are preserved;
these unweighted diagnostic summaries are neither canonical scores nor universal
speedup/parity claims. No canonical evaluator was run. Independent review and
ordinary evaluation remain separate gates.

The selected device was GPU0 H100 UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07, under the owner's
existing GPU/cpu-heavy reservation. CUDA runtime and actual native NVRTC were
13.0; nvcc 12.8/V12.8.93 was inventory, not the pointwise compiler. Python was
3.12.12, Rust 1.92.0 and upstream PyTorch 2.13.0+cu130. Per-process device snapshots,
compiler options and source hashes are in the receipts; idle snapshots alone do
not establish exclusivity. The focused portable diagnostic/documentation suite
passed **25 tests with no skips** against the fresh retained wheel. This Linux
validation does not reproduce macOS or Python 3.14 CI.

The [3,384,584-byte bundle](default-compile-full-call-postcommit-ed614665.tar.xz)
has SHA256 `144f4bd138dae9020a28e9d51c41d86a1cc737ee0448a722e7e3681ec9d97bca`.
Its 12 freshly captured NPZs exactly match array bytes in the
[committed raw archive](default-compile-full-call-20260918-raw.tar.xz); the manifest
maps those bytes without substituting historical timings or provenance.
The unchanged archive-local helper verifies the base and every payload.
[Offline verification](default-compile-full-call-postcommit-ed614665-replay.json)
restored all arrays and passed all six comparisons with original NPZ paths blocked.

```bash
mkdir -p target/full-call-ed614665-replay
tar -xJf docs/diagnostics/default-compile-full-call-postcommit-ed614665.tar.xz -C target/full-call-ed614665-replay
.venv/bin/python -B -I target/full-call-ed614665-replay/profiled-call/bundle.py restore \
  target/full-call-ed614665-replay/profiled-call \
  docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz
for build in accepted incoming retained; do
  .venv/bin/python -B -I scripts/diagnose_compile_full_call.py --compare \
    "target/full-call-ed614665-replay/profiled-call/$build-reference-second.json" \
    "target/full-call-ed614665-replay/profiled-call/$build-native-first.json"
  .venv/bin/python -B -I scripts/diagnose_compile_full_call.py --compare \
    "target/full-call-ed614665-replay/profiled-call/$build-reference-first.json" \
    "target/full-call-ed614665-replay/profiled-call/$build-native-second.json"
done
```
