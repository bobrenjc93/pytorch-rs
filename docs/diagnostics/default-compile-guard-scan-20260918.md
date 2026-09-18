# Default compiler guard traversal: author diagnostics

This author revision starts at `8c9ee82b0e37f6626b05e37d06f66a1a8774b7a8`.
Baseline timings used a fresh release wheel from that clean head; revised timings
used a second release wheel from the **uncommitted source edits**. These are
developer observations, not clean-commit evidence, canonical scores, review or
release approval. Earlier committed measurements remain unchanged.

The change removes temporary owned key/value references from native exact-dict
key validation, inside the existing callback-free critical section. The Python
frontend reads each current Tensor class namespace once per guard pass instead
of once per method. Every key and method identity is still checked on each call;
there is no cached validity verdict. Numerical kernels, preparation/launch,
view planning, mutation, cache publication and supported programs are unchanged.
The baseline profile identified both scans among recurring public-call costs.

## Fixed developer comparison

Five programs (negation, multiply/add/ReLU, trigonometric composition, mixed
computed/view results, and pure views) each traverse scalar, empty, singleton,
257-element vector, 17×31 matrix and 512×512 matrix shapes. Inputs use seed
20540918 and float32 values in [-1, 1], followed by fresh inputs offset by 0.125.
Both frameworks use ordinary `compile(fn)` with no compiler arguments. Each
build runs native/reference and reference/native orders in separate processes:
10 warmups, 17 samples of 16 public calls, the same CUDA runtime device barrier
before each sample and after every call, one host math thread. Input creation
and full output readback are outside timing. Setup calls are recorded separately.
Compiler caches started fresh for this task and were reused across processes;
setup times are therefore **not cold-cache comparisons**. Build order is baseline
then revision, so temporal drift remains a limitation.

The table gives medians across six shapes of the two-order per-cell medians:

| Program | Baseline µs/call | Revised µs/call |
| --- | ---: | ---: |
| Negation | 31.871 | 28.046 |
| Multiply/add/ReLU | 35.761 | 31.969 |
| Trigonometric composition | 36.499 | 33.124 |
| Mixed computed/views | 41.304 | 37.930 |
| Pure views | 24.226 | 20.438 |

All 30 native cells had lower two-order medians, by 2.664–5.199 µs/call. The
reference's median relative shift was -1.96%; all reference samples and slower
individual samples are retained. All 240 process/cell records completed, and
full first/fresh output arrays were byte-identical across builds and frameworks;
shape, stride, offset, dtype, device and gradient metadata matched. The verifier
also applied the existing pointwise `rtol=1e-5`, `atol=1e-6` limits. Repeated view
references retained wrapper identity. This finite diagnostic does not establish
general compiler parity or predict canonical performance scores.

## Checks and provenance

| Check | Outcome |
| --- | --- |
| Focused namespace/method guard tests | 10 passed |
| CUDA-hidden pointwise suite | 208 passed, 195 skipped |
| GPU0 pointwise suite | 393 passed, 9 skipped, one diagnostic interpreter-path failure; that unchanged test passed after local setup recovery |
| CUDA mutation and add/subtract owner tests | 39 passed |
| README contract checks | 12 passed after shortening a Scope cell; original failure retained |
| Release wheels, import/source verification, formatting, Clippy with warnings denied, documentation links | Passed |

The [raw receipts](default-compile-guard-scan-20260918.tar.xz) retain the exact
commands and diagnostic scripts, all samples and full arrays, source diff,
wheel/import hashes, profile, complete test logs and hardware snapshots.
They originate in the worktree's `BURNER_EVALUATION_ARTIFACT_DIR` at
`target/performance-polish/evidence`. Identical payloads are losslessly stored
as tar hard links. No existing diagnostic, benchmark or evaluator was edited.

GPU0 was H100 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07.
The driver UUID agrees with the selected physical inventory. Release builds used
Rust 1.92.0, PyO3 0.29.2 and Python 3.12.12; reference PyTorch was 2.13.0+cu130.
The loaded CUDA runtime and NVRTC were 13.0; installed nvcc 12.6 was inventory,
not the pointwise compiler. Prefix/destination checks and source/wheel/import
hash comparisons passed for the worktree-local environment and both wheels.

Failures are retained: the first dependency export used an unset cache-variable
expansion and failed before installing dependencies; the initial profiler filename
shadowed stdlib `profile`; one GPU diagnostic rejected the standard venv symlink
resolving to the read-only base interpreter; copying just that executable lacked
its standard library. Copying the complete Python distribution into this worktree
and recreating only the venv interpreter/configuration resolved that preflight;
executable bytes stayed identical. This recovery occurred **after** the timing
runs and did not rerun them. The README's word-budget failure required only a
prose correction. No shared environment was repaired or modified.

This Linux/Python 3.12/GPU0 capture does not reproduce macOS or Python 3.14 CI,
or a free-threaded Python run. Frozen evaluator inputs, dependency locks,
Burner-managed progress and all prior diagnostic artifacts remain unchanged.
No canonical evaluation, commit, push or publication was performed.
