# Clean-commit GELU and method-guard evidence

Measured clean commit `9009c8ea6de40ec55c970983602c8fadd8a0447a` on
2026-09-16. The unchanged full gate built and verified a new release wheel;
the public GELU and frozen guard diagnostics used that same installation.
All measurement processes completed before these documentation/evidence edits.
Historical prerequisites, failures, development runs and the prior clean
`2d9e55f8` capture remain unchanged.

| Capture | Result |
| --- | --- |
| Fixed public-default coverage | **21.0%**, 22/112 cells pass |
| Fixed CUDA performance | **33.433414668636416/100**, 22/56 cells pass |
| CUDA common-success geometric mean ratio | 1.1075886576018334 |
| Ordinary native eager / PyTorch eager | 33/33 comparisons pass |
| Public default compile / default PyTorch compile | 236/236 comparisons pass |
| Native eager without NVRTC / PyTorch eager | 33/33 comparisons pass; no NVRTC mapping |
| Guard diagnostic correctness / first-last timing outputs | 56/56 and 40/40 comparisons pass |

The fixed report is valid and non-diagnostic, with every unsupported cell,
five warmups, 17 samples and both CUDA orders retained. **CUDA performance
remains below the supplied 34 baseline.** This run was not repeated to seek a
higher result, and it does not approve the branch or replace canonical qualification.

## Public timing and numerical checks

Steady medians below are microseconds at **257 / 131072 elements**. These
separate eager and compiled diagnostics do not replace the fixed gate.
All first-call costs and steady comparisons are in [public-comparison.json](public-comparison.json);
raw timing samples remain in the indexed reports. First calls can inherit
process warmup from earlier rows, as specified by the unchanged probe.

| Workload | Native | Matching PyTorch mode |
| --- | --- | --- |
| Eager GELU | 9.965 / 10.396 | 10.146 / 10.105 |
| Compiled GELU | 34.792 / 35.634 | 27.011 / 29.064 |
| Eager affine control | 17.547 / 18.038 | 15.032 / 15.233 |
| Compiled affine control | 30.747 / 31.056 | 30.877 / 32.590 |
| Compiled trig control | 34.592 / 33.791 | 31.077 / 31.778 |
| Eager GELU without NVRTC | 10.707 / 11.027 | 10.146 / 10.105 |
| Eager affine without NVRTC | 18.478 / 19.089 | 15.032 / 15.233 |

The [raw-data audit](audit-records.json) verified 1,871 public raw records and
590 retained snapshots, including constant/scalar/shape histories and
realization thresholds. Finite float32 bits and signed zeros match in all
public pairs. The compiled pair has 1,924 NaN-word differences; this is not
a NaN-payload equivalence claim.

## Frozen guard comparison

One fresh original/changed/changed/original run retained all **340 samples**.
Each worker used one unconfigured wrapper over the five-shape history, without
profiling, reset or policy changes. All 20 warm cache snapshots were unchanged.
[Guard timing](guard-timing.json) contains every sample, MAD, range and paired
ratio; [guard correctness](guard-correctness.json) is recorded separately.

| Matrix shape (history order) | Original 1 median µs | Changed 1 | Changed 2 | Original 2 |
| --- | ---: | ---: | ---: | ---: |
| 128×256 | 37.546 | 36.005 | 35.003 | 52.579 |
| 193×256 | 36.295 | 36.104 | 35.894 | 37.517 |
| 202×118 | 38.198 | 37.637 | 37.547 | 37.267 |
| 207×292 | 37.236 | 36.805 | 37.227 | 38.067 |
| 128×256 return | 38.779 | 36.495 | 35.744 | 38.028 |

The second original worker's first-shape median is markedly slower than the
first original worker's. The second 202×118 pairing slightly favors the
original. All orders remain included. Two workers per version, differing
native binaries and order sensitivity do not establish a grouping-only speedup
or a precise confidence interval. Intermediate timed outputs were not recaptured.

## Provenance and retention

All GPU work used physical H100 GPU0
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07, one host thread
and sequential processes. PyTorch was 2.13.0+cu130. Public/guard native runs
used explicit CUDA 13 runtime/NVRTC; the no-NVRTC leg used an absent library.
Actual library/tool hashes, versions, inventory snapshots, commands and exits
are retained. No foreign work was interrupted.

The [wheel check](wheel-check.json) binds all 61 installed package members to
the fresh wheel and clean source. [License byte checks](license-source-check.json)
match all five packaged notices/licenses to source. Wheel SHA256 is
`22b66f5969a614ce4fd6b14f92ab06f007d5961206f37c7ee5d9ac246c00518c`;
native SHA256 is `d335da08693751bc75c187016cae210d45b789903f7d52a3310be68ea55b1ca0`.
The pinned original guard build remains read-only at clean `162abef6`, with
wheel `1f3998184e4e5a542ff480c111e00a241b9167ff35917e9c4825a12fe0f55bf8`
and native `b9db6f35d6129c9681b6d47e190558c5b1897ce997ad1c6c7dbe011249f420c2`.
Interpreter and third-party dependency metadata match; native binaries and
torch-rs package metadata differ. [The declaration](guard-declaration.json)
and worker reports preserve these actual build identities.

[Fixed report](fixed-report.json), [fixed command](fixed-command.json),
[build/run log](fixed.log), and [diagnostic commands](diagnostic-commands.json)
are byte-for-byte copies of generated records. Reproduce with the unchanged
`evaluate_torch_compile_default.sh --metric both --output <local-report>` command,
then the [public probe](../README.md#reproduction-and-generation) and
[guard protocol](../../compile-method-guards/README.md#reproduction-and-retention)
using that installed wheel and fresh worktree-local directories.

[Raw index](raw-index.json.xz) records paths, sizes and SHA256 hashes for the
complete raw worker streams, float32 bytes, source/executor/provider records,
process receipts, wheel and audit sources under this retained worktree.
It is XZ-compressed JSON; no measured content was rewritten. Compiler cache
trees and duplicate bulk raw archives are omitted from the tracked tree.
The indexed raw files and builds must remain available for canonical review.
[SHA256SUMS](SHA256SUMS) covers these compact evidence files. The independent
host audit reproduced all 112 fixed cells and both aggregates with the existing
comparison/scoring functions; it is not an independent implementation review.
No implementation, dependency, test, harness, evaluator or progress file changed.
