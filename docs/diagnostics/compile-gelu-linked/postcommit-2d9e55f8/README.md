# Clean-commit GELU evidence

Measured `2d9e55f820f49efd817f3d52b7cbc9c69894d3ad` with a clean worktree on
2026-09-16. The fixed gate built a fresh release wheel, verified its installed
extension, and ran the unchanged complete corpus. The five public diagnostic
legs then used that same wheel. Documentation and archive assembly began only
after all measurement processes completed. This is measurement evidence, not
independent implementation review or merge approval.

| Check | Measured result |
| --- | --- |
| Fixed weighted public-default coverage | **21.0%**, 22/112 cells pass |
| Fixed CUDA performance | **33.45197861005538/100**, 22/56 cells pass |
| CUDA common-success geometric mean ratio | 1.1075472784632787 |
| Ordinary eager versus PyTorch eager | 33/33 comparisons pass |
| Public default compile versus public default compile | 236/236 comparisons pass |
| Native eager with absent NVRTC versus PyTorch eager | 33/33 comparisons pass; no NVRTC mapping |

Both fixed CUDA GELU variants pass; CPU GELU remains unsupported. Every fixed
cell, unsupported outcome, timing sample and slow result is retained. The fixed
report records `valid=true`, `diagnostic=false`, five warmups, 17 samples and
both CUDA implementation orders. Its performance result is below the supplied
34 baseline; these measurements are not resampled to obtain a higher score.

The public probe uses one unconfigured compiled wrapper for each complete
history, with constants, scalar/shape histories, nested/shared/competing
products, realization thresholds, post-call inputs and retained outputs. Eager
and compiled comparisons remain separate. Exact-byte, metadata, context,
pointer and retention checks accompany the raw captures and audit.
The [independent raw audit](independent-raw-audit.md) verified 1,871 raw files,
590 retained snapshots and committed source identities. All public pairs have
zero finite-bit and signed-zero differences; the compiled pair has 1,924
NaN-word differences, without claiming payload equivalence.

## Timings and environment

Public-probe steady medians below are microseconds at **257 / 131072 elements**.
This separate one-order diagnostic uses five warmups and 17 synchronized
samples; it does not replace the fixed performance gate.

| Public workload | Native | Matching PyTorch mode |
| --- | --- | --- |
| Eager GELU | 10.596 / 11.017 | 10.516 / 10.416 |
| Compiled GELU | 35.403 / 37.968 | 26.059 / 30.967 |
| Eager affine control | 18.618 / 19.069 | 15.985 / 15.904 |
| Compiled affine control | 30.637 / 32.589 | 29.915 / 31.527 |
| Compiled trig control | 45.469 / 35.504 | 31.026 / 31.968 |
| Eager GELU without NVRTC | 10.025 / 10.446 | 10.516 / 10.416 |
| Eager affine without NVRTC | 17.406 / 17.917 | 15.985 / 15.904 |

First small GELU calls were 12.083 ms native versus 7.157 ms PyTorch eager,
and 119.659 ms native versus 1872.217 ms default Inductor. Cold calls include
lazy module loading or compilation/linking; later rows can inherit process
warmup. Full cold and steady observations remain in the reports.

All work used `CUDA_VISIBLE_DEVICES=0`, physical H100 UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07 and PyTorch
2.13.0+cu130. The public native runtime is CUDA 13.0; compiled legs load NVRTC
13.0 and the explicit no-NVRTC eager leg loads none. Tool/library hashes and
before/after GPU snapshots are recorded. No foreign jobs were interrupted.

## Artifacts and reproduction

- [Fixed report](fixed-report.json) is copied byte-for-byte from the successful
  gate, including source/build/import identity, setup durations and all cells.
  [Fixed worker archive](fixed-workers.tar.zst) retains **all six complete raw
  worker JSON byte streams**, logs and the original report, without compiler
  cache directories. [Transport identities](fixed-transport.json) bind original
  gzip paths/hashes to the exact decompressed JSON bytes in the archive. Only
  compression changes; JSON is never parsed/reserialized for transport and
  embedded provenance is unchanged. Original gzip files remain at their
  recorded worktree-local paths. Extract inside a local scratch directory with
  `zstd -dc --long=29 fixed-workers.tar.zst | tar -xf -` (512 MiB window).
  JSON member names omit the original `.gz` suffix. The
  [stream check](transport-stream-check.json) verifies exact identity with
  the original complete tar stream after recompression.
- [Public archive](public-essential.tar.xz) and
  [member identities](public-manifest.json.xz) retain raw float32 storage bytes,
  current/prior retained-output snapshots, declarations, actual process exits,
  source/provider/executor identities and selected passive cache files. The
  archive is a lossless recompression of the existing probe's `pack` output.
  Extract it with `tar -xJf public-essential.tar.xz`.
  [Comparison](public-comparison.json) contains all public timing rows.
- [Wheel check](wheel-check.json) verifies the installed extension, all 60
  Python sources and five vendor/license members against the fresh wheel and
  this clean checkout. Its SHA256 is
  `9e21a312acd09aa155c0d5e2f7984a063300d378110534fc4c9d8781adcb6031`;
  the loaded extension SHA256 is
  `d335da08693751bc75c187016cae210d45b789903f7d52a3310be68ea55b1ca0`.

[Fixed member checks](fixed-member-check.json) verify all 14 retained members
and exact decompression of all six original gzip workers;
[public member checks](public-member-check.json) verify all 2,008 members and
the original tar-stream identity. [SHA256SUMS](SHA256SUMS) covers the final
evidence files. These are archive checks, not additional benchmark runs.

From a clean checkout of the measured commit, use the existing documented
worktree-local environment/cache setup and canonical GPU0 lease:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric both --output target/fixed-report.json
```

The supported `both` mode runs the complete coverage and CUDA-performance
measurement in one invocation; it does not replace Burner's later evaluation
lifecycle. [The command receipt](fixed-command.json) and
[captured build/run log](fixed.log) record this run's setup and exit.
Then follow the [public-probe reproduction commands](../README.md#reproduction-and-generation)
with a fresh output directory and the newly installed local interpreter.
[Exact public commands and observed exits](public-command.json) are retained.
The no-NVRTC leg sets its missing-library override itself. Do not reuse caches,
change a matrix, or configure a reference backend between calls.

Earlier private prerequisites, generation experiments and development captures
remain pinned in the parent directory; their measurements were not rewritten
or used as current performance credit. No implementation, test, dependency,
evaluator, tolerance, denominator or progress artifact changed in this step.
Canonical independent review, full qualification and merge decisions remain
Burner's responsibility.
