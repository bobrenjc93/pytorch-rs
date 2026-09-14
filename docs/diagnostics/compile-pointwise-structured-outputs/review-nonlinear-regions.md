# Shared nonlinear output repair

The reported signed-zero failure is valid. `p=x*y; q=-p; r=p.sin();
return (q,r)` needs rounded `p` before the sine call when the reference joins
those outputs, but independent contraction when it separates them.

The repair scopes normalization, observable uses, call order and contraction
to numerical regions inside one CUDA kernel and launch. It derives horizontal
fusion boundaries from the reference's ten-byte shared-read threshold. Returned
runtime producers connect dependent regions; zero-read outputs stay cheap.
Contractions respect values already rounded before a nonlinear call. Output
allocations, result reconstruction and executable cache identity are unchanged.

The partition certificate covers linear-address graphs with live sine/cosine,
at most ten live original SSA nodes (a conservative thirty-scalar-operation
bound), and at most four buffer-read boundaries per node. Larger graphs retain
joint lowering and admission with the same call-aware contraction rules. This
is a bounded numerical model, not general Inductor scheduler parity.

The permanent differential regression covers seven bodies, four return forms,
sizes 1/2/3/13/257, four value histories and two calls: 1,120 comparisons.
It includes the reviewer's three sizes, both return orders, one-input and
two-input thresholds, returned producers, sine/cosine, independent and shared
arithmetic, subnormals, overflow and finite cancellation. Native wrappers persist
across sizes and retain one executable and one kernel entry. Existing tolerances
and zero-sign assertions are unchanged. Additional controls cover larger graphs,
literal-derived and promoted-runtime-scalar-derived zero-read producers.

## Reference history limitation

A retained expanded diagnostic demonstrates a further parity limitation:
`p=x*x; q=-p; r=p.sin(); return (q,r)` after sizes 1 then 2 can return `q=-0`
at later sizes 3/13/257 under stock default compilation. A fresh compile at
those later sizes returns `q=+0`. The automatic dynamic recompile at size 2
retains its unguarded two-element optimization hint and separate kernels.
The native current-shape plan returns `+0` at sizes at least 3. It cannot match
both reference histories solely from current graph and shape.

The first repair probe, a new-wrapper-only retry and the final-wheel
`history-final.log` each retain 24 failures. New
functions from `exec` also share Dynamo's `<string>:1:f` profile identity.
The permanent cold-reference test calls public `torch.compiler.reset()` before
each shape; it then uses ordinary `torch.compile(fn)` with no overrides and
checks changed values and repeated calls. Reset clears Dynamo code/profile state;
it does not change the reference backend or numerical options. The original
history-dependent failures remain evidence and are not converted into passes.
Full history-sensitive numerical parity remains blocked: it would need a
separately approved account of reference compilation/profile history. No such
state or external-producer modification is added here.

## Provenance and retention

These are development checks of uncommitted repair source based on
`4f974c55f01b8cb22dfe27f39078db7b03116068`. They are not clean-commit
qualification. A final capture is required after Burner commits the repair.
Historical reports, including the clean `4554bbe8` capture, remain unchanged.

All commands, failures, raw outputs, native CUDA/PTX, reference compiler caches,
source contracts and wheels are retained under
`target/default-compile-eval/structured-outputs-nonlinear-repair/`.
The reviewer archive `structured-review-head.cnfJyh-retained.tar.gz` was copied
byte-for-byte into that canonical report root. No worktree cleanup or external
write was performed. Burner's report-root observer owns external archival.
No evaluator definitions, workloads, tolerances, denominators or progress
artifacts changed, and no evaluation score is claimed.

## Final development validation

The locked offline release build and source/import identity checks passed on
2026-09-14. The final wheel SHA256 is
`8c541d02d21983e1036500981d7d89b2df6f9aa314ea82fba3d52ec2454d959c`.
The native path ran before importing PyTorch. Captures record NVRTC 13.0,
CUDA runtime 13000, stock PyTorch `2.13.0+cu130`, the installed CUDA 12.6
`nvcc`, H100 UUIDs, driver, commands and selected library paths.

| Check | Passed | Explicit skips |
| --- | ---: | ---: |
| Pointwise suite, H100, `CUDA_VISIBLE_DEVICES=0` | 287 | 8 two-device tests |
| Dedicated device/restore tests, `CUDA_VISIBLE_DEVICES=0,1` | 8 | 0 |
| Release native pointwise/ownership tests | 47 | 0 |
| Python-conversion failure ownership test | 1 | 0 |
| Portable suite, each CPython version | 143 | 152 hardware tests |

Portable versions are 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5.
Hardware skips are not GPU passes. Native-extension verification,
`cargo fmt --check`, and Clippy with warnings denied passed. Separate fresh
processes captured the reported program at sizes 1, 13 and 257, with matching
values and zero signs, native CUDA/PTX and wheel/runtime provenance.

The [measurement bundle](review-nonlinear-regions-measurements.json.gz) retains
actual commands, source hashes, validation outcomes and runtime captures.
The [raw archive](review-nonlinear-regions-raw.tar.gz) preserves failed and
passing logs, numerical outputs, native code, scripts and copied source
contracts. The [retention manifest](review-nonlinear-regions-retention.json)
identifies the byte-exact source/reference-cache archives, reviewer archive
and every development wheel in the canonical report root. The final-wheel
history diagnostic remains separate from passing validation; no clean-commit
or full history-sensitive parity qualification is claimed.

Nested dispatch diagnostics are also retained byte-for-byte as
`target/default-compile-eval/structured-outputs-nonlinear-repair/nested-dispatch-reports.tar.gz`,
with per-file hashes in `nested-dispatch-retention.json`. This repair's runs are
`target/dispatch-smoke-g26hvqov` and `target/dispatch-smoke-5zn18bjt`; older
included roots retain their historical identities and receive no current credit.
