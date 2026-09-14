# Shared sibling-product contraction repair

The review finding is valid. For `p=x*y; q=(-x)*y;
r=(p+q)+(p-q); return (p,r)`, the previous implementation contracted `p`
into both siblings. Overflow inputs produced NaN instead of the reference's
negative infinity; subnormal inputs exposed a zero-sign mismatch.

The repair compares remaining observable use counts when direct products compete,
including output stores and uses through transparent sign/positive-zero wrappers.
Previously it distinguished only single-use from shared products. The existing
consumer-to-operand updates now select `q` for both sibling contractions while
retaining rounded `p`. Equal-use rank/sign rules and the absence of a blanket
rounding barrier for returned products remain unchanged. Output ownership,
result reconstruction, executable caching and the single-launch ABI are unchanged.

The permanent regression checks five bodies, five return forms, three shapes
(1, 13 and 257), four input histories and two successive calls: 600 strict
native/default-PyTorch comparisons. It includes the reported expression, reversed
sibling and output order, a trigonometric competitor, extra sibling uses and
transparent positive-zero subtraction. Existing tolerances and signed-zero
assertions are unchanged. A native source regression checks both selected FMAs.

## Remaining numerical blocker

Broader diagnostics exposed an additional admitted program whose reference
semantics depend on kernel partitioning:

```python
def program(x, y):
    p = x * y
    q = -p
    r = p.sin()
    return (q, r)
```

For positive float32 inputs `x=y=1e-38`, ordinary PyTorch 2.13 default compilation
returns `q=-0` at shape `(1,)`, using two kernels, and `q=+0` at shapes `(13,)`
and `(257,)`, using one kernel. The final native wheel returns `q=-0` at all three shapes. The standalone
`blocker-confirmed.log` records four failed comparisons at sizes 13 and 257
across cold/warm calls; size 1 matches. The native numerical plan is shape-invariant.
Experiments modeling call boundaries fixed other nonlinear-sharing probes but
still could not match both partitions. Their source snapshots, failures, native
CUDA/PTX and reference compiler caches are preserved; that speculative analysis
is not included in the repair.

This requires a generic account of reference partition-sensitive numerical
semantics while preserving the approved single native launch architecture.
No shape-specific special case, weakened assertion, external producer change or
unsupported parity claim is included. The narrow review finding is repaired,
but the full structured-output numerical milestone remains **blocked** on this
architecture dependency. Passing the regression suite does not resolve it.

## Evidence status and retention

These are development measurements from uncommitted source based on
`fd5ecbc82ba523c05b78d39f18c77c19b9c015e0`. Source and wheel hashes identify the
actual repair bytes. A clean-commit capture remains necessary after Burner
commits a final implementation; these results do not qualify that gate.
Earlier measurements, including the `90de52a9` clean capture, remain unchanged
under their original provenance and do not qualify this repair.

All original failures, experimental sources, commands, generated CUDA/PTX,
reference caches, wheels and runtime provenance remain in
`target/default-compile-eval/structured-outputs-sibling-repair/` inside this
worktree. The reviewer's archive was copied byte-for-byte there as
`structured-review-124bb831-retained.tar.gz`. No disposable checkout was cleaned
up and no external archival write is claimed. Burner's canonical report-root
observer owns external retention. Official evaluators, tolerances, denominators
and progress artifacts were not changed; no score is claimed.

The [measurement bundle](review-sibling-products-measurements.json.gz) records
final build, validation, installed-wheel/runtime provenance and the standalone
blocker invocation. The [raw archive](review-sibling-products-raw.tar.gz) retains
both failures and successful checks, generated native code, diagnostic scripts,
and experimental source snapshots. The [retention manifest](review-sibling-products-retention.json)
identifies the byte-exact reference-cache, final-source and nested diagnostic
archives and all development wheels in the canonical report root.

## Final repair validation

The source-bound locked offline release build passed on 2026-09-14:

| Check | Passed | Explicit skips |
| --- | ---: | ---: |
| Python pointwise suite on H100 (`CUDA_VISIBLE_DEVICES=0`) | 285 | 8 device tests |
| Dedicated device/restore tests (`CUDA_VISIBLE_DEVICES=0,1`) | 8 | 0 |
| Release native pointwise and ownership tests | 42 | 0 |
| Native Python-conversion failure ownership test | 1 | 0 |
| Portable pointwise suite, each CPython version | 143 | 150 hardware tests |

Portable versions were 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware
skips are not GPU passes. The H100 suite includes all 600 new sibling-product
comparisons. Native-extension verification, `cargo fmt --check`, Clippy with
`--all-targets --features python-bindings -- -D warnings`, and source/wheel
identity checks passed. Every interpreter imported the final wheel's exact
Python and extension members. Source hashes stayed unchanged through validation.

The standalone reported-expression capture matches ordinary stock
`torch.compile` for cancellation, subnormal zero signs and overflow: its consumer
values are `[20000002048.0, +0.0, -Infinity]`. The native execution completed before
PyTorch was imported. The separate nonlinear probe remains failing as described
above; the passing suite does not resolve that limitation.

The build used Rust 1.92.0 and Maturin 1.15.0. Final wheel SHA256:
`696dcf602c2a68c46c126d8012210e7160773d1df169a86f4d45a8f565dd12bf`.
The native runtime used NVRTC 13.0 and CUDA runtime 13000 on H100, driver
580.82.07; PATH `nvcc` reported 12.6.85. GPU UUIDs, actual loaded libraries,
import paths, commands and timestamps are in the measurement bundle.
The release build reused Cargo artifacts. The full pointwise run used fresh
report-local reference caches; standalone diagnostics reused report-local
reference caches. No timing or performance score is claimed.

The first standalone blocker invocation failed to import the test helpers.
Its original script and failure log are preserved. Only the diagnostic's local
import path was corrected before the confirmed numerical reproduction; no
production code, test assertion or tolerance changed in that retry.
