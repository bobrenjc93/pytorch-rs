# Returned-product contraction repair

Independent review of `597d1f3` found that storing a returned multiplication
changes the reference's choice between competing FMA candidates. For
`p=(-x)*y; return (p,p+x*y)`, the native consumer produced
`-168.0928955078125` where ordinary PyTorch 2.13 default compilation produced
`+168.0928955078125`. Overflow infinity signs and a trigonometric competitor's
zero sign also differed. Both return orders failed.

The repair counts output stores as observable uses during canonical sign
normalization and normalized contraction selection. A single-use direct product
takes priority over a shared competitor. Shared products remain eligible for
fusion, with the existing rank/sign fallback when neither candidate has priority.
Result topology, allocation ownership and native executable cache keys are unchanged.

Contractions are selected from consumers toward operands. Selecting an FMA
replaces a product edge with its factor edges and releases any unused wrapper
chain before considering inner arithmetic. This matters for a singleton result
such as `p=x*y; q=(-x)*y; return (p+q)+p`: the outer FMA makes the inner product
single-use. Static counts alone failed this case and four existing signed-sharing
subtests. Those original failures are retained.

The permanent differential regression covers 13 bodies and seven return forms,
including singleton, reordered, repeated and separately computed leaves. It uses
257-element inputs with cancellation, overflow, signed zeros, subnormals, infinities,
NaNs and seeded finite tails. Positive-zero subtraction, negative coefficients,
trigonometric competitors and additional arithmetic consumers are included.
Existing tolerances and zero-sign assertions are unchanged. Rust regressions
also check output-use priority, CSE aliases and outer-to-inner use updates.

## Evidence status

This repair is based on `597d1f3` and is validated from uncommitted source.
Its release wheel and source hashes identify those bytes; they are **not** a
clean-commit qualification. A fresh clean-commit capture remains required after
Burner commits the repair. The initial
[36954660 capture](postcommit-36954660/README.md) remains byte-for-byte preserved
under its original provenance and does not qualify this repaired implementation.

Raw commands, failure logs, numerical outputs, native CUDA/PTX, reference caches,
wheel and source provenance are retained under
`target/default-compile-eval/structured-outputs-review-fix/`. The original
reviewer's 1,184 files are byte-exact archived there as `original-review.tar.gz`
with a manifest. All writes remained inside this worktree; no cleanup or external
archive location is claimed. Burner's canonical report-root observer owns external
archival. Official evaluators, scoring, tolerances and denominators are unchanged;
no score is claimed.

## Final repair validation

The final release wheel passed these checks on 2026-09-14:

| Check | Passed | Explicit skips |
| --- | ---: | ---: |
| Complete Python pointwise suite on H100 (`CUDA_VISIBLE_DEVICES=0`) | 284 | 8 device tests |
| Dedicated device/restore checks (`CUDA_VISIBLE_DEVICES=0,1`) | 8 | 0 |
| Release native pointwise and ownership tests | 41 | 0 |
| Portable pointwise suite, each CPython version | 143 | 149 hardware tests |

Portable versions were 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware
skips are not GPU passes. The H100 suite includes all 91 new differential
combinations. Native-extension verification, `cargo fmt --check` and Clippy with
`--all-targets --features python-bindings -- -D warnings` passed.

The source-bound, locked offline release build used Rust 1.92.0 and Maturin 1.15.0.
Final wheel SHA256:
`d62cb19bbcd3048adcbd72bddf960976b7841942c1f1a7e7a8461623603bf794`.
The installed package and extension on every interpreter matched its wheel
members. Source hashes remained unchanged through build and validation.
The final build reused this report's Cargo cache, and correctness runs reused
reference compiler caches; no timing or cold-build performance claim is made.

Ordinary native `compile(fn)` and stock PyTorch `2.13.0+cu130` default
`compile(fn)` now both produce `+168.0928955078125` for the reported cancellation
case. The additional runtime capture verified the same signed zeros and infinity
signs, and confirmed no PyTorch import before native execution. Native kernels
used NVRTC 13.0 and CUDA runtime 13000 on H100, driver 580.82.07; PATH `nvcc`
reported 12.6.85. Exact GPU UUIDs, loaded libraries, commands, timestamps and actual
import paths are retained.

[Measurements](review-returned-products-measurements.json.gz) contain the final
build, validation and runtime provenance. The
[raw archive](review-returned-products-raw.tar.gz) preserves final and failed
runs, materialized numerical comparisons and generated native CUDA/PTX. The
[retention manifest](review-returned-products-retention.json) identifies byte-exact
source, nested diagnostic and reference-cache archives, original review evidence
and every development wheel in the canonical report root. Clean-commit capture
must be refreshed after Burner commits this repair; these development results do
not replace that gate or independent review.
