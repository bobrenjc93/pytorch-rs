# Structured-output numerical history repair

This is development evidence for uncommitted changes after `c13d660d1df7e31a85ecd20e9de40945f96ad772`, not a clean-commit qualification or an evaluation score. The compilation-history finding is repaired. The finite-result finding remains open, with a permanent strict differential regression. The numerical milestone is incomplete.

Each successful logical `Specialization` now freezes its numerical iteration hint. Generalized shape hits retain that hint; current dimensions still control validation, allocation, indexing, launch extent and returned metadata. One additional launch argument carries the hint through the existing native ABI. It does not enter the Graph/device/indexing executable key. Failed reconstruction leaves logical history and all cache orders unchanged. Tests retain both ordinary native and stock PyTorch wrappers across size changes, and separately exercise fresh wrappers.

## Remaining blocker

For `p=x*y; q=p-x; r=p.sin().sin().sin().sin().sin().sin().sin(); return (q,r)`, the native joint fallback still produces `1024.0` instead of reference `1192.0928955078125` at shape `(1,)`, `x=1e10`, `y=1.0000001192092896`. `test_nonlinear_large_unreturned_producer` exposes this without returning `p`, weakening tolerances or marking the failure expected.

Replacing the ten-node certificate needs more than a larger bound. Stock realization uses per-expression CSE operation/read counts, and subsequent fusion can leave rounded internal values between regions. The current output-slot-only lowering cannot represent those imports.

A further H100 probe established that output order is numerical input. Its body starts with `p=x*y; q=p-x; r=p`, then repeats 64 segments of 30 `r=r.sin()` statements followed by `r=r.sin()+r.cos()`, then one final `r=r.sin()`. Both forms are admitted: 2,117 native nodes, 8,211 instructions, identical Graph output slots `(3,2116)`, different immutable result specifications. Ordinary stock `torch.compile`, with fresh wrappers and no backend/mode/fullgraph/dynamic overrides, gives:

| Return order | `q`, at shape `(2,)` with the same finite inputs above |
| --- | ---: |
| `(q,r)` | 1024.0 |
| `(r,q)` | 1192.0928955078125 |

The reference's default inference `fx_passes/post_grad.py::reorder_for_locality` reorders producers from output traversal order. The retained transformed FX and pre/post-fusion IR show the resulting difference. Merely planning from the canonical Graph and element hint cannot distinguish these cases.

Graph-only **executable caching** remains possible in principle. A complete replacement needs an immutable projection of observable root order plus a native representation of realization, region dependencies and rounded imports/exports. One graph-specialized kernel would then need to accept bounded region/contraction plans while preserving one launch and invocation-local ownership. Hiding topology-specific compiled kernels behind one executor would violate the cache requirement. This is a material change from the current static-expression kernel; it has not been implemented or represented as a passing repair.

## Retention and checks

Raw commands, failures, numerical outputs, generated CUDA/PTX, reference compiler artifacts, import/wheel/runtime identities and source hashes are retained beneath the canonical report root:

`target/default-compile-eval/structured-outputs-history-repair/`

`reviewer-original.tar.gz` preserves all 1,944 files from the original review directory byte-for-byte, with a member SHA256 manifest. The order investigation also preserves its rejected hypotheses and the failed unindexed-device diagnostic separately from the retry. No existing historical measurements or canonical evaluation definitions were changed. Clean-commit qualification must follow a Burner commit; these dirty-source runs cannot substitute for it.

The source-matched native audit executed both return orders through one persistent wrapper after replacing only its code's return topology. It reused the same executor object and kept one executable entry, returning `q=1024.0` in both cases. The `(r,q)` call therefore disagrees with the reference. Both native CUDA and PTX files are retained. This is additional blocker evidence, not a performance measurement.

Checks on the source-matched release wheel:

- Pointwise suite: 291 passed, eight explicit two-device skips, one failure—the new size-1 finite regression.
- Targeted history/cache/admission suite: 11 passed.
- Dedicated two-device checks: two passed; native pointwise release checks: 48 passed; conversion ownership: one passed.
- CPython 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5: 145 portable checks passed and 155 hardware checks skipped on each interpreter. These skips are not GPU passes.
- Clippy with warnings denied, formatting, native-extension provenance and archive member hashes passed.

The wheel SHA256 is `e922fe0e0e5800e304a7b648fddba1b1bfa7fe9456ed6128e3e5aa0ba4ffdfc8`. The runtime capture records PyTorch 2.13.0+cu130, native NVRTC 13.0, CUDA runtime 13000, H100 device identity and the selected nvcc/compiler versions. Production source hashes match the release build; three test files changed while that build ran, which is recorded without altering its original provenance.

[Measurements](measurements.json.gz), [focused raw capture](raw-captures.tar.gz), [retention manifest](raw-retention-manifest.json.gz) and [verification](verification.log) accompany this record. The complete byte-exact archive, including compiler caches and all original raw failures, is retained at `target/default-compile-eval/structured-outputs-history-repair/full-archive/raw-captures.tar.gz`; the manifest records its hash. The checked-in capture excludes duplicate compiler caches and binaries to keep review size bounded. No disposable-worktree cleanup was performed.
