# Independent Stage 1 factual audit

Reviewer: `/root/erf_failure_audit`, using Moduler's code-review prerequisite
factual-review allowance. Read-only inspection; no GPU execution or edits by
that reviewer. This report preserves its findings, not an implementation approval.

The prerequisite failed under the unchanged comparator.

- Accounting: 44 cases, 113 calls, 335 output leaves, 2,324 differing float32
  words, exactly four failing leaves. Native/reference input records and scalar
  histories agree. Every output binary matches its recorded SHA256 and words.
  The comparator matches accepted main
  `60202557b4f110d07777f585e804ab5f55e1ff7b` byte-for-byte.
- Failures: `products_first`, steps 0/1, leaf 1; `sum_first`, steps 0/1, leaf 0.
  Each has six violating elements: x=3.75, 4, 4.25, 5 and adjacent float32
  values around 4. The second call reverses positions. At x=3.75, native gives
  -0.0024852752685546875 and reference -0.0024871826171875: error
  1.9073486328125e-6 exceeds permitted 1.024871826171875e-6.
- The explicit SSA matches reference GELU decomposition and competing products.
  Constant materialization, canonical roots, output permutations, native stores,
  and report ordering agree. No mapping mistake explains the failures.
- The diagnostic calls existing `_pointwise_compile`, preparation and native
  execution. Its generated kernel is the generic scalar-program interpreter
  with private Erf. It does not execute the disassembled plan as a substitute
  interpreter. Plan/source consistency and retained-output assertions completed.

The strongest explanation is an upstream Erf difference amplified by
cancellation. Native and linked reference PTX both contract the final sum as
`fma(x, y, rounded_g_squared)`; the separately returned square already differs.

The reference wrapper in the archive is
`attempt-002/reference/products_first/0-wrapper-0.py`. Its generated source
identity, `cghf6jf7xztprdoluoyrxztdrpnufubqv3l3lk5r5jpeaqdjmdrn.py`, matches the
`.file` identity in
`attempt-002/reference/TRITON_CACHE_DIR/B3VUBIGHIPL4NQ5X7V6HVEJS6NWC3S5FWWNDBAGZFABSL4MEBYCA/triton_poi_fused_add_gelu_mul_0.ptx`.
The adjacent `__grp__triton_poi_fused_add_gelu_mul_0.json` retains group links.

Native `products_first/0-native.ptx` and that reference PTX use different
large-argument Erf polynomial coefficients. Recovering Erf from the recorded
`(erf(x*kAlpha)-0.5)*1048576` output exposes one-ULP differences at x=3.75, 4,
4.25 and 5. Separately rounded GELU arithmetic from those values reproduces
each implementation's standalone GELU. This supports the toolchain/Erf
explanation, but establishes no acceptable general fix.

Limits remain material:

- Source/group associations are not per-call selected-kernel or SASS traces.
- The 27 constant cases, one scalar sequence and six threshold sequences are
  finite evidence. Repeated constant lanes are not distinct arguments.
- Native scalar promotion and shape hints are supplied through private owners;
  these are not public frontend cache/guard tests.
- The native report's `extension` field is the package `__init__.py` path.
  The author's separate `build-and-checks.json` hashes the actual copied native
  binary; that later receipt was not part of this reviewer's original check.
- Native NVRTC 13.0 and available nvcc 12.6 are different compiler identities.
- No public GELU, eager image, no-NVRTC eager operation, performance or fixed
  scoring qualification is established.

Preserving this failed experiment and stopping broad surface implementation is
justified. This shows failure of the integrated candidate, not impossibility
of every implementation within the existing architecture.
