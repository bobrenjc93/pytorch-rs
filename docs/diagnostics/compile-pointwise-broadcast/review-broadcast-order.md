# Broadcast product-priority repair

The review reproduction was confirmed on H100 with PyTorch 2.13.0+cu130:

```python
def f(x, y):
    a = x + 1.0
    return x * y + a * a
# x = [[2e38], [-2e38]], y = [[2e38, -2e38]]
```

Before repair, native compilation returned all positive infinities; fresh default
Inductor returned `[[inf, -inf], [-inf, inf]]`. Materializing both inputs as
contiguous 2×2 tensors made both implementations return all positive infinities.
The original output and failing metadata assertion are retained in
[repair evidence](review-broadcast-order.json.gz), alongside the reference LLVM.

The reference emits a separate side-effecting cache-policy instruction before
nonconstant broadcast loads. LLVM's operand ranking counts it when deciding
which product in an addition contracts. Native lowering now derives that extra
materialization step from the checked address mapping. Scalar and linear loads
retain one step; adding singleton axes without expanding elements remains linear.
Existing liveness, sign normalization and FMA selection stay in the same pipeline.
No evaluator, corpus, tolerance or scoring denominator changed.

## Validation and provenance

This is development evidence from uncommitted repairs on `43f06fb`, explicitly
recorded as dirty. It does not replace clean candidate measurements. The archive
retains original failures, commands, test logs, source hashes, the built extension
identity, generated CUDA/PTX and review conclusions. All build, interpreter,
cache and capture paths belong to this worktree. The selected native tools were
NVRTC 13.0, CUDA runtime 13000 and `compute_90`; nvcc 12.6 was queried but not used
for JIT generation. GPU 0 was H100
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07.

The new regressions test arithmetic, ReLU and libdevice producers, both sum and
input orders, scalar/linear/broadcast addresses, singleton-only reshaping,
interior singleton dimensions, larger 17×33 and 17×65 outputs, dead/shared/signed
products, changed values and offset bindings, fresh output storage and warm
execution without Python/eager replay. Rust tests additionally check scalar,
empty and singleton-reshaping address classification.

Passed checks:

- The complete compiler selection: **901 tests in 85 exhaustive, disjoint
  module processes**, with 23 explicit skips on GPU 0. All test IDs and source
  hashes are retained in `compiler/receipt.json` inside the archive.
- All three focused two-device restoration tests on devices `0,1`.
- With CUDA hidden, both new metadata tests passed and all three hardware tests
  skipped explicitly.
- All 30 focused Rust pointwise tests in both default and `python-bindings`
  configurations; default/bindings builds and all-target Clippy with warnings
  denied; formatting and all 12 README documentation tests.
- Independent Moduler design and focused implementation/documentation reviews
  found no actionable issue within the documented concrete-shape contract.

## Reference shape-history limitation

The original persistent-reference test failed three exceptional-value subcases;
those failures remain in `priority-tests.log` inside the archive. This is a
separate, measured difference between fresh and automatically symbolic default
Inductor specializations, not a passing test or a discarded retry.

For the program above, call a single reference wrapper with shapes
`((2,1),(1,2))`, `((1,2),(2,1))`, then `((2,1),(2,2))`, using repeating
`[2e38,-2e38]` input values. The third call returns
`[[inf,-inf],[-inf,inf]]`; a fresh default reference wrapper on the identical
third-call inputs returns all positive infinities. The archive includes the
standalone reference-only `history-probe.py` and its output. Symbolic indexing
changes the reference load policy and contraction order.

Native compilation specializes concrete shapes. Exceptional-value tests therefore
compare each native shape specialization with a fresh default reference, while
keeping the native wrapper alive across shape changes. Separate finite-value
history tests keep both wrappers alive. This does not establish exceptional-value
parity across arbitrary reference symbolic shape histories; see the
[numerical contract](../../compile-pointwise-numerics.md).

## Completed clean captures

Burner committed the repair as `c1f2d380abd012313741e629a5139c651be14cc6`.
The unchanged public-default commands then measured 10% coverage (8/112 cells)
and 20% CUDA performance (8/56 cells), with both reports valid and non-diagnostic.
Generated CUDA/PTX and all five focused broadcast-priority tests were also
captured from that clean commit. The [bundle README](README.md) links the refreshed
reports and post-commit logs. The original failures, development archive above,
and clean `166687a` baseline measurements remain unchanged. These measurements
do not remove the symbolic-history limitation described above.
