# Libdevice product contraction

The repair preserves the reference product ordering when runtime sin/cos
results feed competing multiplication expressions. For
`a=x.sin()+x; return a*a+x*y` (and the cosine counterpart), the square rounds
separately and `x*y` contracts. Large opposing products therefore return
positive infinity, matching default Inductor on both cold and warm calls.
Both sum orders are covered.

The pinned reference LLVM assigns immovable control-flow joins a later rank
region than entry tensor loads. The lowering now preserves that ordering for
runtime libdevice calls. Regions follow live expression evaluation order,
including nested calls; shared calls use one region. Constant-only expressions
have rank zero and their products are not contraction candidates, matching
reference constant folding before FMA selection. Existing sign normalization,
subtraction orientation, and scalar rounding boundaries remain separate.
See the [numerical contract](../../compile-pointwise-jit.md) and its pinned
LLVM source link.

The [development bundle](review-libdevice-order.json.gz) preserves the clean
`e3d7c2ad10bb286c5647ebd66ca7d997cd4212c3` before-fix reproduction, including
its source/build receipt and generated CUDA/PTX. Two of twelve programs failed
on both fresh-input calls. A subsequent constant-product probe also reproduced
a native negative infinity where the reference returns a finite result. These
original observations are retained unchanged. The bundle separately records
the dirty repair's build, source manifests, commands, tests and generated code;
it does not attribute those results to the clean pre-repair implementation.

Permanent regressions cover sin/cos products, both addition orders, fresh cold
and warm inputs, nested and distinct calls, shared and duplicate calls, dead
calls, input encounter order, and constant-only products. They check numerical
results, zero signs, output contracts, unchanged inputs and graph-cache reuse.

Validation passed 96 pointwise tests on H100 GPU 0, with two explicit
two-device skips; both skipped tests then passed on GPUs 0 and 1. All 90
top-level/backend tests, 16 documentation/archive tests, 26 Rust pointwise
tests, and Clippy passed. The separate twelve-program before/after probe now
has zero failures. The source/test manifest is identical before and after the
build and validation. The installed native SHA256 is
`817764896d4e5a62ccdb51f95b969f0421a87cd7ed826fd8335b226b74d24930`;
the generated-code source SHA256 is
`0f0a40416649e53a1154f3bc22ab98af263cda8c5a741d475dfdcde352877f96`.
NVRTC 13.0 generated the kernels for compute capability 9.0 with explicit
FMA and `--ftz=false`; the development codegen capture also verifies warm
code reuse and no installed-PyTorch import.

This is unscored development validation. Clean coverage, CUDA-performance and
generated-code captures remain required after Burner commits this repair.
Run the unchanged commands in the [evidence index](README.md) from that clean
commit, using new output paths. The existing baseline and campaign reports
retain their original identities; no later implementation is credited with
those measurements. Independent review and qualification remain required.
Temporary build/cache paths may disappear during cleanup; the compressed
bundle is the durable record.
