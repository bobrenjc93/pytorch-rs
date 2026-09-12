# Constant folding precision and zero provenance repair

Both review findings reproduced on H100: constant-tensor arithmetic rounded
intermediates and literals to float32 too early, and computed zeros incorrectly
removed additions. The private IR now carries binary64 scalar values. Known
constant arithmetic retains that precision until a float32 runtime consumer or
output. Shared consumers retain their own rounding boundaries.

Zero identities run in the reference's order: addition first, subtraction
second, before numerical lowering. A subtraction-exposed zero cannot remove an
addition retroactively, but later subtractions can still resolve it and expose
products for FMA. Constant arithmetic remains separate from these early graph
identities. Runtime arithmetic, libdevice, and supported-subset limits remain
as described in the [compiler contract](../../compile-pointwise-jit.md).

The [development bundle](review-folding.json.gz) preserves original failures,
before/after output bits and generated CUDA, exact final commands and logs,
source/test/wheel/native hashes, and generated CUDA/PTX provenance. It includes
the initial 145 failing subcases, the first implementation attempt's 18 failures,
and the nested-subtraction overflow failure that ruled out a local provenance
workaround. Superseded attempts retain their recorded paths; their directories
were moved aside before final reruns. No original measured artifact changed.

The expanded tests also exposed two incorrect test assumptions. Accurate native
libdevice sine differed from Inductor by one ULP; the exact arithmetic boundary
test now cancels a shared sine result and retains exact output-bit assertions.
Inductor can promote changed captured floating bindings to runtime scalar
tensors, changing rounding boundaries. The native cache test therefore compares
each warm constant specialization with a fresh constant-specialized reference,
including `16777216.0` versus `16777217.0`. It does not establish equivalence to
Inductor's adaptive warm-call strategy. Both initial observations are retained.

Final validation uses the locked release wheel, Python 3.12.14 and default
PyTorch 2.13.0+cu130 on H100 with `CUDA_VISIBLE_DEVICES=0`:

- All seven pointwise modules: 44 tests, 43 passed and one explicit two-device
  skip. The new module covers precision cancellation, overflow, original literal
  precision, shared consumers, operand orders, zero signs, nested subtraction,
  fresh values/storage and captured scalar cache transitions.
- Thirteen hardware-free Rust IR tests pass with default and Python-bindings
  features. The no-default-features library check, both all-target Clippy
  configurations with warnings denied, and formatting pass.
- Ninety backend-contract tests and twelve documentation tests pass. The
  portable selection passes twelve admission tests with twenty-four explicit
  CUDA skips. All eighteen independent probes match reference output bits on
  cold/warm fresh inputs, excluding NaN payloads; the bundle retains each result.

The JIT uses NVRTC 13.0, runtime 13000 and `compute_90`, without fast math.
Environments, caches and generated files stay inside this worktree. These are
unscored semantic checks from dirty sources based on `c3a8bd08`, not timing or
clean-commit campaign measurements. The [latest clean capture](README.md) stays
pinned to `0c836a49` and requires refresh after Burner commits this repair.
No evaluator, tolerance, denominator or managed progress artifact changed.
