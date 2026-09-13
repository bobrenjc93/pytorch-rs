# CUDA pointwise numerical semantics

This is the numerical contract for the [default CUDA pointwise compiler](compile-pointwise-jit.md).
It follows the measured PyTorch 2.13 Inductor behavior for the supported subset;
the usage guide owns input restrictions, runtime setup and cache behavior.

## Rounding and library functions

Runtime trigonometry uses accurate libdevice `sinf`/`cosf`;
constant-only sine/cosine use double-precision libdevice evaluation between
float32 input and output boundaries, matching reference constant evaluation.
This avoids amplifying single-precision library rounding errors in later
cancellation. `--ftz=false` is used;
fast math is not enabled. Arithmetic explicitly allows FMA contraction
(`--fmad=true`) to match default Inductor, including cancellation and overflow
cases where CUDA eager's separately rounded operations differ. SSA variables
retain expression dependencies and reuse through native compiler optimization.

## Expression sharing and contraction

Numerical lowering shares identical ordered expressions before selecting
explicit `fmaf` operations for each consumer. Self-subtraction established before
sign rewriting uses the shared rounded value: finite values produce positive zero, while infinities and NaNs
produce NaN. Products may contract at multiple consumers while retaining their
rounded value for other uses. Expression deduplication retains operand order,
including for commutative operators.

## Competing products and dependency order

When a sum has two direct positive
products, contraction selection follows the reference's arithmetic/select
dependency ranking: live input loads receive successive ranks, arithmetic adds
one dependency level, and ReLU adds comparison and selection levels. The
lower-ranked product contracts; equal ranks retain expression order. Thus
`x*x + x*y` contracts the square, while `a=x.relu(); a*a + x*y` rounds the
square and contracts `x*y`, including when the sum operands are reversed.
This bounded rule follows LLVM's
[Reassociate operand ordering](https://github.com/llvm/llvm-project/blob/1f126a6dea50d185c0781743a667390037ae88bd/llvm/lib/Transforms/Scalar/Reassociate.cpp#L242)
used by the measured reference. Runtime libdevice sin/cos introduce control-flow
joins, whose ranks follow all entry tensor loads. Distinct live calls receive
successive rank regions in expression evaluation order; shared calls retain one
region and nested calls follow their dependencies. Arithmetic after a call
inherits that region. This models the pinned reference's join/PHI ordering
without reproducing libdevice's internal arithmetic. Constant-only expressions
have rank zero, and their products are excluded from contraction candidates:
the reference folds those products before FMA selection.

## Sign normalization and live uses

Subtraction introduced by sign normalization retains its contraction eligibility
even when the normalized operands coincide. Sign-flipped products retain their
factors: direct products take contraction priority, but an otherwise unpaired
signed product can still contract rather than prematurely overflowing. Shared
tensor products retain that fallback; an exact sign flip of a single-use
tensor product can instead move into its factor and contract directly,
matching the reference's
[negation hoisting](https://github.com/llvm/llvm-project/blob/1f126a6dea50d185c0781743a667390037ae88bd/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L3020).
This counts live canonical operand uses before sign normalization rewrites
consumers; the signed result itself may be shared.
An addition or right-hand subtraction can extract the factor's sign again
only when the signed multiplication has one use. These are separate use counts.
Shared negative doubling retains its rounded value for earlier addition consumers;
the last live consumer and subtraction consumers can expose its factors.
Consumer ordering is determined from live, deduplicated expressions before
sign normalization, including elimination of multiplication by one.
Sign normalization precedes contraction: `-(a-b)` becomes `(b-a)+0`, products
with negative coefficients in sums become subtraction, and unit multipliers
and signed doubling follow the reference's normalization. Negated products use
`fmaf(-a, b, +0)`.
Uncontracted operations use explicit round-to-nearest CUDA intrinsics so NVRTC
cannot choose a different contraction after strength reduction.
Subtraction of positive scalar zero becomes transparent only during contraction
and emission, after expression identity checks. Thus `(a-0.0)-a` can retain an
FMA residual while `a-a` shares one rounded value. Addition of zero and
subtraction of negative zero keep their separate rounding boundaries.
Late factor discovery tracks exact sign flips through zero subtraction and
retains contraction priority for nested signed products; two sign-flipped
candidates retain their left-to-right order.
The complete IR is validated before unused expressions are removed; an unused
local cannot change rounding of the returned expression. Invalid unused
operations remain rejected.

## Scalar kinds and zero identities

Boolean, integer and floating scalars retain distinct IR kinds until their
operator-specific rules and float32 promotion have run. Boolean `False` and
integer zero multiplication produce a known positive-zero tensor, including
for non-finite inputs; floating zero multiplication retains IEEE NaN and zero
signs. The resulting constant tensor is distinct from a scalar operand, so
further tensor multiplication still has float semantics. Only zeros created by
this early integer/Boolean rewrite qualify for tensor addition identities or a
right-hand subtraction identity. Addition identities run before subtraction
identities; a zero exposed by subtraction cannot retroactively remove an addition.
Zeros computed by subsequent arithmetic retain
the arithmetic operation at runtime consumers: `(x*0)+x` preserves an input
negative zero, while `(x*0+0.0)+x` produces positive zero.

## Constant precision and materialization

Scalar literals and add/subtract/multiply/negate of known constant tensors retain
Python binary64 precision and zero signs until float32 materialization. For
example, `((x*0)+16777216.0)+1.0-16777216.0` folds to one. A constant is rounded
to float32 at each runtime consumer or output; other constant consumers of the
same intermediate retain its binary64 value. ReLU/sin/cos stop this propagation
and consume float32 values. Folded negation flips the constant's sign instead of
lowering to positive-zero subtraction or FMA. Runtime arithmetic retains its
existing contraction rules. IEEE bits preserve negative zero and non-finite
values across the private IR bridge.
Integer constants follow default Inductor's Python binary64-to-float32
normalization; large integers at rounding boundaries can differ by one float32
ULP from eager's direct integer conversion.

## Signed zero and subnormal values

The zero-sign contract follows default PyTorch 2.13 Inductor: standalone negation
uses `0 - x`, so negating positive zero produces positive zero; ReLU preserves
negative zero. Contracted negation of a nonzero positive product that underflows
produces negative zero; an exactly zero floating product produces positive zero.
Negating a known zero tensor from Boolean or integer multiplication instead folds
to negative zero. CUDA eager also differs for standalone positive-zero negation
and negative-zero ReLU. Tests assert those differences and check the covered
eager/reference values. Runtime sine flushes subnormal inputs to signed zero
before accurate libdevice evaluation, matching the reference boundary even
when subsequent arithmetic amplifies the result. Constant-only sine expressions
retain gradual underflow, as does other native arithmetic; no global fast-math
or flush-to-zero option is enabled. Regression tests include amplified
subnormal values and exact zero-sign assertions.
