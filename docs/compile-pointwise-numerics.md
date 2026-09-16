# CUDA pointwise numerical semantics

This is the numerical contract for the [default CUDA pointwise compiler](compile-pointwise-jit.md).
It describes the lowering and measured PyTorch 2.13 Inductor compatibility,
including the unequal-shape admission boundary below. The usage guide owns input
restrictions, runtime setup and cache behavior.

## Rounding and library functions

Runtime trigonometry uses accurate libdevice `sinf`/`cosf`;
constant-only sine/cosine use double-precision libdevice evaluation between
float32 input and output boundaries, matching reference constant evaluation.
This avoids amplifying single-precision library rounding errors in later
cancellation. `--ftz=false` is used;
fast math is not enabled. Arithmetic explicitly allows FMA contraction
(`--fmad=true`) to match default Inductor, including cancellation and overflow
cases where CUDA eager's separately rounded operations differ. Declarative
instructions retain dependencies and reuse; register allocation removes dead
products after their consumers contract.

## Realization and observable order

Numerical planning uses the first observable order of computed roots and the
logical specialization's retained iteration hint. After the existing early
algebraic rewrites, independently created tensor producers keep distinct SSA
identities through locality ordering, realization and fusion. Expression
interning supplies operation/read counts without merging those producers.
Realization alone is not a rounding barrier: fused units can inline each other.
Values crossing regions become rounded imports and exports in the native scalar
program. Scheduling and lowering share the producer map, so a consumer of an
exported value reads its rounded import. An independently computed equivalent
expression is not that producer: spelling `x*y` twice does not by itself make
one tensor depend on the other. Scalar-expression CSE within each region remains
available after partitioning.

The version-sensitive reference rules come from PyTorch 2.13
`torch/_inductor/fx_passes/post_grad.py::reorder_for_locality`,
`graph.py::GraphLowering.run_node`, `ir.py::StorageBox`,
`ops_handler.py::OpCounterCSE`, `choices.py::InductorChoices`, and
`scheduler.py::Scheduler`. The repair evidence retains the inspected reference
identity and detailed symbol pointers alongside generated FX and scheduler IR.

One graph-keyed CUDA kernel interprets that bounded program in one launch.
Container topology does not create additional native executables. Instruction
validation and register allocation precede device allocation. Matching warm calls
reuse an immutable validated program and completed instruction upload. Selection
includes exact admitted input shapes, retained numerical hint and observable
output order; rank-zero and length-one inputs are distinct signatures. The Rust
planner remains the only numerical authority, including for oversized ephemeral
preparations. This reuse changes neither instructions nor admission: current
input metadata must still match the checked native signature. Register scratch
and outputs remain fresh invocation-owned storage through launch, completion and
failure. The usage guide describes the bounded data cache and its memory limits;
no performance improvement is implied without a new measurement.

## Expression sharing and contraction

Within each numerical region, lowering shares identical ordered expressions
before selecting explicit `fmaf` operations for each consumer. Self-subtraction
established before sign rewriting uses the shared rounded value: finite values
produce positive zero, while infinities and NaNs produce NaN. Products may
contract at multiple consumers while retaining their
rounded value for other uses. Expression deduplication retains operand order,
including for commutative operators.

## Competing products and dependency order

Within each numerical region, the direct product with fewer remaining uses takes priority.
This includes two products shared by sibling consumers, not only single-use
products. Each canonical output value counts as one observable use after
expression deduplication, even when distinct original computations require
separate output allocations and stores. Repeated result aliases add no uses.
Stores do not prohibit contraction: a shared product can still fuse when there
is no less-used competitor. Transparent sign/positive-zero wrappers contribute
their external uses to the underlying product, including sibling wrappers and
products created by extracting a negative coefficient. These relationships are
used only to count external uses within a basic block; they do not merge values,
allocations or rounding boundaries. Use counts also retain their role in sign rewrites.
Contractions are selected from consumers toward operands. Each selected FMA
replaces its product use with direct factor uses before inner choices are made;
an outer contraction can therefore make an inner product single-use.

The [structured-output evidence index](diagnostics/compile-pointwise-structured-outputs/README.md#numerical-history)
tracks partition-dependent rounding, retained shape-history hints and
realization/order repairs. Historical failures remain distinct from later
passing captures.

When a sum has two direct positive products with equal use priority,
contraction selection follows the reference's arithmetic/select
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

## Unequal-shape numerical boundary

Unequal input shapes admit every returned root with at most one arithmetic
stage, including live sin/cos, or the tensor-leaf multiply-add exception below,
in the original typed IR.
Input/scalar depth is zero; add/subtract/multiply
add one to the maximum operand depth, tensor negation adds one, and ReLU/sin/cos
preserve depth. Scalar signs are metadata, not tensor negation. This check precedes
numerical simplification, so zero/one identities cannot hide a second stage.
Actual input shape equality controls admission, including unused inputs and
unequal shapes whose address maps are linear. Full graph and shape validation
still includes dead expressions; numerical capability depends on the returned
live roots only. Equal-shape admission and lowering remain unchanged.

The one-stage class cannot supply a product to a second arithmetic consumer.
ReLU contributes comparison/selection, and scalar identities, sign normalization
and constant materialization do not introduce another live add/multiply stage.
Consequently this subset has no competing product contraction to select. Address
calculation only chooses input elements; it no longer changes numerical
materialization ranks to approximate a reference autotuner.

The existing numerical planner and generic CUDA instruction executor also own
one-stage trig; admission adds no trigonometric implementation or launch path.
Constant-only trig is not runtime-input evidence, and empty outputs execute no
trig. Finite regression results and generic VM PTX do not prove all-program or
performance parity, nor independently trace a selected scalar plan on-device.
Static scalar `-0.0` may reuse a `+0.0` specialization; tests retain that history
rather than treating each sign as a separately specialized execution.

The sole two-stage exception is `Add(Mul(Input(a), Input(b)), Input(c))` or
`Add(Input(c), Mul(Input(a), Input(b)))`. All leaves must be tensor inputs;
IDs may repeat. The existing lowering emits a single FMA with no competing
product to rank. Scalar leaves of every kind, identity wrappers, subtraction,
negation, ReLU, sin/cos and extra live arithmetic are outside this exception.
No returned root may have live sin/cos when using this exception. In
`p=x*y; return (p+x,p.sin())`, the shared product feeds both an arithmetic and a
trig consumer, so the graph rejects in either output order. Dead trig remains
irrelevant to this live-root check.
Matching the original nodes before identities, CSE and sign normalization keeps
those near misses excluded. Dead nodes and unused arguments still receive full
validation. Preparation checks original admission and addresses; reuse requires
the same exact native shapes and revalidates current input metadata and storage.
The [H100 diagnostic record](diagnostics/compile-pointwise-tensor-madd/README.md)
preserves exact IEEE comparisons, including zero signs, against ordinary default
Inductor; NaN payload equality is not required. This bounded evidence does not
establish general Inductor equivalence.

The earlier broad candidate's `a=x+1.0; x*y+a*a` failures are now explicit
unsupported cases for unequal shapes. H100 runs found finite cancellation errors
on large fresh shapes and persistent shape transitions, as well as wrong infinity
signs. Identical fresh default-Inductor runs could also select different
contractions by timing. The [review record](diagnostics/compile-pointwise-broadcast/review-autotune-blocker.md)
and [earlier repair record](diagnostics/compile-pointwise-broadcast/review-broadcast-order.md)
retain the original measurements and source identities. Narrowing admission does
not repair those numerical discrepancies or establish arbitrary broadcast parity.

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
consumers, including output stores; the signed result itself may be shared.
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
Numerical comparisons retain the existing `rtol=1e-5`, `atol=1e-6` contract;
small subnormal comparisons alone do not establish bit-exact preservation.
