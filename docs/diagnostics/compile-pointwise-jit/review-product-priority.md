# Competing product contraction

The repair selects the reference product for FMA when two multiplication
expressions feed an addition. For `a=x.relu(); return a*a+x*y`, the square
rounds separately and `x*y` contracts, preserving positive infinity for the
reported large-input case in either sum order. Raw products and arithmetic
producers retain their different dependency-order behavior. Expression
identity and subtraction orientation remain separate from this selection.

The rule follows the pinned reference LLVM's operand ranking, rather than a
preference for a particular Python function or shape. A related check found
that exact negation of a single-use tensor product must move into its factor
before consumer normalization. Counting uses in the original live canonical
graph preserves the distinction between a shared positive product and a shared
negated result. See the [numerical contract](../../compile-pointwise-jit.md)
for the source links and the bounded arithmetic/select rank model.

The [development bundle](review-product-priority.json.gz) records the dirty
repair on base `fca257c135683379d0f0096f87919022cdce2084`, including exact commands,
source snapshots before/after the final build and checks, wheel/native hashes,
CUDA/PTX, and the original failures. The operator's and author's initial
three-test runs each reproduced eight failing subcases. Two subsequent
validation attempts exposed four and then two signed-product failures; a
third exposed four cases requiring single-use factor-sign cancellation.
those logs and source snapshots are retained, not replaced by the final run.
An initial Python-bindings Rust launch also failed to locate `libpython3.10`;
the final command explicitly uses the worktree's Python 3.12 interpreter and
library directory.

The permanent regression module covers raw, ReLU and arithmetic products,
both addition orders, reversed input encounter order, deeper/shared/dead
producers, both subtraction orientations, negated-product sharing, and cold
and warm calls with fresh CUDA float32 inputs. It checks non-finite values,
zero signs, output contracts, input preservation and graph-cache reuse.

Final validation passed all sixteen pointwise modules: 91 tests, with two
explicit multi-device skips on H100 GPU 0; those two tests then passed on
GPUs 0 and 1. All 90 top-level/backend tests, 16 documentation/archive tests,
25 Rust pointwise tests and Clippy passed. Verification checked 882 source/test
files, 25 local links, and byte-for-byte preservation of 50 prior artifacts.
The generated-code captures record NVRTC options, native
library identity and CUDA/PTX; the reported overflow case returns positive
infinity on both fresh-input calls in both operand orders without importing
installed PyTorch. The final source snapshots agree before/after the build
and validation.

These are unscored development checks, not clean campaign measurements or
independent review approval. All earlier captures remain at their original
measured identities. The [evidence index](README.md) identifies the outstanding
fresh campaign and generated-code captures after Burner commits this repair.
Temporary build, cache and raw-log paths in the bundle may disappear during
worktree cleanup; the compressed bundle is the durable record.
