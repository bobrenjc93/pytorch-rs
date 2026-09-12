# Scalar-zero contraction and nonfinite cache reuse

Both review findings reproduced on H100. Scalar positive-zero subtraction now
exposes multiplication factors during late contraction and emission, after
expression-identity checks. This preserves the reference residual in
`-a+(a-0.0)` and `(a-0.0)-a` without changing literal self-subtraction,
addition-of-zero or negative-zero boundaries. One factor-discovery pass tracks
exact sign flips and their position relative to zero subtraction, retaining
reference contraction priority for signed doubling and nested compositions.

Captured floats now apply persistent runtime promotion before trying the full
graph-cache guard. New promotion is discovered only on a miss. A previously
compiled static value survives an infinity/NaN-only interlude, while new finite
values and shape misses still consult that history. Actual finite promotion
remains persistent; no separate state or cached output values were introduced.
The final wheel also includes the offset-aware guard update: contiguous storage
offsets are validated and supplied at launch rather than specialized in the
graph key, preserving static binding reuse across offset changes.

The [development bundle](review-zero-boundaries.json.gz) preserves the original
18 failing subcases, intermediate signed-doubling runs with 14 and two failures,
and a four-program nested probe with two mismatches. The initial numerical
probe's import-path failure and successful retry are also retained. Commands,
timestamps, logs, source/test and wheel/native hashes, generated CUDA/PTX and
before/after observations remain attached to their actual attempts.

Final validation used the locked release wheel, Python 3.12.14, default PyTorch
2.13.0+cu130 and H100 GPU 0, with NVRTC 13.0, runtime 13000, `compute_90`, explicit
FMA and `--ftz=false`:

- The main fourteen pointwise modules: 76 tests, 74 passed and two explicit
  multi-device skips under `CUDA_VISIBLE_DEVICES=0`. The new module covers
  cold/warm fresh values, exact sign bits, three nonfinite interludes, shape
  misses, persistent promotion, both operand orders, scalar kinds, signed
  doubling, repeated expressions and nested sign/zero compositions.
- The separately added operator nonfinite-history module passed all six tests
  unchanged after the main selection, including offset transitions and integer
  histories. A later operator update generalized the explicit two-device mask
  guard; its new admission test and both affected restoration tests passed
  under `CUDA_VISIBLE_DEVICES=0,1`. The disjoint selections cover 83 unique
  tests, all passing; the two initial skips passed in that targeted run.
- Eighteen Rust IR tests passed with default and Python-binding features;
  no-default-features library checking, both all-target Clippy configurations
  with warnings denied, and formatting passed.
- Ninety backend-contract and twelve documentation tests passed. Twenty final
  before/after program probes matched default Inductor, including the four
  nested cases. The native-only codegen capture verified two graph entries
  sharing one generated module across fresh inputs and shapes.
- Final source, installed native module and wheel identities were verified.
  All 77 prior measured artifacts remain byte-for-byte unchanged. Earlier
  two-device restoration evidence is also preserved.

These are unscored dirty-source checks based on `1905c865`. Burner committed
the repair as `42959e14`; its [post-commit receipt](postcommit-42959e/postcommit.json)
records clean coverage/performance and the combined fifteen-module selection.
The [current evidence index](README.md) links later clean revisions.
The preceding capture remains pinned to `5fc75c41`; no old report was relabeled.
Temporary wheel/cache/raw-log paths may disappear
at cleanup; this compressed bundle preserves the durable evidence. See the
[compiler contract](../../compile-pointwise-jit.md) and [evidence index](README.md).

## Historical build timeline clarification

The retained `operator-offset-first-validation.log` in the development bundle
started at **22:54:17.715 UTC** with wheel SHA-256 `daa9ddaf…`; it recorded
**94 tests, 12 failures and two skips**, explicitly outside qualification.
The replacement installation began at **22:54:24.204 UTC**, after that process
started. The source fingerprint at test start therefore did not atomically
attest which native binary was already loaded. Those failures remain unchanged.

The operator identifies the earlier native binary as `98f9a08…`, and the
replacement codegen record identifies `6f44f68…`. The earlier wheel has been
cleaned up, so its native-hash association rests on the operator's clarification;
it cannot be reverified from that wheel. The bundle retains both codegen
identities, the installation receipt, the failed operator log and the later
passing 76-test log. These attempts are distinct from the subsequent clean
`42959e14` and `74602c07` captures. The operator also reported a separate passing
three-test run on physical GPUs 6 and 7; that report does not replace the
retained author's `CUDA_VISIBLE_DEVICES=0,1` receipt or a future independent
acceptance run.
