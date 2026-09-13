# Broadcast numerical review: unresolved blockers

Both second-round reviewer findings reproduce on H100. They remain **unfixed**;
this branch is not ready for numerical approval. The earlier passing focused
tests and fixed-corpus scores do not establish the requested broadcast parity.
No implementation, evaluator, tolerance, or external producer changed during
this investigation.

The subsequent review adds blocking regression tests rather than changing the
lowering: `test_compile_pointwise_broadcast_priority.py` now keeps both wrappers
alive throughout every shape sequence, uses cancellation-sensitive finite
values, checks the reported IEEE sequence, and exercises fresh large shapes
in both argument orders. Initial bindings are ordinary contiguous tensors;
changed bindings are offset-contiguous views. No failures are marked expected
or skipped on CUDA-capable hosts.

The [regression results](review-regressions.json.gz) preserve the development
test diff, exact sources, commands, and original logs. On H100, seven tests
completed with **eight failing subtests**, including both reported blockers.
With CUDA hidden, the two metadata tests passed and five hardware tests skipped
explicitly. The test changes remove the masking behavior; they do **not** fix
numerical lowering. The implementation and fixed evaluation artifacts remain
unchanged, and the branch remains blocked.

The [capture bundle](review-autotune-blocker.json.gz) records clean HEAD
`9155d4dd0e82e4c344f0be66e99e4397d1745bda`, whose implementation is unchanged
from `c1f2d380abd012313741e629a5139c651be14cc6`. It includes exact reproduction
scripts, raw results, environment and extension identities, GPU snapshots,
generated reference source/LLVM/PTX, file hashes, and independent design review.
All execution and cache paths are inside this worktree. These are failure
diagnostics, not new scoring measurements. Existing measured artifacts remain
unchanged.

Capture caveat: `review_cases.py` uses the existing `kernel(wrapper)` test
helper, which returns the first cached module. Its raw `native_source` field
therefore describes that module, not necessarily the module dispatched by later
shape calls. The original capture is preserved; later-call source fields must
not be used to attribute a contraction. Per-call numerical outputs are measured
directly and unaffected. Reference LLVM/PTX and fresh-call native source remain
available for the materialization analysis.

## Confirmed candidate failures

For ordinary default compilation of:

```python
def f(x, y):
    a = x + 1.0
    return x * y + a * a
```

with contiguous CUDA float32 inputs filled with `4096` and `-4098`:

| Invocation | Native | Default Inductor | Mismatched elements |
| --- | ---: | ---: | ---: |
| Fresh `(257,1)`, `(257,257)` | 0 | 1 | 66,049 |
| Fresh `(2,1)`, `(2,2)` | 0 | 0 | 0 |
| Persistent wrappers: `(2,1)`, `(1,2)` | 1 | 1 | 0 |
| Same wrappers: `(1,2)`, `(2,1)` | 1 | 1 | 0 |
| Same wrappers: `(2,1)`, `(2,2)` | 0 | 1 | 4 |

The persistent sequence also reproduces an IEEE failure: alternating
`2e38,-2e38` inputs produce two wrong infinity signs on the third call.
Neither framework is reset within either sequence. The earlier test helper's
per-shape reference resets and small finite inputs missed these failures.

The current rank model counts one materialization per linear load and two per
nonconstant broadcast load. Large reference kernels instead issue multiple
loads per thread. Their lane-specific ordering changes the competing products'
ranks. Separately, symbolic shape history can retain a cache-policy instruction
on a load whose final address simplifies to linear. Concrete addresses alone
do not capture either behavior.

## Reference variability blocks a general rank repair

Six fresh processes compiled the same function with untouched default
`torch.compile(f)`, identical `(17,1)`, `(17,33)` inputs, values, and GPU.
Each process used separate empty Inductor/Triton cache directories. No compiler
configuration or reference code was changed.

| Run | Default result (all 561 elements) | Selected XBLOCK | Warps |
| --- | ---: | ---: | ---: |
| 0 | 1 | 256 | 4 |
| 1 | 1 | 256 | 4 |
| 2 | 1 | 256 | 4 |
| 3 | 1 | 256 | 4 |
| 4 | 0 | 128 | 4 |
| 5 | 1 | 256 | 4 |

The retained `.best_config` files and generated LLVM show different FMA choices.
A separate explicitly configured diagnostic launched both automatically
generated candidates and confirmed block 128 returns 0 while block 256 returns
1. That diagnostic is labeled separately from the untouched-default runs.
The installed reference's `triton_heuristics.py` constructs both candidates and
chooses by measured latency without enforcing agreement between their results.

The difference exceeds the unchanged `rtol=1e-5, atol=1e-6` contract. Thus a
deterministic native result cannot agree with both observed default-reference
outcomes. Adding shape history or lane ranks could repair individual cases,
but cannot determine a separately timed reference's autotuning winner. Always
choosing either contraction, disabling contraction, or changing precision also
fails to satisfy both outcomes and risks existing same-shape behavior.

Completion needs a stable numerical policy from the external reference, or an
explicit decision to revise the compatibility requirement. No external patch,
evaluator change, weaker assertion, or additional rank heuristic was applied.
This does not invalidate the review findings; it explains why the complete
requested contract remains blocked.

## Reproduction and validation

The bundle's `files` map contains the original `autotune/probe.py` and
`review_cases.py`, alongside the local environment script and exact commands.
After extracting those scripts under the current worktree, run `probe.py 0`
in separate processes with fresh local `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` paths. It uses only `torch.compile(program(source, torch))`
with defaults. Timing determines the selected configuration, so a later six-run
sample need not contain both outcomes. The preserved runs do; explicit launches
provide deterministic corroboration.

`review_cases.py` checks every element for both original findings and the
persistent IEEE sequence. It records failures rather than claiming parity.
The archive validation checks its embedded hashes, native/source identities,
all six observations and chosen configurations, and the original failure
counts. This documentation-only investigation does not repeat unrelated full
test suites or replace independent review and merge gates.
