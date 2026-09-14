# Review fix: admit skipped bodies before discarding their effects

The required [clean-commit evidence refresh](postcommit-6ef71d3a/README.md) has
completed at `6ef71d3a`. The development checks and original handoff below remain
pinned to their recorded sources.

The reviewer reproduced an admission gap: zero-trip loops skipped typed operator,
helper and data validation. The new cold regressions reproduced reductions,
helper-local loops and higher-order helper arguments being accepted. The warm
regression also reproduced an arbitrary replacement helper being accepted.

Normalization now retains one bounded admission pass for each zero-trip body,
with internal scope markers that cannot be admitted from user bytecode. The
existing frontend validates that pass with a copy of the local frame, then
discards its assignments and temporary IR. Initial-parameter dependency analysis
restores the outer assignment state too. Realized helper/code dependencies and
ignored-data boundaries use the existing specialization guards on warm calls;
no parser, cache owner or execution backend was added. Checked instructions and
nodes count toward the existing shared limits even after their IR is discarded.

New regressions cover cold/warm rejection, callback exclusion, helper code
changes, unchanged caches on failure, recovery without disassembly/lowering,
reset, unchanged locals/output IR, and budgets shared between skipped and
executed bodies. Existing unbound-index and identity-only-return regressions
remain intact. A real CUDA test checks zero-trip helper admission and recovery
without changing outputs or inputs.

## Validation of the uncommitted fix

These are development checks from dirty sources based on `658ccae`, **not clean
commit evidence or a new fixed-corpus measurement**. Frontend SHA256:
`1e5ca6378b7ddb737192b66ef4946d82853463b9d3c721f684a3b109224977c8`.

| Check | Result |
| --- | --- |
| CUDA-hidden compiler suite | 951 tests, 358 explicit skips, no failures |
| Focused loop/helper tests on H100s 0 and 1 | 53 passed |
| Focused loop/helper tests on each CPython 3.10–3.14 | 53 tests, eight hardware skips, no failures per interpreter |
| Fresh native pointwise regressions | 34 passed |
| Source/wheel/import provenance and whitespace checks | Passed |

Burner's canonical `gpu` and `cpu-heavy` records still identified owner
`idea_0efad7e9`. The existing reservation used only GPUs 0 and 1, UUIDs
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1` and
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`, on driver 580.82.07. The local
Python 3.12.12 environment contains PyTorch `2.13.0+cu130`; portable checks used
3.10.19, 3.11.15, 3.13.13 and 3.14.5. A fresh locked offline release wheel was
built with Rust 1.92.0 in `target/loop-review-fix/build`, installed locally, and
byte-checked against checkout/imported Python files and the native extension.
Portable tests extracted that same wheel. Interpreters, dependencies and writable
build/framework/CUDA caches remained within this checkout.

From the author root, with the archived worktree-local environment and canonical
resources held:

```bash
source target/loop-work/env.sh
CARGO_TARGET_DIR="$PWD/target/loop-review-fix/build" \
  maturin build --release --locked --offline --out target/loop-review-fix/wheels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/loop-review-fix/wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -p 'test_compile*.py'
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_loops tests.test_compile_pointwise_helpers
CUDA_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR="$PWD/target/loop-review-fix/build" \
  cargo test --locked --offline --lib pointwise
```

[review-zero-trip-admission.json.xz](review-zero-trip-admission.json.xz) preserves
logs, portable commands/timestamps, dirty source/build identity, resource records,
environment and source snapshots. Its `files` mapping contains original text and
SHA256. `before.log` retains the original admission failures; `source-first.log`
retains an unsuccessful source-path import without the extension; and
`focused-cpu.log` retains a new test fixture's omitted arity argument. These were
followed by successful source-bound checks. The final focused runs additionally
cover hostile and oversized ignored-data replacements added during the broader
compiler sweep. Original raw files remain under `target/loop-review-fix/`.

## Required clean-evidence handoff

The `postcommit-dcfeb27a` measurements predate this implementation change and must
not be used as final evidence for the fix. They are preserved unchanged; no
provenance was rewritten. After Burner commits the implementation, its normal
post-commit evidence step must refresh the candidate's two-H100 persistent-wrapper
diagnostic and unchanged fixed evaluation. Existing clean-main measurements may
be reused only after their source/build/import/raw provenance is revalidated.
The nested clean-main raw directory remains outside the fixed observer's coverage,
as documented in the previous evidence handoff. No raw reports were removed and
no retention owner was changed. Independent review and qualification remain with
Burner; this worker did not commit or create a branch.
