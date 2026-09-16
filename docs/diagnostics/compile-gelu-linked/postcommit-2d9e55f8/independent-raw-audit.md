# Clean-commit raw evidence audit

Passed the requested read-only provenance and raw-result checks for code commit
2d9e55f820f49efd817f3d52b7cbc9c69894d3ad. No new measurement, build, framework,
compiler, test or GPU work was performed. This is not full qualification or
merge approval. Counts and exact report hashes accompany this file in
independent-raw-audit.json.

## Public eager/default-compile/no-NVRTC captures

All five process receipts report exit 0. Every report records the exact clean
commit with empty git_status, the current worktree root, a worktree-local
interpreter with matching file hash, and unchanged before/after source maps.
Independently checked all 126 named source-file hashes against git show at the
committed revision, not merely against current filesystem files.

All native legs record the same mapped and installed extension SHA256
 d335da08693751bc75c187016cae210d45b789903f7d52a3310be68ea55b1ca0,
matching wheel-check.json and the currently installed bytes. The release wheel
hash also matches the checked file and the fixed gate's recorded wheel identity:
9e21a312acd09aa155c0d5e2f7984a063300d378110534fc4c9d8781adcb6031.

Independently rehashed and decoded 1,871 raw files (38,496,436 bytes). Verified
post-call input bytes and pointers, metadata/output offsets, current context,
allocation context/device memory/ordinal for nonempty tensors, native fresh
output nonaliasing, and all 590 retained snapshots with prior/current call IDs
and stable values/pointers. Empty tensors have zero-byte raw records. All scalar
records and input bits match the corresponding declared and paired histories.

The public compiled matrix has 46 formulas, 124 calls, 218 history leaves and
18 timing checks. Its 236 comparisons pass. Each eager matrix has seven formulas,
21 calls, 21 history leaves and 12 timing checks; ordinary eager and no-NVRTC
each pass 33 comparisons against ordinary reference eager. Recomputed the
unchanged tolerance/NaN/signed-zero policy independently from raw words.
All pairs have zero finite-bit and signed-zero differences. The compiled pair
has 1,924 NaN-word differences; both eager pairs have zero. Payload equivalence
is not claimed.

Verified all six historical cancellation inputs in both competing-product
output orders, including x=5, with exact matching output words. Changed timing
inputs actually differ; all recorded 17-sample medians recompute correctly.
Native compiled histories record body/eager replay checks. The deliberately
absent-NVRTC leg has its absent override and no libnvrtc process mapping.

## Fixed full gate

The fixed report is valid=true, diagnostic=false, public-default-compile-v2,
metric=both, and records framework.compile(program). Both source and build
provenance name the exact clean code commit with no working-tree changes.
All six workers use the worktree-local interpreter; CPU and both CUDA orders
are retained. The wheel matches the public capture's checked build.

Independently counted all 112 fixed cells and all 56 CUDA cells: 22 pass overall
and all 22 are CUDA. Both CUDA GELU variants pass; both CPU GELU variants remain
unsupported. Failed cells retain zero ratio. Independently recomputed each
category's weighted contribution from cells, including capped CUDA geometric
aggregation: coverage 21.0 and CUDA performance 33.45197861005538, matching the
report. These are finite-corpus measurements, not general PyTorch percentages
or proof that Burner's merge threshold passes.

Raw transport/archive integrity is a subsequent separate check. Historical
prerequisite/generation records remain pinned and are not substituted for these
fresh final-code captures. Canonical independent review, full qualification and
merge decisions remain Burner's responsibility.

Assembled 2026-09-16T08:01:22.472317+00:00.
