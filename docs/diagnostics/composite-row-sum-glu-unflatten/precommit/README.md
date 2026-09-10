# Precommit integration diagnostics

These are raw precommit correctness/build/test records for the CUDA row-sum,
GLU, and unflatten composite. They are not clean-code evidence, a performance
score, or merge qualification. See the [validation summary](../../../composite-row-sum-glu-unflatten-validation.md).

The `capture-*` files come from the existing repository build tool. Other
`*-receipt.json` files record actual commands, times, environments, exits, and
log hashes. `capture-inputs/` preserves the local command-recording and audit
inputs used for this capture; these are evidence snapshots, not installed tools.
Original absolute worktree paths and dirty flags have not been rewritten.
`python314-full.log` retains the initial setup failure; `python314-final.log`
is the full successful rerun after environment corrections.
