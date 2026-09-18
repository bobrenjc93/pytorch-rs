# Portable `Tensor.add_` descriptor assertion repair

Author-only correction against `3eb6a1861bd4e30108b3f3adbb9c2134ef45310d`: the existing
`tests/test_top_level_add.py` assertion now requires a native method descriptor,
matching `test_tensor_add.py`. The separate module-level `torch_rs.add_` absence
assertion, CPU/gradient/saved-alias rejection tests and implementation are unchanged.

Verification used the exact revised test source and a freshly packaged native
wheel in a fresh worktree-local `.venv`, with installation destinations and
source/wheel/import hashes verified before tests. Local Linux x86_64/Python
3.12.12 used locked PyTorch 2.13.0 and CUDA-hidden suite execution.

- Owner modules and ungated `AddInplaceBindingTests`: 18 passed.
- Previously failing validator smoke test: 1 passed.
- Full portable Python suite: 6,292 run, 5,691 passed, 601 skipped, 0 failures.

The initial full run had 1 failure and 601 skips:
an existing stack-validator smoke test requires dependencies in the real root
`.venv`, while the first environment was under `target/descriptor-ci/venv`.
Recreating the environment at the supported path resolved it without changing
the validator or adding skips. The initial failure, final results, warnings,
commands and actual provenance are retained in the
[author receipts](cuda-add-descriptor-ci-author.tar.gz). Prior committed evidence
is byte-identical and has not been attributed to this revision.

This is dirty author verification, not a clean-commit capture or CI approval.
macOS ARM64/Python 3.12 and Ubuntu/Python 3.14 Actions remain required after
publication. No dependency, implementation, evaluator, managed progress artifact
or unrelated documentation changed; no commit, push or publication was performed.
