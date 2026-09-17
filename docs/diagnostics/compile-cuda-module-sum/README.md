# Positional top-level CUDA row sums: development evidence

This is non-scoring implementation evidence. The frozen baseline was independently
reproduced on clean `a8ea1225406d3aaa0412d74091ecd8aef97ea1a7` before source edits:
144 strict-reference-eligible cells, 96 expected capture gaps, 48 existing positive
controls, and four unrelated-sum-rebinding controls. Every cell, input hash and
control result matches the supplied proof. The original probe bytes are retained
with SHA-256 `9e8e66df128c5b7a7d0a56811dc1d704e6f2f8e4ff3867f3dc74568836f538c6`.

The interpreter alone was copied from the authorized relocatable installation
into an absent worktree-local destination after verifying the complete manifest.
The environment, locked dependencies and release wheels were built here. Python
3.12.14, Rust 1.92.0, PyTorch 2.13.0+cu130 and native CUDA runtime 13.0 were used;
installed nvcc is 12.6.85. Production uses embedded PTX and the driver, not nvcc
or installed PyTorch. GPU 0 is `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`;
GPU 1 is `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. Both are H100s with
driver 580.82.07. Recorded utilization/memory snapshots are observations, not
reservations. Single-GPU commands use mask `0`; restoration uses `0,1` and
hardware-hidden checks use an empty mask, including the leading `env` override.
All environments, caches, test outputs and build artifacts stayed in this worktree.

The first focused attempt is retained: standalone sums still selected the Python
node dispatcher, and one new fixture used unsupported tuple slicing. Single-node
reductions now use the existing native graph bridge; the fixture uses equivalent
chained slices. Assertions were preserved. The existing negative `native.sum(a, dim=-1, keepdim=True)` in
`tests/test_cuda_sum_rows.py` remains unsupported and unchanged; the new tests
explicitly cover the supported positional spelling. No existing fixture or
scoring corpus was changed. Every executed test version can be reconstructed from the
retained source snapshots and the recorded base commit.

The development capture does not represent a clean implementation commit.
Burner must create its managed implementation commit before the separate clean
capture and review/merge gates can run. This agent is prohibited from committing,
opening PRs, pushing, or modifying Burner state. No delivery/review metadata or
blocked-push safeguards were changed. PRs #1970/#1971 remain separate unadopted
human-review campaigns. These correctness checks make no timing or general
compiler, training, Inductor, hardware or summation-exactness claim.

## Final-source results

| Check | Result |
| --- | --- |
| Frozen candidate replay | All 144 cells pass strict eager/Inductor/native checks; 96 gaps closed, 48 controls preserved; all input hashes match the baseline |
| Complete compiler selection | 746 tests across 65 disjoint module processes; 21 reported skips; no failures |
| GPU 0/1 restoration | 2 tests passed, no skips, covering top-level and existing method reductions |
| Native eager and public/reference sum tests | 57 tests; 1 expected two-GPU skip |
| CUDA-hidden frontend/CPU boundaries | 59 tests; 1 expected hardware skip |
| Rust default / Python-binding all-target tests | 395 / 422 passed; zero harness failures or ignored tests |
| Explicit Rust CUDA row sums | 4 passed under mask `0`; no early-return skip messages |
| Rust native graph planner | 7 unit tests passed; no zero-test selector |
| Formatting / Clippy | `cargo fmt --check` and default/Python-binding Clippy with `-D warnings` passed |
| Documentation | 12 smoke tests, relative navigation targets, and the CUDA example passed |

The compiler selection includes every `test_compile*.py` (including
`test_compiled*` and `test_compiler*`) and `test_top_level_compile.py`.
It ran on the final production/test bytes, independently verified against its
recorded source manifest. Reported skips are retained per module, not credited
as GPU execution. Rust harness totals include CPU and capability-gated tests;
the explicit hardware checks above identify the executed row-sum cases.
Reference warnings, including JIT deprecation and Dynamo recompilation messages,
remain verbatim in the logs. No timing credit is taken for process splitting.

[Development identity](development.json) binds the source, native extension,
release wheels, interpreter, runtime, compiler and dependency configuration.
[Commands](commands.json) retain statuses, masks (including leading overrides),
GPU snapshots, warning lines and source snapshot IDs.
[Source history](source-history.json) describes lossless reconstruction against
the base commit. The [raw archive](raw-evidence.tar.gz) contains original probe
and bootstrap bytes, the adapted bootstrap, actual baseline/candidate results,
all per-module logs and receipts, executed capture scripts, source manifests,
and content-addressed copies of every changed source version. Unchanged sources
reconstruct with `git show a8ea1225406d3aaa0412d74091ecd8aef97ea1a7:<path>`;
verify their hashes against archived `setup.json`, then apply the selected
snapshot's changed/added blobs from `sources/<sha256>` and deletions.
`complete-summary.json` and `candidate-closure.json` inside the archive contain
the full module accounting and independently checked input-hash equivalence.
The interpreter copy's command is in archived `bootstrap-command.json`.
The capture recipes use worktree-local paths throughout; a later clean capture
must use a separate output directory and its managed commit/source identities.

The first final navigation audit started before evidence packing completed and
reported the not-yet-created `development.json`. That failed command and its
README fixture are retained; the unchanged audit was rerun after packing.
