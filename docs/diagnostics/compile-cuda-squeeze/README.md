# No-argument CUDA squeeze development capture

This non-scoring development capture starts at verified main
`22d2bd4e9bd211cc2c65e8186b08ee3c827196a2`. It is not a clean implementation-commit
capture and does not change any scoring denominator or historical evidence.

The unchanged corrected baseline probe was copied read-only from
`/tmp/pytorch-rs-compiler-recovery.3UB6mU/probe-compiled-cuda-squeeze.py` into this
worktree's ignored `target/` directory. A fresh exact-main release wheel reproduced
all 15 reference eager/compiled and native eager successes and cold/repeated native
capture rejections, plus nine eager binding controls. Seven programs exercised
no-argument squeeze; explicit-dimension cases were diagnosis only. Input metadata
was asserted equal before comparing outputs, including canonical empty layouts.

The implementation uses the existing parameterless unary-method mechanism,
method identity guards and checked native squeeze. Eager and native graph planning
share the layout calculation. No storage copy, kernel, fusion or Python replay was
added. The [guide](../../compile-cuda-squeeze.md) describes the bounded contract.

`validation.json` records commands, exit statuses, log hashes, GPU snapshots,
source/build/native identities and compact results, including the initial failing
method-guard test. That test caught an omitted guard-registry entry; the registry
was repaired and the unchanged assertions were rerun. Full raw development logs
remain under worktree-local `target/checks/`. Build records are included once per
source revision; identical interpreter/dependency configuration is recorded once.
`inputs.json` binds the test/probe inputs and unchanged helper files by hash.

Final validation used production source digest
`30c8a7ad78fa3e1e8a87b59446a6fa2d759d18c5d66be1b77d5655a993acb4e3`.
The installed wheel's Python files matched the checkout byte-for-byte; its native
extension matched the fresh release build. Native and reference loaded the same
worktree-local CUDA 13.0 runtime. Rust was 1.92.0; installed `nvcc` was 12.6.85,
with no `nvcc` compilation needed for native squeeze (embedded PTX for consumers).

| Check | Result |
| --- | --- |
| Focused H100 squeeze | 11 run, 1 two-device skip |
| Every `test_compile*.py` module (57 modules) | 599 run, 14 two-device skips |
| `test_top_level_compile.py` | 49 passed |
| Two-device squeeze/t/transpose restoration | 3 passed |
| Rust default all-targets, CUDA hidden | 395 passed |
| Rust python-bindings all-targets, explicit local CUDA runtime | 422 passed |
| CUDA-hidden squeeze, CPU/reference regressions, README/docs smoke | 31 run, 10 hardware skips |
| Formatting, Clippy default/python-bindings, four guide examples | Passed |

No rejection fixtures were weakened or replaced. Dimension-bearing squeeze
capture remains unsupported. Full command details and the initial failing
attempt are retained in the receipts.

Environments, downloads, builds, wheels and caches use this worktree:
canonical `.venv`, uv-managed CPython 3.12.14, local `UV_PYTHON_INSTALL_DIR`,
locked dev/reference dependencies and Rust 1.92.0. GPU resource use was declared;
ordinary CUDA checks used `CUDA_VISIBLE_DEVICES=0`, and restoration checks used
only `0,1`. Before/after inventory snapshots are observations, not reservations.
No other jobs were interrupted. These checks measure correctness, not speed.

Setup incident: `uv python install 3.12.14` also created the default executable
symlink `/home/bobren/.local/bin/python3.12` pointing into this worktree. This
violated the requested write boundary despite `UV_PYTHON_INSTALL_DIR` being local.
No further external writes were made; authorization to remove the accidental
symlink was requested. Future setup must use `uv python install --no-bin` or a
worktree-local `UV_PYTHON_BIN_DIR` (now set for subsequent commands).

Burner owns the implementation commit, normal draft PR, independent review,
revision and full-gate delivery. A separate fresh clean-commit capture must follow
that commit; this agent was explicitly prohibited from committing or publishing.
Reuse the existing capture helper and locked manifests, run every
`test_compile*.py` module plus `test_top_level_compile.py`, and publish into a new
capture directory without replacing this development record. No clean-commit
result is claimed here. PR1970/PR1971 remain unadopted separate campaigns.
