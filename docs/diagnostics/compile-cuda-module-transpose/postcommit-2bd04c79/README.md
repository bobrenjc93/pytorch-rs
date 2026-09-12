# Clean-commit positional transpose capture

Fresh, non-scoring correctness evidence measured on clean implementation commit
`2bd04c79fb815ad2c139e85bacb5e23161b93c20`. The release wheel was rebuilt from an exact
commit export using the unchanged repository builder. Every workload receipt
records that commit, an empty Git status before and after, unchanged production
and input manifests, and the worktree-local native extension where applicable.
The subsequent changes contain only this evidence and its documentation link.

The [historical development capture](../README.md), including the exact-main
96-gap probe, original failures, inputs and recipes, remains byte-for-byte
unchanged. This capture completes its deferred clean-commit step. Independent
review, revision and fresh merge gates remain Burner's responsibility.

## Observed results

| Check | Result |
| --- | --- |
| Focused transpose, native/reference; four policies | 13 run: 12 passed, 1 skipped |
| Remaining 62 compiler modules | 708 run: 690 passed, 18 skipped |
| CUDA-hidden / CPU / reference | 148 run: 86 passed, 62 skipped |
| Two-physical-GPU restoration | 5 run: 5 passed, 0 skipped |
| Rust default, all targets (CUDA hidden) | 395 passed, 0 ignored |
| Rust Python-binding graph/planner (H100) | 15 passed, 0 ignored |
| Rust Python-binding graph/planner (CUDA hidden) | 15 passed, 0 ignored; hardware sections return early |
| Documentation/navigation and CUDA example | 12 run: 12 passed, 0 skipped; CUDA example passed |
| Formatting and default/Python-binding Clippy | Passed |
| Native extension, wheel/source and evidence provenance | Passed |

The complete compiler selection contains all 63 modules matching
`tests/test_compile*.py` plus `tests/test_top_level_compile.py`: **721 run,
702 passed, 19 skipped**, with no failures or errors. It was run in two
concurrent fresh processes: the unchanged new transpose module and all remaining
modules. Their selections are disjoint and exhaustive, verified in
[verification.json](verification.json). This process partition keeps the required
capture within the post-commit time limit; tests, inputs, policies and assertions
are unchanged. The transpose matrix covers the original 96 spelling cells under
all four policies, with cold/repeated native and reference execution. Counts in
the table are test functions, not matrix cells. Exact skip reasons and all
warnings remain in the losslessly compressed logs.

## Provenance and reproduction

[commands.json](commands.json) preserves actual commands, timestamps, exit statuses,
GPU snapshots, effective CUDA masks (including leading `env` overrides), and
content-addressed production/input manifests. [environment.json](environment.json)
records Python 3.12.14, Rust 1.92.0, PyTorch 2.13.0+cu130, runtime/driver API 13.0,
NVIDIA driver 580.82.07 and nvcc 12.6.85. Native kernels use driver JIT of embedded
PTX; nvcc is not invoked by the native build. The release profile uses thin LTO,
one codegen unit, extension-module bindings and locked dependencies.

GPU 0 is `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`; GPU 1 is
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`, both H100s. Ordinary GPU commands use
`CUDA_VISIBLE_DEVICES=0`, restoration uses `0,1`, and hidden commands use the
empty mask. Correctness processes shared GPU 0; snapshots are observations, not
reservations. Resources were `gpu` and `cpu-heavy`. Command durations are elapsed
validation time, with no performance or parity credit claimed. Pure Rust commands
do not claim a Python-extension identity.

The existing worktree-local relocatable interpreter was verified again against
the complete 4,927-entry file/mode/internal-link inventory and the supplied
interpreter hash, including isolated executable/stdlib/include identity. No new
interpreter copy was needed. The old development environment was moved to
`target/module-transpose-postcommit/development-venv`; a fresh canonical `.venv`
was created with the verified interpreter. No project environment, wheel or
installed package was copied. The interpreter receipt precedes this replacement
and is setup evidence only. The build receipt starts before wheel installation;
subsequent workload receipts bind the newly built native extension.

[build-record.json.gz](build-record.json.gz) preserves the builder's original bytes,
including the complete exact-commit source manifest, wheel/native hashes, build
paths and setup/build durations. It used an empty per-export Cargo target and
warm worktree-local Cargo/uv download caches. Python/test caches were new under
`target/module-transpose-postcommit` at capture start and shared by these runs.
The unchanged builder explicitly selects its own worktree-local Cargo cache,
`target/tmp`, `target/cuda-cache`, and the per-export build target; see its source
at the measured commit. All executable/import/build/cache paths are inside this
worktree, apart from existing read-only system build tools and driver libraries.

The saved [environment recipe](recipes/env.sh.txt),
[command recorder](recipes/record.py.txt),
[selection recipe](recipes/selection.py.txt), and
[read-only audit](recipes/audit.py.txt) make the capture reproducible.
Restore the recipes to `target/module-transpose-postcommit/` in a clean checkout
of the measured commit, source `env.sh`, verify the local interpreter, then create
`.venv` with `uv venv --python
"$UV_PYTHON_INSTALL_DIR/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12" .venv`.
Run the recorded builder command (`scripts/build_cuda_add_diagnostic.py`,
`--revision HEAD`) and the remaining recorded commands through `record.py`, using
fresh output destinations. `uv sync --locked --no-install-project --group dev
--group reference` is performed by the builder before the release wheel build.
The dependency inventory and setup/build output are retained under `logs/`;
[assets.json](assets.json) records hashes of the original uncompressed bytes.

No evaluator, scoring corpus, benchmark denominator, implementation, dependency,
test or benchmark harness changed during this step. Historical performance
artifacts are not refreshed or used as current-candidate credit. The scope
remains three positional arguments with exact constant integer axes, using the
existing CUDA float32, no-grad, rank-0–2 view planner/executor.
