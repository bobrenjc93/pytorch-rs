# Clean-commit top-level CUDA reshape capture

Measured implementation commit **`32977e5dbab09e4b069dad921286d31697ecadb5`**
on H100 after Burner's implementation commit. Every workload started with an
empty Git status; only this evidence and navigation documentation were written
after all measurements completed. The [historical development capture](../README.md),
its original failures and exact-main baseline remain unchanged.

The frozen replay retains all **144 original cells and four rebinding controls**.
All 96 native capture rejections close: **94 strict-parity closures and two
zero-credit diagnostics**. The two empty add-then-reshape cases still have eager
strides `(1, 1)` and Inductor strides `(0, 0)`; native compiled outputs match eager
values and metadata exactly. Input hashes, programs, reference settings and
classification match the historical baseline. The focused regression test also
runs all 144 cells under all four compile policies, plus view/pack, raw-bit,
lifetime, guard, startup, prevalidation and recovery checks.

## Observed checks

| Check | Result |
| --- | --- |
| Complete compiler selection | 64 modules; 733 run, 713 passed, 20 skipped; 25 warning records |
| Frozen replay | 144 cells + 4 rebinding controls; 96 closures, 94 strict eligible, 2 zero-credit diagnostics |
| Focused CUDA/reference regressions | 12 run, 11 passed, 1 skipped |
| CUDA-hidden frontend/CPU/reference boundaries | 84 run, 36 passed, 48 skipped |
| Two-device restoration | 4 run, 4 passed, 0 skipped |
| Default Rust CUDA packing/reshape | 1 passed on H100; hidden run returned early |
| Python-binding Rust graph/planner | 15 passed on H100; 15 passed with CUDA hidden (hardware sections return early) |
| Formatting and default/binding Clippy | All passed |
| Documentation/navigation | 12 run, 12 passed, 0 skipped; CUDA guide example passed |
| Interpreter, frozen-input and installed-wheel audits | Passed; all 60 installed package members verified |

The [compiler partition](compiler-partition.json) is exhaustive and disjoint:
every `tests/test_compile*.py` file plus `tests/test_top_level_compile.py`, one
subprocess per file, with at most three file processes running concurrently.
[Totals and exit statuses](compiler-summary.json) count test functions, not
subcases. Exact skips and warnings remain in the logs and receipts. The default
`--lib cuda_graph` attempt selected **zero tests** (180 filtered out) and earns
no validation credit; graph tests require Python bindings. The default
`cuda_contiguous` integration test supplies the native layout check. Hardware
sections return early in CUDA-hidden Rust runs; their successful status is not
hardware coverage.

## Source and environment

The unchanged repository [release builder](../../../../scripts/build_cuda_add_diagnostic.py)
exported the exact commit into a new worktree-local directory, built with
`maturin build --release --locked`, and installed the resulting wheel. It used an
empty per-export Cargo build directory and warm local download caches. The
existing isolated `.venv` was synchronized with locked dev/reference dependencies;
no sibling environment or wheel was copied. The already local CPython 3.12.14
interpreter was reverified against the original full file/mode/internal-link
inventory and binary SHA-256, including isolated executable/stdlib identity.

[Build provenance](build-record.json.gz) retains the full source inventory,
wheel/native hashes, source/export paths, cache state and build durations.
[Command receipts](commands.json.gz) retain timestamps, complete commands,
effective masks including leading `env` overrides, native/interpreter identity,
source snapshots, GPU observations, statuses, warnings and raw-log hashes.
[Source snapshots](source-snapshots.json.gz) reconstruct every executed
production/test file from the measured commit; no test-source edits occurred.
Unmodified raw receipt text is retained by command name in
[raw receipts](raw-receipts.json.gz); the command index omits
incidental pre-command Python imports for the build and standalone Cargo runs.
All 60 installed package members matched the wheel, and Python source members
matched the commit. Standalone Cargo receipts make no Python-extension claim.

Ordinary GPU runs used `CUDA_VISIBLE_DEVICES=0`; restoration used `0,1`.
GPU 0: `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`; GPU 1:
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. Driver 580.82.07, selected local CUDA
runtime 13.0 (`13000`), PyTorch 2.13.0+cu130, Rust 1.92.0 and nvcc 12.6.85 were
observed. [Runtime provenance](logs/provenance.log.gz) records actual mapped
runtime and compiler versions. GPU snapshots are observations, not reservations.
Correctness processes overlapped and reference caches were warm. **No timing,
performance, scoring or general compiler/Inductor/training parity is claimed.**

## Retained recipes and verification

Exact executed recipes are gzip files under `recipes/`. [Adaptation records](recipe-adaptations.json)
and unified diffs retain changes from committed development recipes: only
path/head provenance guards and file-level orchestration changed. The frozen
program/input/reference workload is unchanged. [Historical input bindings](historical-inputs.json)
link the original probes, survey, failure and baseline without relabeling them as
new measurements. To reconstruct the capture, decompress these recipes and the
linked historical bootstrap/frozen inputs into the recorded worktree-local
staging paths; source `recipes/env.sh`, then use the recorded commands. A new
build requires an absent local export directory and corresponding path guards.

[Publication verification](verification.json) checks log hashes, source/wheel
identity, unchanged historical measurement bytes, the full compiler partition,
links and the evidence-only follow-up diff. [Artifact inventory](artifact-manifest.json)
binds every retained file. This capture supplies post-commit evidence; independent
review, evaluations and merge gates remain Burner's responsibility.
