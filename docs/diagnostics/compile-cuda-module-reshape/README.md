# Native top-level CUDA reshape validation

Non-scoring development evidence for exactly two positional arguments to native
`reshape(x, constant_shape)` and genuine imported aliases. Shapes are exact
constant integer tuples/lists; the existing CUDA float32, no-grad, rank-0–2
planner and executor remain unchanged. See the [supported contract](../../compile-cuda-reshape.md).

Before production edits, a fresh locked release wheel at main
`6772c3405a8bc2baf70ca80106a27a390b19e9a0` reproduced all 144 frozen cells:
96 native capture rejections, 48 positive controls and four additional unrelated
rebinding controls. The retained v2 probe changes only external artifact paths.
It independently enforces the original reference classification: 94 eligible
gaps and two empty add-then-reshape diagnostics with eager strides `(1,1)` versus
Inductor `(0,0)`. Both diagnostics remain present and earn zero strict-parity
credit. Candidate replay closes all 96 native rejections while preserving all
inputs, reference settings and native eager metadata checks. The [regression matrix](../../../tests/test_compile_cuda_module_reshape.py)
additionally exercises all four policies, cold/repeated/lowered reuse,
raw bits, views/packs, identities, mutation/lifetime, guards and error boundaries.

The original probe, failure, reference survey, v2 probe, audit receipts and recipes
are retained byte-for-byte as gzip files. Their original SHA-256 hashes are in
`frozen-manifest.json`. No scoring corpus, evaluator definition, benchmark evidence,
weight, hardware matrix or managed progress artifact is changed. PR1970/PR1971
remain separate unadopted human-review campaigns. No timing or score credit is claimed;
reference caches were warm on some reruns and correctness processes overlapped.

## Results

| Check | Result |
| --- | --- |
| Complete compiler selection, 64 modules | 733 run, 713 passed, 20 skipped; 25 warning records |
| Frozen exact-main baseline | 96 expected rejections, 48 positive and 4 rebinding controls passed |
| Candidate frozen replay | All 96 rejections closed; 94 strict closures, 2 zero-credit diagnostics |
| New CUDA/reference regression module | 12 run, 11 passed, 1 two-device skip |
| CUDA-hidden frontend/CPU/reference boundaries | 84 run, 36 passed, 48 hardware skips |
| Two-device restoration | 4 passed |
| Rust default all-targets | 395 passed, CUDA hidden |
| Rust Python-binding graph/planner | 15 passed on H100; 15 passed with CUDA hidden |
| Formatting and default/Python-binding Clippy | Passed |
| Documentation/navigation and guide example | 12 tests and CUDA example passed |
| Source/wheel/frozen-input audit | Passed |

Counts are test functions, not generated subcases. The 144-cell matrix runs under
all four policies (576 cell/policy combinations); the two reference diagnostics
retain zero strict credit in every policy. Exact skips, warnings and prior failed
attempts are retained in command logs. The [complete compiler summary](compiler-summary.json)
and [partition](compiler-partition.json) account for every `test_compile*.py`
file plus `test_top_level_compile.py`, in disjoint subprocesses with no omissions.
All 64 exit statuses are zero. Warning records are preserved verbatim in the
receipts/logs; they are not suppressed or treated as performance evidence.

## Reproduction and provenance

[Command receipts](commands.json) record commands, statuses, effective masks (including leading
`env` overrides), test totals and full logs. [Source manifests](source-manifests.json) and content
blobs preserve every executed production/test revision as deltas against the
retained base manifest at the commit above. Unchanged files are reconstructed
with `git show <base>:<path>`; changed files use the SHA-named gzip blobs. [Build records](builds.json) bind immutable source exports, wheels and installed extension hashes.
The final source/native verification checks all 60 installed package members
against the release wheel and production source. Standalone Cargo results describe Rust commands, not Python extension tests.

Only the verified relocatable interpreter was copied, into an absent worktree
path, after verifying the full file/mode/internal-link manifest. Dependencies
were freshly installed from the lockfiles; all environments, caches, source
exports and build artifacts stayed inside this worktree. Runtime provenance and
GPU before/after snapshots are observations, not reservations. Tests used H100
GPU 0 (`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`), adding GPU 1
(`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`) only for restoration. Driver
580.82.07, worktree-local CUDA runtime 13.0, PyTorch 2.13.0+cu130,
Python 3.12.14, Rust 1.92.0 and default nvcc 12.6.85 are recorded. The reshape
change introduces no CUDA compiler invocation or kernel; existing private-kernel
regressions in the full sweep retain their nvcc-based checks.

Retained failures include an early frontend launch before package installation;
the first implementation's list-consumption rejection; and test fixtures with
two helper calls (the existing limit is one) and a singleton incorrectly treated
as noncontiguous. The corrected fixtures preserve those compiler boundaries.
The two existing negative module-reshape placeholders now use `shape=`; the
method test also uses its actual `m` module binding, so rejection exercises the
unsupported signature instead of an undefined global. That fixture was corrected
before its sweep subprocess (earlier sweep modules do not import it); both source
versions are retained. Positive
positional tuple/list coverage is explicit in the new test module.

The separate [clean-commit capture](postcommit-32977e5b/README.md) measures
`32977e5dbab09e4b069dad921286d31697ecadb5` after Burner's managed implementation
commit. These development measurements remain pinned to their original source
and build identities. Burner owns independent review, draft delivery and fresh
merge gates. No branch, commit, push or PR was created by this implementation session.
