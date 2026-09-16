# Broadcast trig: compact redelivery

This replaces the delivery in [PR #2000](https://github.com/bobrenjc93/pytorch-rs/pull/2000),
not an additional independently creditable feature. The 11 implementation,
test and documentation files come from reviewed source
[`6255306`](https://github.com/bobrenjc93/pytorch-rs/commit/6255306165615d9502a4aa491187bcf626e9d585)
(tree `de3d425c8468ab595f79efd2d11f6c3cac0c3a99`), directly based on
[`9a36b4a`](https://github.com/bobrenjc93/pytorch-rs/commit/9a36b4a4642528d5af7533b309a8562b74447084).
All seven changed/new test files remain byte-identical. Beyond the permitted
compiler-guide scalar-leaf example and supported-surface opening corrections,
the indexing rejection loop now has the equivalent CI repair described below.

The [compiler guide](../../compile-pointwise-jit.md#supported-programs) owns
admission; the [numerical guide](../../compile-pointwise-numerics.md#unequal-shape-numerical-boundary)
owns rounding and numerical limits. Unequal actual argument shapes admit every
returned root at arithmetic depth at most one, including live sin/cos. The
exact tensor-leaf multiply-add exception requires no live trig in any returned
root. Numerical planning and the generic native CUDA executor are unchanged.

## Current clean-commit evidence

Measured on 2026-09-16 UTC at clean repaired commit
[`9ef4515`](https://github.com/bobrenjc93/pytorch-rs/commit/9ef4515fe7537d8cc0ac1d11a2f1ad52856f7313).
All 13 command receipts record that commit and empty Git status before and after
execution. Source and tests remained unchanged throughout; evidence and this
index were added afterward. There were no test failures.

| Check | Result |
| --- | --- |
| Both CI Clippy configurations, with warnings denied | Passed |
| Rust original-IR indexing | 11 passed |
| Focused H100 Python selection, including all seven trig tests | 21 passed |
| H100 ownership, structured I/O and scalar histories | 31 passed; 2 explicit two-device skips |
| CUDA-hidden trig suite | 2 passed; 5 explicit GPU skips |
| Formatting, installed-extension provenance and diff whitespace | Passed |

[Build receipt](postcommit-9ef4515/build.json),
[runtime identities](postcommit-9ef4515/runtime.json),
[commands](postcommit-9ef4515/commands.json) and
[complete text outcomes](postcommit-9ef4515/outcomes.log) total about 58 KiB.
Native Rust compilation occurred in an empty target (50.948 seconds Maturin
wall time). The existing worktree-local locked environment, interpreter and
Cargo registry were reused; CUDA/Inductor/Triton cache directories were new.
The wheel and imported extension match, and installed Python sources match the
commit. Python was 3.12.14, PyTorch `2.13.0+cu130`, NVRTC 13.0 and CUDA runtime
13000. Installed nvcc 12.6 was not used for the pointwise JIT. The build tool's
embedded-PTX label concerns its base CUDA inventory; the runtime receipt records
the pointwise NVRTC path exercised here.

GPU 0 was H100 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver
580.82.07. Before/after snapshots showed 0% utilization and 4 MiB used.
This is single-GPU evidence, with unchanged public `torch_rs.compile(fn)` and
`torch.compile(fn)` defaults, compiler limits and tolerances.

Reproduce with the [locked setup](../../../CONTRIBUTING.md#locked-setup), the
[repository build tool](../../../scripts/capture_depth_concat_build.py), and
the recorded test argument vectors. Export the recorded environment with
current worktree paths and a fresh build output directory; keep all writable
caches local. The full Rust suites already checked during repair were not
repeated here; the focused indexing checks and both Clippy configurations were.

Canonical qualification is pending. These are focused checks, not candidate
scores or independent review. The [canonical evaluation contract](../../torch-compile-default-evaluator.md#evidence-retention)
owns fixed evaluations and raw archives. No full fixed benchmark or duplicate
`--metric both` capture was run here. Reproduction does not require recovering
ignored files after delivery. Burner owns CI, publication and merge qualification.
Retirement of PR #2000 remains maintainer-owned; this record does not claim it
is closed or its commits deleted.

## PR #2001 CI repair

The [required `test` job](https://github.com/bobrenjc93/pytorch-rs/actions/runs/35046757556/job/104638120458)
failed Rust 1.92 Clippy: `Graph::indexing` had 101 lines against the 100-line
limit. Replacing its rejection loop with a short-circuit `any` predicate keeps
the same condition, validation order and error message, without a lint exemption.

The [repair receipt](ci-repair.json) preserves the local reproduction of that
failure and subsequent working-tree checks: both CI Clippy configurations,
formatting, 464 Rust tests without Python bindings and 493 with bindings passed.
A newly compiled local wheel passed 21 focused and 31 additional H100 tests,
plus two portable trig tests; seven hardware skips remain explicit. Tests,
compiler defaults, tolerances and evaluator definitions are unchanged.

The repair was committed as `9ef4515`. Its clean focused evidence is recorded
above; the working-tree repair receipt remains a separate development record.
External required checks and canonical qualification remain Burner-owned.

## Earlier clean capture at `876627d`

The [build](postcommit-876627d/build.json),
[runtime](postcommit-876627d/runtime.json),
[commands](postcommit-876627d/commands.json) and
[outcomes](postcommit-876627d/outcomes.log) remain unchanged and pinned to
[`876627d`](https://github.com/bobrenjc93/pytorch-rs/commit/876627dbc12662aa0d3357bf8055ed1ee1137b68).
That clean capture preceded the CI repair: 11 Rust indexing tests, 21 focused
H100 tests, 31 additional H100 tests, two portable trig tests and 94 portable
admission/cache tests passed, with seven explicit hardware skips. Formatting,
import provenance and whitespace checks passed. Its native compile took
52.240 seconds in an empty target and produced the same native hash as the
pre-delivery compile. `db3df93` subsequently added only evidence/index updates.
These earlier measurements do not qualify the repaired source; the current
clean capture above supplies the required refreshed focused evidence.

## Historical evidence, not current qualification

The original pre-delivery [commands](commands.json), [identities](provenance.json)
and [outcomes](outcomes.log) remain unchanged and explicitly describe working-tree
checks before `876627d`. That same focused matrix passed, with the same seven
explicit skips; its native compile took 51.663 seconds. Those earlier results
do not substitute for the clean-commit rerun above.

The `6255306` postcommit record reported 11 Rust indexing tests and 21 focused
H100 Python tests passing, plus two portable passes and five explicit GPU skips.
Coverage was 19.5 with 20/112 successful cells; CUDA performance was
33.7010604577856 with 20/56 successful cells. Both release-wheel packaging
steps reused the same native artifact and reported 0.02 seconds: neither was
a clean native rebuild. Subsequent
[`5fd18af`](https://github.com/bobrenjc93/pytorch-rs/commit/5fd18af02c4cad407c888c65daac6f0b2e627588)
delivery measured CUDA performance 33.1 and polish 77. None qualifies this head
or establishes general Inductor equivalence. Its 27 evidence files are not
copied into this delivery, and that commit is not added to ancestry.

The campaign retains clean raw evidence in the host-local ignored archive
`.burner/diagnostic-archives/broadcast-trig-6255306165615d9502a4aa491187bcf626e9d585/clean-evidence-5fd18af.tar`
(161904640 bytes, SHA256
`e7bf4dc54111c4c492e9b2705eac9a65daf7c8b506441ddf194082a5d8e0d92f`).
Its adjacent `manifest.json` has SHA256
`eb9c04b1a414dc5f6a283234409413d83415f4f792caf0fda3da95c1b7e4c454`.
Historical development archives and their preservation receipt remain in that
directory, including failed attempts. Delivery coverage/performance archives
are `.burner/evaluation-runs/evalrun_57fe6ef0` and `evalrun_9136cef0`.
These locations and hashes are historical preservation facts supplied for this
redelivery, not newly generated captures or a replicated public artifact service.
Public commits and PR references remain available; the focused tests and locked
setup above are the reproduction path. No old capture is modified or rerun.
