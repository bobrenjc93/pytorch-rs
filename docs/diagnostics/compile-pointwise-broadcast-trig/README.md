# Broadcast trig: compact redelivery

This replaces the delivery in [PR #2000](https://github.com/bobrenjc93/pytorch-rs/pull/2000),
not an additional independently creditable feature. The 11 implementation,
test and documentation files come from reviewed source
[`6255306`](https://github.com/bobrenjc93/pytorch-rs/commit/6255306165615d9502a4aa491187bcf626e9d585)
(tree `de3d425c8468ab595f79efd2d11f6c3cac0c3a99`), directly based on
[`9a36b4a`](https://github.com/bobrenjc93/pytorch-rs/commit/9a36b4a4642528d5af7533b309a8562b74447084).
Executable source and all seven changed/new test files are byte-identical.
Only the permitted compiler-guide scalar-leaf example and supported-surface
opening are corrected beyond those reviewed contents.

The [compiler guide](../../compile-pointwise-jit.md#supported-programs) owns
admission; the [numerical guide](../../compile-pointwise-numerics.md#unequal-shape-numerical-boundary)
owns rounding and numerical limits. Unequal actual argument shapes admit every
returned root at arithmetic depth at most one, including live sin/cos. The
exact tensor-leaf multiply-add exception requires no live trig in any returned
root. Numerical planning and the generic native CUDA executor are unchanged.

## Current candidate

Canonical qualification is pending. These are pre-delivery working-tree
checks, not post-commit results or candidate scores. Burner owns the commit,
clean-source focused rerun, ordinary fixed evaluations, archival, review, CI
and merge qualification. This record does not claim PR #2000 is closed or
its commits deleted. The replacement publication should identify #2000;
retirement through Burner's withdrawal API remains maintainer-owned.

Measured here on 2026-09-16 UTC, with no test failures:

| Check | Result |
| --- | --- |
| Rust original-IR indexing | 11 passed |
| Same focused H100 Python selection, including all seven trig tests | 21 passed |
| Additional H100 ownership, structured I/O and scalar histories | 31 passed; 2 explicit two-device skips |
| CUDA-hidden trig suite | 2 passed; 5 explicit GPU skips |
| Additional CUDA-hidden admission/cache regressions | 94 passed |
| Formatting, installed-extension provenance and diff whitespace | Passed |

The release build compiled native Rust in an empty local target (51.663 seconds
wall time); it did not reuse a native artifact. The locked local environment
used Python 3.12.14, PyTorch `2.13.0+cu130`, NVRTC 13.0 and CUDA runtime 13000.
Installed nvcc 12.6 was not used for the native JIT. GPU 0 was H100 UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07; before/after
snapshots both showed 0% utilization and 4 MiB used. This is single-GPU evidence.

[Command receipts](commands.json), [identities](provenance.json) and
[complete text outcomes](outcomes.log) retain elapsed times, hashes and skips
in about 60 KiB, with no raw tensors or evaluator reports. Reproduce with the
locked environment in [CONTRIBUTING.md](../../../CONTRIBUTING.md#locked-setup),
a worktree-local release wheel, and the receipt's test selections. Keep writable
Cargo, uv, Python, CUDA, Inductor, Triton and temporary directories inside the
worktree. Use `CUDA_VISIBLE_DEVICES=0` for H100 checks and an empty value for
portable checks; two-device tests explicitly skip without a reservation.
Public comparisons use unchanged `torch_rs.compile(fn)` and `torch.compile(fn)`.
No compiler-limit, backend or tolerance override is part of this redelivery.

The [canonical evaluation contract](../../torch-compile-default-evaluator.md#evidence-retention)
owns candidate scores and full raw archives. No full fixed benchmark or duplicate
`--metric both` capture belongs to this author record. Ignored local build files
are disposable; reproduction does not require recovering them after delivery.

## Historical evidence, not current qualification

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
