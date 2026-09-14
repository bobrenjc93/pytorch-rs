# Clean structured-output capture at 232cc734

Measured implementation: `232cc73423e6622aac6d16af71c5c55d3928933d`, with `main` at `30a3b504ef4d43bf2958998cc39545996cc09970`. This supplies the clean-commit capture deferred in the [duplicate-root repair record](../review-duplicate-roots/README.md). Every measured command recorded a clean checkout before and after execution, and post-command byte fingerprints matched the capture baseline for all 6,797 tracked files.

## Results

The unchanged committed compiler tests ran against a fresh source-bound release wheel. They compare ordinary `torch_rs.compile(fn)` with ordinary stock `torch.compile(fn)`, preserving existing tolerances, zero-sign/nonfinite assertions, persistent wrappers, aliases and no-original/helper-replay checks.

| Check | Measured result |
| --- | --- |
| Pointwise suite on H100, device 0 | 305 total: 297 passed, 8 two-device skips |
| Required two-device checks, devices 0 and 1 | 8 passed, no skips |
| Release native pointwise/ownership checks | 59 passed |
| Python conversion-failure ownership | 1 passed |
| Portable checks, each CPython 3.10–3.14 | 145 passed, 160 hardware skips |
| Clippy, formatting and native extension verification | Passed |
| Installed wheel identities on all five interpreters | Passed |

Portable versions were 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware skips are not GPU passes.

The duplicate-product regression passes all 720 comparisons across five product constructions, three return topologies, sizes 1/13/257, eight value histories, and first/repeated calls. It also verifies separate allocations, repeated aliases and unchanged retained prior outputs. For `p=x*y; p2=x*y; q=(-x)*y; r=p+q`, both implementations return `r=+168.0928955078125` with `x=1e10, y=1.0000001192092896`, and `r=+Infinity` with `x=2e38, y=-2e38`, at all three sizes on first and repeated calls. `numerical-summary.json` in the raw archive links these observations to their original files and hashes.

The existing large-graph rounding, observable return-order/executor reuse, nonlinear persistent-history, helper/loop/branch composition, metadata, empty/offset storage and failure-atomicity regressions also pass. These bounded checks do not establish general Inductor parity or approve the branch.

## Provenance and retention

Wheel SHA256: `102e6e67bf4e9edb6b0f7fe5e1daaeef8ac3d1be51b84b54050b395d23e85a2b`.

The locked offline release build used a fresh Cargo target. Reference, Triton and CUDA caches were initially empty and remained inside this worktree; later checks reused these report-local caches. Build/runtime records identify Rust 1.92.0, PyTorch `2.13.0+cu130`, H100/driver 580.82.07, NVRTC 13.0 and CUDA runtime 13000. PATH `nvcc` reports 12.6.85; native kernels were compiled by NVRTC. The runtime probe executed native compilation before importing Torch, verified structured aliases and compared materialized outputs with the ordinary default reference.

[Measurements](measurements.json.gz) retain timestamped commands, source hashes, clean status, environments and return codes. [Raw captures](raw-captures.tar.gz) retain numerical JSON, generated native CUDA/PTX and instruction plans, logs, wheel/import/runtime/GPU identities and capture orchestration. Identical files use tar hard links while retaining every original path. The [raw manifest](raw-manifest.json.gz) and [verification](verification.json) verify all 10,952 packaged files and archive hashes.

The canonical root `target/default-compile-eval/structured-outputs-postcommit-232cc734/` additionally retains the wheel, byte-verified committed-source archive, original reference compiler caches and `retention-manifest.json.gz` covering 45,218 files. Burner's current report-root observer does not scan this custom directory. Its wheel, source archive and compiler-cache artifacts need explicit external retention before worktree cleanup. The capture performed no cleanup or external write and records no external archival path. Every earlier report, development failure and historical measurement remains unchanged, including the [523d51d7 capture](../postcommit-523d51d7/README.md).

Only this evidence and its compiler-guide link change. Implementation, tests, dependencies, benchmark harnesses, evaluation definitions and managed progress artifacts remain unchanged. No official scoring corpus or unrelated repository suite was rerun; this capture awards no performance or coverage score.
