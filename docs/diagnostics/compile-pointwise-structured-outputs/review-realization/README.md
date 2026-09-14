# Realization and observable-order repair

This record describes development validation after `0f7e6296470c1e6791fdc9c82934db8ab51e63b5`, from the worktree and source hashes recorded in the accompanying evidence. The implementation was uncommitted during these runs. It is not a clean-commit qualification or a fixed-corpus performance measurement; Burner must commit the implementation before a fresh post-commit capture.

The review's finite cancellation case exceeded the former ten-node certificate and fell back to joint lowering. The separate 2,117-node case required different rounding for `(q,r)` and `(r,q)` despite identical canonical arithmetic. The repair replaces the certificate/fallback with bounded realization and fusion planning, carries first-observable computed-root order from the immutable result specification, and preserves canonical expression identity across rounded imports. Public output allocations still follow original SSA identities.

A typed scalar instruction plan executes all numerical regions in one native CUDA kernel. The graph-keyed executable is reused when output order changes. Invocation-owned instruction and register buffers remain live through synchronization and failure. Scratch is capped at 64 MiB by limiting workers and using a grid-stride loop. Instruction dispatch and scratch traffic add costs; these correctness runs establish no speedup or evaluator score.

The regressions cover the reported finite result, output-order-only changes with executor reuse, cancellation, overflow, signed zero, scalar and vector shapes, nested repeated containers, persistent wrappers, retained prior outputs, duplicate imported expressions, and the 4,096-node/64-output native bounds. Existing numerical tolerances and zero-sign assertions remain unchanged.

## Final source-bound checks

| Check | Result |
| --- | --- |
| H100 pointwise suite, `CUDA_VISIBLE_DEVICES=0` | 296 passed; eight two-device tests skipped |
| Two-device checks, `CUDA_VISIBLE_DEVICES=0,1` | 8 passed |
| Release native pointwise tests | 58 passed, including real CUDA ownership and maximum bounds |
| Python conversion-failure ownership | 1 passed |
| CPython 3.10.19, 3.11.15, 3.12.12, 3.13.13, 3.14.5 | Each: 145 portable passes, 159 hardware skips |
| Clippy, all targets and Python bindings, warnings denied | Passed |
| Formatting, native extension verification, installed wheel/source identity | Passed |

At shape `(1,)`, the seven-sine unreturned-producer case returns `1192.0928955078125` on both implementations. At shape `(2,)`, the 2,117-node order case returns `q=1024.0` for `(q,r)` and `q=1192.0928955078125` for `(r,q)` on both implementations, reusing the same native executor. Cancellation, overflow and subnormal variants pass strict comparisons. The native maximum test executes 4,096 nodes and 64 outputs in both orders with 4,225 instructions, 104 registers and 106,496 scratch bytes for 256 workers.

The release wheel was built with locked, offline Maturin/Cargo from the recorded source snapshot (Rust 1.92.0, Maturin 1.15.0). The H100 reference was PyTorch 2.13.0+cu130; native JIT used NVRTC 13.0 and loaded CUDA runtime 13.0. The default `nvcc` reported 12.6.85. Exact GPU UUIDs, driver, loaded library paths and hashes are retained in runtime/toolchain provenance. Tests compare ordinary `torch_rs.compile(fn)` and ordinary stock `torch.compile(fn)` without backend, mode, fullgraph or dynamic overrides. The runtime probe checks native execution before Torch is imported, then checks outputs against stock Torch. Reference/build caches are confined to the report root and reused across development attempts; these are not cold-performance measurements.

## Evidence and retention

[measurements.json.gz](measurements.json.gz) retains exact commands, timestamps, environments, dirty state and source hashes for all wrapper-recorded runs, including failures and superseded builds. [raw.tar.gz](raw.tar.gz) retains the raw logs, numerical JSON, generated CUDA/PTX and executable-plan listings, runtime/import/toolchain provenance and design reviews. Identical captured files use tar hard links; their full original paths and byte hashes remain in the manifest. [raw-manifest.json.gz](raw-manifest.json.gz) verifies each packaged byte; [verification.json](verification.json) records archive hashes, final-check validation and historical-file preservation.

The canonical report root is `target/default-compile-eval/structured-outputs-realization-repair/` in the recorded worktree. It additionally retains the byte-exact original reviewer archive, all wheels, the exact source snapshot, original reference compiler caches, the superseded packaging attempt, and `retention-manifest.json.gz` for Burner's outer archival observer. No report root was deleted or written outside the worktree. All 42 pre-existing checked-in structured-output evidence files remain unchanged.

The original failed compile, old-source assertion failures, style failures, and the initially failing design gate remain retained. The separate gate recheck passes after sharing canonical identity across planning and lowering. A suspected constant-count discrepancy was disproved by a reference-only FX probe; both the initial hypothesis and correction are retained, and no speculative constant-analysis change was made. None of these intermediate records is presented as final-source validation.

This repair changes native execution strategy and carries instruction-dispatch, upload and scratch costs. The unchanged canonical evaluator alone determines coverage or performance credit. No evaluation definitions, benchmark workloads, tolerances, managed progress artifacts or historical measurements were changed.
