# Clean-commit structured-input correctness refresh

Measured implementation: `c6488789c0360f8d754b35c6108d9c9bc0dc55e3`, based on
`2f15b4ce53dc45918ea644a3370257f340747755`. This capture includes the structural
hash and direct flat-admission repair, plus the earlier CPython 3.10 tuple-unpack
fix. Every command recorded this same clean commit/tree before and after running.
Capture interval: **2026-09-15 19:27:02.917906–19:27:30.985448 UTC**. Reports were
written afterward.

**110 test executions passed; zero failures and zero skips.** These include
repeated checks across interpreters, not 110 distinct tests.

| Interpreter | Committed selection | Passed |
| --- | --- | ---: |
| CPython 3.12.14 | Complete structured-input module, including ten paired GPU history tests | 29 |
| CPython 3.10.21, 3.11.16, 3.13.15, 3.14.7 | Five portable structured-input classes | 19 each |
| CPython 3.10–3.14 | Malformed loop-stack regression | 1 each |

The portable classes are `SourceIdentity`, `InputAdmission`, `InputContracts`,
`InputLanguage` and `InputTransactions`. The additional selection is
`LoopAdmission.test_malformed_exit_backedge_and_body_stack`. Exact selected IDs,
assertion results, commands and process exits are preserved in the raw records.
No tests, workloads, tolerances or reference settings changed.

The GPU tests retain persistent ordinary `torch_rs.compile(fn)` and untouched
`torch.compile(fn)` wrappers through scalar/shape history, dict mutation, ABI
reorder, unequal Tensor replacements, aliases/views, helpers, local unpacking
and failed-call recovery. They compare output trees, metadata, float32 bits
(excluding NaN payload identity) and current-owner relationships, with native
body replay forbidden by existing helpers. Portable checks cover hash equality
and collisions, flat/tree transitions, bounds, callbacks, lazy observation,
retention and failure-atomic cache publication. The broader pointwise suite was
already run during development and was not repeated for this evidence refresh.

## Build and hardware

The existing `scripts/evaluate_torch_compile_default.sh --setup-only` rebuilt
and installed a release wheel with `maturin build --release --locked`. Cargo/uv
and interpreter environments were reused; this was not a cold build. CUDA,
Inductor, Triton and XDG capture caches began empty. All builds, imports and
caches are inside the current `full-leaf-agent_ea088824` worktree. Every wheel
Python source matched the committed checkout, imported tests matched the commit,
and all five installed native extensions matched the wheel.

- Wheel SHA256: `8c0b1b7eda45d3753916b341f7c5b1e9f3750995038eb3078d215de0f3b1990e`.
- Native extension SHA256: `29c47a9b6058b7036e8e15f3d7ae68665e7789000dc1522b0445318c145d1c04`.
- Frontend SHA256: `43b1a283edc9d5bddd2ba3b9972463ac74947c0846d685454879835b1c830e33`.
- `CUDA_VISIBLE_DEVICES=0`; H100 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`,
  capability 9.0, driver 580.82.07; one visible GPU and one host thread.
- Reference PyTorch `2.13.0+cu130`; loaded CUDA runtime and NVRTC 13.0;
  host nvcc 12.6.85. Dynamo was enabled, suppression false, recompile limit eight.

## Evidence and limits

[evidence.json](evidence.json) preserves 75 raw files with exact UTF-8 text,
byte lengths and SHA256 hashes. Its `raw_files` include all 21 command receipts,
setup/install logs, selected test results, source identities, before/after GPU
snapshots, actual import/executable/library paths and the orchestration source.
The orchestration invokes existing build and unittest tooling; it does not
replace tests or scoring. Reproduce with its recorded commands and worktree-local
environment settings from a clean checkout before adding reports.

Evidence SHA256: `705c9959bf560751d385eb96d96bbd442bad3fb733d6f0842502cde98c084526`.
Packaging verified raw-file byte identity, clean source identities, successful
process exits, expected test counts, wheel sources and installed native bytes.

This refresh replaces the [85fe118 capture](../postcommit-85fe118/README.md) as
current correctness evidence. Earlier captures and all
[performance-repair diagnostics](../performance-repair/README.md), including
interruptions and unfavorable timings, remain unchanged. This capture measures
no latency or score and does not establish recovery from the rejected canonical
CUDA result. Burner still owns independent review and unchanged qualification.
The [documented unsupported surface](../../../compile-pointwise-jit.md#bounded-positional-input-trees)
remains unchanged; no general pytree, Dynamo, Inductor or accelerator parity is
claimed.
