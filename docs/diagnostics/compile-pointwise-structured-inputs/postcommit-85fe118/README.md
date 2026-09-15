# Structured input clean-commit correctness refresh

**Superseded by the [c648878 clean-commit refresh](../postcommit-c648878/README.md):** these measurements predate
the source-hash/flat-admission changes and retain their original worktree identity.
They do not validate the revised candidate. The
[repair record](../performance-repair/README.md) preserves the development
diagnostics. Original measurements and raw evidence below remain unchanged.

Measured implementation: `85fe11870a02349278087c5500df1146cd1bca90`
(base `2f15b4ce53dc45918ea644a3370257f340747755`). Capture interval:
`2026-09-15T18:13:38.356670+00:00` through `2026-09-15T18:14:07.899610+00:00`.
All commands recorded the same clean commit/tree before and after execution.
Evidence files were added only after the capture completed.

This refresh includes the CPython 3.10 `ROT_TWO`/`ROT_THREE` fix and its committed
two-/three-item tuple-assignment regressions. It replaces the
[pre-fix capture](../postcommit-3f9bc4d/README.md) as current-candidate evidence;
the earlier measured records remain unchanged. Dirty-source development runs
are not used as clean-commit proof.

**85 test executions passed; zero failures and zero skips.**

| Interpreter | Selection | Passed |
| --- | --- | ---: |
| CPython 3.12.14 | Complete structured-input module | 24 |
| CPython 3.10.21 | Four portable structured-input classes | 14 |
| CPython 3.11.16 | Four portable structured-input classes | 14 |
| CPython 3.13.15 | Four portable structured-input classes | 14 |
| CPython 3.14.7 | Four portable structured-input classes | 14 |
| CPython 3.10–3.14 | One malformed loop-stack regression on each version | 5 |

The four portable classes are `InputAdmission`, `InputContracts`,
`InputLanguage` and `InputTransactions`. The additional test is
`LoopAdmission.test_malformed_exit_backedge_and_body_stack`, including the
rotation cases that must not reach below a loop body's own stack.
The workloads and assertions are unchanged from the measured commit.

The complete module includes ten persistent paired GPU tests: 74 successful
native/reference calls and four expected native rejection calls interleaved
with valid calls for recovery checks. Each history retains its compiled
wrappers across calls. Tuple assignments now cover two/three items at the root,
in data-only helpers and in loops; portable checks also cover zero-trip loops.
The remaining histories retain precision-sensitive list/tuple/length and
dictionary mutation, signed-zero freezing, unequal replacement Tensor data,
noncommutative role changes, ABI reorder, aliases/views, shapes, helper
composition and failed-call recovery. Portable contracts retain callback-free
complete admission, limits, lazy observation, retention and transactional
cache/recency checks.

Tests call ordinary `torch_rs.compile(fn)` and untouched `torch.compile(fn)`,
without backend, fullgraph, dynamic, mode or options overrides. Reference
Dynamo was enabled with suppression false and the default recompile limit of
eight. Existing helpers forbid native original/helper body replay and check
output trees, metadata, exact float32 bits except NaN payload identity, and
input identities/signed zeros where applicable. This is focused correctness
evidence, not universal fallback detection or general compiler parity.

## Build, import and GPU provenance

The existing `scripts/evaluate_torch_compile_default.sh --setup-only` command
built and installed a new worktree-local release abi3 wheel using
`maturin build --release --locked`. Existing Cargo/uv dependencies and Python
environments were reused; this was not a cold build. The same wheel was installed
into the four compatibility environments. Every packaged Python source matched
the checkout, every tested native extension matched the wheel, and imported
tests matched the measured commit.

- Worktree: `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_ea088824`
- Wheel SHA256: `c1705165f2d7baa66681150fafd46ab0c987cbdd6fc5aa6d4645d9face060197`
- Native extension SHA256: `1b036e7d62783b9d55c5a05fa363a43aec8e7d59b775705803f5205595169a3e`
- Frontend SHA256: `f9bac574aacc14ba493d797459259691562754ff41baec12f1e3e2416fb52309`
- GPU 0: NVIDIA H100, capability 9.0,
  UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07.
- `CUDA_VISIBLE_DEVICES=0`; one visible device and one host thread.
- Reference: PyTorch `2.13.0+cu130`, CUDA build 13.0.
  Actual loaded CUDA runtime and NVRTC: 13.0. Host nvcc: 12.6.85;
  the native JIT uses NVRTC rather than that host compiler.

The raw records preserve before/after physical GPU inventory snapshots,
compiler versions, interpreter/package/build paths and hashes, loaded CUDA
library paths/hashes and queried runtime versions. Build, import, temporary and
cache paths are inside this worktree; the system NVIDIA driver is an external
runtime dependency. CUDA/Inductor/Triton/XDG capture caches began empty. No
latency or cold/steady performance conclusion follows from these correctness runs.

## Raw evidence and checks

[evidence.json](evidence.json) preserves all 75 raw capture files under
`raw_files`, each with its original name, exact UTF-8 text, byte length and SHA256.
Receipts contain exact argv/environment overrides, timestamps, exit codes and
before/after git identities. The records include setup/install logs, individual
unittest results and selected IDs, source identities, hardware/runtime metadata,
and the one-off orchestration source invoking the existing build/unittest tools.
JSON records can be read by parsing `raw_files[name].text`.

To reproduce, run the setup-only command, install its wheel into worktree-local
compatibility environments and use the standard unittest loader with the
selections above. Exact executed commands, including the provenance wrapper and
worktree-local cache configuration, are retained in the receipts. Capture from
a clean code commit before adding the resulting report files.

Evidence JSON SHA256:
`1beee0dbe99ca1813a046e5ce40ef1f602cc3e8e23f4402563c71a09fa13c26c`.
Packaging checks verified raw-file byte identity, all successful command exits,
expected test counts with zero skips, unchanged clean source identities, source
hashes against the commit and installed extension bytes against the wheel.

No canonical scoring command was run. Burner still owns coverage/CUDA evaluation
and independent review. Shared/cyclic caller containers, original-container
result passthrough, runtime/captured selectors, dict/starred unpacking and all
other [documented limits](../../../compile-pointwise-jit.md#bounded-positional-input-trees)
remain unchanged. This report makes no general pytree, Dynamo, Inductor,
CPU compiler, training or accelerator parity claim. The independent audit's
historical 39 reference-only calls remain separate evidence.
