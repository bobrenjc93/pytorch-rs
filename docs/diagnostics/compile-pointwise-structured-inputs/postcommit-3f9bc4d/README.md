# Structured input clean-commit correctness capture

**Refresh required before final review:** this capture predates the CPython 3.10
`ROT_TWO`/`ROT_THREE` compatibility fix and its tuple-assignment regressions.
It does not validate the revised candidate. Preserve these measured records;
after Burner commits the fix, repeat the clean-commit capture with the expanded
test module and record the new source/build identities. Dirty-source development
checks do not replace that outstanding capture.

Measured implementation: `3f9bc4dc6599a9a5393a5254edb3fa5ee222473d`
(base `2f15b4ce53dc45918ea644a3370257f340747755`). Captured on
2026-09-15 at 17:54–17:55 UTC, before adding this evidence. Every command's
before/after git status was empty and its commit and tree were unchanged.
This directory adds evidence and documentation only.

All 69 selected test executions passed, with **zero failures and zero skips**:

| Interpreter | Committed tests selected | Passed |
| --- | --- | ---: |
| CPython 3.12.14 | Complete `tests.test_compile_pointwise_structured_inputs` | 21 |
| CPython 3.10.21 | `InputAdmission`, `InputContracts`, `InputLanguage`, `InputTransactions` | 12 |
| CPython 3.11.16 | Same four portable classes | 12 |
| CPython 3.13.15 | Same four portable classes | 12 |
| CPython 3.14.7 | Same four portable classes | 12 |

The 3.12 run includes nine persistent paired GPU history tests: 56 successful
native/reference calls and four expected native rejection calls interleaved
with valid calls to check recovery. Each history keeps its compiled wrappers across changes.
The tests use ordinary `torch_rs.compile(fn)` and untouched `torch.compile(fn)`;
they do not override backend, fullgraph, dynamic, mode or options. Reference
Dynamo was enabled, error suppression was false, and the recompile limit was
the default eight. Native function/helper body replay is forbidden by the
existing test helper. Assertions compare output trees, metadata and float32
bits (excluding NaN payload identity), with explicit input-identity and
signed-zero checks where applicable.

The GPU histories cover precision-sensitive list/tuple/length transitions,
negative-index normalization, dictionary insertion order and unrelated-key
mutation, frozen signed zero, unequal replacement Tensor data, noncommutative
role changes, runtime scalar and Tensor ABI reorder, aliases and offset views,
shape/rank changes, helpers, local selection/unpacking, and failed-call
recovery. Portable tests cover complete callback-free admission, bounds
(including flat calls above 4096 slots), lazy observation, unsupported forms,
retention and transactional cache failures/recency. These are the committed
tests, without changed assertions or added workloads.

## Build and device provenance

The existing `scripts/evaluate_torch_compile_default.sh --setup-only` command
built and installed a release abi3 wheel from this worktree using
`maturin build --release --locked`. Cargo/uv dependencies and worktree-local
Python environments were reused; this was not a cold build. The new wheel was
also installed into each compatibility environment. Every packaged Python
source matched the checkout, and every tested native extension matched the
new wheel. Imported test files matched the measured commit.

- Worktree: `/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_ea088824`
- Wheel SHA256: `803c8d4bd23a4fde6ff110b5eeb8b6ffb91e0d245f9c480a5f73ece507945751`
- Native extension SHA256: `1b036e7d62783b9d55c5a05fa363a43aec8e7d59b775705803f5205595169a3e`
- Frontend SHA256: `1ecf4af7b00bb592e116008b8ce4552d9862cdd6b4e2acdd5dfc21d236f82755`
- Physical GPU 0: NVIDIA H100, capability 9.0,
  `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07.
- `CUDA_VISIBLE_DEVICES=0`; the test process saw one GPU and one host thread.
  GPU 0 reported 0% utilization and 4 MiB allocated in both surrounding snapshots.
- Reference: PyTorch `2.13.0+cu130`, CUDA build 13.0. Loaded CUDA runtime and
  NVRTC were both 13.0. Host `nvcc` reported 12.6.85; native runtime compilation
  uses NVRTC, not that host compiler.

Interpreter, package, frontend, extension and loaded library paths/hashes are
recorded in the evidence. Build, import, temporary and cache paths are inside
this worktree; the system NVIDIA driver is recorded as an external runtime
dependency. CUDA/Inductor/Triton/XDG capture cache directories began empty.

## Raw evidence and reproduction

[evidence.json](evidence.json) preserves all 55 raw capture files as UTF-8 text
under `raw_files`, each with its original name, byte length and SHA256. This
includes exact command arguments, environment overrides, timestamps and exit
codes; setup/install logs; individual unittest results and selected test IDs;
source identities; and GPU/compiler/runtime provenance. JSON receipts can be
read by parsing the corresponding `raw_files[name].text`. The build record also
preserves the one-off orchestration source; it invokes the existing build and
unittest loaders and adds provenance collection, without replacing a workload
or modifying the committed test harness.

To repeat the workloads, use the setup-only command above, then the installed
worktree interpreter with `python -m unittest -v
tests.test_compile_pointwise_structured_inputs`. For the additional Python
versions, install the same wheel and select the four portable classes listed
above. Exact commands used here, including provenance collection and all
worktree-local cache settings, are in the receipts. Clean-commit capture should
precede writing the resulting reports.

The evidence JSON SHA256 is
`86ed640ce05410693dbc90181bd45d0f195b1dac896a500641521670dbaebc89`.
Packaging checks verified every preserved file against the original capture,
all command exit codes, all expected test counts and clean source identities,
and the installed extension against the wheel. No historical artifact was
replaced. Earlier dirty-source author runs are not used as clean-commit proof.

This is focused correctness evidence, with no latency, throughput or coverage
score. Canonical coverage/CUDA scoring remains Burner's later merge gate.
It does not establish general pytree, Dynamo, Inductor, CPU compiler, training
or accelerator parity. Shared/cyclic caller containers, original-container
result passthrough, runtime/captured selectors, dict/starred unpacking and the
other [documented limits](../../../compile-pointwise-jit.md#bounded-positional-input-trees)
remain unsupported. The independent reference audit's historical 39 calls
remain reference-only evidence.
