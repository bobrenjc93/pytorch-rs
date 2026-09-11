# Compiled CUDA view: clean revision capture

Fresh measurements of clean implementation commit
`d682311eb60dd0908116c12340deac75fb3f8120`, including both rounds of
public-error precedence review fixes. The [release record](release/build-record.json) and [audit](audit.json)
bind source, wheel, installed Python, native extension, interpreter and commands.
Source SHA-256: `92c6837dd4dd350390e37e047971efe146bcc5db1823646c9d86ad60937f5756`.
All implementation measurements precede publication and record clean status
before and after execution.

| Check | Result | Log |
| --- | --- | --- |
| Complete `test_compile*.py` / `test_top_level_compile.py` sweep, including seeded H100 view and exact-error regressions | 624 passed; 13 device skips | [compiler](compiler.log) |
| View/reshape context restoration, GPUs 0,1 | 2 passed | [devices](two-device.log) |
| CUDA-hidden metadata/planner and CPU view/reshape/reference | 80 passed; 16 hardware skips | [portable](portable.log) |
| Rust graph planning / CUDA storage boundaries | 13 / 7 passed | [graph](rust-graph.log), [storage](rust-storage.log) |
| Clippy default / Python bindings, warnings denied; formatting | passed | [default](clippy-default.log), [bindings](clippy-bindings.log), [format](format.log) |
| Native import/source verification; view/reshape/add examples | passed | [imports](native-verification.log), [examples](guide-examples.log) |

[Setup](setup.sh.txt) reuses the canonical `.venv`, worktree-local uv-managed
CPython 3.12.14 and local Cargo/uv caches. It rechecks locked dev/reference
dependencies before building. The [environment](env.sh.txt), [setup log](setup.log)
and build record show the actual paths and cache state. The repository command
`.venv/bin/python scripts/capture_depth_concat_build.py --output target/view-postcommit-d682311e/release`
ran without `--allow-dirty`, using a fresh empty build target, pinned Rust/Cargo
1.92.0, locked offline compilation, release thin LTO and one codegen unit.
All build, import, interpreter and cache paths stay inside this worktree.

[Preflight](preflight.log) records PyTorch 2.13.0+cu130, NumPy 2.5.1, CUDA runtime
13.0 and driver 580.82.07. GPU 0 is NVIDIA H100, capability 9.0, UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`; restoration checks add only GPU 1,
UUID `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. nvcc 12.6.85 is installed but
unused: execution uses driver-JIT PTX and runtime cuBLAS. Each command receipt
includes before/after GPU identity, utilization and memory snapshots. Snapshots
do not reserve devices. These diagnostics make no latency or scoring claim.

The committed [recorder](capture.py.txt), configured for this commit/directory,
checks clean status and unchanged source/native/command/input hashes. Exact
[GPU](validate-gpu.sh.txt) and [portable](validate-portable.sh.txt) commands use
the committed tests unchanged. The [publication audit](audit-publish.py.txt)
verifies every receipt, all 59 installed Python sources, and 295 historical
artifacts. Binaries, wheels and caches remain local and untracked.

The unchanged input manifest is reused via the [original mapping](../measured-inputs.json)
and the entry in [second review input index](../review-revision-2/input-manifest-index.json)
keyed by SHA-256 `0f35e9e0b9bacf419db011b961d5e87772738b869335c456426adf11f5bf31c5`.
Apply its replacements/removals and serialize with
`json.dumps(mapping, indent=2, sort_keys=True) + '\n'` to reproduce that hash.
No duplicate mapping or delta is published. The [inventory](inventory.json)
hashes this bundle.

The [initial development history](../README.md), [first clean capture](../postcommit-12ac70fd/README.md),
[first review development](../review-revision/README.md),
[second review development](../review-revision-2/README.md), and
[preceding clean capture](../postcommit-e426d1f6/README.md), including all failed
attempts, remain unchanged. This fresh capture completes the revision's
deferred post-commit evidence step; it does not replace independent review or
Burner's delivery/full gates. Implementation, dependencies, tests, evaluators,
scoring corpora and managed progress artifacts are unchanged by this publication.

After publication, [12 README/docs smoke tests](publication-docs.log) passed.
Their separate [receipt](publication-docs.receipt.json) records the evidence/guide-only
dirty status, with unchanged implementation and inputs. This documentation check
is separate from the 11 clean implementation measurements. The
[publication recorder](publication-capture.py.txt) is the committed development
recorder configured for this commit and output directory.
