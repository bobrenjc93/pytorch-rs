# Compiled CUDA view: clean implementation-commit capture

Fresh measurements of `12ac70fd1e7ab039d9317866ddaf269ef0c2d484`, taken from a
clean checkout before adding this evidence. The [release record](release/build-record.json)
and [audit](audit.json) bind the source, wheel, native extension, installed Python,
interpreter and commands. Source SHA-256 is
`fa2bc2ed12827fb87cebf1406c94bc1bcbcd12cef02f8973c943c03139c40875`.
The [development history](../README.md), baseline and failed attempts are unchanged.

| Check | Result | Log |
| --- | --- | --- |
| Complete `test_compile*.py` and `test_top_level_compile.py` sweep, including seeded native/reference CUDA view differentials | 619 passed; 13 device skips | [compiler](compiler.log) |
| View/reshape two-device context restoration | 2 passed | [two-device](two-device.log) |
| Native eager CUDA view/packing | 11 passed; 1 device skip | [storage](cuda-storage.log) |
| CUDA-hidden view metadata/planner | 3 passed; 14 hardware skips | [hidden](hidden.log) |
| CPU view/layout/reference and README/docs smoke | 113 passed | [CPU/docs](cpu-view-docs.log) |
| Rust whole-graph planner and zero-execution prevalidation | 12 passed | [graph](rust-graph.log) |
| Rust CUDA storage/boundaries | 7 passed | [storage](rust-storage.log) |
| Clippy default and Python bindings, warnings denied; formatting | passed | [default](clippy-default.log), [bindings](clippy-bindings.log), [format](format.log) |
| Exact-source native import verification; view/reshape/add guide examples | passed | [imports](native-verification.log), [examples](guide-examples.log) |

The canonical `.venv` was recreated with worktree-local uv-managed CPython
3.12.14 and locked dev/reference dependencies. The release build used pinned
Rust/Cargo 1.92.0, thin LTO, one codegen unit and `extension-module`; its target
started empty. It was captured with
`.venv/bin/python scripts/capture_depth_concat_build.py --output target/view-postcommit-12ac70fd/release`
without `--allow-dirty`. Setup reused the populated worktree-local uv cache and copied
the local Cargo registry, with locked offline compilation. The previous local
development environment was retained under `target`. [Setup commands](setup.sh.txt),
[setup log](setup.log) and [environment](env.sh.txt) record the new environment;
all build, import, interpreter and cache paths are inside this worktree.

[Preflight](preflight.log) records PyTorch 2.13.0+cu130, NumPy 2.5.1, native CUDA
runtime 13.0 and driver 580.82.07. GPU 0 is NVIDIA H100, capability 9.0, UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`. Context checks add only GPU 1,
UUID `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. Installed nvcc is 12.6.85;
execution uses driver-JIT PTX and runtime cuBLAS, not nvcc compilation.
Each receipt preserves GPU identity, utilization and memory snapshots before/after;
snapshots do not reserve devices. No latency or scoring result is claimed.

The [GPU](validate-gpu.sh.txt) and [portable](validate-portable.sh.txt) command lists
use the committed tests unchanged. The existing [capture wrapper](capture.py.txt)
was configured for this commit/output directory with an additional clean-status
assertion. Receipts retain actual commands, timestamps, masks, source/native and
command-file hashes. They reuse the unchanged [input manifest](../measured-inputs.json)
(SHA-256 `865c3aaa1a280d78079b64c57d874efd3df6b85de4f1f607c6e94ef54bd13e4e`);
no duplicate manifest is published. The preflight and guide command scripts are
unchanged copies of [preflight](../preflight.py.txt) and [examples](../guide_examples.py.txt).
The [publication audit](audit-publish.py.txt) verifies receipts and all 105 historical
artifacts before copying the new records. The [inventory](inventory.json) hashes this bundle.

This capture completes the deferred post-commit evidence step. It does not replace
independent review or Burner's delivery/full gates. Implementation, tests, dependencies,
benchmark/evaluator definitions and managed progress artifacts remain unchanged.

After publishing the bundle and its guide link, the [publication docs smoke](publication-docs.log)
passed 12 tests. Its separate [receipt](publication-docs.receipt.json) truthfully
records the evidence/documentation-only dirty status; it is not part of the clean
implementation measurements above. The [publication wrapper](publication-capture.py.txt)
uses the committed development recorder with only commit/output configuration.

The first [publication link check](publication-check-first.log) ran before the
inventory had been written. Writing the inventory before checking its link
resolved this packaging-order error; no implementation measurement failed.
