# CUDA view review revision diagnostics

This bundle records fixes for both independent-review findings on top of
`4a0a77d4d73e70898e7b073dbc150a7d2a1a0562`. It is development evidence from
uncommitted revision sources, not a clean-commit capture. The prior
[clean capture](../postcommit-12ac70fd/README.md) remains pinned to `12ac70fd`;
its records and the earlier development history are unchanged. Burner must
commit this revision before a separate fresh clean-commit capture is made.

The [reproduction](reproduced-review-findings.log) confirms all four reported
mismatches on clean `4a0a77d`: native/reference eager report public TypeError or
RuntimeError while all four compiler policies report unsupported capture.

List-extension bytecode now preserves exact internal container structure for
validation without iterating user objects. Unsupported binary expressions stay
opaque, including surrounding expressions that use those opaque values. Their
pending rejection survives unused or overwritten locals, so no new expression
is admitted to capture. Known dimension type/range errors are checked first;
then known negative dimensions and repeated inference are checked before the
exact-integer/rank capture restrictions. Accepted shapes still use the shared
native checked resolver and alias-only stride planner.

New regressions compare exact exception types against both eager implementations
under every policy, repeat invalid calls with empty-cache checks, and prohibit
original-body, native-graph and per-node Python execution. Portable metadata and
bytecode tests cover the same four findings without a GPU. Existing view tests
now require exact RuntimeError types or matching diagnostics, preventing
CompileTraceUnsupportedError's RuntimeError ancestry from satisfying them.

| Final check | Result | Log |
| --- | --- | --- |
| Complete current-source compiler sweep | 622 passed; 13 device skips | [compiler](compiler-current-source.log) |
| Exact-error and hardware-free review regressions | 8 passed | [focused](review-regressions-current-source.log) |
| CUDA-hidden view/reshape/reference and metadata tests | 79 passed; 15 hardware skips | [portable](portable-current-source.log) |
| View/reshape context restoration on GPUs 0,1 | 2 passed | [devices](two-device.log) |
| Native Rust graph planning and prevalidation | 12 passed | [graph](rust-graph.log) |
| Clippy with Python bindings and warnings denied; format | passed | [Clippy](clippy-current-source.log), [format](format-current-source.log) |
| README/docs smoke after publishing the guide evidence link | 12 passed | [docs](docs-final.log) |
| Installed native verification and view/reshape/add examples | passed | [imports](native-verification-final.log), [examples](guide-examples.log) |

Final command results and exact source/native identities are in [audit.json](audit.json),
with the [release build](release-final/build-record.json) and command receipts.
The full `test_compile*.py` / `test_top_level_compile.py` sweep runs after the
last source and test edits. These are non-scoring diagnostics, with no changed
scoring definitions, corpora, workloads or managed progress artifacts.

The canonical `.venv` retains worktree-local uv-managed CPython 3.12.14 and
locked dev/reference dependencies. Release builds use the repository's
`scripts/capture_depth_concat_build.py --allow-dirty` with fresh local target
directories, Rust/Cargo 1.92.0, offline locked compilation, thin LTO and one
codegen unit. Existing worktree-local Cargo/uv caches are reused. The
[environment](setup.sh.txt) sources the unchanged prior capture environment;
all generated files and build/import/cache paths stay in this worktree.
[Preflight](preflight.log) records actual runtime/compiler selection. Ordinary
GPU tests use H100 GPU 0 (UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`);
restoration checks use only 0,1. Driver 580.82.07 and native/reference CUDA 13.0
are selected. nvcc 12.6.85 is installed but unused: kernels use driver-JIT PTX. Receipts contain UUID,
utilization and memory snapshots; these do not reserve devices.

The [capture wrapper](capture.py.txt) is the committed recorder configured for
this base commit and output directory. [Input deltas](input-manifest-deltas.json)
reuse the unchanged [original manifest](../measured-inputs.json): apply each
entry's replacements/removals and serialize with
`json.dumps(mapping, indent=2, sort_keys=True) + '\n'` to reproduce receipt hashes.
The [inventory](inventory.json) hashes this bundle. Source/test deltas preserve
intermediate inputs alongside the final checked-in implementation.

Preserved intermediate attempts:

- The first focused view run passed 18 tests with one device skip. An additional
  portable bytecode regression was then added.
- The first [compound-expression probe](compound-probe.log) omitted an import;
  its [corrected probe](compound-probe-verified.log) exposed surrounding `+`/`*`
  expressions masking the overflow. Both scripts and outputs remain recorded.
- The [initial compiler sweep](compiler-final.log) was deliberately interrupted
  after that diagnosis (receipt exit status -2). Its first-source build and
  source/test snapshots remain available. It is not a completed acceptance run.
  A fresh release build and complete sweep follow the propagation fix.
