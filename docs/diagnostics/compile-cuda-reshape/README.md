# Bounded compiled CUDA reshape diagnostics

The clean PR1977 baseline `7ee52792de45df190efecf0e5812c2416e4f96ea` was rebuilt
in an empty worktree-local release target before implementation. The
[baseline reproduction](baseline-gap.log) shows reference compiled and native
eager reshape succeeding on H100 for both view and pack layouts, while native
compiled reshape rejects capture. The [baseline build](baseline/build-record.json)
is evidence for that gap, not for the implementation.

The [candidate build](verified-build/build-record.json) and [audit](audit.json)
bind modified sources, installed Python files, wheel and native binary. These
are **development diagnostics**. Fresh clean-commit validation remains pending
Burner's delivery commit: this agent was instructed not to commit, push or open
PRs. Publish a separate capture after that commit; do not relabel this bundle.

| Check | Result | Evidence |
| --- | --- | --- |
| Reshape shapes/layouts/guards/ownership/composition | 10 passed; 1 device skip | [reshape](reshape-final.log) |
| Existing CPU/CUDA compiler and frozen corpus regressions | 222 passed; 10 device skips | [compiler](compiler-regressions-final.log) |
| Two-device context/restoration and unused captures | 4 passed | [two-device](two-device.log) |
| Rust graph prevalidation and packing, pinned CUDA runtime | 9 + 1 passed | [graph](rust-runtime-pinned.log), [packing](rust-packing-runtime-pinned.log) |
| Python CUDA packing regressions | 6 passed; 1 device skip | [packing](cuda-packing.log) |
| CPU view/layout/reference and docs regressions | 132 passed | [CPU/docs](cpu-layout.log) |
| CUDA-hidden portability | 2 passed; 9 hardware skips | [hidden](hidden.log) |
| Clippy, formatting, installed imports and guide example | passed | [Clippy](clippy-final.log), [format](format.log), [imports](python-imports.log), [example](docs.log) |

Seeded H100 graphlets compare reference compiled and native compiled values,
metadata and aliasing for scalar/vector/non-square, empty/singleton/large-valid-empty,
offset/slice/transpose and view/copy inputs. Checks cover constant/inferred shapes,
raw signed-zero/NaN bits, independent/shared mutations, retained lifetimes,
distinct wrappers, repeated output identity, cold/warm caches, dynamic view/pack
transitions and composition with transpose, packing, neg/scalar/add/matmul/row sums.
Invalid bindings, shape/type/rank/device/gradient declarations, cached nodes and
repeated output/metadata pairs reject before execution. Dedicated tests block
reference imports, Python-body replay and method redispatch. Rust test-only
accounting verifies zero native operations for malformed early/late nodes.
These are correctness diagnostics, not performance measurements.

The canonical local `.venv` uses uv-managed CPython 3.12.14 installed inside
this worktree, with locked dev/reference dependencies (NumPy 2.5.1 and PyTorch
2.13.0+cu130). Rust/Cargo 1.92.0 release builds use thin LTO, one codegen unit,
`extension-module`, an empty target and a local Cargo registry copy. Native and
reference CUDA runtimes are 13.0; the GPU is NVIDIA H100, capability 9.0, with
driver 580.82.07. nvcc 12.6.85 is installed but unused: kernels use driver-JIT PTX.
Rust checks also passed with `TORCH_RS_CUDART` explicitly pinned to the local
CUDA 13 library and `--nocapture` confirming hardware checks ran.
[Preflight](preflight.log), [setup](setup-commands.txt), [setup log](setup.log) and
[environment](environment.sh.txt) record the selection. Receipts include physical
UUID/index, utilization and memory before/after every measured command.
Ordinary tests use GPU 0, with only context checks using 0,1. Snapshots are not
exclusive reservations, and no other jobs were interrupted.

Failures remain preserved:

- [First reshape run](reshape-first.log): one binder bug classified `shape=1`
  as unsupported instead of `TypeError`; two fixtures used unsupported multi-range
  tuple indexing or expected the wrong scalar-input exception. The binder and
  fixtures were corrected. [Initial source patch](first-source-from-baseline.patch)
  applies to the baseline; [test delta](first-test-fixture.patch) reconstructs
  the first fixture from the final test.
- [First Rust run](rust-first.log): new tests used nonexistent `select`/`to_vec`
  method names; corrected to `select_dimension`/`try_to_vec`.
  [Delta](rust-first.patch) reconstructs the failed source from the final file.
- [Clippy run](clippy-second.log): test pointer comparisons required explicit raw
  borrows. [Delta](clippy-first.patch) reconstructs that source.
- [First compiler regression run](compiler-regressions.log): two existing
  assertions still excluded rank-1 compiled reshape. Only these assertions were
  updated to exclude rank-3 reshape; all unrelated assertions remain intact.
  [Delta](old-boundaries.patch) reconstructs those tests.

The [capture wrapper](capture.py.txt), [validation commands](validate.py.txt),
[remaining validation commands](resume.py.txt) and [publication audit](publish.py.txt)
reuse existing repository provenance/build helpers. Final receipts also hash
command files before/after execution. Earlier receipts used the preserved
[initial wrapper](capture-initial.py.txt). One [input manifest](measured-inputs.json)
serves final receipts; the [manifest index](manifest-index.json) records small
replacement/removal deltas for earlier attempts. Serialize reconstructed mappings
with `json.dumps(mapping, indent=2, sort_keys=True) + '\n'` to verify receipt hashes.
The [inventory](inventory.json) hashes published artifacts. Wheels, binaries,
environments and caches remain local and untracked.

For the later clean-commit capture, recreate the canonical locked environment,
use the existing `scripts/capture_depth_concat_build.py` with a fresh output
under `target` and without `--allow-dirty`, and run the same requested commands.
Copy the capture wrapper to that new local directory, set its `COMMIT` to the
new commit and `OUT` to a fresh evidence directory, and verify clean status
before/after measurements. Include Clippy and the complete compiler and device
checks. Publish under a separate commit-named directory only after measurement.

Frozen38, performance workloads, evaluator definitions/weights, observer,
hardware matrix, `.burner` contracts and managed README/history/progress artifacts
are unchanged. PR1970/PR1971 remain separate unadopted human-review campaigns.
This bundle does not replace independent review or Burner's merge gates.
