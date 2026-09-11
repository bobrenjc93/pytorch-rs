# Compiled CUDA t validation

Non-scoring diagnostics for parameterless rank-0/1/2 CUDA float32 `Tensor.t()`
and packing composition. Started from clean main
`19e6c31e4371cb34647d37b6badcae8f94acad38` (PR1975). No scoring corpus,
performance workload, evaluator/observer/hardware contract or Burner-managed
progress artifact changed. PR1970/PR1971 remain separate unadopted campaigns.

The fresh [clean baseline build](baseline-release-python312/build-record.json)
and [baseline log](baseline-python312.log) show native eager and reference
compiled `t()` agreeing, then native compilation rejecting `Tensor.t`.
The [final release receipt](release-ordered/build-record.json),
[audit](audit.json), [deduplicated input manifests](input-manifests.json) and
[inventory](inventory.json) bind the development source, build, native binary,
installed Python sources, test/tool inputs and raw command logs.

| Final artifact | SHA-256 |
| --- | --- |
| Production source | `7398e4f13c0878e303e0743da9f86f6685f06c907aa52cdd7b79c05f72deb546` |
| Production diff against PR1975 | `318fe9c7365a7cd3b98fa45bd6527b7d8be08e3f2f0d823f9e4b985c9fb1d011` |
| Native extension, installed and source package | `4a46c3adde899f52f851a125efa54215e37bfa5857f6bf34793586b123076ef7` |

**This is development evidence, not a clean implementation-commit capture.**
The task prohibits creating commits or PRs. After Burner commits the candidate,
a fresh clean capture using the unchanged
[`capture_depth_concat_build.py`](../../../scripts/capture_depth_concat_build.py)
without `--allow-dirty`, followed by these same checks, remains required in its
normal same-branch draft/review workflow. The clean baseline is not a substitute
for that candidate capture.

## Environment and checks

Final [preflight](preflight.log) records canonical worktree-local `.venv`, locked
dev/reference dependencies, CPython 3.12.14, NumPy 2.5.1, PyTorch 2.13.0+cu130,
Rust/Cargo 1.92.0, NVIDIA H100 (compute capability 9.0), and driver 580.82.07.
`TORCH_RS_CUDART` explicitly selects the local CUDA 13.0 runtime (13000).
nvcc 12.6.85 is installed but unused; kernels use driver-JIT embedded PTX.
Builds use release, thin LTO, one codegen unit, locked/offline Cargo dependencies
and fresh Cargo targets. All artifacts and caches stay in the worktree;
read-only Cargo registry contents were copied locally. Ordinary GPU checks use
mask `0`; only restoration uses `0,1`. Receipts retain GPU UUID/index, memory
and utilization before/after commands. Correctness suites may overlap; their
wall times are not performance evidence.

| Check | Result | Raw log |
| --- | --- | --- |
| New t differentials, identity, ownership, guards | 10 passed; 1 device skip | [t](t-ordered.log) |
| Existing CPU/CUDA compiler and corpus regressions | 201 passed; 8 device skips | [compiler](compiler-ordered.log) |
| Device/context restoration, unused mixed-device captures | 4 passed | [two-device](two-device-ordered.log) |
| Rust with Python bindings, GPU 0 | 409 passed | [Rust](rust-bindings-final.log) |
| Rust default, CUDA hidden | 392 passed; hardware tests return early | [Rust default](rust-default.log) |
| Eager CUDA regressions | 60 passed; 10 device skips | [eager](eager-cuda.log) |
| CPU layouts, t/reference and docs | 132 passed | [CPU/docs](cpu-layout-ordered.log) |
| CUDA-hidden portability | 1 passed; 10 hardware skips | [portability](no-device-ordered.log) |
| Clippy, warnings denied, all targets with bindings | passed | [Clippy](clippy-final.log) |
| Formatting / installed wheel verification | passed | [format](formatting.log), [imports](imports-ordered.log) |
| Compiler guide example | passed | [example](docs-example.log) |

The final Python-only repair restores existing device-error precedence while
checking exact metadata types before equality. Compiler, t, CPU, device and
portability checks were repeated afterward. Rust sources and the native binary
are unchanged from the successful Rust/eager/Clippy checks, as verified by the
audit. Hardware skips receive no correctness credit. No new performance or
frozen38 coverage score is claimed.

## Retained attempts and reproduction

Failed setup/build/check attempts remain unmodified: the initial Python 3.14
diagnostic import-path error; host Python 3.12 lacking a static library for
Rust tests (including one stale Cargo build-config retry); missing error mapping
and formatting-variable repairs; Clippy function-length failures; two fixtures
using unsupported eager constructors/indexing; an incorrectly named CPU test
module; and the existing corpus test exposing changed device/gradient error
ordering. [That failure](compiler-final.log) was repaired in production code;
no existing unsupported-program test was edited. Earlier source snapshots are
retained as `.txt` files. `ordering-reproduction` used source-package imports,
as disclosed in its [environment note](ordering-reproduction.environment.txt).
Final acceptance checks use the installed wheel.

[Environment](environment.sh.txt), [capture wrapper](capture.py.txt),
[baseline program](baseline.py.txt), [preflight program](preflight.py.txt) and
per-command receipts give exact commands and paths. The wrapper reuses the
existing source-provenance helper and preserves each attempt. Identical input
manifests are shared; `input-manifests.json` stores one baseline plus changed
hashes/removals for each snapshot. Reconstruct a snapshot from the baseline and
its delta, then serialize with `json.dumps(files, indent=2, sort_keys=True) +
"\n"` to recover the `measured_inputs_sha256` in its receipts. The
[publication audit source](publish.py.txt) records this verification. Build
receipts retain original `target/compile-t` paths; their logs/JSON files are
mirrored here under the same build-directory names. Wheels, native binaries,
virtualenvs and Cargo outputs remain untracked under the worktree.
