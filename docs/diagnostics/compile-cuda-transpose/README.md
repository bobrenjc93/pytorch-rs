# Bounded compiled CUDA transpose diagnostics

The clean PR1976 baseline `1ff6e6b3a57c69fec9f8b1292bf6ed30e00dec0b`
was rebuilt in a fresh release target before implementation. The
[baseline reproduction](baseline-gap.log) confirms eager native transpose and
reference `torch.compile` succeed on H100 while native capture rejects
`Tensor.transpose` before execution. The [baseline build](baseline/build-record.json)
is clean-commit evidence for that gap, not for the new implementation.

The [candidate build](verified-build/build-record.json) and [audit](audit.json)
bind the final modified sources, installed Python package, wheel and native
extension. **Candidate clean-commit validation remains pending Burner's delivery
commit:** this implementation agent was instructed not to commit, push or open
PRs. All candidate results below are development diagnostics.

| Check | Result | Evidence |
| --- | --- | --- |
| Transpose axis/layout/policy differentials and guards | 10 passed; 1 device skip | [transpose](transpose-verified.log) |
| Existing CPU/CUDA compiler and frozen corpus regressions | 212 passed; 9 device skips | [compiler](compiler-regressions.log) |
| Device/context restoration and unused mixed-device captures | 5 passed | [two-device](two-device.log) |
| Rust graph planning, shared storage and prevalidation | 7 passed | [Rust](rust-final.log) |
| Rust and Python CUDA packing regressions | 1 Rust + 6 Python passed; 1 device skip | [Rust packing](rust-packing.log), [CUDA packing](cuda-packing.log) |
| CPU view/layout/reference and docs regressions | 132 passed | [CPU/docs](cpu-final.log) |
| Hardware-hidden portability | 1 passed; 10 hardware skips | [hidden](no-device.log) |
| Clippy, formatting, installed imports and guide example | passed | [Clippy](clippy.log), [format](fmt.log), [imports](imports.log), [example](docs-verified.log) |

Seeded graphlets exercise every valid rank-0/1/2 axis pair, negative/same axes,
non-square/scalar/vector/empty/singleton/large-valid-empty inputs, offset and
strided views, raw signed-zero/NaN bits, mutation/lifetimes, output identity,
cache reuse, explicit frozen axes under static/dynamic policies, packing and
neg/scalar/add/matmul/row-sum composition. Invalid bindings, axis types/ranges,
metadata, rank/device/gradient declarations and patched methods are rejected
before native execution. Reference imports and Python-body/method replay are
blocked in dedicated checks. These are correctness checks, not timings.

The canonical worktree-local `.venv` uses uv-managed CPython 3.12.14 installed
under this worktree and locked dev/reference dependencies: NumPy 2.5.1 and
PyTorch 2.13.0+cu130. Rust/Cargo 1.92.0 release builds use thin LTO, one codegen
unit, `extension-module`, an empty build target and a populated local copy of
the Cargo registry. The selected native and reference CUDA runtimes are 13.0;
H100 compute capability is 9.0 and the driver is 580.82.07. nvcc 12.6.85 is
installed but unused: native kernels use driver-JIT embedded PTX.
[Preflight](preflight-verified.log), [setup commands](setup-commands.txt),
[setup log](setup.log) and [environment](environment.sh.txt) record the details.
Single-GPU checks use `CUDA_VISIBLE_DEVICES=0`; only restoration uses `0,1`.
Receipts include physical UUID/index, utilization and memory snapshots.

Two failed attempts are retained: [first](transpose-first.log) exposed a test
fixture that tried to construct an unsupported CUDA gradient tensor; [second](transpose-final.log)
caught the omitted package-level transpose method guard. The fixture now uses
an existing CPU gradient tensor, while CUDA gradient declarations are rejected
in metadata tests. The guard repair covers functions and descriptors on cold
and cached calls. The focused [guard repair](guard-repair.log) used
`PYTHONPATH=$PWD/python`; the complete successful suite used the installed wheel.
[First-attempt](first-attempt.patch) and [guard-failure](guard-failure.patch)
patches reconstruct those earlier sources from the final source tree.
Intermediate build records remain under `development/` and `final-build/`.

The [capture wrapper](capture.py.txt) and [publication audit](publish.py.txt)
reuse existing repository provenance helpers. One [input manifest](measured-inputs.json)
serves final receipts; the [manifest index](manifest-index.json) identifies
small deltas for earlier attempts. Apply each delta's removals and replacements,
then serialize with `json.dumps(mapping, indent=2, sort_keys=True) + '\n'`
to recover its receipt hash. The [inventory](inventory.json) hashes published
artifacts. Wheels, binaries, environments and caches remain untracked locally.

No scoring corpus, performance workloads, evaluator definitions/weights,
observer/hardware matrix, Burner contracts or managed progress artifacts changed.
PR1970/PR1971 remain separate unadopted review campaigns. These diagnostics do
not replace independent review or Burner's clean-commit validation and merge gates.
