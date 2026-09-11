# Clean-commit CUDA transpose capture

Measured clean implementation commit
`fef2ad8258a09c6e3d2554d9477526500447fe27`, based on main
`1ff6e6b3a57c69fec9f8b1292bf6ed30e00dec0b` (PR1976). This completes the
clean-commit capture deferred in the [development bundle](../README.md).
The checkout stayed clean before and after the build and every captured command;
this evidence and the compiler-guide link were published afterward.

The [release receipt](release/build-record.json), [command receipts](audit.json)
and [candidate inspection](candidate-inspection.json) bind the complete
90-file candidate diff, source, imports and native binary to this worktree.
The source digest matches the development candidate. The freshly hashed input
manifest is byte-identical to the existing [shared manifest](../measured-inputs.json);
[its reference](manifest-reference.json) avoids publishing a duplicate.
Historical baseline/development records and both implementation-stage failures
are unchanged, including their original paths and provenance.

| Clean-commit check | Result | Evidence |
| --- | --- | --- |
| Transpose axis/layout/policy differentials, identity and guards | 10 passed; 1 device skip | [transpose](transpose.log) |
| Existing CPU/CUDA compiler and unchanged frozen corpus regressions | 212 passed; 9 device skips | [compiler](compiler-regressions.log) |
| Device/context restoration and unused mixed-device captures | 5 passed | [two-device](two-device.log) |
| Rust planning, shared storage and zero-operation prevalidation | 7 passed | [Rust graph](rust-graph.log) |
| Rust and Python CUDA packing regressions | 1 Rust + 6 Python passed; 1 device skip | [Rust packing](rust-packing.log), [CUDA packing](cuda-packing.log) |
| CPU layout/view/reference and docs regressions | 132 passed | [CPU/docs](cpu-docs.log) |
| CUDA-hidden portability | 1 passed; 10 hardware skips | [hidden](no-device.log) |
| Installed native extension and compiler-guide example | passed | [imports](imports.log), [example](docs-example.log) |

The unchanged seeded graphlets cover all valid rank-0/1/2 axes, positive/negative
and same-axis views, non-square/scalar/vector/empty/singleton/large-empty shapes,
offsets and strided inputs, raw signed-zero/NaN bits, mutation and lifetimes,
repeated references, cache hits, explicit constant axes under static/dynamic
policies, packing and arithmetic composition. They also check strict early/late
node and output declarations, argument/type/rank/device/gradient boundaries,
repeated-output metadata validation, blocked reference imports and Python
body/method redispatch. No hardware skip earns correctness credit.

[Setup](setup.log) created a fresh canonical worktree-local `.venv` using the
existing worktree-local uv-managed CPython 3.12.14 installation and locked dev
and reference dependencies (NumPy 2.5.1, PyTorch 2.13.0+cu130). The prior `.venv`
was retained under this capture's local `target/` directory. Dependency caches
were reused; build targets and runtime/JIT/Python caches started empty for this
capture. Rust/Cargo 1.92.0 built with `--release --locked --offline`, thin LTO,
one codegen unit and `extension-module` through the unchanged repository helper.

[Preflight](preflight-clean.log) verifies H100 capability 9.0 and native/reference
CUDA 13.0; physical inventory receipts record driver 580.82.07, UUID/index,
utilization and memory snapshots. nvcc 12.6.85 was recorded but unused; native
kernels use driver-JIT embedded PTX. Ordinary checks used only
`CUDA_VISIBLE_DEVICES=0`; restoration used `0,1`. These are correctness
diagnostics, not performance measurements.

One [failed preflight](preflight.log) is preserved: a temporary inspection helper
named `inspect.py` shadowed Python's standard library before the workloads ran.
Renaming that capture-only helper fixed the retry; implementation, tests and
measurement workloads were unchanged. All workload commands passed.

The [capture wrapper](capture.py.txt), [command runner](run.py.txt),
[environment](environment.sh.txt), [setup commands](setup.sh.txt) and
[publication audit](publish.py.txt) reuse the committed build/provenance helpers.
The [inventory](inventory.json) hashes all newly published artifacts. Wheels,
binaries, environments and caches remain untracked inside this worktree.
Clippy and unrelated full suites were not repeated; their earlier results remain
in the development bundle.

No required clean-commit capture remains deferred. Implementation, dependencies,
tests, benchmark/evaluator definitions, weights, corpora, unsupported outcomes,
observer/hardware matrix and Burner-managed artifacts are unchanged.
PR1970/PR1971 remain separate unadopted campaigns. This capture does not approve
the branch or replace independent review, evaluation or Burner merge gates.
