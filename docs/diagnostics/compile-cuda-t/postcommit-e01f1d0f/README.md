# Clean-commit CUDA t capture

Measured clean implementation commit
`e01f1d0f69dffa8018e1b333bb5ab642ce9fdcb9`, based directly on main
`19e6c31e4371cb34647d37b6badcae8f94acad38` (PR1975). This completes the
previously deferred candidate capture. Git status was empty before and after
the build and every measured command. Evidence and documentation were added
only after all measurements finished; implementation, tests, dependencies and
measurement harnesses are unchanged.

The [release receipt](release/build-record.json), [audit](audit.json),
[shared measured-input manifest](measured-inputs.json) and [inventory](inventory.json)
bind source, commands, imports, build and native binaries to this worktree.
The [candidate inspection](candidate-inspection.json) verifies the complete
125-file diff and preservation of existing evidence. All earlier baseline,
development and failed-attempt records remain unchanged in the
[original bundle](../README.md); they retain their original source identities.

| Artifact | SHA-256 |
| --- | --- |
| Production source | `7398e4f13c0878e303e0743da9f86f6685f06c907aa52cdd7b79c05f72deb546` |
| Production diff against measured commit (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Installed and source-package native extension | `4a46c3adde899f52f851a125efa54215e37bfa5857f6bf34793586b123076ef7` |

## Setup and results

A fresh canonical worktree-local `.venv` used locked dev/reference dependencies;
its predecessor was preserved under `target/t-postcommit-e01f1d0f`. The unchanged
repository build helper ran `maturin build --release --locked --offline` without
`--allow-dirty`, with thin LTO, one codegen unit and a fresh Cargo target.
Local dependency caches and the managed interpreter were reused. Runtime,
Python, CUDA JIT and compiler cache directories started empty and were shared
by this capture's commands. See [setup](setup.receipt.json),
[environment](environment.sh.txt) and [preflight](preflight.log).

Preflight verified CPython 3.12.14, NumPy 2.5.1, PyTorch 2.13.0+cu130,
Rust/Cargo 1.92.0, NVIDIA H100, compute capability 9.0 and driver 580.82.07.
`TORCH_RS_CUDART` selected the new `.venv`'s CUDA 13.0 runtime (13000), matching
reference CUDA 13.0. nvcc 12.6.85 was installed but unused; native kernels use
driver-JIT embedded PTX. Ordinary GPU checks used mask `0`; only restoration
used `0,1`. Receipts record physical UUID/index, utilization and memory snapshots.
Correctness suites could overlap; their wall times are not performance evidence.

| Clean-commit check | Result | Log |
| --- | --- | --- |
| t differentials, identities, ownership and guards | 10 passed; 1 device skip | [compiled t](compiled-t.log) |
| Existing CPU/CUDA compiler and corpus regressions | 201 passed; 8 device skips | [compiler](compiler-regressions.log) |
| Device/context restoration and unused mixed-device captures | 4 passed | [two-device](two-device.log) |
| Rust graph planning, storage sharing and zero-operation negatives | 5 passed | [Rust graph](rust-graph.log) |
| Rust CUDA packing integration | 1 passed | [Rust packing](rust-packing.log) |
| Eager CUDA packing/views | 11 passed; 1 device skip | [CUDA packing](cuda-packing.log) |
| CPU layout, t/reference and docs regressions | 132 passed | [CPU/docs](cpu-layout.log) |
| CUDA-hidden portability | 1 passed; 10 hardware skips | [no device](no-device.log) |
| Installed wheel verification and compiler guide example | passed | [imports](imports.log), [example](docs-example.log) |

The existing seeded graphlets cover non-square matrices, scalar/vector and
empty/singleton views, large valid empty extents, slices/offsets/transposed
inputs, signed-zero/NaN bits, shared mutation and lifetimes, cache hits, blocked
reference imports/redispatch, packing/arithmetic composition and strict early/late
metadata rejection. Every command and the provenance audit passed; this capture
had no failed attempts. Hardware skips receive no correctness credit.

[Capture wrapper](capture.py.txt) and [publication audit](publish.py.txt) preserve
commands and verification logic. The wrapper reuses the committed provenance
helper with only this capture's commit/output location and clean-status checks;
workloads and assertions are unchanged. One input manifest serves all receipts.
Build logs retain their original paths under this worktree's
`target/t-postcommit-e01f1d0f/release` and are mirrored under `release/` here.
Wheels, binaries, environments and caches remain untracked in this worktree.

No required clean capture remains deferred. Unrelated full Rust suites, Clippy
and performance workloads were not repeated. These are non-scoring diagnostics;
the frozen 38-case corpus, weights, observer, hardware matrix, Burner-managed
artifacts and separate PR1970/PR1971 campaigns are unchanged. Independent review,
evaluation and normal same-branch Burner merge gates remain required.
