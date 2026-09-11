# Clean-commit CUDA t review-repair capture

Measured clean implementation commit
`2ce3dfbf627c59a57d1e37131663a26c2ebfccad`, based on main
`19e6c31e4371cb34647d37b6badcae8f94acad38` (PR1975). This completes the
clean-code capture deferred by the repeated-output metadata repair. Git status
was empty before and after the build and every captured check. Evidence and
documentation were published afterward; implementation, tests, dependencies and
measurement harnesses remain unchanged.

The [release receipt](release/build-record.json), [audit](audit.json),
[shared input manifest](measured-inputs.json) and [inventory](inventory.json)
bind source, commands, imports and native binaries to this worktree. The
[candidate inspection](candidate-inspection.json) inventories the complete
197-file diff against main and verifies that earlier evidence is preserved.
Production source, test inputs and native binary match the
[review-repair development capture](../review-output-metadata/README.md).
The [initial clean capture](../postcommit-e01f1d0f/README.md), original
[baseline/development evidence](../README.md) and failed attempts remain pinned
to their original identities and unchanged.

## Setup and results

The existing helper created a fresh release wheel using
`maturin build --release --locked --offline`, thin LTO, one codegen unit and an
empty build target. A fresh canonical worktree-local `.venv` uses the locked
dev/reference dependencies; its predecessor remains under this capture's
`target/` directory. Local dependency caches and the managed interpreter were
reused. Python, CUDA JIT and compiler caches started empty, then were shared by
these commands. See [setup](setup.receipt.json), [environment](environment.sh.txt)
and [preflight](preflight.log).

Verified CPython 3.12.14, NumPy 2.5.1, PyTorch 2.13.0+cu130, Rust/Cargo 1.92.0,
NVIDIA H100 with compute capability 9.0 and driver 580.82.07. The selected
`TORCH_RS_CUDART` runtime reports CUDA 13.0, matching reference CUDA 13.0.
nvcc 12.6.85 was installed but unused; native kernels use driver-JIT embedded
PTX. Ordinary GPU checks used mask `0`; restoration alone used `0,1`.
Receipts preserve physical GPU UUID/index, utilization and memory snapshots.
Concurrent correctness checks are not performance measurements.

| Clean-commit check | Result | Evidence |
| --- | --- | --- |
| CUDA t views, packing, guards and repeated-output repair | 11 passed; 1 device skip | [t](compiled-t.log) |
| Existing CPU/CUDA compiler and frozen corpus regressions | 201 passed; 8 device skips | [compiler](compiler-regressions.log) |
| Device/context restoration and unused mixed-device captures | 4 passed | [two-device](two-device.log) |
| Rust planning, shared storage and zero-operation negatives | 5 passed | [Rust graph](rust-graph.log) |
| Rust CUDA packing integration | 1 passed | [Rust packing](rust-packing.log) |
| Eager CUDA contiguous layout/ownership checks | 6 passed; 1 device skip | [CUDA packing](cuda-packing.log) |
| CPU layout, t/reference and docs checks | 132 passed | [CPU/docs](cpu-layout.log) |
| CUDA-hidden portability | 1 passed; 11 hardware skips | [no device](no-device.log) |
| Installed extension and compiler guide example | passed | [imports](imports.log), [example](docs-example.log) |

The repair regression checks 48 malformed second-occurrence declarations across
shared lists, tuples and nested containers in static/dynamic cache hits, asserting
zero native calls. Independently allocated valid metadata retains returned
container/tensor identity. Seeded graphlets also cover non-square, scalar/vector,
empty/singleton and large-empty shapes, sliced/offset/transposed views, raw
signed-zero/NaN bits, mutation and lifetimes, packing/arithmetic composition,
cache reuse, blocked reference imports and method redispatch, and malformed
early/late metadata. Hardware skips receive no correctness credit.

[Capture wrapper](capture.py.txt) and [publication audit](publish.py.txt) reuse
the existing repository provenance helper and verify clean status, build/import
consistency, unchanged inputs and earlier evidence inventories. One manifest
serves every receipt. Build logs retain actual paths under this worktree's
`target/t-postcommit-2ce3dfbf/release` and are mirrored under `release/` here.
Environments, wheels, binaries and caches remain untracked inside the worktree.
All captured commands passed; this capture had no failed attempts.

No required clean capture remains deferred. Unrelated full Rust suites, Clippy
and performance workloads were not repeated. These are non-scoring diagnostics;
the frozen 38-case denominator, workloads, weights, observer, hardware matrix,
Burner-managed artifacts and separate PR1970/PR1971 campaigns are unchanged.
Independent review, evaluation and same-branch Burner merge gates remain required.
