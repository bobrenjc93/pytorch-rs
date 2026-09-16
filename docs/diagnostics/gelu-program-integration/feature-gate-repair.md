# Default-feature test compilation repair

Review found that the selected/eager GELU image-separation test called
`Tensor::gelu_cuda`, which requires `python-bindings`, without requiring that
feature itself. Default `cargo check --tests --offline --locked` reproduced
E0599. The repair adds the same feature gate to that test, preserving its
assertions and its eager-route positive control.

Default-feature check, all-target tests (486 passed, zero failed/ignored),
Clippy with warnings denied, and formatting passed. Python-bindings test
compilation and all-target Clippy also passed. The exact gated test ran on H100
GPU0 and passed (one executed, 300 filtered out). NVRTC 13.0 and CUDA runtime
13000 pins, GPU identity, source snapshots, commands, streams, return codes and
executed native test binaries are retained in the
[repair archive manifest](feature-gate-repair-manifest.json), including the
original failed check.
The [packaging verification](feature-gate-repair-verification.json.gz) retains
the initial checker's unsplit-file assumption and the successful part checks.

These are repair-validation runs from `1771291632d7697fe1dda4f4a9b88d8e5ac80020`
plus the recorded uncommitted test-only patch. Release production code,
dependencies, consumer/tooling and workloads did not change. Existing clean
measurements remain bound to `5bf3e9b0`; their provenance and bytes are unchanged.
No timing rerun or new performance claim accompanies this test-compilation fix.
