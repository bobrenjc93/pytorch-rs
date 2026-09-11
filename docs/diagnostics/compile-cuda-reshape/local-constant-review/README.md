# Reshape local-constant validation revision

These **development measurements** use `e8a4140048fcdfe3617201426a1fa7e6f9e3fee2` plus
[the recorded Python/test revision](revision.patch), with source hash
`199c5415583dc51919af8fa04216cd3603ddbd095655029eee4f3c1d6614d5c9`. They do not claim a clean-commit capture.
The existing [release build helper](../../../../scripts/capture_depth_concat_build.py)
built a fresh release wheel in an empty target. The canonical worktree-local
CPython 3.12.14 environment, locked dependencies and local Cargo registry were
reused. Installed Python bytes and native imports were verified before tests.
See the [build record](release/build-record.json), [preflight](preflight.log),
[runner](validate.py.txt) and individual command receipts.

The new regression compares local and inline None, string, bytes, complex and
Ellipsis constants against reference PyTorch for seven invalid argument forms.
All four policies, early/late nodes and repeated calls are checked, with native
execution and Python-body replay blocked. Local and inline native error messages
must match. Unused or overwritten unsupported constants, scalar output leaves,
invalid multiplication and otherwise valid reshape graphs still reject.

| Check | Result |
| --- | --- |
| [validation-after](validation-after.log) | 1 passed; 0 skipped |
| [output-errors](output-errors.log) | 2 passed; 0 skipped |
| [reshape-final](reshape-final.log) | 13 passed; 1 skipped |
| [compiler-regressions](compiler-regressions.log) | 222 passed; 10 skipped |
| [two-device](two-device.log) | 4 passed; 0 skipped |
| [cuda-packing](cuda-packing.log) | 6 passed; 1 skipped |
| [hidden](hidden.log) | 2 passed; 12 skipped |
| [cpu-layout](cpu-layout.log) | 132 passed; 0 skipped |

Installed imports, Python source/install equality and the documentation example
also passed. Ordinary GPU checks expose only physical GPU 0; context restoration
checks expose 0,1. The H100 runtime is native/reference CUDA 13.0 with driver
580.82.07 and PyTorch 2.13.0+cu130. The build records Rust 1.92.0 and installed
nvcc 12.6.85 (unused; native kernels use driver JIT of embedded PTX). Receipts
include physical UUIDs, utilization and memory snapshots, not GPU reservations.
All environments, caches, builds and artifacts remain inside this worktree.
This fix changes Python lowering only; Rust planners and CUDA kernels are unchanged.

The [pre-fix log](local-before.log.gz) preserves all **560 errors**, losslessly
compressed. Its decompressed SHA-256 matches the unchanged
[receipt](local-before.receipt.json). That run used the committed frontend and
previous clean native build with the same new regression test. It is not counted
as passing. A single [input manifest](measured-inputs.json) and a verified delta
retain both attempts; [manifest-index](manifest-index.json) maps receipt hashes.
The [publisher](publish.py.txt) verifies source/build/import/command/log hashes.

All previous evidence remains unchanged, including the
[preceding clean capture](../postcommit-3d44687d/README.md), which does not validate
this fix. **A fresh clean-commit capture is required after Burner commits this
revision.** These non-scoring diagnostics do not change evaluators, workloads,
managed progress, or the separate unadopted PR1970/PR1971 campaigns. Independent
review and merge gates remain required.
