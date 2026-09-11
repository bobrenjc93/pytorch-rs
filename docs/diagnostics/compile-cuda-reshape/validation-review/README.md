# Reshape partial-argument validation revision

These are **development measurements**, based on `d3c2545604a20074a9fcde74e9b585d68fccc3ca` plus
[the recorded Python/test revision](revision.patch), with source hash
`5ae61d6c8c8ca60f242fc3e3d5b931924ef898007873bccd1755b7c76033ceab`. A fresh final release wheel was built with the
existing [build helper](../../../../scripts/capture_depth_concat_build.py).
Installed Python files and native imports were verified before the final tests.
The canonical worktree-local CPython 3.12.14 environment, locked dependencies,
CUDA caches and local Cargo registry were reused; the release build target was
empty. See the [build record](release/build-record.json), [preflight](preflight-verified.log),
command receipts and [runner](validate-verified.py.txt). All writes stayed in this worktree.

The differential regression covers known-invalid bool/float/overflow dimensions
alongside Tensor arguments, mixed literal/local tuples with invalid binding,
all four policies, early/late nodes, repeated rejected calls and empty caches.
It blocks native execution and Python-body replay. Nonconstant shapes, mixed
output pytrees, tuple helper arguments/arithmetic and unused mixed tuples stay
unsupported. Unknown values are opaque during validation, with no conversion.

| Check | Result |
| --- | --- |
| [output-errors-verified](output-errors-verified.log) | 2 passed; 0 skipped |
| [validation-after-verified](validation-after-verified.log) | 1 passed; 0 skipped |
| [reshape-final-verified](reshape-final-verified.log) | 12 passed; 1 skipped |
| [compiler-regressions-verified](compiler-regressions-verified.log) | 222 passed; 10 skipped |
| [two-device-verified](two-device-verified.log) | 4 passed; 0 skipped |
| [cuda-packing-verified](cuda-packing-verified.log) | 6 passed; 1 skipped |
| [hidden-verified](hidden-verified.log) | 2 passed; 11 skipped |
| [cpu-layout-verified](cpu-layout-verified.log) | 132 passed; 0 skipped |

Installed imports, source/installed Python byte equality and the guide example
also passed. This revision changes Python validation only; native planners and
kernels are unchanged. Ordinary CUDA checks use physical GPU 0, an H100 with
driver 580.82.07 and native/reference CUDA 13.0 (PyTorch 2.13.0+cu130); only
context/restoration checks expose 0,1. Build metadata records Rust 1.92.0;
nvcc 12.6.85 was installed but unused. Receipts include physical UUID, utilization
and memory snapshots, which do not reserve hardware.

The [pre-fix reproduction log](validation-before.log.gz) preserves all 256 errors
losslessly compressed with gzip. Its decompressed SHA-256 matches the unchanged
[receipt](validation-before.receipt.json); [reproduction-tests.patch](reproduction-tests.patch)
reconstructs that test revision. The first [compiler regression run](compiler-regressions.log)
also preserved two failures: rejection of scalar output leaves lost its established
leaf-specific diagnostic. The implementation restored that diagnostic without
changing the pre-existing assertions. [output-diagnostic.patch](output-diagnostic.patch)
reconstructs the failed implementation, bound to [its build](intermediate-release/build-record.json).
The verified run and table above use a new release build after that correction.
Neither failed run is counted as passing.
One [input manifest](measured-inputs.json) and verified deltas cover all attempts;
[manifest-index](manifest-index.json) maps every receipt hash. The
[publisher](publish.py.txt) checks source/build/import/log hashes before copying.

All prior development failures and clean captures remain unchanged and pinned
to their original code, including [the preceding clean capture](../postcommit-7533cbbc/README.md).
They do not validate this revision. **A fresh clean-commit capture remains required
after Burner commits this fix**; this development bundle does not satisfy that step.
Scoring corpora, evaluators, workloads, managed progress and Burner contracts are
unchanged. Independent review and merge gates remain required.
