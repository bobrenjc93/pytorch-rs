# Reshape binding review revision

These are **development measurements**, based on `0c9e1397040b60c82c20b589ab5b79e26dd399a9` plus
[the recorded Python/test revision](revision.patch), with source hash
`c4a5f260b136edea02470d2f01c325ae45b1349599453a9196c24b44dee906bd`. A fresh release wheel was built with the existing
[build helper](../../../../scripts/capture_depth_concat_build.py); installed Python
files and native imports were verified before the final checks. The existing
worktree-local CPython 3.12.14 environment and locked dependencies were reused;
Cargo used an empty release target and populated local registry. See the
[build record](release/build-record.json), [preflight](preflight.log), command
receipts and [runner](run.py.txt). All writes and caches stayed in this worktree.

The reviewer regression exercises seven Tensor-valued invalid call forms,
early/late nodes, all four supported policies and repeated rejected calls with
empty caches. Reference PyTorch raises TypeError; native capture must match before
native execution or Python-body replay. Valid nonconstant dimension forms remain
unsupported. The dedicated test passed after the fix.

| Check | Result |
| --- | --- |
| [binding-after](binding-after.log) | 1 passed; 0 skipped |
| [reshape](reshape.log) | 11 passed; 1 skipped |
| [compiler](compiler.log) | 98 passed; 4 skipped |
| [hidden](hidden.log) | 2 passed; 10 skipped |
| [reshape-reference](reshape-reference.log) | 19 passed; 0 skipped |

Installed imports, source/installed Python byte equality and the guide example
also passed. This Python-only revision did not change native kernels or planners.
CUDA checks use GPU 0, an H100 with driver 580.82.07 and native/reference CUDA
13.0 (PyTorch 2.13.0+cu130). Build metadata records Rust 1.92.0; nvcc 12.6.85 was
present but unused. Each receipt records physical UUID/utilization/memory
snapshots; snapshots do not reserve hardware.

All failed attempts are preserved. [regression-before](regression-before.log)
failed because the initial test used the unsupported `ones(device='cuda:0')`
factory; [first-fixture.patch](first-fixture.patch) reconstructs that fixture.
After correcting the fixture, [binding-before](binding-before.log) reproduced
112 errors from the original constant-first rejection. An early validation
launch preceded completion of wheel installation: [python-imports](python-imports.log)
correctly rejected stale installed Python. Validation stopped there and was
rerun after the build completed; [python-imports-final](python-imports-final.log)
verified the installed revision. These failures are not counted as passing runs.

One [input manifest](measured-inputs.json) plus hash-verified deltas covers all
attempts; [manifest-index](manifest-index.json) maps their receipt hashes.
The previous development and [clean-commit capture](../postcommit-1ad5fd62/README.md)
remain unchanged and pinned to their recorded code. They do not validate this
revision. **A fresh clean-commit capture of this binding fix remains required
after Burner commits it**; this development bundle does not satisfy that step.
No scoring corpora, evaluators, workloads, managed progress or Burner contracts
were changed. Independent review and merge gates remain required.
