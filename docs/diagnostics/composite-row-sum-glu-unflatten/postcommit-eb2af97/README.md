# Clean composite evidence: eb2af97

Measured implementation commit: `eb2af972ac6fd538951024bb109db89079a6c77e`,
including the 32-bit indexing-boundary repair. Every build, evaluator, regression
and audit capture began and ended with an empty `git status --porcelain`.
This evidence bundle and its documentation were published only afterward.
The [integration summary](../../../composite-row-sum-glu-unflatten-validation.md)
separates this capture from the unchanged earlier measurements.

The existing repository build tool ran without `--allow-dirty`, using an empty
Cargo target, populated local registry, locked dependencies and offline build.
The fresh release wheel was installed into this composite's real `.venv`;
installed and source-package extension bytes match the wheel. The unchanged
six-case evaluator used its original selected seeds `7763153567161607008` and
`2618969910755569448`, retaining every case and unsupported outcome.

| Capture | Result |
| --- | --- |
| [Build receipt](capture-build-record.json), [commands](capture-commands.json), [build log](capture-build.log), [install log](capture-install.log) | Clean code; fresh release build; source unchanged |
| [Unchanged CUDA math evaluator](cuda-math.json) | 5/6 at both seeds, including row sums; unsupported matmul zero |
| [Focused compatibility tests](focused-python.log) | 53 tests: 51 passed, two two-device mask skips |
| [Separate two-GPU tests](two-gpu.log) | Both restoration tests passed, including split launches; no skips |
| [Numerical reproduction](numerical-reproduction.log) | 46 row-sum comparisons and finite GLU gradient passed |
| [Evidence audit](evidence-audit.json) | Source, wheel, native binary, interpreter, evaluator, runtime hashes and local worker paths verified |

The numerical [candidate](numerical-candidate.json) and
[reference](numerical-reference.json) records come from separate processes.
Candidate processes blocked PyTorch imports. Nonfinite values are explicit
`NaN`, `Infinity`, and `-Infinity` strings in strict JSON.

The 46 comparisons include 30 exact cancellation/overflow cases (3e38 and
float32 maximum, row counts 1/2/17, both keepdim forms), 12 wide-decimal cases
(widths 65,539/1,000,000/1,000,003), and four indexing-boundary cases. The reviewed
sparse width 536,870,916 returns `-1` in both implementations with either keepdim
form; the width 536,870,912 control returns `0`. GLU's large-upstream example
produces identical finite gradients `[206115373056.0, 2.0611537240190114e31]`.
The committed focused tests additionally cover random inputs, row partitions,
all offset alignments, nested odd splits, layouts, aliasing, gradients, errors,
fresh storage and strict documentation assertions. No memory skips occurred.

Ordinary work used only GPU 0; the two-device checks used 0,1. The evaluator
records H100, driver 580.82.07, PyTorch 2.13.0+cu130, and the actually mapped
worktree-local CUDA runtime 13000. Python was 3.12.13, Rust/Cargo 1.92.0. Release
build settings were thin LTO, one codegen unit, and `extension-module`. Available
nvcc 12.6.85 was unused: the driver JITs embedded PTX. Build outputs and the
wheel remain at the verified local paths in the receipt.

`*.receipt.json` files retain actual commands, UTC times, environments, exits,
clean statuses and log hashes. Empty outer build/evaluator stdout logs reflect
tools writing to their named files. `capture-inputs/` preserves the supplemental
capture/audit commands as evidence snapshots, not installed tools or evaluator
changes. [SHA256.json](SHA256.json) covers this bundle. No capture failed.

This replaces the stale 02535d5 capture as current-implementation evidence and
fulfills the clean capture deferred by the indexing repair. Earlier raw records,
including development failures and full-suite results, remain unchanged. Full
Rust and Python compatibility suites were not repeated during this evidence
step. These are correctness measurements, not performance credit, independent
review, evaluation scores, exact-head CI or merge approval. All ten evaluation
gates and final exact-head CI remain required before managed merge.
