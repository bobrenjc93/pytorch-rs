# Clean composite evidence: 02535d5

Measured implementation commit: `02535d5fb191285b0e2ca62677a1ec5d29edbb01`.
All build, evaluator, regression, and audit commands began and ended with an
empty `git status --porcelain`. This bundle was published only afterward.
The [integration summary](../../../composite-row-sum-glu-unflatten-validation.md)
separates these records from the unchanged historical and precommit evidence.

The existing repository build tool produced a fresh release wheel without
`--allow-dirty`, using an empty Cargo target, populated worktree-local registry,
locked dependencies, and offline compilation. The unchanged six-case evaluator
ran in separate reference/candidate processes at its previously selected seeds
`7763153567161607008` and `2618969910755569448`.

| Capture | Result |
| --- | --- |
| [Fresh build receipt](capture-build-record.json), [commands](capture-commands.json), [build log](capture-build.log), [install log](capture-install.log) | Clean code; wheel, installed extension and source extension verified |
| [Original CUDA math evaluator](cuda-math.json) | 5/6 at both seeds; all six denominator slots retained; unsupported matmul zero |
| [Focused Python compatibility](focused-python.log) | 49 tests, 48 passed, one two-device mask skip |
| [Separate two-GPU restoration](two-gpu.log) | 1 passed, no skips |
| [Numerical reproduction](numerical-reproduction.log) | 30 exact cancellation/overflow cases, 12 wide-decimal cases, finite GLU backward passed |
| [Evidence audit](evidence-audit.json) | Source/build/wheel/interpreter/evaluator/runtime hashes and local package paths verified |

The numerical [candidate](numerical-candidate.json) and
[reference](numerical-reference.json) observations came from separate processes;
the candidate blocked PyTorch imports. Their nonfinite observations use explicit
`NaN`, `Infinity`, and `-Infinity` strings so the raw records remain strict JSON.
The focused unittest suite independently checks the committed random, alignment,
layout, nonfinite, gradient, error, and documentation regressions.

Ordinary GPU jobs used only GPU 0; the separate restoration test used 0,1.
H100, driver 580.82.07, PyTorch 2.13.0+cu130, and the actually mapped local CUDA
runtime 13000 are recorded. Python was 3.12.13; Rust/Cargo were 1.92.0. The native
release build uses thin LTO and one codegen unit. Available nvcc 12.6.85 was
unused: the driver JITs embedded PTX. The wheel and build output remain at the
worktree-local paths in the receipt; their bytes and hashes were verified.

`*.receipt.json` files retain actual commands, UTC timestamps, environments,
exit statuses and log hashes. The fresh build and evaluator write their output
to named files, so their outer stdout logs are empty. `capture-inputs/` contains
snapshots of the supplemental receipt/reproduction/audit commands, not installed
tools or evaluator changes. [SHA256.json](SHA256.json) covers this bundle.
No capture failed or required an implementation correction. Earlier development
failures remain in the unchanged precommit bundle.

These are correctness measurements, not performance measurements, independent
evaluation scores, exact-head CI, or merge qualification. No full compatibility
suite was repeated during this evidence step. Independent review, all ten
baseline evaluation gates, and final exact-head CI remain Burner's gates.
