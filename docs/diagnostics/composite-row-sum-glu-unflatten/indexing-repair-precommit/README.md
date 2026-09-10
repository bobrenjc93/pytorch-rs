# Indexing-boundary repair: precommit diagnostics

The reviewer finding was reproduced and repaired after implementation commit
`02535d5fb191285b0e2ca62677a1ec5d29edbb01` and its evidence-only successor
`e360a656775367c60993ac3099d7e8d7d53554d2`. These records measure uncommitted
repair sources. The final receipt explicitly records `clean_checkout=false`
and `measurement_kind=precommit-diagnostic`; source and diff fingerprints bind
the actual code. **Burner must commit this repair and run a new clean capture.**
The earlier clean 02535d5 evidence is stale for this repair and remains untouched.
See the [integration summary](../../../composite-row-sum-glu-unflatten-validation.md).

The [before log](reproduce-before.log) reproduces native `0` versus reference
`-1` at width 536,870,916, with agreement at 536,870,912, on the prior verified
extension. [Final numerical reproduction](numerical-reproduction.log) confirms
both widths now match, along with the earlier cancellation/overflow, wide-row,
and GLU regressions. [Candidate](numerical-candidate.json) and
[reference](numerical-reference.json) observations are from separate processes;
the candidate blocks PyTorch imports. Nonfinite values use explicit strings in
strict JSON. Neither correctness results nor wall-clock command durations are
performance credit.

| Check | Result |
| --- | --- |
| [Final native build](final-build-record.json), [commands](final-commands.json), [build log](final-build.log), [install log](final-install.log) | Fresh release wheel; dirty-source diagnostic |
| [Formatting](fmt.log), [Clippy](clippy-final.log), [Clippy with Python bindings](clippy-python.log) | Passed |
| [Full Rust targets](rust.log), [with Python bindings](rust-python.log) | 374 and 385 reported passes; two-device section exercised separately |
| [Separate Rust two-device guard](two-gpu-rust.log) | 1 passed, no skip |
| [Focused Python 3.12](focused-python.log), [Python 3.14](focused-python314.log) | Each: 53 tests, 51 passed, two device-mask skips |
| [Separate Python 3.12 two-device checks](two-gpu312.log), [Python 3.14](two-gpu314.log) | Each: 2 passed, no skips |
| [No-GPU portability](no-gpu.log) | All four new hardware tests explicitly skipped |
| [Unchanged CUDA math evaluator](cuda-math.json) | 5/6 at both original selected seeds; unsupported matmul stays zero |
| [Separate-process numerical reproduction](numerical-reproduction.log) | 46 row-sum comparisons and finite GLU gradient passed |
| [Source/build/runtime audit](evidence-audit.json) | Final source, wheel, both interpreters, local worker paths and runtime hashes verified |

All three new single-GPU tests ran on H100 without memory skips. Their sparse
fixtures exercise the actual signed 32-bit byte boundary, multiple row counts,
offset alignments 0–3, nested odd splits, and cancellation/nonfinite values.
Inputs are initialized on device without giant host tensors. The largest pair
uses about 12 GiB; machines lacking GPU memory skip with the required/free byte
counts. No supported size limit or tolerance was changed.

Both interpreters use the same fresh abi3 extension bytes. Python 3.14 uses the
existing real `.venv` in a refreshed source snapshot inside this worktree. No
checkout/environment identity checks were weakened. Hardware was NVIDIA H100,
driver 580.82.07; ordinary work used GPU 0, and two-device checks used only 0,1.
The evaluator records PyTorch 2.13.0+cu130 and the actually mapped local CUDA
runtime 13000. Rust/Cargo 1.92.0 built with release optimization, thin LTO and one
codegen unit. Available nvcc 12.6.85 was unused; the driver JITs embedded PTX.

The initial [Clippy failure](clippy.log) caught doc markup and a test expression;
both were corrected before the final build/checks. Initial build receipts and
the first [boundary test run](indexing-python.log) are retained separately.
No numerical failure occurred after the algorithm repair.
`*.receipt.json` files contain exact commands, times, environment, exits, dirty
status and log hashes. `capture-inputs/` preserves the supplemental commands as
evidence snapshots, not installed tooling. [Reference provenance](reference-and-preservation.json)
binds the read-only TensorIterator.cpp reference to PyTorch's installed git
revision and records protected/prior artifact hashes. [SHA256.json](SHA256.json)
covers the bundle.

Full Python compatibility suites were not repeated for this CUDA-only review
repair; their earlier records remain pinned to their original captures. These
diagnostics do not replace the required post-commit clean build/evaluator and
regression capture, independent review, ten evaluation gates, or exact-head CI.
