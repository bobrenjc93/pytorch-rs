# Numerical review round two

All four reported failures reproduce on the previous wheel. The new
[regression module](../../../tests/test_compile_pointwise_rounding.py) checks
exact finite results, NaNs/infinities and zero signs against ordinary default
PyTorch Inductor, with reversed inputs and held-out coefficients. No tolerance,
evaluator, corpus or supported-feature denominator changed.

- Identical ordered expressions share their rounded value before contraction.
  Self-subtraction remains positive zero for finite values and NaN for infinities
  or NaNs. Commuting operands does not merge expressions: the reference can
  contract those separately even when their mathematical values agree.
- Repeated Tensor objects share input expressions. The wrapper guards this
  identity relationship separately from metadata, without retaining objects,
  pointers or data. Cold/warm transitions, fresh values, equal-valued distinct
  tensors, storage-sharing views, reset and recompile limits are tested.
- Sign normalization precedes contraction. Negated subtraction reverses its
  operands and retains the positive-zero correction. Negative coefficients,
  unit multipliers and signed doubling retain the reference's rounding order.
- Shared products can contract for each consumer. Other uses still observe
  their rounded value. Explicit round-to-nearest intrinsics prevent NVRTC from
  introducing a different contraction after the compiler's decisions.

The numerical passes live in
[pointwise_lowering.rs](../../../src/pointwise_lowering.rs), separately from
whole-graph admission in `pointwise_ir.rs`. All nodes, including unused ones,
are validated before normalization and dead-code removal. The lowering uses
operator and coefficient rules, without function-name, shape or program
recognition; warm execution still launches cached native code.

The [development evidence bundle](review-round2.json.gz) preserves the original
24 failing regression subcases, before/after output bits, generated native CUDA
and PTX, reference LLVM/PTX probes, build logs and final checks. It records the
actual dirty source manifest, wheel/native hashes and worktree-local commands.
This is unscored development evidence based on `2b148c46`, not a clean-commit
campaign capture. The reference probe files were generated locally with locked
PyTorch 2.13.0+cu130; installed PyTorch remains test-only. NaN payload bits are
recorded but are not a numerical equivalence requirement.

Checks on the final release wheel and source:

- Four pointwise modules: 27 tests, 26 pass and one explicit two-device skip.
  The operator-added liveness and signature regressions remain included.
- Separate two-device restoration and module-ownership check: one pass with
  `CUDA_VISIBLE_DEVICES=0,1`; ordinary CUDA work uses `CUDA_VISIBLE_DEVICES=0`.
- Portable Python 3.14: nine hardware-free tests pass, 18 hardware tests skip.
- Backend/default-resolution/disable/entrypoint/static-admission selection:
  96 tests pass on the final wheel.
- Rust all-target suites: 404 default and 431 Python-bindings tests pass across
  all 11 executables in each configuration. Seven hardware-free IR tests pass.
- Both all-target Clippy configurations with warnings denied pass, as do
  `cargo check --locked --lib --no-default-features`, formatting and whitespace.
- Documentation: 12 Python quickstart checks pass; rustdoc completes with zero
  doctests. Local documentation links resolve.
- All 16 before/after probe expressions, both input orders, and the repeated
  input probe now match reference output bits except immaterial NaN payloads.

Commands and logs are in the bundle. Cargo, Python environments, UV caches,
temporary directories, CUDA cache paths and Inductor/Triton caches stay within
the worktree; driver disk caching is disabled. The generated kernel selected
NVRTC 13.0, CUDA runtime 13000 and `compute_90` on H100, with no fast math.
The recorded `nvcc` 12.6 is queried for provenance and does not generate the JIT.
Initial Clippy complaints about exact floating-constant comparisons are retained;
the final implementation compares coefficient bits exactly.

Burner committed these fixes as `53c10058`. The subsequent
[post-commit capture](README.md) supplies fresh clean coverage/performance reports,
generated-code evidence and all four pointwise regression modules. The earlier
`60abd863` campaign measurements and this unscored development bundle remain
unchanged at their original source identities. Current-candidate scores use
only the new committed measurements.
