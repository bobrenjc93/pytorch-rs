# Scalar admission and sign provenance review fixes

All four findings reproduced on the preceding release wheel. The new
[regression module](../../../tests/test_compile_pointwise_scalar_admission.py)
covers their fixes through ordinary default compilation, including cold/warm
binding and code changes, reversed signs, held-out coefficients, non-finite
values, zero signs, fresh output storage and unchanged inputs.

- Sign rewriting preserves the distinction between an original shared-value
  subtraction and a subtraction introduced by normalization. Only the former
  requires one rounded value. Signed products retain their factors for FMA when
  no direct product takes priority; signed doubling no longer imposes rounding
  on every consumer. Existing repeated-expression and shared-product tests pass.
- Boolean, integer and float scalar kinds survive admission. Scalar-specific
  simplification precedes float32 promotion and expression deduplication.
  `False` and integer zero multiplication produce positive-zero tensors;
  floating zero multiplication retains IEEE non-finite and signed-zero behavior.
  A known zero tensor remains distinct from a scalar zero in subsequent
  multiplication, addition and negation. Changed captured scalar types are guarded.
- Scalar admission compares types by identity, without metaclass equality.
  Rejected cold and warm bindings cannot invoke user callbacks or consume a
  cache entry, and restoring a supported binding works.
- Constant pools are validated before disassembly can format objects with
  `repr`. Custom and nested custom constants are rejected without callbacks,
  including after replacing a warmed function's code. Exact strings and `None`
  remain allowed as compiler metadata; scalar operand rules are unchanged.

The [development evidence bundle](review-round3.json.gz) contains the original
30 failing subcases, intermediate failures, final commands and logs, source and
wheel hashes, before/after numerical output bits, native CUDA/PTX, and reference
Triton/LLVM/PTX probes. The initial six-test module was expanded to nine tests
while investigating composed scalar behavior; its earlier logs are preserved.
The original local probe-import failure and successful self-contained retry are
also retained. Default Clippy initially found that the integer variant lacked a
Rust test constructor; an integer-zero IR assertion now covers it and both
configurations were checked again. Reference probes inspect installed PyTorch
read-only; production execution still uses only native generated code.

Final validation:

- Six pointwise modules: 37 tests, 36 pass and one explicit two-device skip with
  `CUDA_VISIBLE_DEVICES=0` on H100. Both operator-added regression modules remain
  included, along with the separately supplied
  [integer-scalar regression module](../../../tests/test_compile_pointwise_integer_scalars.py).
  Three hardware-free and six CUDA regression methods were added.
- Separate two-device restoration/module-ownership check: one pass with
  `CUDA_VISIBLE_DEVICES=0,1`.
- Portable Python 3.14: 12 hardware-free tests pass, 25 hardware tests skip.
- Backend/default-resolution/disable/entrypoint/static-admission selection:
  96 tests pass.
- Rust all-target suites: 406 default and 433 Python-bindings tests pass across
  all 11 executables in each configuration, including nine IR tests.
- Both all-target Clippy configurations pass with warnings denied, as do
  `cargo check --locked --lib --no-default-features` and formatting.
- Documentation: 12 Python quickstart tests pass; rustdoc completes with zero
  doctests. Documentation links and whitespace checks pass.
- All 20 numerical probes, in both input orders, match default Inductor's output
  bits except NaN payloads. Exact finite and zero-sign assertions were retained.

The bundle records actual worktree-local environments, commands, timestamps,
cache paths and source identities. Ordinary GPU work used physical H100 index 0;
only the separate restoration test exposed devices 0 and 1. NVRTC 13.0 generated
`compute_90` code with CUDA runtime 13000, gradual underflow and no fast math.
`nvcc` 12.6 was queried for provenance and did not generate these kernels.
Reference compiler caches were reused between development probes; these are
semantic checks, not timing or cold-compilation measurements.

This is unscored development evidence on dirty sources based on `4f0faea`, not a
clean-commit campaign capture. All earlier measurements remain unchanged. The
`53c10058` reports predate these fixes and cannot score the revised implementation.
Burner committed these fixes as `dbd1a0f6`. The subsequent
[post-commit capture](README.md) supplies fresh clean coverage/performance reports,
generated-code evidence and all six pointwise regression modules. This original
unscored development bundle remains unchanged at its original source identity.
No evaluator, corpus, tolerance, denominator, dependency or managed progress
artifact changed. This validation does not replace independent review.
