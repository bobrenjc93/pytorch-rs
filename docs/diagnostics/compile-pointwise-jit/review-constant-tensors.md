# Composed constant-tensor zero signs

The reported cold/warm failures reproduced on the preceding release wheel.
Arithmetic after integer/Boolean zero multiplication lost its constant-tensor
identity, so negation emitted positive-zero subtraction or FMA. Lowering now
retains constant tensor results through add/subtract/multiply. Negation flips
their sign bits. Tensor/tensor zero-addition identities remain distinct from
tensor/scalar arithmetic; runtime operands and unary/libdevice paths keep their
existing rules.

The [development bundle](review-constant-tensors.json.gz) preserves the original
48 failing subcases, before/after numerical observations, generated CUDA/PTX,
source and wheel hashes, exact final build/test commands and logs. Initial
formatting/Clippy feedback and a standalone probe import failure followed by a
self-contained retry are retained. All earlier captured artifacts are unchanged.

Validation on the final locked release wheel:

- Six pointwise modules: 38 tests, 37 passed and one explicit two-device skip
  with `CUDA_VISIBLE_DEVICES=0` on H100. The new regression checks exact zero
  signs, fresh inputs, unchanged inputs, new output storage, reversed operands,
  repeated expressions and captured integer/Boolean/float transitions.
- All 15 independent numerical probes match default PyTorch 2.13 Inductor's
  output bits on cold/warm calls, except immaterial NaN payloads. The bundle
  includes generated kernels for the formerly failing compositions.
- Ten hardware-free Rust IR tests pass in both default and Python-bindings
  configurations. The no-default-features library check, both all-target Clippy
  configurations with warnings denied, and formatting pass.

The JIT used NVRTC 13.0, CUDA runtime 13000 and `compute_90`, with gradual
underflow and no fast math. Environments, caches and build artifacts remained
inside the worktree. These are semantic checks with reusable reference caches,
not timing measurements.

This is unscored dirty-source development evidence based on `1ba5ebcf`.
The `dbd1a0f6` campaign reports predate this fix. After Burner commits it,
repeat the unchanged gates and `capture.py` using the
[documented workflow](README.md). No evaluator, tolerance, denominator or
managed progress artifact changed. Independent review remains required.
