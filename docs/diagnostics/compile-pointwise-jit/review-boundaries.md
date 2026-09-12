# Shared products, runtime sine and warm scalar repair

All three review findings reproduced on H100. Signed negative doubling now
retains its original expression identity and rounds at earlier live addition
consumers. The last live consumer, subtraction and negation can expose its
factors for contraction. Consumer analysis excludes dead nodes, deduplicates
equivalent expressions, and removes multiplication by one after constant
folding and scalar materialization.

Runtime sine explicitly converts subnormal inputs to signed zero before
accurate libdevice evaluation. Constant-only sine retains gradual underflow,
as does other arithmetic. The compiler still uses `--ftz=false` and does not
enable fast math or approximate range reduction.

Changed captured floats now become runtime float32 kernel parameters after a
successful static specialization. Promotion belongs to each binding and
persists across earlier values and integer interludes; reset clears it.
Scalar values are passed by value, never cached as results or embedded in the
promoted code. Tests keep the same native and default Inductor wrappers across
binding changes, removing the earlier reference-reset workaround.

The [development bundle](review-boundaries.json.gz) retains the original
before-fix probes, 16 failing numerical subcases, the intermediate attempt's
three failures, and an additional sharing probe that exposed multiplication
by one's effect on expression identity. That probe's initial import-path
failure is also preserved. It contains exact commands, logs, final source/test
and wheel/native hashes, generated CUDA/PTX, reference LLVM/PTX excerpts, and
before/after output bits. No prior captured artifact was overwritten.

Final checks used the locked release wheel, Python 3.12.14, default PyTorch
2.13.0+cu130, H100 and NVRTC 13.0 (runtime 13000, `compute_90`):

- All ten pointwise modules: 59 tests, 57 passed and two explicit device-count
  skips with `CUDA_VISIBLE_DEVICES=0`. Both skipped tests passed separately
  with `CUDA_VISIBLE_DEVICES=0,1`.
- Exact cancellation, overflow and zero-sign checks pass on cold and warm
  fresh inputs. Tests include amplified runtime sine, constant-only sine,
  repeated products, dead consumers, unit identities, globals/closures,
  non-finite and type transitions, independent captures, empty/scalar shapes,
  failed compilation, runtime argument admission, reset and concurrent launches.
  The operator's warm-binding sequence module is included unchanged.
- Seventeen Rust IR tests pass with default and Python-binding features; the
  focused Rust CUDA storage/admission test passes on GPU 0. The library check
  without default features, both all-target Clippy configurations with warnings
  denied, and formatting pass.
- Ninety backend-contract tests and twelve documentation tests pass. Portable
  validation passes fourteen admission tests with thirty-seven explicit CUDA
  skips. Native-only execution with imports of PyTorch blocked covers both
  static and promoted scalar kernels.
- Generated-code captures verify one reused code module across changed input
  shapes and a reused runtime-parameter kernel across changed scalar values.
  All eleven before/after program probes and the four-step warm-binding
  sequence agree with the reference after the repair (NaN payloads excluded).

These are unscored development checks of dirty sources based on `e61173f0`,
not clean-commit campaign measurements. Burner committed this repair as
`f745c45c`. The [post-commit capture](README.md) supplies fresh clean coverage/performance,
generated-code evidence and all ten pointwise regression modules. All earlier
measurements and failures remain unchanged. The supported subset and remaining exclusions are
in the [compiler contract](../../compile-pointwise-jit.md).
