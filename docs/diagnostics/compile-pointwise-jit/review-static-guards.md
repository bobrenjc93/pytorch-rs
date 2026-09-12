# Globals keys, static zero guards and constant unary precision

Both review findings reproduced. Admission now checks every globals key by
exact type before dictionary lookup, preventing colliding custom keys from
invoking equality hooks. Static captured-float guards equate positive and
negative zero, retaining the first specialization's sign. New graphs use the
current binding; literal zeros and promoted runtime parameters retain their
actual signs. Regressions cover cold/warm rejection, unsupported graphs,
globals/closures, both zero-transition directions, shape changes and reset.

The separately added constant-transcendentals regression module exposed
amplified sine rounding errors after the guard repair. Constant-only sine and
cosine now use double-precision libdevice evaluation between float32 input
and output boundaries. Runtime trigonometry and its existing subnormal rules
are unchanged; fast math remains disabled. Tests cover cancellation, held-out
constants, shared consumers, non-finites, zero signs and runtime promotion.

The [development bundle](review-static-guards.json.gz) preserves original
failures, commands, timestamps, logs, source/test and wheel/native hashes,
generated CUDA/PTX, reference compilation excerpts and before/after output
bits. The original guard run had 33 failing subcases. The intermediate
guard-only wheel passed its 65-test selection, then failed eight operator
constant-sine subcases. All fourteen independent trigonometric probes agree
with the reference after the repair; four differed before it. No prior
captured artifact was overwritten: 68 existing artifact hashes were verified.

Final checks used the rebuilt release wheel, Python 3.12.14, default PyTorch
2.13.0+cu130, H100 GPU 0 and NVRTC 13.0 (runtime 13000, `compute_90`):

- Thirteen pointwise modules: 70 tests, 68 passed and two explicit device-count
  skips under `CUDA_VISIBLE_DEVICES=0`. A separate three-test unary run passed,
  including one subsequently added promotion test: 71 unique tests overall,
  69 passed and two skipped. No two-device rerun was needed for these changes.
- Seventeen Rust IR tests passed with both default and Python-binding
  features. The no-default-features library check, both all-target Clippy
  configurations with warnings denied, and formatting passed.
- Ninety backend-contract and twelve documentation tests passed. Portable
  guard validation passed two admission tests with four explicit CUDA skips
  on the intermediate wheel; that Python guard source is unchanged in the
  final wheel.
- The native-only generated-code capture verified fresh outputs, two graph
  entries sharing one code module and no installed-PyTorch import. Build,
  installed module and source identities were verified against the bundle.

These are unscored dirty-source checks based on `a14afaeb`, not clean campaign
measurements. Existing reports remain pinned to their actual measured commits.
Fresh coverage/performance captures await Burner's next clean implementation
commit. Temporary wheel/cache/log paths may disappear during cleanup; the
compressed bundle retains the durable evidence. See the bounded
[compiler contract](../../compile-pointwise-jit.md) and
[latest clean capture](README.md).
