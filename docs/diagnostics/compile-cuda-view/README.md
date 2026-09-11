# Compiled CUDA view development diagnostics

This is development evidence for main
`69ceb6f912fdba7dcdb536a3601b8150a603fbf6` plus the recorded source changes.
The [final release build](final-build/build-record.json) and [audit](audit.json)
bind the source, wheel, native extension and installed Python package.
It is not a clean implementation-commit capture. Burner owns committing and
publishing this work; after that commit, publish a separate fresh clean-commit
capture without changing this history.

The [fresh baseline](baseline-gap-verified.log) was reproduced before production
edits using the [main release build](baseline/build-record.json). Native eager
and reference eager/compiled view succeeded for `view(-1)`, `view((3,-1))`,
`(x@w).relu().view(-1)` and `view(size=(21,))`; native compilation rejected all
four. Transposed flattening failed for eager view while reshape copied. The
supplied root probe was diagnosis only, not acceptance evidence.

| Final check | Result | Log |
| --- | --- | --- |
| H100 view shapes, aliasing, guards, binding and recovery | 16 passed; 1 two-device skip | [view](view-validation-verified.log) |
| Complete current-source `test_compile*.py` and top-level compiler sweep | 619 passed; 13 device skips | [compiler](compiler-current-source.log) |
| View/reshape device and context restoration, GPUs 0/1 | 2 passed | [two-device](two-device-current-source.log) |
| CUDA-hidden frontend/planner portability | 3 passed; 14 hardware skips | [hidden](hidden-current-source.log) |
| CPU view/layout/reference and README/docs smoke | 113 passed | [CPU/docs](cpu-layout-docs.log) |
| Native eager CUDA view/packing regressions | 11 passed; 1 device skip | [storage](cuda-storage.log) |
| Rust whole-graph planning/prevalidation on H100 | 12 passed | [graph](rust-gpu-graph-final.log) |
| Rust CUDA packing/storage boundaries on H100 | 7 passed | [storage](rust-gpu-storage-verified.log) |
| Rust default all-targets / Python bindings library, CUDA hidden | 395 / 204 passed; hardware branches skip explicitly | [default](rust-default.log), [bindings](rust-bindings-hidden.log) |
| Clippy default / bindings, warnings denied; formatting | passed | [default](clippy-default.log), [bindings](clippy-current-source.log), [format](format-current-source.log) |
| Installed extension/source verification and guide examples | passed | [imports](native-verification-final.log), [examples](guide-examples.log) |

Seeded native/reference eager and compiled graphlets cover scalar/vector/matrix,
empty/singleton/large empty, offsets, slices, transposes and noncanonical strides.
They verify raw bits, shared mutations and ownership after input deletion,
distinct operation wrappers, repeated outputs/containers, cold/warm calls and
static/dynamic policies. Compositions include reshape, transpose/t, contiguous,
arithmetic, matmul, row sums and ReLU. Compatible → incompatible → compatible
cache transitions reject before execution; Rust test-only counters independently
prove zero native operations for early and late incompatible view nodes.
Tests block original Python bodies, per-node replay, method redispatch and
reference imports. Invalid cached payloads, metadata pairs, unused inputs/captures
and method/global/helper guards retain their rejection assertions.

View accepts exact constant positional dimensions or one flat tuple, including
`size=(...)`. The [reference binding probe](reference-binding.jsonl) records its
different overload/error order from reshape. Lists and bounded exact integer
arithmetic are retained only for validation, never accepted into capture, even
if unused or overwritten. Known binding/type/range/shape errors therefore remain
visible beside unsupported arguments, without user conversions. The sole old
rejection-placeholder edit changes `x.view(-1)` to `x.view_as(x)` in the reshape
suite; the same rejection assertion and every other boundary check remain.

The canonical `.venv` uses worktree-local uv-managed CPython 3.12.14 and locked
dev/reference dependencies (NumPy 2.5.1, PyTorch 2.13.0+cu130). Rust/Cargo 1.92.0
release builds use thin LTO, one codegen unit, `extension-module`, fresh empty
build targets, a local Cargo registry copy and offline locked compilation.
[Setup commands](setup-commands.txt), [environment](env.sh.txt),
[setup log](setup.log), [preflight](preflight.log) and build records preserve
selection and provenance. The selected GPU 0 is NVIDIA H100, capability 9.0,
UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07. Native and
reference runtimes are CUDA 13.0. nvcc 12.6.85 is installed but unused; kernels
use driver-JIT PTX and matmul uses cuBLAS. Ordinary tests use GPU 0; context
checks use only 0,1. Every command receipt contains physical index/UUID,
utilization and memory before/after. Snapshots do not reserve devices, and no
other jobs were interrupted. No new latency or scoring measurements are claimed.

All failed attempts remain available:

- [First baseline probe](baseline-gap.log) omitted the repository import path;
  [second](baseline-gap-retry.log) used the unsupported CUDA `ones` factory.
  The preserved probe scripts show both corrections; the verified probe uses
  the public CPU-to-CUDA transfer.
- [First view test](view-first.log) caught missing method-identity guard
  registration. [Initial source patch](first-source.patch) applies to main;
  [first-test delta](first-test.patch) reconstructs the test file from the final one.
- [Second view test](view-final.log) used a nonexistent public `native.float64`
  fixture; the [delta](second-test.patch) preserves that test. Same-dtype overload
  rejection remains tested without adding a dtype.
- The first [complete compiler sweep](compiler-final.log) passed 631 tests with
  13 skips before a [targeted probe](partial-argument-probe.log) exposed early
  rejection of lists/computed integers masking public argument errors. The
  [source](validation-source.patch) and [test](validation-test.patch) deltas
  reconstruct that earlier state. A fresh final wheel and complete 632-test
  sweep followed the validation fix, alongside a 17-test view suite.
- [First Rust storage command](rust-gpu-storage.log) named a nonexistent test
  target. The corrected command ran the existing `cuda_contiguous` and
  `cuda_native_boundaries` targets; expected caught panics in rejection tests
  are retained in the successful log.

The [capture wrapper](capture.py.txt) records exact commands, source/native
hashes and command-file hashes before/after execution. One
[input manifest](measured-inputs.json) serves unchanged final inputs; the
[manifest index](manifest-index.json) stores only replacement/removal deltas
for earlier mappings. Reconstruct those mappings and serialize with
`json.dumps(mapping, indent=2, sort_keys=True) + '\n'` to verify receipt hashes.
The [inventory](inventory.json) hashes published artifacts. Binaries, wheels,
interpreters, caches and targets stay local and untracked. Rust sources did not
change after their passing default/bindings checks; final frontend changes
were followed by the complete current-source compiler sweep.

For Burner's later clean-commit capture, recreate the local locked environment,
run `scripts/capture_depth_concat_build.py` with a fresh `target` output and
without `--allow-dirty`, and repeat the receipt commands. Set the capture
wrapper's `COMMIT` and `OUT` to that new commit/directory, verify clean status
before/after measurements, and publish separately. This bundle does not replace
independent review or Burner's delivery/full gates.

Evaluator definitions, frozen38, scoring corpora, performance workloads,
observer/hardware contracts, managed README/history/SVG and `.burner` contracts
are unchanged. PR1970/PR1971 remain separate unadopted human-review campaigns.
