# Clean composite capture: CUDA matmul, unflatten and L1 gradients

**Refresh required:** the subsequent native-size validation repair changes the
implementation measured here. These raw records remain unchanged and are stale
for the current candidate. Burner must commit the repair before a fresh
clean-commit capture; see [review validation](../../../composite-matmul-unflatten-l1-validation.md#review-repair-native-sizes-validation).

Measured code commit: **`06a249663e969fde8c65af384277a6d15ea7f39d`**, against
main `046b7a21e4e2fb7b59b56ea8a9679a9d9c5b0981`, on 2026-09-10 UTC.
The native build, checks, workloads and integrity verification all completed
with empty tracked/untracked Git status. These files were copied for publication
only afterward. This publication changes evidence and documentation only; it
is not independent review, a ten-gate evaluation or merge approval.

The [inspection](inspection.json) inventories the complete candidate diff and
confirms that the compiler corpus, CUDA scoring harness, hardware matrix and
lockfiles are unchanged. The [source manifest](source-manifest.json) identifies
the committed files. Original source-branch reports, development runs, historical
baselines and unsuccessful attempts remain unchanged; only this fresh capture
measures the combined committed implementation.

## Results

| Check or workload | Result | Evidence |
| --- | --- | --- |
| Fresh release wheel | Passed; absent build target, locked offline build and matching installed/source extension | [Build record](build-record.json), [commands](commands.json), [build](build.log), [install](install.log), [receipt](fresh-build.receipt.json) |
| Local interpreter, imports and native identity | Passed; 59 installed package sources matched; 1,192 loaded module files resolved locally | [Provenance](provenance.json), [receipt](provenance.receipt.json), [isolated wheel verifier](wheel-verifier.log) |
| Unflatten and L1 public/reference regressions | 76 passed, including sizes-before-dim conversion, replaced Tensor methods, nested-mode restoration, aliases and weighted gradients | [Log](focused-surfaces.log), [receipt](focused-surfaces.receipt.json) |
| Compiled/eager CUDA matmul regressions | 25 run, 23 passed; two device-mask skips covered separately | [Log](compiled-regressions.log), [receipt](compiled-regressions.receipt.json) |
| Independent compiled-program proof | Passed: installed wheel under `-I`, PyTorch imports and original-body execution blocked, changed data checked | [Log](compiled-proof.log), [receipt](compiled-proof.receipt.json) |
| Separate GPUs 0,1 restoration | Both compiled and eager tests passed, including mixed-device rejection | [Log](two-gpu.log), [receipt](two-gpu.receipt.json) |
| Native Rust bridge and matmul | One bridge and two matmul tests passed on GPU 0 | [Bridge](rust-bridge.log), [matmul](rust-matmul.log) |
| Fixed CUDA math | All six existing cases passed at all three prescribed seeds: 18/18 trials | [Raw report](fixed-math.json), [receipt](fixed-math.receipt.json) |
| Fixed compiler corpus | 38/38 reference-eligible cases passed | [Report/log](frozen-compiler.log), [receipt](frozen-compiler.receipt.json) |
| Fixed four-shape CUDA scoring workload | 4/4 eligible; existing capped score 100%, common-success speed ratio 1.3543x | [Raw report](fixed-scoring.json), [receipt](fixed-scoring.receipt.json), [same-process imports/libraries](scoring-provenance.json) |
| Separate compiled matmul timing diagnostic | 12/12 correctness passes; **80.85% capped geometric parity**, retaining all six slower composed cells | [Raw report](compiled-timings.json), [receipt](compiled-timings.receipt.json) |
| Integrity checks | Passed: accounting recomputed with committed tools; all source, native, library and receipt hashes verified | [Verification](verification.json), [log](evidence-validation.log), [receipt](evidence-validation.receipt.json) |
| Publication checks (after measurement) | 12 documentation tests passed; links, copied hashes and evidence-only diff verified | [Publication check](publication-check.json), [log](publication-docs.log), [receipt](publication-docs.receipt.json) |

The compiled regressions cover operator, positional method and supported
positional top-level capture; both fullgraph policies; fresh changed inputs;
shape, stride, offset, capture and callable guards; exception recovery; explicit
unsupported boundaries; overlapping inputs, aliases, empty and zero-K products;
wide-K decimals through one million; overflow and nonfinite classifications;
and completion/lifetimes. Unflatten and L1 checks verify the integration repair
against the same freshly built wheel and local PyTorch 2.13 reference.
The conversion-order regressions include competing errors and dimension hooks
that replace or clear sizes, through both positional and keyword calls.

The two performance reports measure different workloads. The fixed scoring
workload remains the existing private pointwise/reduction lane, with four equal
weights, five warmups, 17 samples and three calls per sample. Its existing
summary/checksum report format is preserved. Native eager-backend matmul capture
is an unfused graph path, not general Inductor parity; its separate diagnostic
uses the committed 12-cell matrix and seed, ten warmups, two execution orders,
and 31 five-call samples per backend/order. All 155 individual call timings and
output-check counts per backend/order, aggregate samples, dispersion, first-call
and wrapper costs, input preservation and changed-data checks are retained.
No slow cell was removed or reweighted. Bounded scores establish no universal
compiler, backend or training coverage.

## Identities and setup

- Native extension SHA-256: `419d2322f31bc6ec63742c9b19fcf3fdfbc2e8905a13de9f3d62343578e77e23`.
- Production source SHA-256: `70160c594f56eab5c449494f2e772f62a77a104b5fc3ffdfa29164339c8e6c79`;
  production diff SHA-256 is the empty-diff hash.
- Worktree-local CPython 3.12.12, NumPy 2.5.1 and PyTorch 2.13.0+cu130;
  the matching release wheel was installed before tests, including isolated
  subprocesses. No copied parent editable installation was used.
- H100, capability 9.0, driver 580.82.07, GPU 0 except the separate GPUs 0,1
  test. Selected local CUDA runtime 13000 and cuBLAS 13.1.1 (130101).
- Rust 1.92.0, release, thin LTO, one codegen unit, `extension-module`, abi3-py310.
  Native matmul uses cuBLAS and pointwise operations use embedded driver-JIT PTX;
  the native extension build does not invoke nvcc.
- The fixed scoring workload's existing private kernels used nvcc 12.6.85.
  Reference Triton selected the local PTX assembler 12.8.93. Exact paths,
  versions, hashes and PyTorch build configuration are in the provenance files.
- The native build target was absent. Local Python packages, Cargo registry and
  uv download cache were reused, as disclosed in the build record. CUDA/JIT
  caches were local; the CUDA driver cache was shared with preceding checks.
  Scoring and matmul used separate initially empty Triton and Inductor caches; later cells/orders reuse those caches, so reversed-order
  first calls are not cold disk-cache measurements.
- Prior private scoring-kernel cache contents were archived unchanged locally
  before the scoring run; both kernel targets were absent for its fresh build.
  [Preparation record](scoring-cache-preparation.json) preserves those identities.

## Attempts and reproduction

Every command in this capture completed successfully on its first attempt.
The [previous capture](../postcommit-208e9bf/README.md) and its failed attempts
remain unchanged apart from its documentation pointer to this replacement.
Its measurements predate the conversion-order repair and provide no performance
credit for this candidate. This fresh capture supplies the required replacement.

[Artifact hashes](artifact-hashes.json) bind each published raw file to its
original current-worktree capture; provenance was not rewritten. Command
receipts record actual argv, timestamps, exit status, log hashes, source
identities, clean status and cache environment. The exact evidence-only
orchestration sources are preserved as text in `command-sources/`; they invoke
committed repository tools. The scoring wrapper runs the unchanged CLI and
then records resolved dependencies outside its measurement intervals.

Use the [committed capture procedure](../../../compile-cuda-matmul.md#reproduction-and-delivery)
and the actual commands in these receipts with new output/cache directories.
The fresh build used `scripts/capture_depth_concat_build.py`; math used
`scripts/evaluate_cuda_math.py`; the compiler corpus used
`scripts/evaluate_torch_compile_coverage.py --subset full`; the fixed scoring
CLI was `scripts/benchmark_compile_cuda.py` with its defaults; the separate
matmul diagnostic used `scripts/diagnose_compile_cuda_matmul.py` without
`--allow-dirty`. The unrelated full Python/Rust suites were not repeated here;
the author's development results remain documented separately.

Independent exact-head review, all ten non-regressing current-definition gates,
exact-head CI, confirmed source delivery/dispatch pause and managed merge remain
Burner-owned requirements. This capture does not waive them.
