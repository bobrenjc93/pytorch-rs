# Native CUDA float32 matrix multiplication

Eager `torch_rs.matmul(a, b)`, `a.matmul(b)`, and `a @ b` execute native CUDA
matrix multiplication for contiguous rank-2 float32 operands on the same
device. Rust exposes the same operation through `Tensor::matmul`. Rectangular
matrices, contiguous offset views, overlapping input views, and contiguous
singleton layouts are supported. The result has fresh contiguous device
storage, shape `(a.shape[0], b.shape[1])`, offset zero, and no gradients.

Native cuBLAS SGEMM computes the product using float32 accumulation with TF32
disabled. The row-major matrices are passed as the equivalent column-major
`Bᵀ Aᵀ` product without copies; the 64-bit cuBLAS API retains wide dimensions.
This replaces the source PR's serial accumulator, which lost accuracy for long
decimal sums and disagreed with PyTorch on intermediate overflow. Float64
accumulation would also change observable float32 overflow behavior.

cuBLAS is loaded lazily only for nonempty, nonzero-inner-dimension matmul.
Install NVIDIA cuBLAS 12 or 13 alongside the CUDA runtime, or set
`TORCH_RS_CUBLAS` to its shared library. Discovery checks the optional NVIDIA
runtime wheel locations and system library names without importing PyTorch.
CPU operations and the existing CUDA kernels do not require cuBLAS. No nvcc,
NVRTC, Python PyTorch, or libTorch operator forwarding is used. Handles are
cached per CUDA context and serialized during submission; the library and
handles remain alive for the process lifetime. The legacy stream and shared
completion path protect input/output lifetimes, including failed submissions.

Empty outputs require no launch. A zero inner dimension writes positive zeros
without forming or reading input addresses. Checked dimension products and
storage bounds protect both input offsets and the output allocation. The
existing storage helper guards the device across allocation, launch, and
legacy-stream synchronization. Both inputs stay alive until completion, and
launch/completion failures use the established cache-disable/error path. The
caller’s current device is restored, including on error and destruction.

Noncontiguous inputs, mixed devices, other dtypes, vector or batched products,
and CUDA gradients remain unsupported. [Compiled CUDA matmul](compile-cuda-matmul.md) now captures the same tensor scope through positional forms. Inner-dimension
mismatches report an error before launch. The existing CPU implementation and
other compiler support boundaries are unchanged.

## Validation

The [Rust tests](../tests/cuda_matmul.rs), storage bounds unit test in
`src/cuda.rs`, and [H100 differentials](../tests/test_cuda_matmul.py) cover
generated dimensions and values, rectangular products, offsets, singleton and
empty views, zero inner dimensions, output metadata, source preservation,
independent allocations, retained views, thread/context lifetimes, stream
completion, invalid inputs, and compiled-operation acceptance. A rectangular
output exceeding 1,048,576 elements exercises large output indexing. Special
values include NaNs, infinities, subnormals, signed zeros, and exact binary
fraction products. Wide decimal products cover K=4096 through 1,000,000,
including the reported 65,539 and 262,144 failures, generated rectangular
products and overlapping offsets. Adversarial tests compare finite/infinite/NaN
classification and subnormal bits, including the K=1031/2048 intermediate
overflow cancellation probes. A separate two-device test checks guard restoration and
mixed-device rejection. Hardware-only tests skip clearly without CUDA.

## Clean composite capture

The repaired combined implementation was measured from clean commit
**`cc42fdb265af856d8110404e91ff6cf8dd70a5e7`** using a fresh release extension.
The [capture index](diagnostics/cuda-matmul/composite-cc42fdb2/README.md) links
all raw reports, command/build receipts, source/native/runtime identities,
verification, and reproduction commands.

All six unchanged CUDA math cases passed at all three required seeds (18 trials).
All 22 public matmul timing cells validated, with **91.62% capped geometric parity**
for this diagnostic. The report retains generated square/rectangular shapes,
offsets, raw samples and dispersion, including slower native cells. Forty
explicit numerical reproductions passed, including the reported wide-K decimals
and both overflow cancellation widths. Fresh checks also passed 43 focused
Python tests, eight Rust tests, 19 evaluator tests, and the separate two-GPU test.
One expected two-device mask skip occurred in the single-GPU Python run.

Capture and verification completed before artifact publication. These are scoped
measurements; independent review, all ten baseline gates, and exact-head CI remain
required. No implementation, tests, dependencies, harnesses, evaluators, corpora,
or Burner-managed progress artifacts changed in this evidence step.

## Historical source PR capture

These records belong to PR #1952, before the composite repair. They are kept
unchanged with their original source/build identities and paths. They do not
qualify the combined implementation or establish its performance. The clean
composite capture above supplies the new measured evidence separately.

The source [correctness report](diagnostics/cuda-matmul/postcommit-a267ab89/evaluation.json)
measures clean implementation commit **`a267ab89a31573e68d0045f1cbcf55d8ea243307`**.
The unchanged six-case evaluator passed **6/6 cases at all three seeds**, including
`cuda_f32_matmul`. The existing repository capture procedure selects seeds
`8503945240872567646`, `8613321571747136749`, and `4480763905421893394`.
The fixed `atol=1e-6`, `rtol=1e-5`, denominator, device-pointer inspections,
input-preservation checks, and candidate PyTorch-import blocking are unchanged.
This is correctness evidence for that cell, not an overall heterogeneity score
or a performance measurement.

The [build receipt](diagnostics/cuda-matmul/postcommit-a267ab89/build-record.json),
[build log](diagnostics/cuda-matmul/postcommit-a267ab89/build.log), and
[run receipt](diagnostics/cuda-matmul/postcommit-a267ab89/run-record.json) record
an initially absent Cargo target, a fresh release extension, empty tracked
checkout status, and unchanged source across build and measurement. The
[verification record](diagnostics/cuda-matmul/postcommit-a267ab89/evidence-checks.json)
checks source/binary binding, log/report hashes, unchanged tooling and matrix,
all worker paths, runtime versions, and recomputed six-case accounting.
The production fingerprint is
`707ec5d124910b4c4247e84024fbf7179e78bb5144467548eb64dea2a80e9502`;
the extension SHA-256 is
`78943559de1dc2fdf37d4ef20065e7f523b2900f90fdeb72e12108a85bde4fd9`.

All captures and focused checks completed while the tracked checkout was clean.
Reports were first written under `target/cuda-matmul-postcommit/` and copied
byte-for-byte into the evidence directory afterward. The source PR’s subsequent publication
contained only evidence and documentation. Implementation, tests, dependencies,
benchmark harnesses, evaluators, and Burner-managed artifacts were not changed
in that publication.

Host: NVIDIA H100, compute capability 9.0, driver 580.82.07. Both workers used
worktree-local CPython 3.12.12; the reference used worktree-local PyTorch
2.13.0+cu130. Both loaded the explicitly selected worktree-local CUDA runtime
reporting version 13000. Rust/Cargo 1.92.0 built release
`extension-module`/`abi3-py310` with thin LTO and one codegen unit. Available nvcc
was 12.6.85 and was unused. The Cargo registry was reused; the native build
target was empty and the evaluator created a new temporary CUDA JIT cache.
The local Python environment was installed from the unchanged lockfile.
Existing toolchain executables were read-only; every generated artifact and
cache stayed inside the original source worktree.

Focused clean-commit checks are bound to the same source and extension by the
[checks receipt](diagnostics/cuda-matmul/postcommit-a267ab89/checks-record.json):

| Check | Result |
| --- | --- |
| [Python matmul differentials](diagnostics/cuda-matmul/postcommit-a267ab89/python-matmul.log), GPU 0 | 6 passed; two-device test skipped |
| [Rust matmul integration](diagnostics/cuda-matmul/postcommit-a267ab89/rust-matmul.log), GPU 0 | 2 passed |
| [Rust storage bounds](diagnostics/cuda-matmul/postcommit-a267ab89/rust-bounds.log), GPU 0 | 1 passed |
| [Two-device Python test](diagnostics/cuda-matmul/postcommit-a267ab89/python-two-device.log), GPUs 0,1 | 1 passed |
| [Existing CUDA math evaluator tests](diagnostics/cuda-matmul/postcommit-a267ab89/evaluator-tests.log) | 19 passed |

## Original author validation

The original [development report](diagnostics/cuda-matmul/evaluation.json),
[build receipt](diagnostics/cuda-matmul/build-record.json), and
[build log](diagnostics/cuda-matmul/build.log) remain unchanged as superseded
records. They measured base `96205cb01e85` plus the uncommitted implementation,
with the same production fingerprint above, using the original paths recorded
there. They do not supply the required clean-commit capture; the historical clean-source capture
above did for that source revision. Their original three-seed 6/6 result is preserved without rewriting
its source, executable, import, or runtime identities.

The author's broader checks below were also preserved unchanged. They were not
rerun for this evidence step and are not presented as post-commit measurements.

| Original author check | Recorded result |
| --- | --- |
| [Rust default](diagnostics/cuda-matmul/rust-default.log) / [Python bindings](diagnostics/cuda-matmul/rust-bindings.log), all targets on GPU 0 | 377 / 388 passed |
| [Python matmul differentials](diagnostics/cuda-matmul/python-matmul-final.log), GPU 0 | 6 passed; two-device test skipped |
| [Two-device Python test](diagnostics/cuda-matmul/python-two-device.log), GPUs 0,1 | Passed |
| [Focused Python regression suite](diagnostics/cuda-matmul/python-regression.log) | 95 passed; 11 device-specific skips |

The original regression suite covered CPU matmul, eager CUDA
add/multiply/negate/copy, CUDA views/storage, generated row sums, and existing
CUDA compiler boundaries. The author also reported passing Rustfmt/Clippy and
clear hardware skips with no visible GPU; this step does not recapture those
checks or claim new full-suite validation.

## Reproduction

Follow the existing
[build-receipt procedure](hardware-heterogeneity-evaluator.md#reproduce-and-bind-a-native-build)
and [local environment setup](cuda-neg-validation.md#reproduce-from-this-checkout).
Use a clean checkout, a worktree-local Python executable and reference packages,
local CUDA runtime, local cache/temporary directories, and an absent
`CARGO_TARGET_DIR`. Run the unchanged repository capture helper with a new output
directory; it records the fresh build and all six cases at its predeclared seeds:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python .venv/bin/python -B - <<'PYTHON'
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "capture", "docs/diagnostics/composite-cuda-neg/reproduce.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
capture.OUT = Path("target/cuda-matmul-recapture").resolve()
capture.OUT.mkdir(parents=True, exist_ok=False)
capture.main()
PYTHON
```

With that extension, run the focused commands recorded verbatim in the checks
receipt: `cargo test --locked --test cuda_matmul`, the named storage-bounds unit
test, `tests.test_cuda_matmul` on GPU 0, its device test on GPUs 0,1, and
`tests.test_cuda_math_evaluator`. Publish artifacts only after measurement, so
documentation writes do not dirty the measured checkout. Independent review and
normal merge gates remain separate from this evidence step.

## Composite repair validation and timings

After Burner commits implementation, tests, and harness, run the clean capture
above with the worktree-local environment. Then, using that build receipt:

```bash
export TRITON_CACHE_DIR="$PWD/target/matmul-triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/target/matmul-inductor-cache"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python .venv/bin/python -B \
  scripts/benchmark_cuda_matmul.py \
  --build-record target/cuda-matmul-recapture/build-record.json \
  --output target/cuda-matmul-recapture/public-timings.json
```

The [timing harness](../scripts/benchmark_cuda_matmul.py) requires clean code
by default. `--allow-dirty` records development diagnostics explicitly and
cannot produce clean-commit qualification. It refuses stale build receipts,
nonlocal imports/runtime/cache paths, and overwriting a previous attempt.
It reports every fixed and generated square/rectangular shape at offsets zero
and three, including 512-, 1024-, and 2048-square products. Both implementations
use the public `@` API with equal float32 inputs and TF32 disabled, ten warmups
per order, two reversed orders, 31 five-call blocks per order, and symmetric
synchronization. Outputs and original inputs are materialized and checked.
Raw samples, medians and dispersion accompany the fixed equal-weight geometric
mean of capped reference/native median ratios. This is a scoped diagnostic,
not a substitute for Burner's performance gate.

Run the full Rust suites with and without `python-bindings`, the full Python
compatibility suite, the focused matmul/rsqrt/linear differentials, and the
separate two-GPU matmul test. Retain failed attempts and corrected receipts.
The source linear fixes already preserve omitted versus explicit empty keyword
arguments and registered-callable identity; its saved-original and wrapper
regressions remain part of combined validation. CPU rsqrt retains first-order
saved-output backward, now reflected in the feature-coverage contract.

Final qualification remains Burner's responsibility: clean combined-commit
capture, independent review, all ten current-definition gates against main
(including correctness 100, polish 97, performance 97, hardware 4), and exact-head
CI. Source PR captures above and uncommitted development runs do not waive those
requirements. Publish evidence only after capture and change no measured code
or harness without a new capture. No evaluator or corpus changes are needed.
