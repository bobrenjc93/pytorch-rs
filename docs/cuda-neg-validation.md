# Native contiguous CUDA negation validation

Public `-x`, `x.neg()`, `x.negative()`, `torch_rs.neg(x)` and
`torch_rs.negative(x)` dispatch contiguous float32 CUDA tensors to a native
64-bit grid-stride PTX kernel. Scalars, empty tensors and contiguous offset
views are supported. The kernel flips the sign bit into fresh device storage;
it does not compute on the CPU or import PyTorch. Allocation, device guards,
context initialization and legacy-stream completion follow the existing CUDA
storage contracts. Inputs remain alive through completion, including errors.
Noncontiguous CUDA negation, CUDA autograd and compiled CUDA negation remain
explicitly unsupported.

## Composite validation and provenance

The durable [raw six-case report](diagnostics/composite-cuda-neg/evaluation.json),
[build receipt](diagnostics/composite-cuda-neg/build-record.json),
[run receipt](diagnostics/composite-cuda-neg/run-record.json),
[build log](diagnostics/composite-cuda-neg/build.log), and
[run log](diagnostics/composite-cuda-neg/run.log) retain the complete local
measurement, including all failure slots. They supersede the leaf's report,
whose raw files were only in its disposable worktree.

This capture is **uncommitted integration validation**, not a clean final-commit
attestation. The receipt identifies the actual checkout commit, production
source fingerprint, production diff hash, dirty status, extension hash, build
configuration, commands, timestamps, and local paths. The task forbids staging
outside the worktree and making commits; Burner must finish its pending merge
and repeat this capture after committing the integrated implementation. Do not
attribute these results to an unchanged clean HEAD or award current final-commit
performance credit from them. The capture procedure below works for that clean
rerun without changing evaluator definitions.

The unchanged `scripts/evaluate_cuda_math.py` uses all three recorded seeds:
`8503945240872567646`, `8613321571747136749`, `4480763905421893394`.
Its denominator is always six: negation, same-shape addition, trailing-vector
addition, scalar multiplication, axis reduction, and matrix multiplication.
A bounded correctness result does not establish general CUDA support, compiler
parity, accelerator training support, or a performance improvement.

The [integration check record](diagnostics/composite-integration/README.md)
retains full/focused test logs, compile evaluation, and baseline failure reproductions.

## Recorded result

All 18 reference executions passed. Native negation and same-shape addition
passed all three seeds: **2/6 fixed cases**. Trailing-vector addition, scalar
multiplication, axis reduction, and matrix multiplication remain unsupported;
all twelve failing candidate slots are retained with zero credit.

Checkout base: `174cccaa30e362a8de752db527916c7ce2d71ec1` with uncommitted integrated changes.
Measured production fingerprint: `a0e6ac3f54ad339bc95ca4d89fd875aa6c336ff78fdc35ea59f92f53a894f87d`.
Native extension SHA-256: `e2e9c8bea89c8b8963cff95851ae2c4b2ce2038185ce630058f0dd2955abc7bd`.
Build: `2026-09-09T19:34:48.881328+00:00` to `2026-09-09T19:35:30.471814+00:00`;
evaluator: `2026-09-09T19:35:30.629810+00:00` to `2026-09-09T19:36:51.984870+00:00` (exit 0).
Source stability, receipt/report/log hashes, unchanged evaluator/matrix hashes,
all worker package/interpreter/extension/runtime paths, and successful-worker
input preservation were independently checked after capture. Every worker used
the worktree-local CUDA runtime 13000; candidate workers loaded no PyTorch modules.

## Reproduce from this checkout

All downloaded dependencies, Python interpreters, caches, and build outputs
stay inside the checkout. Existing Rust/uv/toolkit executables are read-only.
Start with a fresh build target; the capture script refuses to reuse one.

```bash
mkdir -p target/cuda-neg-validation/tmp target/cargo-home
export UV_CACHE_DIR="$PWD/target/cuda-neg-validation/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/target/cuda-neg-validation/python"
export TMPDIR="$PWD/target/cuda-neg-validation/tmp"
uv sync --locked --no-install-project --group dev --group reference \
  --python 3.12 --managed-python
export CARGO_HOME="$PWD/target/cargo-home"
export CARGO_TARGET_DIR="$PWD/target/cuda-neg-validation/build"
export RUSTC="$(rustup which --toolchain 1.92.0 rustc)"
export RUSTDOC="$(rustup which --toolchain 1.92.0 rustdoc)"
export PATH="$(dirname "$RUSTC"):$PATH"
export PYO3_PYTHON="$PWD/.venv/bin/python"
export CUDA_CACHE_PATH="$PWD/target/cuda-neg-validation/cuda-cache"
export XDG_CACHE_HOME="$PWD/target/cuda-neg-validation/xdg-cache"
export TORCH_RS_CUDART="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python docs/diagnostics/composite-cuda-neg/reproduce.py
```

The [capture script](diagnostics/composite-cuda-neg/reproduce.py) follows the
[repository build-receipt procedure](hardware-heterogeneity-evaluator.md#reproduce-and-bind-a-native-build),
checks source stability across build and run, copies the fresh extension into
`python/torch_rs`, and retains only text receipts/reports/logs under docs.
The evaluator independently checks native device pointers, immutable inputs,
materialized outputs, and separate candidate/reference processes, with PyTorch
imports blocked in candidate workers. It gives each run a new CUDA JIT cache.

For source-copy builds, `PYTHONPATH=python` is required for these test commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python .venv/bin/python -m unittest \
  tests.test_cuda_neg tests.test_cuda_add tests.test_compile_cuda_boundary \
  tests.test_cuda_native_views tests.test_top_level_neg tests.test_top_level_negative \
  tests.test_cuda_math_evaluator
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python .venv/bin/python -m unittest \
  tests.test_cuda_neg.CudaNegDeviceTests tests.test_cuda_add.CudaAddDeviceTests
cargo test --locked --all-targets --features python-bindings
```

Negation tests cover generated shapes, block/grid tails, signed zeros,
subnormals, infinities, NaNs, offsets, empty/scalar inputs, input preservation,
fresh output storage, independent-stream reads, cross-thread ownership,
allocation reuse and outputs larger than the 64 MiB front cache. Hardware-only
cases skip clearly when the required CUDA devices are unavailable.

Host configuration: NVIDIA H100, compute capability 9.0, driver 580.82.07;
local CPython 3.12 with PyTorch 2.13.0+cu130 and CUDA runtime 13.0 explicitly
selected above. Available nvcc 12.6.85 is not used: embedded PTX 6.0 targeting
`sm_50` is JIT-compiled by the driver. Rust/Cargo 1.92.0 build a release
`extension-module`/`abi3-py310` library with thin LTO and one codegen unit.
Actual versions, resolved interpreter/package/extension/runtime paths and GPU
UUIDs are recorded in the receipts and raw worker results.
