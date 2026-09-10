# Native CUDA float32 matrix multiplication

Eager `torch_rs.matmul(a, b)`, `a.matmul(b)`, and `a @ b` execute native CUDA
matrix multiplication for contiguous rank-2 float32 operands on the same
device. Rust exposes the same operation through `Tensor::matmul`. Rectangular
matrices, contiguous offset views, overlapping input views, and contiguous
singleton layouts are supported. The result has fresh contiguous device
storage, shape `(a.shape[0], b.shape[1])`, offset zero, and no gradients.

One embedded PTX kernel uses a bounded grid with 64-bit grid-stride indexing.
Each thread computes an output element with float32 round-to-nearest fused
multiply-adds in increasing inner-dimension order, without flushing subnormals.
The driver JIT compiles PTX 6.0 targeting `sm_50`; execution needs neither nvcc,
NVRTC, cuBLAS, nor PyTorch. Accumulation order can differ from cuBLAS, so general
finite results are compared with tolerances rather than bitwise equality.
This implementation establishes correctness coverage, with no performance claim.

Empty outputs require no launch. A zero inner dimension writes positive zeros
without forming or reading input addresses. Checked dimension products and
storage bounds protect both input offsets and the output allocation. The
existing storage helper guards the device across allocation, launch, and
legacy-stream synchronization. Both inputs stay alive until completion, and
launch/completion failures use the established cache-disable/error path. The
caller’s current device is restored, including on error and destruction.

Noncontiguous inputs, mixed devices, other dtypes, vector or batched products,
CUDA gradients, and compiled matmul remain unsupported. Inner-dimension
mismatches report an error before launch. The existing CPU implementation and
compiler support boundaries are unchanged.

## Validation

The [Rust tests](../tests/cuda_matmul.rs), storage bounds unit test in
`src/cuda.rs`, and [H100 differentials](../tests/test_cuda_matmul.py) cover
generated dimensions and values, rectangular products, offsets, singleton and
empty views, zero inner dimensions, output metadata, source preservation,
independent allocations, retained views, thread/context lifetimes, stream
completion, invalid inputs, and compiled-operation rejection. A rectangular
output exceeding 1,048,576 elements exercises grid-stride iteration. Special
values include NaNs, infinities, subnormals, signed zeros, and exact binary
fraction products. A separate two-device test checks guard restoration and
mixed-device rejection. Hardware-only tests skip clearly without CUDA.

Final checks on this source:

| Check | Result |
| --- | --- |
| [Rust default](diagnostics/cuda-matmul/rust-default.log) / [Python bindings](diagnostics/cuda-matmul/rust-bindings.log), all targets on GPU 0 | 377 / 388 passed |
| [New Python matmul differentials](diagnostics/cuda-matmul/python-matmul-final.log), GPU 0 | 6 passed; two-device test skipped |
| [Two-device Python test](diagnostics/cuda-matmul/python-two-device.log), GPUs 0,1 | Passed; Rust matmul integration tests also passed on both devices |
| [Existing focused Python regression suite](diagnostics/cuda-matmul/python-regression.log) | 95 passed; 11 device-specific skips |
| Rustfmt and Clippy with warnings denied, default and Python-bindings configurations | Passed |
| Python matmul tests with no visible GPU | All 7 hardware cases skipped clearly |

The regression suite covers CPU matmul, eager CUDA add/multiply/negate/copy,
CUDA views/storage, generated row sums, and existing CUDA compiler boundaries.

The [raw correctness report](diagnostics/cuda-matmul/evaluation.json) is from the
unchanged six-case `scripts/evaluate_cuda_math.py`. **All six cases passed at
all three automatically selected seeds**, including `cuda_f32_matmul`:
`8382850184713162372`, `7206935938653639621`, and `5395145091457407776`.
The fixed evaluator’s `atol=1e-6`, `rtol=1e-5`, denominator, pointer inspections,
input-preservation checks, and candidate PyTorch-import blocking were unchanged.
This is implementation validation, not an overall heterogeneity score or final
merge evaluation.

The [build receipt](diagnostics/cuda-matmul/build-record.json) and
[build log](diagnostics/cuda-matmul/build.log) record an initially absent Cargo
target and a fresh release extension built from base `96205cb01e85` plus this
uncommitted implementation. Source fingerprints were checked before and after
build/evaluation; the local extension hash matches every candidate worker.
The receipt binds the production source, binary, compiler, command, profile,
and actual paths. The report retains all reference/candidate outputs and runtime
provenance. These are new correctness records; existing benchmark evidence and
Burner-managed progress artifacts were not edited.

Host: NVIDIA H100, compute capability 9.0, driver 580.82.07. Both workers used
CPython 3.12.12, the reference used PyTorch 2.13.0+cu130, and the explicitly
selected CUDA runtime reported version 13000. Rust/Cargo 1.92.0 built release
`extension-module`/`abi3-py310` with thin LTO and one codegen unit. Available nvcc
was 12.6.85 and was unused. Existing external Python/runtime/toolchain files were
read-only; build outputs, caches, test artifacts, and edits stayed in this worktree.

For reproduction, follow the existing
[build-receipt procedure](hardware-heterogeneity-evaluator.md#reproduce-and-bind-a-native-build)
with a new worktree-local Cargo target, temporary/cache directories, and a fresh
source-matched extension. Set `TORCH_RS_CUDART` to the desired CUDA runtime and
`PYO3_PYTHON` to an interpreter with the stable reference installed. Then run:

```bash
CUDA_VISIBLE_DEVICES=0 cargo test --locked --all-targets
CUDA_VISIBLE_DEVICES=0 cargo test --locked --all-targets --features python-bindings
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python "$PYO3_PYTHON" -B -m unittest -v tests.test_cuda_matmul
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python "$PYO3_PYTHON" -B -m unittest -v \
  tests.test_cuda_matmul.CudaMatmulDeviceTests
CUDA_VISIBLE_DEVICES=0,1 cargo test --locked --test cuda_matmul
CUDA_VISIBLE_DEVICES=0 "$PYO3_PYTHON" -B scripts/evaluate_cuda_math.py \
  --build-record target/cuda-matmul/build-record.json \
  --output target/cuda-matmul/evaluation.json
```
