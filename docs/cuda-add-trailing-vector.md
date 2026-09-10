# Native CUDA matrix plus trailing vector

Eager `x + v`, `x.add(v)` and `torch_rs.add(x, v)` support contiguous
float32 CUDA tensors shaped `(M, N)` and `(N,)` on the same device, in either
operand order. Empty dimensions and contiguous offset views are supported.
The output owns fresh CUDA storage and neither input is changed. Singleton
and empty view output strides follow the existing elementwise layout rules.

A general 64-bit grid-stride PTX kernel indexes the vector by column and uses
round-to-nearest float32 addition without flushing subnormals. It uses native
storage allocation, independent input bounds checks, device/context guards,
and legacy-stream completion, including cleanup after launch errors. Empty
outputs form no input pointers and launch no kernel. Same-shape addition keeps
its existing scalar/vectorized kernels and replay cache.

Other broadcasts, noncontiguous inputs, mixed devices, other dtypes, nonunit
`alpha`, `out=`, and autograd remain unsupported. Compiler CUDA addition still
requires equal shapes, including the private native trace entrypoint. This
change adds no compiler, reduction, matrix multiplication, or performance claim.

## Historical source validation (PR #1936)

The records below measure the original source PR, not the integrated composite.
Their original source/build identities and paths remain pinned and unchanged.
See [combined validation](composite-cuda-vector-dstack.md) for fresh clean-commit
measurements and the remaining independent review and delivery gates.

`tests/test_cuda_add_trailing_vector.py` compares against real PyTorch CUDA for
generated rectangles, grid-stride loop tails, both operand orders, public call
forms, empty dimensions, singleton strides, offset and overlapping views,
IEEE values, fresh outputs, input preservation, independent-stream reads,
cross-thread lifetimes, and unsupported shapes/layouts. A separate two-device
test checks device restoration and cross-device rejection. Hardware tests skip
clearly without the required devices and explicit visibility mask. Native Rust
tests exercise independent storage bounds and empty views with extreme offsets.

The unchanged six-case math evaluator runs from a fresh source-matched release
extension. Raw results and commit/source/build/runtime provenance are retained
in [the validation report](diagnostics/cuda-add-trailing-vector/evaluation.json),
[build receipt](diagnostics/cuda-add-trailing-vector/build-record.json), and
[run receipt](diagnostics/cuda-add-trailing-vector/run-record.json).
The post-commit capture freshly built clean implementation commit
`953309445979c1340a9adffde93b2d1aef09893c` in its original source worktree. The receipt
records clean checkout status, an empty production diff, and hashes for every
production source file. Build, evaluation, and focused checks all completed
before any tracked evidence was replaced. Outputs were first written under
`target/post-commit-95330944/evidence/` and copied here byte-for-byte.
The committed [capture helper](diagnostics/composite-cuda-neg/reproduce.py)
supplied build/run receipts and command logging; the unchanged evaluator used
the same three previously evaluator-selected seeds. This refresh changes only
evidence and its documentation; Burner owns subsequent commits and review.

The post-commit run passed **4/6 fixed math cases**, including
`cuda_f32_add_trailing_vector` at all three evaluator-selected seeds:
`5027103527016457416`, `3727445652986941402`, and `6885146872121100961`.
All 18 reference executions passed. Same-shape addition, negation, and scalar
multiplication retained credit; axis reduction and matrix multiplication
remained unsupported with zero credit. Candidate workers loaded no PyTorch
modules, and the source fingerprint remained unchanged through build and run.

The measured code commit is `953309445979c1340a9adffde93b2d1aef09893c`, with
production fingerprint
`6cd0f5d735f488bbbf9eed39d29f9b9b38041ccecb606f16452e2d90b8c0dba3`
and native extension SHA-256
`fde6fabde331f77f2943d4a58ecf379d187880d630df2c16255f9aa0057a2ad2`.
Hardware was NVIDIA H100 (compute capability 9.0), driver 580.82.07, with
PyTorch `2.13.0+cu130`. Both workers selected the local CUDA 13.0 runtime
(`cudaRuntimeGetVersion=13000`). The release build used Rust 1.92.0, thin LTO,
one codegen unit and `extension-module`. Installed nvcc was 12.6.85; it was
unused because the driver JIT compiles embedded PTX.

The [check receipt and linked logs](diagnostics/cuda-add-trailing-vector/checks.json)
record 76 passing Python tests with nine two-device tests skipped under the
single-GPU mask, one passing focused two-GPU test, and 17 passing Rust CUDA
tests. Clippy with warnings denied and `cargo fmt --check` passed. Source,
extension and log hashes were rechecked after validation. Evaluators, the fixed
corpus, existing benchmark evidence and Burner-managed artifacts are unchanged.

## Reproduce

Use a worktree-local Python 3.12 environment with the locked development and
PyTorch reference dependencies. Set `CARGO_HOME`, `CARGO_TARGET_DIR`, `TMPDIR`,
`CUDA_CACHE_PATH`, and `XDG_CACHE_HOME` inside the checkout; choose a new empty
Cargo target for the release build. Use the pinned Rust 1.92.0 toolchain,
`PYO3_PYTHON=$PWD/.venv/bin/python`, and select the CUDA runtime explicitly with
`TORCH_RS_CUDART` (this capture uses the environment's `libcudart.so.13`).

```bash
cargo build --locked --release --features extension-module
cp "$CARGO_TARGET_DIR/release/libpytorch_rs.so" python/torch_rs/torch_rs.abi3.so
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python .venv/bin/python -m unittest -v \
  tests.test_cuda_add_trailing_vector tests.test_cuda_add \
  tests.test_cuda_neg tests.test_cuda_mul_scalar tests.test_cuda_native_views \
  tests.test_compile_cuda_boundary tests.test_compile_cuda_neg \
  tests.test_tensor_add tests.test_tensor_add_reference \
  tests.test_top_level_add tests.test_top_level_add_reference
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python .venv/bin/python -m unittest -v \
  tests.test_cuda_add_trailing_vector.CudaAddTrailingVectorDeviceTests
CUDA_VISIBLE_DEVICES=0 cargo test --locked --lib cuda:: -- --test-threads=1
CUDA_VISIBLE_DEVICES=0 cargo test --locked \
  --test cuda_add --test cuda_mul_scalar --test cuda_native_boundaries
```

Record a fresh receipt with the unchanged evaluator's `source_provenance()`
fields, exact build command, `rustc`/`cargo` versions, and extension SHA-256;
verify the source fingerprint and clean checkout before and after building.
To repeat the recorded inputs, run the unchanged evaluator with these seeds
(omit the seed arguments when a fresh evaluator-selected draw is desired):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_cuda_math.py \
  --seed 5027103527016457416 --seed 3727445652986941402 \
  --seed 6885146872121100961 \
  --build-record target/build-record.json --output target/cuda-math-result.json
```

Keep every case and failure in the report. Reusing this branch's receipt for a
different source revision or extension does not establish reproducible credit.
