# Native contiguous CUDA negation validation

Public `-x`, `x.neg()`, `x.negative()`, `torch_rs.neg(x)` and
`torch_rs.negative(x)` dispatch contiguous float32 CUDA tensors to a native
64-bit grid-stride PTX kernel. Scalars, empty tensors and contiguous offset
views are supported. The kernel flips the sign bit into fresh device storage;
it does not compute on the CPU or import PyTorch. Allocation, device guards,
context initialization and legacy-stream completion follow the existing CUDA
storage contracts. Inputs remain alive through completion, including errors.
Noncontiguous CUDA inputs, CUDA autograd and compiled CUDA negation remain
explicitly unsupported.

## H100 validation, 2026-09-09

The unchanged `scripts/evaluate_cuda_math.py` selected these seeds itself:
`8503945240872567646`, `8613321571747136749`, `4480763905421893394`.
All reference executions passed. Both `cuda_f32_neg` and
`cuda_f32_add_same_shape` passed every seed: **2/6 fixed cases**. Trailing-vector
addition, scalar multiplication, axis reduction and matrix multiplication
remain unsupported and receive zero. This is correctness evidence for this
cell, without a claim about overall heterogeneity score or performance.

The candidate used this checkout's Python package and a fresh native extension
built in an initially empty `target/cuda-neg-validation/build` directory:

```sh
cargo build --release --locked --features extension-module \
  --target-dir target/cuda-neg-validation/build
cp target/cuda-neg-validation/build/release/libpytorch_rs.so python/torch_rs/torch_rs.abi3.so
CUDA_VISIBLE_DEVICES=0 "$REFERENCE_PYTHON" scripts/evaluate_cuda_math.py \
  --build-record target/cuda-neg-validation/build-record.json \
  --output target/cuda-neg-validation/evaluation.json
```

The build receipt records the build command, compiler versions, extension hash
and the evaluator's `source_provenance()` before/after the build. The raw report
and receipt are local validation artifacts under ignored `target/`. The
evaluator verified unchanged source during execution, native device pointers,
input preservation, materialized results and independent candidate/reference
processes. Candidate workers loaded no PyTorch modules.

- GPU: NVIDIA H100, compute capability 9.0, driver 580.82.07;
  device 0 UUID `8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`.
- Reference: CPython 3.12.12, PyTorch 2.13.0+cu130, CUDA 13.0.
- Both workers actually mapped CUDA runtime 13000 from
  `/data/users/bobren/a/pytorch-rs-burner/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`,
  explicitly selected with `TORCH_RS_CUDART`.
- Rust 1.92.0 (`ded5c06cf`), Cargo 1.92.0 (`344c4567c`), release profile,
  thin LTO, one codegen unit, `extension-module`, `abi3-py310`.
- Embedded PTX 6.0 targeting `sm_50` was JIT-compiled by the NVIDIA driver.
  Available nvcc 12.6.85 and NVRTC were unused.
- The existing reference environment was read only. Build products, native
  extension, Cargo/download/JIT caches and temporary files stayed in this worktree.

The measured source is an uncommitted implementation overlay on
`31310e8fd1a1ff4f8dadcca93566dba70d115241`, identified by these SHA-256 hashes:

| Artifact | SHA-256 |
| --- | --- |
| Evaluator production-source fingerprint | `19dc571beb5bd41c58d8ef69122df51ec2b9f6c3883640bbe4af7148fa186499` |
| Native extension | `15bead05125d20899f23feb42e5cd5911d1d6ecd9db8d60febbabf64c34f0997` |
| Unchanged evaluator | `29a97d07bad0d82c261e28aa880de68cdf46df0731d5ff0aa9802ec5bb8e605b` |
| Unchanged matrix | `33ce3417e04bc87f73e5b95acc7480295c476d406d4dc04ae6b7d1e4e6a6e513` |

## Checks

- `cargo test --locked --features python-bindings --lib --test cuda_add --test cuda_native_boundaries`:
  177 passed on device 0, including native bitwise negation and storage bounds.
- `cargo clippy --locked --all-targets --features python-bindings -- -D warnings`,
  `cargo fmt --check` and `git diff --check` passed.
- With `CUDA_VISIBLE_DEVICES=0`, the fresh extension passed 45 tests across
  `test_cuda_neg`, `test_cuda_add`, `test_compile_cuda_boundary`,
  `test_cuda_native_views`, `test_top_level_neg` and `test_top_level_negative`.
  Five tests skipped because they require another device or execution mode.
- With `CUDA_VISIBLE_DEVICES=0,1`, all four negation/addition device-guard and
  context-cache tests passed.

Negation tests cover generated shapes, block/grid tails, signed zeros,
subnormals, infinities, NaNs, offsets, empty/scalar inputs, input preservation,
fresh output storage, independent-stream reads, cross-thread ownership,
allocation reuse and outputs larger than the 64 MiB front cache. Hardware-only
cases skip clearly when the required CUDA devices are unavailable. Existing
evaluator definitions, benchmark files and managed progress artifacts are unchanged.
