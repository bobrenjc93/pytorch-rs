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

## Post-commit H100 validation, 2026-09-09

Measured clean implementation commit: `5fa1017bff6bc8608ce3ff6ee9bf79e87c02680c`.
The worktree was clean before and after both the build and evaluator run.
Only this evidence document changed afterward; implementation, dependencies,
tests, evaluator and workload definitions are unchanged.

The unchanged `scripts/evaluate_cuda_math.py` reused all three previously
selected evaluator seeds: `8503945240872567646`, `8613321571747136749`, `4480763905421893394`.
All 18 reference executions passed. Both `cuda_f32_neg` and
`cuda_f32_add_same_shape` passed every seed: **2/6 fixed cases**. Trailing-vector
addition, scalar multiplication, axis reduction and matrix multiplication
remain unsupported and receive zero. This is correctness evidence for this
cell, without a claim about overall heterogeneity score or performance.

The repository-documented native build and receipt procedure in
[hardware-heterogeneity-evaluator.md](hardware-heterogeneity-evaluator.md#reproduce-and-bind-a-native-build)
was used from this clean checkout. Locked dependencies were installed with
`uv sync --locked --no-install-project --group dev --group reference --python 3.12`.
The new environment is `.venv`; its interpreter resolves to
`target/postcommit-neg/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12` inside this worktree.
`UV_PYTHON_INSTALL_DIR`, `UV_CACHE_DIR`, `TMPDIR`, `CUDA_CACHE_PATH` and
`XDG_CACHE_HOME` were all rooted under `target/postcommit-neg/`.
The Cargo download cache at `target/cargo-home` was warm; the Cargo build target
was initially empty. The Python environment and uv download cache were new.
The evaluator gives each run its own temporary CUDA JIT cache.

```sh
export PYO3_PYTHON="$PWD/.venv/bin/python"
export CARGO_HOME="$PWD/target/cargo-home"
export CARGO_TARGET_DIR="$PWD/target/postcommit-neg/build"
cargo build --locked --release --features extension-module
cp target/postcommit-neg/build/release/libpytorch_rs.so python/torch_rs/torch_rs.abi3.so
export TORCH_RS_CUDART="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_cuda_math.py \
  --seed 8503945240872567646 --seed 8613321571747136749 --seed 4480763905421893394 \
  --build-record target/postcommit-neg/build-record.json \
  --output target/postcommit-neg/evaluation.json
```

The fresh release build ran from `2026-09-09T18:58:55.141474+00:00` to
`2026-09-09T18:59:36.590527+00:00`. The evaluator ran from
`2026-09-09T18:59:44.593436+00:00` to `2026-09-09T19:01:07.574968+00:00` and exited 0.
The build receipt records the actual command, compiler versions, extension hash
and the evaluator's `source_provenance()` verified before/after the build.
The raw report, build receipt, run receipt and logs are local validation
artifacts under ignored `target/postcommit-neg/`.
The evaluator verified unchanged source during execution, native device
pointers, input preservation, materialized results and independent
candidate/reference processes. Candidate workers loaded no PyTorch modules.
Every recorded interpreter, package, extension and CUDA runtime path was
verified to resolve inside this worktree.

- GPU: NVIDIA H100, compute capability 9.0, driver 580.82.07;
  device 0 UUID `8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`; `CUDA_VISIBLE_DEVICES=0`.
- Reference: CPython 3.12.14, PyTorch 2.13.0+cu130, CUDA 13.0.
- Both workers actually mapped CUDA runtime 13000 from
  `.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`, relative to this worktree,
  explicitly selected with `TORCH_RS_CUDART`.
- Rust 1.92.0 (`ded5c06cf`), Cargo 1.92.0 (`344c4567c`), release profile,
  thin LTO, one codegen unit, `extension-module`, `abi3-py310`.
- Embedded PTX 6.0 targeting `sm_50` was JIT-compiled by the NVIDIA driver.
  Available nvcc 12.6.85 and NVRTC were unused.

The source and extension hashes match the original implementation validation;
the newly measured receipt now binds them to the clean implementation commit.

| Artifact | SHA-256 |
| --- | --- |
| Evaluator production-source fingerprint | `19dc571beb5bd41c58d8ef69122df51ec2b9f6c3883640bbe4af7148fa186499` |
| Native extension | `15bead05125d20899f23feb42e5cd5911d1d6ecd9db8d60febbabf64c34f0997` |
| Unchanged evaluator | `29a97d07bad0d82c261e28aa880de68cdf46df0731d5ff0aa9802ec5bb8e605b` |
| Unchanged matrix | `33ce3417e04bc87f73e5b95acc7480295c476d406d4dc04ae6b7d1e4e6a6e513` |
| Refreshed raw evaluation report | `c501ce7e0b5195a376442a0715732d890cf163e42e1f772bd8109f48a87324f7` |

## Evidence refresh checks

The full six-case evaluation was checked against the unchanged accounting
function, including every seed and all unsupported outcomes. Clean source,
extension hash, local paths and source stability checks passed.
`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest tests.test_cuda_neg.CudaNegTests -v`
passed all seven targeted tests on the rebuilt extension. `git diff --check`
also passed. No unrelated full test suite was rerun.

## Previously completed implementation checks

These author checks predate this post-commit evidence refresh. Their source
fingerprint and extension hash match the fresh build above; their results are
preserved below rather than presented as newly executed checks.

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
