# Clean compiled CUDA matmul capture

Measured implementation commit: **`d8374ec16f6c13fb09b18723c491d06fdaeaafaa`**.
All builds, workloads and verification finished with an empty tracked checkout
status before these reports and their documentation were published. This step
changes evidence/documentation only and does not approve the candidate.

The [candidate inspection](inspection.json) records the complete diff against
main `046b7a21e4e2fb7b59b56ea8a9679a9d9c5b0981`, file hashes and outstanding
capture requirements. The original [development capture](../development/README.md)
remains unchanged, including its unsuccessful attempts; it does not supply
clean-commit performance credit.

## Results

| Required capture/check | Result | Raw evidence |
| --- | --- | --- |
| Fresh release wheel and installed/source binary identity | Passed; empty Cargo target, locked offline build | [Build record](build-record.json), [build/install commands](commands.json), [build log](build.log), [install log](install.log) |
| Worktree-local interpreter, imports and runtime identities | Passed; every installed `torch_rs` Python source matched this checkout | [Provenance](provenance.json), [wheel verification](wheel-verifier.log) |
| Compiled matmul H100 regressions | 12 passed; two-GPU test skipped under the single-GPU mask | [Verbose test log](compiled-regressions.log), [receipt](compiled-regressions.receipt.json) |
| Separate GPUs 0,1 | Device restoration, mixed-device rejection and device guards passed | [Log](two-gpu.log), [receipt](two-gpu.receipt.json) |
| Rust compiled bridge and native matmul | One bridge test and two matmul tests passed on GPU 0 | [Bridge](rust-bridge.log), [matmul](rust-matmul.log) |
| Unchanged fixed CUDA math workload | Six cases passed at all three existing seeds: 18/18 trials | [Report](fixed-math.json), [receipt](fixed-math.receipt.json) |
| Unchanged compiler corpus | 38/38 reference-eligible cases passed; bounded existing score 100 | [Report/log](frozen-compiler.log), [receipt](frozen-compiler.receipt.json) |
| Separate compiled timing diagnostic | 12/12 cells passed; **79.48% capped geometric parity** | [Raw report](compiled-timings.json), [receipt](compiled-timings.receipt.json) |
| Post-capture integrity verification | Passed: source/binary identities, current-worktree paths, raw sample counts, math accounting and fixed timing aggregation | [Verification](verification.json), [receipt](evidence-validation.receipt.json) |

The compiled tests exercise both supported fullgraph modes, changed input data,
shape/stride/offset and callable guards, rejection/recovery, aliases, overlapping
views, empties, singleton/zero-inner products, wide-K decimal drift and overflow
counterexamples. The isolated `-I` test blocks PyTorch imports and Python-body
execution. The timing diagnostic independently blocks re-lowering and body
execution on changed-data calls outside its timed intervals.

The timing matrix, seed 798431, tolerances, warmups and sampling policy are
unchanged. Both execution orders retain 31 five-call samples per implementation
per cell; medians, p10/p90 and extrema are retained. All six pure matmul cells
reached the cap in this run. All six composed cells were slower and remain in
the fixed equal-weight geometric aggregation. First-call and wrapper costs are
separate from warmed synchronized timings. Inductor and Triton caches started
empty; disk caches are reused across later cells/orders as documented by the
committed harness. This scoped diagnostic is not a new Burner score or a claim
of general `torch.compile` parity.

## Source, binary and hardware

- Native extension SHA-256: `e591b9edecc721aefacce130ce7b9771589e3e758e11b1ed1d8ce524a4b50d2d`.
- CPython 3.12.14 and PyTorch 2.13.0+cu130 are installed inside this worktree.
- H100 GPU 0, compute capability 9.0, driver 580.82.07; only the separate device
  test exposes GPUs 0,1.
- Selected local CUDA runtime: 13000; cuBLAS version: 13.1.1. Exact mapped paths
  and library hashes are in the provenance and timing reports.
- Rust 1.92.0, release profile, thin LTO, one codegen unit, `extension-module`
  and abi3-py310. Available nvcc 12.6 was unused: matmul uses native cuBLAS;
  pointwise operations use driver-JIT embedded PTX.
- The native build target was absent before the build. The locked local Python
  environment and Cargo/uv download caches were reused. Native code was freshly
  built and installed as a wheel, with identical bytes copied to the source
  package required by the unchanged math evaluator.

[Artifact hashes](artifact-hashes.json) bind the published raw files to their
original current-worktree `target/postcommit-d8374` captures. No provenance was
rewritten during publication. The command receipts include argv, timestamps,
exit status, source hashes, clean status and relevant environment. The source
and harnesses stayed unchanged during measurement and verification. No failures
occurred in this post-commit campaign; earlier development failures remain in
the unchanged development directory.

## Reproduce

Use the locked worktree-local environment and local caches described in the
[compiler guide](../../../compile-cuda-matmul.md). Set `CUDA_VISIBLE_DEVICES=0`,
local `TORCH_RS_CUDART`/`TORCH_RS_CUBLAS`, and fresh local CUDA, Inductor and
Triton cache directories. Start from a clean checkout of the measured commit.
With a new output path, invoke the existing repository tools:

```bash
.venv/bin/python -B scripts/capture_depth_concat_build.py --output target/compiled-matmul-clean/native
.venv/bin/python -B .github/scripts/verify_native_extension.py
.venv/bin/python -B -m unittest -v tests.test_compile_cuda_matmul tests.test_compile_cuda_matmul_diagnostic
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -B -m unittest -v tests.test_compile_cuda_matmul.CompileCudaMatmulDeviceTests
cargo test --locked --features python-bindings --lib compiled_matmul_bridge
cargo test --locked --test cuda_matmul
.venv/bin/python -B scripts/evaluate_cuda_math.py \
  --seed 8503945240872567646 --seed 8613321571747136749 --seed 4480763905421893394 \
  --build-record target/compiled-matmul-clean/native/build-record.json \
  --output target/compiled-matmul-clean/math.json
.venv/bin/python -B scripts/evaluate_torch_compile_coverage.py --subset full
.venv/bin/python -B scripts/diagnose_compile_cuda_matmul.py \
  --build-record target/compiled-matmul-clean/native/build-record.json \
  --output target/compiled-matmul-clean/timings.json
```

The build tool creates its own empty native target and records all build/install
commands. Point `CARGO_TARGET_DIR` at that local target for the Rust checks.
Campaign command receipts use the existing `capture` helper in
`docs/diagnostics/composite-cuda-neg/reproduce.py`; workload implementations and
measurement definitions were not changed. Publish copies only after capture
and source verification have completed.

Independent review, all ten non-regressing gates and exact-head CI remain
required before merge. Unrelated full test suites were not rerun in this
focused evidence step. No implementation, dependency, test, harness, evaluator,
corpus, threshold, weight or Burner-managed progress artifact changed.
