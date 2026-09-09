# Native CUDA neg/add capture validation

The shared marker-free eager graph path accepts unary minus, `Tensor.neg()` and
`Tensor.negative()` over exact native contiguous float32 CUDA tensors, composed
with same-shape addition. This is unfused bounded capture, not a general
Inductor compiler or a performance-parity claim. See the [owning guide](compile-cuda-add.md).

Raw reports and logs are retained under
[docs/diagnostics/compile-cuda-neg/](diagnostics/compile-cuda-neg/build-record.json),
including the [production patch](diagnostics/compile-cuda-neg/production.patch)
and [source audit](diagnostics/compile-cuda-neg/provenance-audit.log). The baseline was
freshly built from clean commit `e4cddf0ecb45c23c683953fa2f8c45904b170a80`.
The candidate is a fresh release build bound to that commit plus the recorded
production patch and source fingerprint. It is **not clean-commit candidate
evidence**: this implementation task prohibits creating commits. Burner must
commit the candidate and repeat the commands below before claiming that milestone.

All dependency installations, Cargo targets, CUDA/compiler caches, and temporary
files were confined to this worktree. The selected runtime is the worktree-local
CUDA 13 runtime from PyTorch 2.13.0+cu130; H100 compute capability 9.0 and driver
580.82.07 were used. Available nvcc is 12.6.85; native negation/addition use
embedded PTX 6.0 targeting sm_50 through driver JIT, without nvcc. Rust/Cargo
1.92.0 built release extension-module/abi3-py310 with thin LTO and one codegen unit.
The unchanged private CUDA performance benchmark records its own compiler and
runtime selection separately.

## Results and limits

| Check | Result | Raw evidence |
| --- | --- | --- |
| Rust default / Python bindings | 354 / 365 passed | [default](diagnostics/compile-cuda-neg/rust-default-tests.log), [bindings](diagnostics/compile-cuda-neg/rust-tests.log) |
| Formatting / Clippy | passed | [fmt](diagnostics/compile-cuda-neg/fmt.log), [Clippy](diagnostics/compile-cuda-neg/clippy.log) |
| Final focused suite | 126 tests; passed, 6 device-specific skips | [log](diagnostics/compile-cuda-neg/focused-final.log) |
| Two-device suite, GPUs 0,1 | 6 passed | [log](diagnostics/compile-cuda-neg/two-device-final.log) |
| Full Python suite | 5,311 tests; 5 failures, 11 skips; see below | [log](diagnostics/compile-cuda-neg/python-full.log) |
| Unchanged compile evaluator | 38/38 existing cases passed | [report](diagnostics/compile-cuda-neg/compile-evaluation.json) |
| Unchanged CUDA performance benchmark | 4/4 existing shapes passed | [fresh-cache report](diagnostics/compile-cuda-neg/cuda-performance-fresh.json) |
| Unchanged CUDA math evaluator | 2/6 cases passed on all 3 seeds | [report](diagnostics/compile-cuda-neg/cuda-math.json) |
| Unchanged addition diagnostic | 58 passed, 24 unsupported on GPU 0; 4 passed, 2 unsupported on GPUs 0,1 | [GPU 0](diagnostics/compile-cuda-neg/addition-diagnostic.json), [GPUs 0,1](diagnostics/compile-cuda-neg/addition-diagnostic-multi.json) |

The full run found one obsolete compiled-negation rejection assertion in
`test_cuda_neg.py`. Only that assertion was replaced with a differential success
check; its layout and autograd rejections remain. The final focused run passes
that corrected test and all the added CUDA capture tests. The full suite was
not repeated after this test-only correction. The earlier
[focused run](diagnostics/compile-cuda-neg/focused.log) preserves the same obsolete
assertion failure; the [initial run](diagnostics/compile-cuda-neg/focused-initial.log)
preserves two documentation smoke failures subsequently fixed.

The remaining four full-run failures are the existing factory keyword ordering
and noncanonical boolean-buffer differentials. A separate environment using the
fresh clean-baseline wheel reproduced both boolean-buffer failures
([baseline](diagnostics/compile-cuda-neg/baseline-buffer.log)) and factory-order
failures at `PYTHONHASHSEED=6`
([baseline](diagnostics/compile-cuda-neg/baseline-factory-seed6.log)). Factory
ordering is intermittent: the isolated candidate seed-6 rerun passed
([log](diagnostics/compile-cuda-neg/candidate-factory-seed6.log)); the candidate
buffer rerun still failed both boolean cases
([log](diagnostics/compile-cuda-neg/candidate-baseline-failures-seed6.log)).
The relevant production Python files are byte-identical between the baseline
and candidate wheels; only `_compile_trace.py` differs. See the baseline
[build receipt](diagnostics/compile-cuda-neg/baseline-build-record.json) and
[environment receipt](diagnostics/compile-cuda-neg/baseline-repro-provenance.json).
These unrelated tests and implementations were not changed.

The compile corpus remains fixed at 38 cases and does not measure the new CUDA
capture surface. CUDA math remains fixed at six cases; broadcasting, scalar
multiplication, axis reduction and matmul retain zero credit. No new overall
coverage, hardware-family, training or performance-parity claim is made. The
addition diagnostic's two negation acceptances are its only expectation
mismatches; see the frozen-expectation note below.

The final unchanged performance benchmark used new CUDA, Triton and Inductor
cache directories and rebuilt its private pointwise library with
`/usr/local/cuda-12.6/bin/nvcc`, targeting `sm_90`; its runtime is libcudart 13.0.
Five warmups, 17 samples and three repeats per sample retain the existing
synchronization and materialization policy. The raw report includes the compiler
command, timings, variance and memory pressure. Its four-shape result measures
the unchanged private benchmark path, not generic neg/add capture. The earlier
[cache-reusing run](diagnostics/compile-cuda-neg/cuda-performance.json) is retained
separately. Neither timing run is clean-commit candidate evidence.

## Repeat after Burner commits

Use the worktree-local dependency and cache setup in the
[kernel validation guide](cuda-neg-validation.md#reproduce-from-this-checkout).
Select a new, unused `CARGO_TARGET_DIR` for the candidate build. With that
configuration and `CUDA_VISIBLE_DEVICES=0`:

```bash
.venv/bin/maturin build --release --locked --out target/compile-neg-wheels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/compile-neg-wheels/*.whl
cp "$CARGO_TARGET_DIR/release/libpytorch_rs.so" python/torch_rs/torch_rs.abi3.so
.venv/bin/python .github/scripts/verify_native_extension.py
cargo fmt --all -- --check
cargo clippy --locked --all-targets --features python-bindings -- -D warnings
cargo test --locked --all-targets
cargo test --locked --all-targets --features python-bindings
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m unittest -v tests.test_compile_cuda_neg \
  tests.test_compile_cuda_boundary tests.test_cuda_neg tests.test_cuda_add \
  tests.test_cuda_native_views tests.test_top_level_compile \
  tests.test_readme_quickstart tests.test_cuda_math_evaluator \
  tests.test_hardware_heterogeneity_evaluation
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v \
  tests.test_compile_cuda_neg.CompileCudaNegDeviceTests \
  tests.test_compile_cuda_boundary.CompileCudaDeviceTests \
  tests.test_cuda_neg.CudaNegDeviceTests tests.test_cuda_add.CudaAddDeviceTests
bash scripts/evaluate_torch_compile_coverage.sh > target/compile-neg-coverage.json
.venv/bin/python scripts/benchmark_compile_cuda.py \
  --include-unprepared-comparison --output target/compile-neg-performance.json
```

Before running the unchanged CUDA math evaluator, create a build receipt using
the [existing receipt procedure](hardware-heterogeneity-evaluator.md#reproduce-and-bind-a-native-build).
Then run:

```bash
.venv/bin/python scripts/evaluate_cuda_math.py \
  --seed 9173 --seed 260909 --seed 903217 \
  --build-record target/compile-neg-build-record.json \
  --output target/compile-neg-cuda-math.json
.venv/bin/python scripts/diagnose_compile_cuda_add.py \
  --output target/compile-neg-addition-diagnostic.json
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/diagnose_compile_cuda_add.py \
  --output target/compile-neg-addition-diagnostic-multi.json
```

The unchanged addition diagnostic's GPU-0 `reject_neg` expectation is obsolete:
its raw negation outputs must now match the reference even though the script
returns exit 1 for that successful case. Keep that diagnostic's report intact;
it is separate from the scoring evaluators. Do not infer a failure of native
negation from its old expectation flag.
