# Run from the worktree root; hold Burner's gpu and cpu-heavy resources first.
# These are author correctness checks, not the paired scoring/qualification gate.
source docs/diagnostics/compile-input-admission/env.sh
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

# This command initially failed downloading Python (setup.log). The author
# copied the existing Python 3.12.14, Cargo registry and locked environment into
# worktree-local target paths, as recorded below, then repeated setup-only.
bash scripts/evaluate_torch_compile_default.sh --setup-only

# Actual read-only source copies used to recover from that download failure:
# cp -a --reflink=auto /data/users/bobren/a/pytorch-rs-burner/target/uv-python/cpython-3.12.14-linux-x86_64-gnu target/uv-python/
# cp -a --reflink=auto /data/users/bobren/a/pytorch-rs-burner/target/cargo-home/. target/cargo-home/
# cp -a --reflink=auto /data/users/bobren/a/pytorch-rs-burner/target/default-compile-eval/venv target/default-compile-eval/venv
# ln -sfn "$PWD/target/uv-python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12" target/default-compile-eval/venv/bin/python
# bash scripts/evaluate_torch_compile_default.sh --setup-only
# After cargo fmt, setup-only ran again to rebuild the final source.

PYTHONPATH="$PWD/tests" target/default-compile-eval/venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_admission \
  tests.test_compile_pointwise_helpers.HelperAdmission \
  tests.test_compile_pointwise_helpers.HelperCache \
  tests.test_compile_pointwise_scalar_admission.ScalarAdmission \
  tests.test_compile_pointwise_scalar_admission.PositionalBindingAdmission \
  tests.test_compile_pointwise_scalar_admission.SharedCacheGuards \
  tests.test_compile_pointwise_structured_outputs.StructuredAdmission \
  tests.test_compile_pointwise_structured_outputs.StructuredCache \
  tests.test_compile_pointwise_prepared.PreparedExports \
  tests.test_compile_pointwise_prepared.PreparedCache \
  tests.test_compile_pointwise_jit.Admission

cargo test --locked --offline --lib pointwise

PYTHONPATH="$PWD/tests" target/default-compile-eval/venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_prepared.PreparedHardware \
  tests.test_compile_pointwise_broadcast.Metadata \
  tests.test_compile_pointwise_broadcast.Broadcast \
  tests.test_compile_pointwise_jit.Hardware.test_generated_expression_trees_default_inductor_and_fresh_data \
  tests.test_compile_pointwise_jit.Hardware.test_offsets_constants_code_bindings_guards_and_reset \
  tests.test_compile_pointwise_jit.Hardware.test_invalid_metadata_failure_retry_and_concurrency

PYTHONPATH="$PWD/tests" target/default-compile-eval/venv/bin/python -m unittest -v \
  tests.test_compile_corpus.CompileCorpusTraceTests \
  tests.test_compile_cuda_boundary.CompileCudaBoundaryTests

cargo fmt --all -- --check
git diff --check
