# Compiled CUDA matrix/vector addition

Ordinary `torch_rs.compile(..., backend="eager")` captures contiguous float32
`(M,N)+(N,)` on the same explicit CUDA device in either operand order. Operator
`+`, positional `.add`, `.__add__` and `.__radd__` compose with existing scalar
multiplication, negation, helpers, global captures and tensor output containers.
The [compiler guide](compile-cuda-add.md) defines options and rejected syntax.
The shared native kernel supplies fresh storage, independent input offsets,
completion before return, and elementwise output strides for singleton/empty
views. General CPU broadcasting and dynamic specialization policy are unchanged.

`tests.test_compile_cuda_trailing_vector` compares both fullgraph modes and all
existing dynamic policies with PyTorch's eager backend, using independently
generated shapes and two fixed data seeds. It checks private bridge rejection,
cache reuse/reset/recompile limits, changed strides/offsets/ranks/devices,
captured scalar/tensor/callable changes, malformed graph metadata, aliasing,
source/output lifetimes and blocked Python-body/PyTorch-import execution.
The two-device class checks CUDA ordinals and current-device restoration.

## Reproduce in the evidence phase

Run from the clean committed implementation, with a real canonical `.venv` in
this worktree for the selected Python version. Follow the locked environment
setup in [CONTRIBUTING](../CONTRIBUTING.md); do not share a `.venv` symlink across
checkouts or Python versions. Each version needs its own checkout, build and logs.
Set worktree-local caches and bounded numerical-library threads before setup:

```bash
mkdir -p target/tmp target/cache
export TMPDIR="$PWD/target/tmp" XDG_CACHE_HOME="$PWD/target/cache"
export UV_CACHE_DIR="$PWD/target/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python"
export CARGO_HOME="$PWD/target/cargo-home" CARGO_TARGET_DIR="$PWD/target/build"
export CUDA_CACHE_PATH="$PWD/target/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$PWD/target/cache/inductor" TRITON_CACHE_DIR="$PWD/target/cache/triton"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python scripts/build_cuda_add_diagnostic.py --name trailing-final --revision HEAD
.venv/bin/python -m unittest tests.test_readme_quickstart
.venv/bin/python scripts/validate_top_level_stack_benchmark.py \
  --cases-per-category 1 --max-elements 64 --warmups 1 --samples 2 --threads 1 \
  --output target/stack-preflight.json
.venv/bin/python -m unittest tests.test_compile_cuda_trailing_vector \
  tests.test_compile_cuda_trailing_vector_diagnostic
.venv/bin/python -m scripts.diagnose_compile_cuda_trailing_vector \
  --case-set trailing_vector_v1 \
  --build-record target/cuda-add-diagnostic/trailing-final/build-record.json \
  --output target/trailing-vector-single.json
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest \
  tests.test_compile_cuda_trailing_vector.CompileCudaTrailingVectorDeviceTests
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m scripts.diagnose_compile_cuda_trailing_vector \
  --case-set trailing_vector_v1 \
  --build-record target/cuda-add-diagnostic/trailing-final/build-record.json \
  --output target/trailing-vector-multi.json
```

`trailing_vector_v1` is a separate correctness diagnostic. Reference options are
explicit: PyTorch 2.13, `backend="eager"`, `fullgraph=True/False`, `dynamic=None`.
Each eligible reference must match reference eager execution. All results are
materialized and synchronized, with value hashes and full output metadata.
Failures, unsupported cases and reference-ineligible cases receive zero local
credit. Unexpected failures return nonzero. Completed cases are checkpointed;
setup errors and interruptions are retained. Missing hardware is a diagnostic
failure, while hardware-only unit tests clearly skip on portable hosts.

The diagnostic verifies canonical environment containment, installed Python
source hashes and native library identity against the source-matched locked
release build record. It records the source manifest, harness hashes, Python
version/build/compiler, Rust toolchain, PyTorch/CUDA version, driver/GPU, native
runtime path/version and available nvcc. Native kernels use driver-JIT embedded
PTX; nvcc is not used by this implementation.

No performance measurement or universal `torch.compile` coverage is claimed.
The frozen 38-case scoring corpus, private four-workload CUDA benchmark, feature
weights and historical evidence are unchanged. Historical `neg_add_v1` retains
four obsolete matrix/vector rejection expectations; `mul_neg_add_v1` also retains
two. Preserve their raw failures, and distinguish them from the current diagnostic.

Burner must run and preserve clean committed final diagnostics, independent
review, current-definition no-regression gates and exact-head CI in its evidence
and publication phases. An implementation-worktree run with an uncommitted
source export is development evidence only. Record full-suite totals separately
from focused reruns, and retain failed/interrupted/setup captures. This branch
must not commit, publish, merge, or modify Burner-managed progress artifacts.
