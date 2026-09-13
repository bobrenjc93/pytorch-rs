export TASK_ROOT="$(pwd -P)"
export CARGO_HOME="$TASK_ROOT/target/tensor-madd/cargo-home"
export CARGO_TARGET_DIR="$TASK_ROOT/target/tensor-madd/build"
export RUSTUP_TOOLCHAIN=1.92.0
export CARGO_BUILD_JOBS=8
export UV_CACHE_DIR="$TASK_ROOT/target/tensor-madd/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/target/tensor-madd/python"
export UV_LINK_MODE=copy
export TMPDIR="$TASK_ROOT/target/tensor-madd/tmp"
export XDG_CACHE_HOME="$TASK_ROOT/target/tensor-madd/cache"
export CUDA_CACHE_PATH="$TASK_ROOT/target/tensor-madd/cuda-cache"
export TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/target/tensor-madd/inductor-cache"
export TRITON_CACHE_DIR="$TASK_ROOT/target/tensor-madd/triton-cache"
export PYTHONPYCACHEPREFIX="$TASK_ROOT/target/tensor-madd/pycache"
export VIRTUAL_ENV="$TASK_ROOT/.venv"
export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH CONDA_PREFIX
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
