export TMPDIR="$PWD/target/admission/tmp"
export XDG_CACHE_HOME="$PWD/target/admission/cache"
export UV_CACHE_DIR="$PWD/target/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/target/uv-python"
export CARGO_HOME="$PWD/target/cargo-home"
export CARGO_TARGET_DIR="$PWD/target"
export CUDA_CACHE_PATH="$PWD/target/admission/cuda-cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/target/admission/inductor-cache"
export TRITON_CACHE_DIR="$PWD/target/admission/triton-cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CARGO_BUILD_JOBS=4
export CUDA_VISIBLE_DEVICES=0 UV_SYSTEM_CERTS=true
