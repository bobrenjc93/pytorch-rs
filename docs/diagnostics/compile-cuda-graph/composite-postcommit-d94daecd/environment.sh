export UV_CACHE_DIR="$PWD/target/integration/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/target/integration/uv-python"
export UV_PYTHON_BIN_DIR="$PWD/target/integration/uv-python-bin"
export UV_TOOL_BIN_DIR="$PWD/target/integration/uv-tool-bin"
export UV_TOOL_DIR="$PWD/target/integration/uv-tools"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export CARGO_HOME="$PWD/target/integration/cargo"
export CARGO_TARGET_DIR="$PWD/target/integration/rust"
export CARGO_HTTP_PROXY=http://fwdproxy:8080
export TMPDIR="$PWD/target/integration/tmp"
export XDG_CACHE_HOME="$PWD/target/integration/cache"
export CUDA_CACHE_PATH="$PWD/target/integration/cuda"
export TRITON_CACHE_DIR="$PWD/target/integration/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/target/integration/inductor"
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 RUSTUP_TOOLCHAIN=1.92.0
export VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python"
export CUDA_HOME=/usr/local/cuda-12.6 CUDA_VISIBLE_DEVICES=0
export TORCH_RS_CUDART="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
export TORCH_RS_CUBLAS="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublas.so.13"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CONDA_PREFIX PYTHONPATH
