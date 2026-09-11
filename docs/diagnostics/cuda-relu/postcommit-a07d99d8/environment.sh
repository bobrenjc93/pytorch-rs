source docs/diagnostics/cuda-relu/development/environment.sh
export TMPDIR="$PWD/target/relu-postcommit-a07d99d8/tmp"
export XDG_CACHE_HOME="$PWD/target/relu-postcommit-a07d99d8/xdg-cache"
export CUDA_CACHE_PATH="$PWD/target/relu-postcommit-a07d99d8/cuda-cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/target/relu-postcommit-a07d99d8/inductor-cache"
export TRITON_CACHE_DIR="$PWD/target/relu-postcommit-a07d99d8/triton-cache"
export TORCH_RS_CUDART="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
export PYTHONPATH="$PWD"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
