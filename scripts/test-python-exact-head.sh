#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

cd "$repository_root"

# Do not let an activated environment, source-tree import path, or optimized
# interpreter select stale code or disable assertions. Exact paths are set
# explicitly where needed.
unset \
    CONDA_PREFIX \
    GIT_DIR \
    GIT_INDEX_FILE \
    GIT_WORK_TREE \
    VIRTUAL_ENV \
    PYTHONHOME \
    PYTHONOPTIMIZE \
    PYTHONPATH \
    PYO3_PYTHON \
    UV_CONFIG_FILE \
    UV_PROJECT \
    UV_PROJECT_ENVIRONMENT \
    UV_PYTHON \
    UV_WORKING_DIR

mkdir -p "$repository_root/target"
run_directory="$(mktemp -d "$repository_root/target/exact-head-run.XXXXXX")"
checkout="$run_directory/checkout"
virtualenv="$checkout/.venv"
python="$virtualenv/bin/python"
maturin="$virtualenv/bin/maturin"
wheel_directory="$run_directory/wheels"

cleanup() {
    cd "$repository_root"
    rm -rf -- "$run_directory"
}
trap cleanup EXIT

mkdir -p "$checkout" "$wheel_directory"
head_commit="$(git -C "$repository_root" rev-parse --verify 'HEAD^{commit}')"
git -C "$repository_root" archive --format=tar "$head_commit" | tar -x -C "$checkout"
echo "testing exact HEAD $head_commit"

export CARGO_HOME="$repository_root/target/cargo-home"
export CARGO_TARGET_DIR="$run_directory/cargo-target"
export UV_CACHE_DIR="$repository_root/target/uv-cache"
export UV_PYTHON_INSTALL_DIR="$repository_root/target/uv-python"
export UV_PROJECT_ENVIRONMENT="$virtualenv"
export PYTHONNOUSERSITE=1

cd "$checkout"
uv venv --clear --python 3.12 "$virtualenv"
uv sync \
    --locked \
    --python "$python" \
    --no-install-project \
    --group reference

TMPDIR="$run_directory" \
VIRTUAL_ENV="$virtualenv" \
PYO3_PYTHON="$python" \
    "$maturin" build --release --locked --out "$wheel_directory"

shopt -s nullglob
wheels=("$wheel_directory"/torch_rs-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "expected exactly one torch-rs wheel, found ${#wheels[@]}" >&2
    exit 1
fi

uv pip install \
    --python "$python" \
    --force-reinstall \
    --no-deps \
    "${wheels[0]}"

"$python" .github/scripts/verify_native_extension.py
"$python" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, got {sys.version}")
if sys.flags.optimize != 0:
    raise SystemExit(f"expected Python optimization level 0, got {sys.flags.optimize}")

import torch


torch_version = torch.__version__.split("+", 1)[0]
if torch_version != "2.13.0":
    raise SystemExit(f"expected PyTorch 2.13.0, got {torch.__version__}")

if torch.cuda.is_available():
    cuda_status = f"available ({torch.cuda.get_device_name(0)})"
else:
    cuda_status = "unavailable; CUDA-only tests will skip"
print(f"verified Python 3.12 and PyTorch {torch.__version__}; CUDA is {cuda_status}")
PY

# PyTorch formats argument-parser errors differently when stderr is a terminal.
# Keep differential error strings stable while preserving the unittest status
# through Bash's pipefail setting.
"$python" -u -m unittest discover -s tests -p 'test_*.py' 2>&1 | cat
