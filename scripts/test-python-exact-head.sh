#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
virtualenv="$repository_root/.venv"
python="$virtualenv/bin/python"
maturin="$virtualenv/bin/maturin"

cd "$repository_root"

# Do not let an activated environment or source-tree import path select a stale
# extension. The exact workspace environment is set explicitly where needed.
unset CONDA_PREFIX VIRTUAL_ENV PYTHONHOME PYTHONPATH PYO3_PYTHON

mkdir -p "$repository_root/target"
export CARGO_HOME="$repository_root/target/cargo-home"
export CARGO_TARGET_DIR="$repository_root/target"
export UV_CACHE_DIR="$repository_root/target/uv-cache"
export UV_PROJECT_ENVIRONMENT="$virtualenv"
export PYTHONNOUSERSITE=1

uv venv --clear --python 3.12 "$virtualenv"
uv sync \
    --locked \
    --python "$python" \
    --no-install-project \
    --group reference
uv pip install --python "$python" 'maturin>=1.14,<2'

wheel_directory="$(mktemp -d "$repository_root/target/exact-head-wheel.XXXXXX")"
cleanup() {
    rm -rf -- "$wheel_directory"
}
trap cleanup EXIT

TMPDIR="$repository_root/target" \
VIRTUAL_ENV="$virtualenv" \
PYO3_PYTHON="$python" \
    "$maturin" build --release --out "$wheel_directory"

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

import torch


if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, got {sys.version}")

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
