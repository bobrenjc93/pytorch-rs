#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
virtualenv="$repository_root/.venv"
target_path="$repository_root/target"
wheel_directory=

cleanup() {
    if [[ -n "${wheel_directory:-}" ]]; then
        rm -rf -- "$wheel_directory"
    fi
}
trap cleanup EXIT

cd "$repository_root"

if [[ -L "$virtualenv" ]]; then
    echo "refusing symlinked virtual environment: $virtualenv" >&2
    exit 1
fi
if [[ -L "$target_path" ]]; then
    echo "refusing symlinked target directory: $target_path" >&2
    exit 1
fi
mkdir -p "$target_path"
target_directory="$(cd "$target_path" && pwd -P)"
if [[ "$target_directory" != "$target_path" ]]; then
    echo "target directory resolved outside the worktree: $target_directory" >&2
    exit 1
fi

export CARGO_HOME="$target_directory/cargo-home"
export CARGO_TARGET_DIR="$target_directory"
export UV_CACHE_DIR="$target_directory/uv-cache"
export UV_PYTHON_INSTALL_DIR="$target_directory/uv-python"
export UV_PROJECT_ENVIRONMENT="$virtualenv"
export PYTHONNOUSERSITE=1

if [[ ! -x "$virtualenv/bin/python" ]]; then
    uv --no-config venv --python 3.12 "$virtualenv" >/dev/null
fi
uv --no-config sync \
    --locked \
    --python "$virtualenv/bin/python" \
    --no-install-project \
    --group dev \
    --group reference \
    >/dev/null

wheel_directory="$(mktemp -d "$target_directory/compile-coverage-wheels.XXXXXX")"
TMPDIR="$target_directory" \
VIRTUAL_ENV="$virtualenv" \
PYO3_PYTHON="$virtualenv/bin/python" \
    "$virtualenv/bin/maturin" build --release --locked --out "$wheel_directory" \
    >/dev/null

shopt -s nullglob
wheels=("$wheel_directory"/torch_rs-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "expected exactly one torch-rs wheel, found ${#wheels[@]}" >&2
    exit 1
fi

uv --no-config pip install \
    --python "$virtualenv/bin/python" \
    --force-reinstall \
    --no-deps \
    "${wheels[0]}" \
    >/dev/null

"$virtualenv/bin/python" .github/scripts/verify_native_extension.py >/dev/null
"$virtualenv/bin/python" scripts/evaluate_torch_compile_coverage.py "$@"
