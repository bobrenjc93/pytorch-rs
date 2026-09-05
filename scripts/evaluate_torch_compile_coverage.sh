#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
target_path="$repository_root/target"
wheel_directory=

cleanup() {
    if [[ -n "${wheel_directory:-}" ]]; then
        rm -rf -- "$wheel_directory"
    fi
}
trap cleanup EXIT

cd "$repository_root"

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

evaluator_path="$target_directory/torch-compile-coverage"
if [[ -L "$evaluator_path" ]]; then
    echo "refusing symlinked evaluator directory: $evaluator_path" >&2
    exit 1
fi
mkdir -p "$evaluator_path"
evaluator_directory="$(cd "$evaluator_path" && pwd -P)"
if [[ "$evaluator_directory" != "$evaluator_path" ]]; then
    echo "evaluator directory resolved outside the worktree: $evaluator_directory" >&2
    exit 1
fi

virtualenv="$evaluator_directory/venv"
python="$virtualenv/bin/python"
maturin="$virtualenv/bin/maturin"

if [[ -L "$virtualenv" ]]; then
    echo "refusing symlinked virtual environment: $virtualenv" >&2
    exit 1
fi
if [[ -e "$virtualenv" && ! -d "$virtualenv" ]]; then
    echo "virtual environment path is not a directory: $virtualenv" >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "missing flock command required to serialize evaluator setup" >&2
    exit 1
fi

exec {setup_lock_fd}<"$evaluator_directory"
flock "$setup_lock_fd"

export CARGO_HOME="$target_directory/cargo-home"
export CARGO_TARGET_DIR="$target_directory"
export UV_CACHE_DIR="$target_directory/uv-cache"
export UV_PYTHON_INSTALL_DIR="$target_directory/uv-python"
export UV_PROJECT_ENVIRONMENT="$virtualenv"
export PYTHONNOUSERSITE=1

if [[ ! -x "$python" ]] ||
    ! "$python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' \
        >/dev/null 2>&1
then
    uv --no-config venv --clear --python 3.12 "$virtualenv" >/dev/null
fi
uv --no-config sync \
    --locked \
    --python "$python" \
    --no-install-project \
    --group dev \
    --group reference \
    >/dev/null

wheel_directory="$(mktemp -d "$evaluator_directory/compile-coverage-wheels.XXXXXX")"
TMPDIR="$evaluator_directory" \
VIRTUAL_ENV="$virtualenv" \
PYO3_PYTHON="$python" \
    "$maturin" build --release --locked --out "$wheel_directory" \
    >/dev/null

shopt -s nullglob
wheels=("$wheel_directory"/torch_rs-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "expected exactly one torch-rs wheel, found ${#wheels[@]}" >&2
    exit 1
fi

uv --no-config pip install \
    --python "$python" \
    --force-reinstall \
    --no-deps \
    "${wheels[0]}" \
    >/dev/null

TORCH_RS_VERIFY_VIRTUALENV="$virtualenv" "$python" \
    .github/scripts/verify_native_extension.py >/dev/null
"$python" scripts/evaluate_torch_compile_coverage.py "$@"
