#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
virtualenv="$repository_root/.venv"
python="$virtualenv/bin/python"
maturin="$virtualenv/bin/maturin"
provenance_check="$repository_root/.github/scripts/verify_native_extension.py"
target_path="$repository_root/target"

cd "$repository_root"

if [[ ! -x "$python" || ! -x "$maturin" ]]; then
    echo "missing .venv Python or Maturin; run the README environment setup first" >&2
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

wheel_directory="$(mktemp -d "$target_directory/python-test-wheels.XXXXXX")"
cleanup() {
    rm -rf -- "$wheel_directory"
}
trap cleanup EXIT

env -u CONDA_PREFIX \
    TMPDIR="$target_directory" \
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

"$python" "$provenance_check"

if "$python" -m unittest discover -s tests -p 'test_*.py'; then
    :
else
    status=$?
    echo "Python suite failed; resolved native-extension paths:" >&2
    "$python" "$provenance_check" >&2 || true
    exit "$status"
fi
