#!/usr/bin/env bash
# Build from this worktree, then run the public-default comparison on real hardware.
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repository_root"
for relative in target target/default-compile-eval target/default-compile-eval/venv \
    target/cargo-home target/uv-cache target/uv-python; do
    if [[ -L "$repository_root/$relative" ]]; then
        echo "refusing symlinked build/evaluation path: $relative" >&2
        exit 1
    fi
done
evaluator_directory="$repository_root/target/default-compile-eval"
mkdir -p "$evaluator_directory"
if [[ "${TORCH_RS_DEFAULT_COMPILE_LOCKED:-}" != 1 ]]; then
    if [[ -L "$evaluator_directory/run.lock" ]]; then
        echo "refusing symlinked evaluation lock" >&2
        exit 1
    fi
    exec "${PYTHON:-python3}" scripts/run_with_unix_lock.py \
        "$evaluator_directory/run.lock" -- env TORCH_RS_DEFAULT_COMPILE_LOCKED=1 \
        bash scripts/evaluate_torch_compile_default.sh "$@"
fi

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export CARGO_HOME="$repository_root/target/cargo-home"
export CARGO_TARGET_DIR="$repository_root/target"
export UV_CACHE_DIR="$repository_root/target/uv-cache"
export UV_PYTHON_INSTALL_DIR="$repository_root/target/uv-python"
export UV_PROJECT_ENVIRONMENT="$evaluator_directory/venv"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "select exactly one real CUDA GPU for both implementations" >&2
    exit 1
fi

python="$UV_PROJECT_ENVIRONMENT/bin/python"
setup_started_ns="$(date +%s%N)"
if [[ ! -x "$python" ]]; then
    uv --no-config venv --managed-python --python 3.12 "$UV_PROJECT_ENVIRONMENT" >&2
fi
uv --no-config sync --locked --python "$python" --no-install-project \
    --group dev --group reference >&2
dependencies_finished_ns="$(date +%s%N)"
build_identity="$("$python" scripts/evaluate_torch_compile_default.py --source-identity)"
wheel_directory="$(mktemp -d "$evaluator_directory/wheels.XXXXXX")"
VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" PYO3_PYTHON="$python" \
    "$UV_PROJECT_ENVIRONMENT/bin/maturin" build --release --locked \
    --out "$wheel_directory" >&2
build_finished_ns="$(date +%s%N)"
shopt -s nullglob
wheels=("$wheel_directory"/torch_rs-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "expected one newly built native wheel, got ${#wheels[@]}" >&2
    exit 1
fi
uv --no-config pip install --python "$python" --force-reinstall --no-deps \
    "${wheels[0]}" >&2
TORCH_RS_VERIFY_VIRTUALENV="$UV_PROJECT_ENVIRONMENT" \
    "$python" .github/scripts/verify_native_extension.py >&2
setup_finished_ns="$(date +%s%N)"
if [[ "${1:-}" == --setup-only ]]; then
    echo "setup complete; no evaluation or score produced; wheel: ${wheels[0]}" >&2
    exit 0
fi
exec "$python" scripts/evaluate_torch_compile_default.py --wheel "${wheels[0]}" \
    --build-identity "$build_identity" \
    --setup-timestamps "$setup_started_ns,$dependencies_finished_ns,$build_finished_ns,$setup_finished_ns" "$@"
