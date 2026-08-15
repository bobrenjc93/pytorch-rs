#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

cd "$repository_root"

# Do not let an activated environment, source-tree import path, or optimized
# interpreter select stale code or disable assertions. Exact paths are set
# explicitly where needed.
unset \
    CONDA_PREFIX \
    CARGO_BUILD_RUSTC \
    CARGO_BUILD_RUSTC_WRAPPER \
    CARGO_BUILD_RUSTFLAGS \
    CARGO_BUILD_TARGET \
    CARGO_ENCODED_RUSTFLAGS \
    RUSTC \
    RUSTC_WORKSPACE_WRAPPER \
    RUSTC_WRAPPER \
    RUSTDOC \
    RUSTDOCFLAGS \
    RUSTFLAGS \
    RUSTUP_TOOLCHAIN \
    TAR_OPTIONS \
    VIRTUAL_ENV \
    PYO3_PYTHON

# Git and Python both expose behavior-changing settings through prefixed
# environment variables. Clear them before selecting the exact inputs below.
for environment_name in "${!GIT_@}" "${!PYTHON@}"; do
    unset "$environment_name"
done
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1

# uv exposes most command-line settings through UV_* environment variables.
# Clear all of them before setting the few paths controlled by this command.
for environment_name in "${!UV_@}"; do
    unset "$environment_name"
done

mkdir -p "$repository_root/target"
run_directory="$(mktemp -d "$repository_root/target/exact-head-run.XXXXXX")"
checkout="$run_directory/checkout"
bare_repository="$run_directory/repository.git"
head_tree_manifest="$run_directory/head-tree"
extracted_tree_manifest="$run_directory/extracted-tree"
virtualenv="$checkout/.venv"
python="$virtualenv/bin/python"
maturin="$virtualenv/bin/maturin"
wheel_directory="$run_directory/wheels"

cleanup() {
    cd "$repository_root"
    rm -rf -- "$run_directory"
}
trap cleanup EXIT

# Cargo walks from the build directory to the filesystem root and merges every
# parent .cargo/config file it finds. All artifacts must remain in this
# worktree, so reject parent configuration instead of placing the checkout in
# an external temporary directory.
cargo_config_parent="$(dirname "$checkout")"
while :; do
    for cargo_config_name in .cargo/config .cargo/config.toml; do
        cargo_config="$cargo_config_parent/$cargo_config_name"
        if [[ -e "$cargo_config" || -L "$cargo_config" ]]; then
            echo "refusing parent Cargo configuration outside exact HEAD: $cargo_config" >&2
            exit 1
        fi
    done
    if [[ "$cargo_config_parent" == / ]]; then
        break
    fi
    cargo_config_parent="$(dirname "$cargo_config_parent")"
done

mkdir -p "$checkout" "$wheel_directory"
head_commit="$(git -C "$repository_root" rev-parse --verify 'HEAD^{commit}')"
git -c core.attributesFile=/dev/null clone \
    --bare \
    --no-hardlinks \
    --no-tags \
    "$repository_root" \
    "$bare_repository"
git --git-dir="$bare_repository" cat-file -e "$head_commit^{commit}"
git --git-dir="$bare_repository" ls-tree -rz --full-tree "$head_commit" \
    > "$head_tree_manifest"
git -c core.attributesFile=/dev/null --git-dir="$bare_repository" \
    archive --format=tar "$head_commit" | tar -x -C "$checkout"

expected_file_count=0
while IFS= read -r -d '' tree_entry; do
    metadata="${tree_entry%%$'\t'*}"
    relative_path="${tree_entry#*$'\t'}"
    read -r expected_mode expected_type expected_object <<< "$metadata"
    destination="$checkout/$relative_path"
    expected_file_count=$((expected_file_count + 1))

    case "$expected_mode:$expected_type" in
        100644:blob|100755:blob)
            if [[ ! -f "$destination" || -L "$destination" ]]; then
                echo "exact-HEAD export is missing regular file: $relative_path" >&2
                exit 1
            fi
            if [[ "$expected_mode" == 100755 && ! -x "$destination" ]]; then
                echo "exact-HEAD export lost executable mode: $relative_path" >&2
                exit 1
            fi
            if [[ "$expected_mode" == 100644 && -x "$destination" ]]; then
                echo "exact-HEAD export added executable mode: $relative_path" >&2
                exit 1
            fi
            actual_object="$(
                git --git-dir="$bare_repository" hash-object \
                    --no-filters -- "$destination"
            )"
            ;;
        120000:blob)
            if [[ ! -L "$destination" ]]; then
                echo "exact-HEAD export is missing symbolic link: $relative_path" >&2
                exit 1
            fi
            actual_object="$(
                printf '%s' "$(readlink "$destination")" |
                    git --git-dir="$bare_repository" hash-object --stdin
            )"
            ;;
        *)
            echo "unsupported exact-HEAD tree entry: $tree_entry" >&2
            exit 1
            ;;
    esac

    if [[ "$actual_object" != "$expected_object" ]]; then
        echo "exact-HEAD export content mismatch: $relative_path" >&2
        exit 1
    fi
done < "$head_tree_manifest"

find "$checkout" -mindepth 1 ! -type d -print0 > "$extracted_tree_manifest"
actual_file_count=0
while IFS= read -r -d '' _; do
    actual_file_count=$((actual_file_count + 1))
done < "$extracted_tree_manifest"
if [[ "$actual_file_count" -ne "$expected_file_count" ]]; then
    echo \
        "exact-HEAD export file count mismatch: expected $expected_file_count, got $actual_file_count" \
        >&2
    exit 1
fi
echo "testing exact HEAD $head_commit"
echo "verified $expected_file_count exact-HEAD files"

export CARGO_HOME="$run_directory/cargo-home"
export CARGO_TARGET_DIR="$run_directory/cargo-target"
export UV_CACHE_DIR="$repository_root/target/uv-cache"
export UV_PYTHON_INSTALL_DIR="$repository_root/target/uv-python"
export UV_PROJECT_ENVIRONMENT="$virtualenv"
export PYTHONNOUSERSITE=1

cd "$checkout"
uv --no-config venv --clear --python 3.12 "$virtualenv"
uv --no-config sync \
    --locked \
    --python "$python" \
    --no-install-project \
    --group dev \
    --group reference

toolchain_channel="$("$python" - <<'PY'
import re
import tomllib


with open("rust-toolchain.toml", "rb") as toolchain_file:
    channel = tomllib.load(toolchain_file)["toolchain"]["channel"]
if not isinstance(channel, str) or re.fullmatch(r"\d+\.\d+\.\d+", channel) is None:
    raise SystemExit(f"expected a pinned Rust release channel, got {channel!r}")
print(channel)
PY
)"
export RUSTUP_TOOLCHAIN="$toolchain_channel"
rustc_version="$(rustc --version)"
cargo_version="$(cargo --version)"
if [[ "${rustc_version#rustc }" != "$toolchain_channel "* ]]; then
    echo "expected rustc $toolchain_channel, got $rustc_version" >&2
    exit 1
fi
if [[ "${cargo_version#cargo }" != "$toolchain_channel "* ]]; then
    echo "expected cargo $toolchain_channel, got $cargo_version" >&2
    exit 1
fi
echo "verified Rust toolchain: $rustc_version; $cargo_version"

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

uv --no-config pip install \
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
if sys.flags.dev_mode:
    raise SystemExit("expected Python development mode to be disabled")
if sys.warnoptions:
    raise SystemExit(f"expected default Python warning policy, got {sys.warnoptions!r}")

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
