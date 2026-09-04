from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TextIO

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 support.
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_VENV = (REPOSITORY_ROOT / ".venv").resolve()
MODULE_NAMES = ("torch_rs", "torch_rs.torch_rs")


def module_path(module: ModuleType) -> Path:
    spec = module.__spec__
    if spec is None or spec.origin is None:
        raise RuntimeError(f"{module.__name__} has no import origin")
    return Path(spec.origin).resolve(strict=True)


def require_inside_virtualenv(name: str, path: Path) -> None:
    try:
        path.relative_to(WORKSPACE_VENV)
    except ValueError as error:
        raise RuntimeError(
            f"{name} resolved outside the workspace virtualenv: {path}"
        ) from error


def resolved_origin(name: str) -> str:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return "<not found>"
        if spec.origin is None:
            return "<no origin>"
        return str(Path(spec.origin).resolve(strict=True))
    except Exception as error:  # Keep diagnostics available for broken imports.
        return f"<unresolved: {type(error).__name__}: {error}>"


def print_resolved_paths(*, file: TextIO = sys.stdout) -> None:
    print(f"interpreter: {sys.executable}", file=file)
    print(f"virtualenv: {Path(sys.prefix).resolve()}", file=file)
    for name in MODULE_NAMES:
        print(f"{name}: {resolved_origin(name)}", file=file)


def verify() -> str:
    if Path(sys.prefix).resolve() != WORKSPACE_VENV:
        raise RuntimeError(
            f"expected interpreter prefix {WORKSPACE_VENV}, "
            f"got {Path(sys.prefix).resolve()}"
        )

    package = importlib.import_module("torch_rs")
    native = importlib.import_module("torch_rs.torch_rs")
    package_path = module_path(package)
    native_path = module_path(native)
    require_inside_virtualenv("torch_rs", package_path)
    require_inside_virtualenv("torch_rs.torch_rs", native_path)

    package_spec = package.__spec__
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("torch_rs did not resolve as a package")

    native_spec = native.__spec__
    if native_spec is None or not isinstance(
        native_spec.loader, importlib.machinery.ExtensionFileLoader
    ):
        raise RuntimeError("torch_rs.torch_rs is not a native extension module")
    if not native_path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        raise RuntimeError(
            f"native module does not have a recognized ABI extension suffix: {native_path}"
        )
    abi3_suffixes = tuple(
        suffix
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if "abi3" in suffix
    )
    if abi3_suffixes and not native_path.name.endswith(abi3_suffixes):
        raise RuntimeError(f"native module is not using the stable ABI: {native_path}")

    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)["project"]
    expected_version = project["version"]
    versions = {
        f"installed distribution {project['name']}": importlib.metadata.version(
            project["name"]
        ),
        "torch_rs": getattr(package, "__version__", None),
        "torch_rs.torch_rs": getattr(native, "__version__", None),
    }
    mismatches = {
        name: version
        for name, version in versions.items()
        if version != expected_version
    }
    if mismatches:
        details = ", ".join(
            f"{name}={version!r}" for name, version in mismatches.items()
        )
        raise RuntimeError(
            f"versions do not match pyproject.toml ({expected_version!r}): {details}"
        )
    return expected_version


def main() -> int:
    try:
        version = verify()
    except Exception as error:
        print(f"native-extension provenance check failed: {error}", file=sys.stderr)
        print_resolved_paths(file=sys.stderr)
        return 1

    print(f"verified native-extension provenance for torch-rs {version}")
    print_resolved_paths()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
