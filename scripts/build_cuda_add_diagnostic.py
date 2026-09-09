#!/usr/bin/env python3
"""Build an isolated source export for diagnose_cuda_add.py without changing Git.

--revision HEAD exports a committed baseline. Without --revision, overlay the
current tracked and unignored source files onto that export. The latter
is an immutable, hashed source snapshot, NOT a claim that dirty code is HEAD.
No commits, Git index/object writes, or paths outside this worktree are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    return subprocess.check_output(args, cwd=ROOT, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, **kwargs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if not args.name.replace("-", "").isalnum():
        parser.error("name must contain only letters, digits and hyphens")
    destination = ROOT / "target" / "cuda-add-diagnostic" / args.name
    destination.resolve().relative_to(ROOT)
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "source"
    source.mkdir()
    revision = run(["git", "rev-parse", args.revision or "HEAD"], text=True).strip()
    archive = run(["git", "archive", revision])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    status = run(["git", "status", "--short"], text=True)
    patch = run(["git", "diff", "HEAD", "--binary"])
    if not args.revision:
        paths = run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]).decode().split("\0")[:-1]
        for name in paths:
            # Burner owns these artifacts; they are irrelevant to this build.
            if name in ("docs/burner-evaluation-history.json", "docs/burner-evaluation-progress.svg"):
                continue
            original = ROOT / name
            target = source / name
            if original.is_file():
                original.resolve().relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
            elif target.exists():
                target.unlink()
    files = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()}
    manifest = json.dumps(files, sort_keys=True).encode()
    env = os.environ.copy()
    env.update({"UV_CACHE_DIR": str(ROOT / "target/uv-cache"),
                "UV_PYTHON_INSTALL_DIR": str(ROOT / "target/uv-python"),
                "UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv"),
                "CARGO_HOME": str(ROOT / "target/cargo-home"),
                "CARGO_TARGET_DIR": str(destination / "build"),
                "TMPDIR": str(ROOT / "target/tmp"),
                "CUDA_CACHE_PATH": str(ROOT / "target/cuda-cache"),
                "PYTHONDONTWRITEBYTECODE": "1", "RUSTUP_TOOLCHAIN": "1.92.0",
                "VIRTUAL_ENV": str(ROOT / ".venv"), "PYO3_PYTHON": str(ROOT / ".venv/bin/python")})
    env.pop("CONDA_PREFIX", None)
    env.pop("PYTHONPATH", None)
    durations = {}
    commands = [
        ("dependency_setup", ["uv", "sync", "--locked", "--no-install-project", "--group", "dev", "--group", "reference"], ROOT),
        ("build", [str(ROOT / ".venv/bin/maturin"), "build", "--release", "--locked", "--out", str(destination / "wheels")], source),
    ]
    for label, command, cwd in commands:
        start = time.perf_counter()
        with (destination / f"{label}.log").open("w") as log:
            subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        durations[label] = time.perf_counter() - start
    wheel, = (destination / "wheels").glob("*.whl")
    start = time.perf_counter()
    subprocess.run(["uv", "pip", "install", "--python", str(ROOT / ".venv/bin/python"),
                    "--force-reinstall", "--no-deps", str(wheel)], env=env, check=True)
    durations["wheel_install"] = time.perf_counter() - start
    info = json.loads(subprocess.check_output([str(ROOT / ".venv/bin/python"), "-c",
        'import torch_rs,torch_rs.torch_rs as n,json; print(json.dumps([torch_rs.__file__,n.__file__]))'], env=env, text=True))
    native = Path(info[1]).resolve()
    native.relative_to(ROOT)
    with zipfile.ZipFile(wheel) as zipped:
        member, = [name for name in zipped.namelist() if name.endswith(".so")]
        assert hashlib.sha256(zipped.read(member)).hexdigest() == digest(native)
    # Recheck the export after building. Builds may write only to the separate
    # build directory, never alter the recorded source inputs.
    assert all(digest(source / name) == value for name, value in files.items())
    record = {"source_root": str(source), "source_files": files,
              "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
              "base_commit": revision, "measured_code_commit": revision if args.revision else None,
              "source_kind": "clean commit export" if args.revision else "immutable current-source export; uncommitted overlay, not exact HEAD",
              "origin_status": status, "origin_patch_sha256": hashlib.sha256(patch).hexdigest(),
              "source_matches_commit": bool(args.revision), "durations_seconds": durations,
              "build_cache": "empty per-export Cargo target; shared worktree Cargo/uv download caches warm",
              "profile": "release, thin LTO, codegen-units=1, extension-module, locked dependencies",
              "wheel": str(wheel), "wheel_sha256": digest(wheel),
              "installed_package": info[0], "installed_native": str(native),
              "installed_native_sha256": digest(native), "build_target": env["CARGO_TARGET_DIR"]}
    (destination / "origin.patch").write_bytes(patch)
    (destination / "build-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(destination / "build-record.json")


if __name__ == "__main__":
    main()
