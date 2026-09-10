#!/usr/bin/env python3
"""Capture a fresh local release wheel build for depth-concat/CUDA diagnostics.

Requires the locked dependencies in a real canonical .venv and a populated
worktree-local CARGO_HOME. Refuses dirty source unless --allow-dirty is explicit.
It does not commit, stage, or overwrite any previous capture.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluate_cuda_math import sha256, source_provenance

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    assert Path(sys.prefix).resolve() == ROOT / ".venv"
    assert not (ROOT / ".venv").is_symlink()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT / "target"), "capture under target only"
    assert not output.exists(), "use a fresh capture directory"
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status and not args.allow_dirty:
        parser.error("clean committed candidate required; --allow-dirty records a precommit diagnostic")
    before = source_provenance()
    for key in ("CARGO_HOME", "UV_CACHE_DIR", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
        assert Path(os.environ[key]).resolve().is_relative_to(ROOT), key
    output.mkdir(parents=True)
    env = {**os.environ, "CARGO_TARGET_DIR": str(output / "build"),
           "VIRTUAL_ENV": str(ROOT / ".venv"), "PYO3_PYTHON": sys.executable}
    commands = []

    def run(command, name):
        started, tick = stamp(), time.perf_counter()
        log = output / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        commands.append({"command": command, "started_at": started, "ended_at": stamp(),
                         "wall_seconds": time.perf_counter() - tick, "exit_status": result.returncode,
                         "log": str(log), "log_sha256": sha256(log)})
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result.check_returncode()

    command = [str(ROOT / ".venv/bin/maturin"), "build", "--release", "--locked", "--offline",
               "--out", str(output / "wheels")]
    run(command, "build")
    wheels = list((output / "wheels").glob("*.whl"))
    assert len(wheels) == 1
    run(["uv", "pip", "install", "--python", sys.executable, "--force-reinstall", "--no-deps",
         str(wheels[0])], "install")
    import torch_rs
    extension = Path(torch_rs._C.__file__).resolve()
    assert extension.is_relative_to(ROOT / ".venv")
    # The unchanged CUDA math evaluator requires the checkout's source package;
    # stack validators require the installed wheel. Give both identical bytes.
    source_extension = ROOT / "python/torch_rs" / extension.name
    shutil.copyfile(extension, source_extension)
    assert sha256(source_extension) == sha256(extension)
    assert before == source_provenance()
    assert status == subprocess.check_output(["git", "status", "--porcelain"], text=True)
    receipt = {
        **before, "git_status_before_build": status, "clean_checkout": not bool(status),
        "measurement_kind": "precommit-diagnostic" if status else "clean-code",
        "build_command": command, "commands": commands,
        "source_unchanged_during_build": True,
        "extension_path": str(extension), "extension_sha256": sha256(extension),
        "source_extension_path": str(source_extension),
        "wheel_path": str(wheels[0]), "wheel_sha256": sha256(wheels[0]),
        "python": sys.version, "python_executable": sys.executable,
        "interpreter_path": str(Path(sys.executable).resolve()),
        "interpreter_sha256": sha256(Path(sys.executable).resolve()),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "cargo": subprocess.check_output(["cargo", "-V"], text=True),
        "rustc_path": subprocess.check_output(["rustup", "which", "rustc"], text=True).strip(),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "nvcc_used": False, "cuda_compilation": "driver JIT of embedded PTX",
        "profile": {"name": "release", "lto": "thin", "codegen_units": 1, "features": ["extension-module"]},
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "environment": {key: env.get(key) for key in ("CARGO_HOME", "CARGO_TARGET_DIR", "PYO3_PYTHON",
                         "VIRTUAL_ENV", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "UV_CACHE_DIR",
                         "CUDA_VISIBLE_DEVICES", "TORCH_RS_CUDART", "RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS")},
        "cache_state": "empty build target; populated local Cargo registry and locked .venv; offline build",
    }
    receipt["rustc_sha256"] = sha256(receipt["rustc_path"])
    (output / "build-record.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
