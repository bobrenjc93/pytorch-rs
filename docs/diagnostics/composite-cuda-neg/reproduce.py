#!/usr/bin/env python3
"""Build and run the unchanged six-case evaluator, retaining honest receipts.

Run from the repository root with its local Python 3.12 environment. The caller
must configure local CARGO_HOME, CARGO_TARGET_DIR, TMPDIR, CUDA_CACHE_PATH,
XDG_CACHE_HOME, PYO3_PYTHON and TORCH_RS_CUDART as in the validation guide.
The target directory must be absent for a fresh build. This script does not
commit or stage files; a dirty checkout is recorded explicitly, never as HEAD.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_cuda_math import source_provenance, sha256


def stamp():
    return datetime.now(timezone.utc).isoformat()


def capture(command, label):
    start, tick = stamp(), time.perf_counter()
    log = OUT / f"{label}.log"
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return {"command": command, "cwd": str(ROOT), "started_at": start,
            "ended_at": stamp(), "wall_seconds": time.perf_counter() - tick,
            "exit_status": result.returncode, "log": log.name,
            "log_sha256": sha256(log)}


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def main():
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    assert Path.cwd().resolve() == ROOT
    for key in ("CARGO_HOME", "CARGO_TARGET_DIR", "TMPDIR", "CUDA_CACHE_PATH",
                "XDG_CACHE_HOME", "PYO3_PYTHON", "TORCH_RS_CUDART"):
        assert Path(os.environ[key]).resolve().is_relative_to(ROOT), key
    assert Path(sys.executable).resolve().is_relative_to(ROOT)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    target = Path(os.environ["CARGO_TARGET_DIR"])
    assert not target.exists(), "fresh build requires an absent CARGO_TARGET_DIR"
    before = source_provenance()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    command = ["cargo", "build", "--locked", "--release", "--features", "extension-module"]
    build = capture(command, "build")
    assert build["exit_status"] == 0, build
    assert source_provenance() == before
    extension = ROOT / "python/torch_rs/torch_rs.abi3.so"
    shutil.copyfile(target / "release/libpytorch_rs.so", extension)
    receipt = {**before, **build, "build_command": " ".join(command),
               "source_unchanged_during_build": True, "git_status_before_build": status,
               "clean_checkout": not bool(status),
               "measurement_kind": "clean-code" if not status else "uncommitted-integration-validation",
               "extension_path": str(extension), "extension_sha256": sha256(extension),
               "rustc": subprocess.check_output([os.environ["RUSTC"], "--version"], text=True).strip(),
               "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
               "nvcc": "unused; embedded PTX JIT-compiled by NVIDIA driver",
               "available_nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
               "python_executable": str(Path(sys.executable).resolve()), "python": sys.version,
               "environment": {key: os.environ.get(key) for key in (
                   "CARGO_HOME", "CARGO_TARGET_DIR", "RUSTC", "RUSTDOC", "PYO3_PYTHON",
                   "TMPDIR", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "TORCH_RS_CUDART",
                   "CUDA_VISIBLE_DEVICES", "PATH")},
               "cache_state": "empty Cargo build target; previously fetched Cargo registry; local Python environment; evaluator allocates a new temporary CUDA JIT cache",
               "profile": {"name": "release", "lto": "thin", "codegen_units": 1,
                           "features": ["extension-module"], "pyo3_abi": "abi3-py310"}}
    write("build-record.json", receipt)
    run = capture([sys.executable, "scripts/evaluate_cuda_math.py",
                   "--seed", "8503945240872567646", "--seed", "8613321571747136749",
                   "--seed", "4480763905421893394",
                   "--build-record", str(OUT / "build-record.json"),
                   "--output", str(OUT / "evaluation.json")], "run")
    run.update({"source_before": before, "source_after": source_provenance(),
                "report_sha256": sha256(OUT / "evaluation.json"),
                "build_record_sha256": sha256(OUT / "build-record.json"),
                "evaluator_sha256": sha256(ROOT / "scripts/evaluate_cuda_math.py"),
                "matrix_sha256": sha256(ROOT / "docs/hardware-heterogeneity-matrix-v1.json")})
    write("run-record.json", run)
    assert run["exit_status"] == 0 and run["source_after"] == before
    print(json.dumps(json.loads((OUT / "evaluation.json").read_text())["accounting"], indent=2))


if __name__ == "__main__":
    main()
