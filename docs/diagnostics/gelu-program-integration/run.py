"""Retain development command streams and source bindings inside the worktree.

Usage: python3 .../run.py LABEL COMMAND [ARG ...]
This is a command recorder, not an evaluator or a timing consumer.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "target/gelu-program-integration"


def main():
    label, *command = sys.argv[1:]
    folder = OUT / f"{time.time_ns()}-{label}"
    folder.mkdir(parents=True)
    env = dict(os.environ)
    local = ROOT / "target"
    env.update(
        CARGO_HOME=str(local / "cargo-home"),
        CARGO_TARGET_DIR=str(local),
        UV_CACHE_DIR=str(local / "uv-cache"),
        UV_PYTHON_DOWNLOADS="never",
        TMPDIR=str(local / "tmp"),
        CUDA_CACHE_PATH=str(local / "cuda-cache"),
        TORCHINDUCTOR_CACHE_DIR=str(local / "inductor-cache"),
        TRITON_CACHE_DIR=str(local / "triton-cache"),
        XDG_CACHE_HOME=str(local / "cache"),
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYO3_PYTHON=str(ROOT / ".venv/bin/python"),
        PATH="/home/bobren/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:"
        + str(ROOT / ".venv/bin") + ":" + env["PATH"],
    )
    cuda = ROOT / ".venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    if cuda.is_dir():
        env.setdefault("TORCH_RS_CUDART", str(cuda / "libcudart.so.13"))
        env.setdefault("TORCH_RS_NVRTC", str(cuda / "libnvrtc.so.13"))
        env.setdefault("TEST_PROGRAM_IDENTITY_REAL_NVRTC", str(cuda / "libnvrtc.so.13"))
    for name in ("tmp", "cuda-cache", "inductor-cache", "triton-cache", "cache"):
        (local / name).mkdir(exist_ok=True)
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    (folder / "source.patch").write_bytes(git("diff", "HEAD", "--", ".", ":!.burner"))
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    sources = {}
    for name in paths:
        p = ROOT / name
        if name and not name.startswith((".burner/", "docs/diagnostics/")) and p.is_file():
            sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with tarfile.open(folder / "source.tar.gz", "w:gz") as archive:
        for name in sources:
            archive.add(ROOT / name, arcname=name)
        for tool in Path(__file__).parent.glob("*.py"):
            archive.add(tool, arcname=str(tool.relative_to(ROOT)))
    receipt = dict(command=command, cwd=str(ROOT), start_ns=time.time_ns(),
                   head=git("rev-parse", "HEAD").decode().strip(),
                   status=git("status", "--short").decode(), sources=sources,
                   environment={k: env[k] for k in env if k.endswith("_DIR") or k in (
                       "CARGO_HOME", "CARGO_TARGET_DIR", "CUDA_VISIBLE_DEVICES", "PATH", "PYO3_PYTHON",
                       "TORCH_RS_CUDART", "TORCH_RS_NVRTC", "TEST_PROGRAM_IDENTITY_REAL_NVRTC",
                       "TMPDIR", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        receipt["returncode"] = result.returncode
    except BaseException:
        receipt.update(returncode=None, failure=traceback.format_exc())
        raise
    finally:
        receipt["end_ns"] = time.time_ns()
        (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(folder.relative_to(ROOT), "exit", result.returncode, flush=True)
    for stream in ("stdout", "stderr"):
        lines = (folder / f"{stream}.log").read_text(errors="replace").splitlines()
        print("\n".join(lines[-35:]))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
