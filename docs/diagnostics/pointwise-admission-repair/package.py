"""Verify a real sdist and run its mandatory archive controls without Git.

Usage: python package.py SDIST OUTPUT_DIRECTORY
All inputs, extraction, temporary files and retained outputs stay in this worktree.
This checks package selection; it does not claim current-source build identity.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[3]
HISTORY = {
    "docs/diagnostics/compile-pointwise-jit/history.tar.gz",
    "docs/diagnostics/compile-pointwise-jit/history-later/history.tar.gz",
    "docs/diagnostics/compile-pointwise-tensor-madd/history.tar.gz",
}
LICENSES = {
    "LICENSE", "src/cuda/NOTICE.md",
    "docs/diagnostics/compile-gelu-vendor/notices/requested-CUDA11.8-LICENSE.txt",
    "docs/diagnostics/compile-gelu-vendor/notices/compiler-CUDA12.8-LICENSE.txt",
    "docs/diagnostics/compile-gelu-linked/notices/CUDA13-LICENSE.txt",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    package, output = (Path(arg).resolve() for arg in sys.argv[1:])
    assert package.is_relative_to(ROOT) and output.is_relative_to(ROOT)
    assert package.is_file() and not output.exists()
    output.mkdir(parents=True)
    tool = Path(__file__).read_bytes()
    (output / "package.py").write_bytes(tool)
    record = {"passed": False, "package": {"path": str(package),
              "size": package.stat().st_size, "sha256": sha(package.read_bytes())},
              "toolSha256": sha(tool), "scope": "Actual sdist selection and offline archive tests; no current-source build claim"}
    try:
        with tarfile.open(package) as archive:
            members = archive.getmembers()
            prefixes = {PurePosixPath(member.name).parts[0] for member in members}
            assert len(prefixes) == 1, prefixes
            prefix = prefixes.pop()
            files = {}
            inventory = []
            for member in members:
                path = PurePosixPath(member.name)
                assert not path.is_absolute() and ".." not in path.parts, member.name
                assert member.isfile() or member.isdir(), member.name
                if member.isfile():
                    relative = path.relative_to(prefix).as_posix()
                    assert relative not in files, relative
                    data = archive.extractfile(member).read()
                    files[relative] = member
                    inventory.append({"path": relative, "size": len(data), "sha256": sha(data)})
            record["members"] = inventory
            binary_archives = {name for name in files
                               if ".tar." in name or name.endswith((".tar", ".whl", ".zip", ".so", ".a", ".rlib", ".o"))}
            record["binaryArchives"] = sorted(binary_archives)
            assert binary_archives == HISTORY, sorted(binary_archives ^ HISTORY)
            required = {"Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md", *LICENSES}
            required.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*")
                            if path.is_file() and path.suffix in (".rs", ".ptx", ".cu", ".ll"))
            required.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "python").rglob("*.py"))
            record["requiredBuildInputs"] = sorted(required)
            assert required <= files.keys(), sorted(required - files.keys())
            immutable = LICENSES | HISTORY | {name for name in required if name.endswith((".ptx", ".cu", ".ll"))}
            for name in immutable:
                assert archive.extractfile(files[name]).read() == (ROOT / name).read_bytes(), name
            extracted = output / "extracted"
            extracted.mkdir()
            archive.extractall(extracted, filter="data")
        source = extracted / prefix
        temporary = output / "tmp"
        temporary.mkdir()
        command = [str(ROOT / ".venv/bin/python"), str(source / "tests/test_compile_evidence_archive.py"), "-v"]
        env = {**os.environ, "PATH": "", "TMPDIR": str(temporary),
               "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""}
        record["offline"] = {"command": command, "cwd": str(source),
                             "environment": {key: env[key] for key in ("PATH", "TMPDIR", "CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH")}}
        result = subprocess.run(command, cwd=source, env=env, capture_output=True)
        (output / "stdout.log").write_bytes(result.stdout)
        (output / "stderr.log").write_bytes(result.stderr)
        record["offline"]["returncode"] = result.returncode
        stderr = result.stderr.decode(errors="replace")
        count = re.search(r"Ran (\d+) tests?", stderr)
        skipped = re.search(r"skipped=(\d+)", stderr)
        record["offline"].update(tests=int(count[1]) if count else None,
                                skipped=int(skipped[1]) if skipped else 0)
        print(stderr, end="")
        assert result.returncode == 0, "Offline archive test failed; streams retained"
        assert count, "Missing unittest test count"
        record["passed"] = True
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Verified {len(record['members'])} packaged files and exactly three mandatory historical archives")


if __name__ == "__main__":
    main()
