"""Seal executed worktree-local evidence, including binary bytes, before handoff.

Usage: python3 archive.py NAME PATH [PATH ...]
Every input must be inside this worktree. Existing archives are never replaced.
"""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

PART_BYTES = 64 * 1024 * 1024

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    name, *paths = sys.argv[1:]
    assert name and "/" not in name
    destination = HERE / f"{name}.tar.gz"
    manifest_path = HERE / f"{name}-manifest.json"
    assert not destination.exists() and not manifest_path.exists()
    files = {}
    for arg in paths:
        path = (ROOT / arg).resolve(strict=True)
        assert path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / ".burner")
        members = path.rglob("*") if path.is_dir() else [path]
        for member in members:
            if member.is_file():
                assert member.resolve().is_relative_to(ROOT)
                relative = member.relative_to(ROOT).as_posix()
                if member.name == "receipt.json":
                    assert "returncode" in json.loads(member.read_text()), f"Pending command: {relative}"
                files[relative] = member
    assert files
    inventory = []
    with tarfile.open(destination, "x:gz") as archive:
        for relative, path in sorted(files.items()):
            data = path.read_bytes()
            inventory.append({"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            archive.add(path, arcname=relative, recursive=False)
    with tarfile.open(destination) as archive:
        assert archive.getnames() == [row["path"] for row in inventory]
        for row in inventory:
            data = archive.extractfile(row["path"]).read()
            assert len(data) == row["size"] and hashlib.sha256(data).hexdigest() == row["sha256"]
    manifest = {"archive": destination.name, "size": destination.stat().st_size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "members": inventory,
                "scope": "Executed inner commands and artifacts; no future outer-controller receipt"}
    from capture import PINS
    for path, digest in PINS.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    manifest["protocolBindings"] = PINS
    if destination.stat().st_size > PART_BYTES:
        # Keep required binary evidence in Git without exceeding GitHub's
        # per-file size limit. Concatenating parts recovers the verified gzip.
        parts = []
        with destination.open("rb") as stream:
            while data := stream.read(PART_BYTES):
                part = HERE / f"{destination.name}.part-{len(parts):03d}"
                with part.open("xb") as output:
                    output.write(data)
                parts.append({"path": part.name, "size": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()})
        manifest["parts"] = parts
        reconstructed = hashlib.sha256()
        for part in parts:
            data = (HERE / part["path"]).read_bytes()
            assert len(data) == part["size"] and hashlib.sha256(data).hexdigest() == part["sha256"]
            reconstructed.update(data)
        assert reconstructed.hexdigest() == manifest["sha256"]
        destination.unlink()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Sealed and re-read {len(inventory)} files: {destination.name}")


if __name__ == "__main__":
    main()
