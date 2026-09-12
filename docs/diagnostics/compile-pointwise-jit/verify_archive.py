"""Verify historical evidence without extracting or executing archive members."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "history-manifest.json").read_text())
    if manifest["schema"] != 1 or manifest["archive"]["path"] != "history.tar.gz":
        raise ValueError("unsupported evidence archive manifest")
    archive = directory / "history.tar.gz"
    data = archive.read_bytes()
    if (len(data) != manifest["archive"]["size"] or
            hashlib.sha256(data).hexdigest() != manifest["archive"]["sha256"]):
        raise ValueError("evidence archive hash or size mismatch")
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts or
                    str(path) != member.name or member.name in seen or
                    member.name not in manifest["files"]):
                raise ValueError("unexpected or unsafe evidence archive member")
            expected = manifest["files"][member.name]
            with bundle.extractfile(member) as stream:
                data = stream.read()
            if (len(data) != expected["size"] or
                    hashlib.sha256(data).hexdigest() != expected["sha256"]):
                raise ValueError("evidence member hash or size mismatch: " + member.name)
            seen.add(member.name)
    if seen != set(manifest["files"]):
        raise ValueError("evidence archive is missing manifest members")
    return manifest


if __name__ == "__main__":
    result = verify(Path(__file__).resolve().parent)
    print(f"Verified {len(result['files'])} unchanged evidence files from {result['source_commit']}")
