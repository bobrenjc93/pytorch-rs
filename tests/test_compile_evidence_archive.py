"""Portable integrity checks for the checked-in historical compiler evidence."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


EVIDENCE = Path(__file__).resolve().parents[1] / "docs/diagnostics/compile-pointwise-jit"
SPEC = importlib.util.spec_from_file_location("verify_compile_evidence", EVIDENCE / "verify_archive.py")
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


class EvidenceArchive(unittest.TestCase):
    def fixture(self, directory, members=("capture/report.json",), listed=("capture/report.json",)):
        payload = b'{"valid":false,"original_failure":"preserved"}\n'
        path = directory / "history.tar.gz"
        with tarfile.open(path, "w:gz") as bundle:
            for name in members:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
        data = path.read_bytes()
        manifest = {
            "schema": 1, "source_commit": "test-fixture",
            "archive": {"path": path.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()},
            "files": {name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                      for name in listed},
        }
        (directory / "history-manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_committed_archive(self):
        manifest = ARCHIVE.verify(EVIDENCE)
        self.assertEqual(manifest["source_commit"], "ed6ad9eaaea17ee500f5464183c00984115e7fb3")
        self.assertEqual(len(manifest["files"]), 56)

    def test_committed_later_archive(self):
        manifest = ARCHIVE.verify(EVIDENCE / "history-later")
        self.assertEqual(manifest["source_commit"], "91fd9524fb15a8d46e7c165dd3c87de763826921")
        self.assertEqual(len(manifest["files"]), 56)

    def test_default_command_verifies_both_archives(self):
        result = subprocess.run([sys.executable, str(EVIDENCE / "verify_archive.py")],
                                capture_output=True, text=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        self.assertIn("56 unchanged evidence files from ed6ad9eaaea17ee500f5464183c00984115e7fb3", result.stdout)
        self.assertIn("56 unchanged evidence files from 91fd9524fb15a8d46e7c165dd3c87de763826921", result.stdout)

    def test_collection_requires_later_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            with self.assertRaises(FileNotFoundError):
                ARCHIVE.verify_all(directory)

    def test_archive_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            with (directory / "history.tar.gz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "archive hash or size mismatch"):
                ARCHIVE.verify(directory)

    def test_member_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.fixture(directory)
            manifest["files"]["capture/report.json"]["sha256"] = "0" * 64
            (directory / "history-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "member hash or size mismatch"):
                ARCHIVE.verify(directory)

    def test_missing_duplicate_and_unsafe_members_rejected(self):
        for members, listed in (
                ((), ("capture/report.json",)),
                (("capture/report.json",) * 2, ("capture/report.json",)),
                (("unexpected",), ("capture/report.json",)),
                (("../escape",), ("../escape",)),
                (("/absolute",), ("/absolute",))):
            with self.subTest(members=members), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.fixture(directory, members, listed)
                with self.assertRaises(ValueError):
                    ARCHIVE.verify(directory)


if __name__ == "__main__":
    unittest.main()
